from . import SentenceEvaluator
import logging
import os
import csv
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from ..readers import InputExample
import torch
import torch.nn.functional as F
from ..util import SiameseDistanceMetric, get_best_distance_threshold
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


def auc_ci_stratified_bootstrap(
    labels: np.ndarray,
    scores: np.ndarray,
    B: int = 5000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Stratified bootstrap CI for ROC-AUC.

    labels: 0/1 array
    scores: higher => more positive
    Returns: (auc, ci_low, ci_high)
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)

    # AUC undefined if single-class
    if np.unique(labels).size < 2:
        return float("nan"), float("nan"), float("nan")

    base_auc = float(roc_auc_score(labels, scores))

    rng = np.random.default_rng(seed)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    aucs = np.empty(B, dtype=float)
    for b in range(B):
        idx = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
        ])
        aucs[b] = roc_auc_score(labels[idx], scores[idx])

    lo = float(np.percentile(aucs, 100 * (alpha / 2)))
    hi = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
    return base_auc, lo, hi


def _to_numpy(x) -> np.ndarray:
    """Safely convert torch / list / np to np.ndarray."""
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class BinaryClassificationEvaluator(SentenceEvaluator):
    """
    Evaluates a model using embedding distances for binary classification (0/1 labels).

    - Accuracy: using best distance threshold (computed on this eval set)
    - AUC: ROC-AUC using score = -distance
    - AUC CI: 95% CI via stratified bootstrap on the eval set
    """

    def __init__(
        self,
        sentences1: List[str],
        sentences2: List[str],
        labels: List[int],
        distance_metric: SiameseDistanceMetric = SiameseDistanceMetric.EUCLIDEAN,
        normalize_embeddings: bool = False,
        name: str = "",
        batch_size: int = 32,
        margin: Optional[float] = None,
        show_progress_bar: bool = False,
        write_csv: bool = True,
        ci_bootstrap_B: int = 5000,
        ci_seed: int = 42,
        ci_alpha: float = 0.05,
        permutation_steps: int = 1000,
        bonferroni_correction: float = 1.0,
    ):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels
        self.distance_metric = distance_metric
        self.normalize_embeddings = normalize_embeddings
        self.margin = margin

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.labels)
        for lab in labels:
            assert lab in (0, 1)

        self.write_csv = write_csv
        self.name = name
        self.batch_size = batch_size

        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO
                or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        # CI config
        self.ci_bootstrap_B = int(ci_bootstrap_B)
        self.ci_seed = int(ci_seed)
        self.ci_alpha = float(ci_alpha)

        # P-value config
        self.permutation_steps = int(permutation_steps)
        self.bonferroni_correction = float(bonferroni_correction)

        self.csv_file = name + ".csv"
        self.csv_headers = [
            "epoch",
            "steps",
            "loss",
            "accuracy",
            "accuracy_thr",
            "auc",
            "auc_ci_low",
            "auc_ci_high",
            "auc_pval",
            "fpr",
            "tpr",
            "roc_thrs",
        ]

    def _calculate_p_value(self, y_true, y_score, base_auc, B=1000):
        if B <= 0:
            return float("nan")
        if not np.isfinite(base_auc):
            return float("nan")
        if base_auc <= 0.5:
            return 1.0  # one-sided test for AUC > 0.5

        rng = np.random.default_rng(self.ci_seed)
        y_true = np.asarray(y_true)
        count = 0
        target = base_auc - 0.5

        for _ in range(B):
            perm_auc = roc_auc_score(rng.permutation(y_true), y_score)
            if (perm_auc - 0.5) >= target:
                count += 1

        return (count + 1) / (B + 1)


    @classmethod
    def from_input_examples(cls, examples: List["InputExample"], **kwargs):
        sentences1, sentences2, scores = [], [], []
        for ex in examples:
            sentences1.append(ex.texts[0])
            sentences2.append(ex.texts[1])
            scores.append(ex.label)
        return cls(sentences1, sentences2, scores, **kwargs)

    def __call__(
        self,
        model,
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1
    ):
        scores, preds_and_trues = self.compute_metrics(model)

        loss = scores["loss"]
        acc = scores["accuracy"]
        best_acc_thr = scores["accuracy_thr"]
        auc = scores["auc"]
        auc_lo = scores["auc_ci_low"]
        auc_hi = scores["auc_ci_high"]
        auc_pval = scores["auc_pval"]
        fpr = scores["fpr"]
        tpr = scores["tpr"]
        roc_thresholds = scores["roc_thrs"]

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            write_header = not os.path.isfile(csv_path)

            with open(csv_path, newline="", mode="a" if not write_header else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(self.csv_headers)

                writer.writerow([
                    epoch,
                    steps,
                    f"{loss:.4f}",
                    f"{acc:.4f}",
                    f"{best_acc_thr:.6f}" if np.isfinite(best_acc_thr) else "nan",
                    f"{auc:.4f}",
                    f"{auc_lo:.4f}",
                    f"{auc_hi:.4f}",
                    f"{auc_pval:.4e}",
                    fpr.tolist() if hasattr(fpr, "tolist") else fpr,
                    tpr.tolist() if hasattr(tpr, "tolist") else tpr,
                    roc_thresholds.tolist() if hasattr(roc_thresholds, "tolist") else roc_thresholds,
                ])

        # keep your original contract
        return acc, preds_and_trues

    def compute_metrics(self, model):
        # Build unique sentence list (keep deterministic order)
        sentences = []
        seen = set()
        for s in (self.sentences1 + self.sentences2):
            if s not in seen:
                seen.add(s)
                sentences.append(s)

        embeddings = model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True
        )
        emb_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}
        embeddings1 = [emb_dict[s] for s in self.sentences1]
        embeddings2 = [emb_dict[s] for s in self.sentences2]

        distances = self.distance_metric(embeddings1, embeddings2)
        distances = _to_numpy(distances).astype(float)

        labels = np.asarray(self.labels).astype(int)

        # --- Accuracy (best threshold on this set) ---
        try:
            best_distance_threshold = float(get_best_distance_threshold(distances, labels))
            preds_best = (distances < best_distance_threshold).astype(int)
            acc_by_best = float((preds_best == labels).mean())
        except Exception:
            best_distance_threshold = float("nan")
            acc_by_best = float("nan")
            preds_best = np.zeros_like(labels)

        # --- Contrastive-style validation loss (robust to margin=None) ---
        t_labels = torch.from_numpy(labels)
        t_scores = torch.from_numpy(distances)

        if self.margin is None:
            # Only positive term (distance^2) when label=1
            valid_losses = 0.5 * (t_labels.float() * t_scores.pow(2))
        else:
            m = float(self.margin)
            valid_losses = 0.5 * (
                t_labels.float() * t_scores.pow(2)
                + (1 - t_labels).float() * F.relu(m - t_scores).pow(2)
            )
        loss_mean = float(valid_losses.mean().item()) if valid_losses.numel() > 0 else float("nan")

        # --- preds_and_trues (based on best threshold) ---
        preds_and_trues = [(int(p), int(y)) for p, y in zip(preds_best.tolist(), labels.tolist())]

        # --- AUC (+ CI) ---
        # Score must be higher for positive class -> use -distance
        y_score = -distances

        try:
            fpr, tpr, roc_thresholds = roc_curve(labels, y_score)
            AUC = float(roc_auc_score(labels, y_score))
            AUC, auc_lo, auc_hi = auc_ci_stratified_bootstrap(
                labels,
                y_score,
                B=self.ci_bootstrap_B,
                alpha=self.ci_alpha,
                seed=self.ci_seed
            )
            
            # P-value calculation
            raw_p = self._calculate_p_value(labels, y_score, AUC, B=self.permutation_steps)
            # Bonferroni correction (capped at 1.0)
            auc_pval = min(1.0, raw_p * self.bonferroni_correction)
        except ValueError:
            fpr = np.array([])
            tpr = np.array([])
            roc_thresholds = np.array([])
            AUC = float("nan")
            auc_lo = float("nan")
            auc_hi = float("nan")
            auc_pval = float("nan")

        # --- Embedding Norms ---
        norms = np.linalg.norm(embeddings, axis=1)
        norm_min = float(norms.min())
        norm_mean = float(norms.mean())
        norm_max = float(norms.max())

        print(
            f"{self.name:<5} "
            f"Loss = {loss_mean:>8.4f}   "
            f"Accuracy = {acc_by_best:>6.4f} (Threshold: {best_distance_threshold:>8.4f})   "
            f"AUC = {AUC:>6.4f}   "
            f"95% CI = [{auc_lo:>6.4f}, {auc_hi:>6.4f}]   "
            f"P-val = {auc_pval:.4e}"
        )

        print(f"{self.name:<5} Norms: Min={norm_min:.4f} Mean={norm_mean:.4f} Max={norm_max:.4f}   ")

        output_scores: Dict[str, Any] = {
            "loss": loss_mean,
            "accuracy": acc_by_best,
            "accuracy_thr": best_distance_threshold,
            "auc": AUC,
            "auc_ci_low": auc_lo,
            "auc_ci_high": auc_hi,
            "auc_pval": auc_pval,
            "fpr": fpr,
            "tpr": tpr,
            "roc_thrs": roc_thresholds,
            "norm_min": norm_min,
            "norm_mean": norm_mean,
            "norm_max": norm_max
        }

        return output_scores, preds_and_trues
