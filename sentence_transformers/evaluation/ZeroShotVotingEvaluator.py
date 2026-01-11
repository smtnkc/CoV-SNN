from . import SentenceEvaluator
import logging
import os
import csv
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from ..util import SiameseDistanceMetric
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


def auc_ci_stratified_bootstrap(
    labels: np.ndarray,
    scores: np.ndarray,
    B: int = 5000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float, float]:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)

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


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def pairwise_euclidean(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    # (n,d) vs (m,d) -> (n,m)
    X = np.asarray(X, dtype=np.float32)
    A = np.asarray(A, dtype=np.float32)
    x2 = np.sum(X * X, axis=1, keepdims=True)          # (n,1)
    a2 = np.sum(A * A, axis=1, keepdims=True).T        # (1,m)
    cross = X @ A.T                                     # (n,m)
    d2 = np.maximum(x2 + a2 - 2.0 * cross, 0.0)
    return np.sqrt(d2)


def pairwise_manhattan(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    A = np.asarray(A, dtype=np.float32)
    return np.sum(np.abs(X[:, None, :] - A[None, :, :]), axis=2)


def pairwise_cosine_distance(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    A = np.asarray(A, dtype=np.float32)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    sim = Xn @ An.T
    return 1.0 - sim


class ZeroShotVotingEvaluator(SentenceEvaluator):
    """
    BinaryClassificationEvaluator-style zero-shot voting evaluator.

    Key requirement from you:
      - remove normalize_metric
      - computations are driven by self.distance_metric (a callable like SiameseDistanceMetric.EUCLIDEAN)

    Important:
      SiameseDistanceMetric.* callables are typically "paired" metrics (1D output).
      For voting we need an (n_samples x n_anchors) matrix. Therefore we compute pairwise
      distances using the selected metric semantics, but we select WHICH pairwise routine
      strictly based on the callable identity (self.distance_metric == SiameseDistanceMetric.EUCLIDEAN, etc).
    """

    def __init__(
        self,
        anchors: List[str],
        positives: List[str],
        negatives: List[str],
        distance_metric: SiameseDistanceMetric = SiameseDistanceMetric.EUCLIDEAN,
        normalize_embeddings: bool = False,
        name: str = "Zero",
        batch_size: int = 32,
        margin: Optional[float] = 2.0,
        show_progress_bar: bool = False,
        write_csv: bool = True,
        threshold_search_steps: int = 100,
        ci_bootstrap_B: int = 5000,
        ci_seed: int = 42,
        ci_alpha: float = 0.05,
        permutation_steps: int = 1000,
        bonferroni_correction: float = 1.0,
    ):
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives

        self.distance_metric = distance_metric  # callable
        self.normalize_embeddings = normalize_embeddings
        self.name = name
        self.batch_size = int(batch_size)
        self.margin = margin

        self.write_csv = write_csv
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO
                or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.threshold_search_steps = int(threshold_search_steps)

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
            "majority_cutoff",
            "fpr",
            "tpr",
            "roc_thrs",
        ]

    def _calculate_p_value(self, y_true, y_score, base_auc, B=1000):
        if B <= 0 or not np.isfinite(base_auc):
            return float("nan")
        # one-sided test: AUC > 0.5
        if base_auc <= 0.5:
            return 1.0

        rng = np.random.default_rng(self.ci_seed)
        y_true = np.asarray(y_true)
        target = base_auc - 0.5

        count = 0
        for _ in range(B):
            perm_auc = roc_auc_score(rng.permutation(y_true), y_score)
            if (perm_auc - 0.5) >= target:
                count += 1

        return (count + 1) / (B + 1)


    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1):
        scores = self.compute_metrics(model)

        loss_val = scores["loss"]
        acc_val = scores["accuracy"]
        best_acc_thr = scores["accuracy_thr"]
        auc_val = scores["auc"]
        auc_lo = scores["auc_ci_low"]
        auc_hi = scores["auc_ci_high"]
        auc_pval = scores["auc_pval"]
        majority_cutoff = scores["majority_cutoff"]
        fpr = scores["fpr"]
        tpr = scores["tpr"]
        roc_thresholds = scores["roc_thrs"]

        print(
            f"{self.name:<5} "
            f"Loss = {loss_val:>8.4f}   "
            f"Accuracy = {acc_val:>6.4f} (Threshold: {best_acc_thr:>8.4f})   "
            f"AUC = {auc_val:>6.4f}   "
            f"95% CI = [{auc_lo:>6.4f}, {auc_hi:>6.4f}]   "
            f"P-val = {auc_pval:.4e}   "
            f"Majority Cutoff = {majority_cutoff:>8.4f}"
        )
    
        print(f"{self.name:<5} Norms: Min={scores.get('norm_min', 0):.4f} "
              f"Mean={scores.get('norm_mean', 0):.4f} Max={scores.get('norm_max', 0):.4f}")

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            _ensure_dir(os.path.dirname(csv_path))

            write_header = not os.path.isfile(csv_path)
            with open(csv_path, newline="", mode="a" if not write_header else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(self.csv_headers)

                writer.writerow([
                    epoch,
                    steps,
                    f"{loss_val:.4f}",
                    f"{acc_val:.4f}",
                    f"{best_acc_thr:.6f}" if np.isfinite(best_acc_thr) else "nan",
                    f"{auc_val:.4f}",
                    f"{auc_lo:.4f}",
                    f"{auc_hi:.4f}",
                    f"{auc_pval:.4e}",
                    f"{majority_cutoff:.3f}",
                    fpr.tolist() if hasattr(fpr, "tolist") else fpr,
                    tpr.tolist() if hasattr(tpr, "tolist") else tpr,
                    roc_thresholds.tolist() if hasattr(roc_thresholds, "tolist") else roc_thresholds,
                ])

        return float(acc_val)

    # ---- core: compute pairwise matrix based on the callable ----
    def _pairwise(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        """
        Return (len(X), len(A)) distance matrix consistent with self.distance_metric.
        """
        if self.distance_metric == SiameseDistanceMetric.EUCLIDEAN:
            return pairwise_euclidean(X, A)
        if self.distance_metric == SiameseDistanceMetric.MANHATTAN:
            return pairwise_manhattan(X, A)
        if self.distance_metric == SiameseDistanceMetric.COSINE:
            return pairwise_cosine_distance(X, A)

        # Fallback: try to use the callable on all pairs (may be slow)
        # and may fail if the callable is paired-only.
        # We'll implement a safe slow fallback that works for ANY callable
        # that can accept two 1D vectors and return a scalar distance.
        X = np.asarray(X, dtype=np.float32)
        A = np.asarray(A, dtype=np.float32)
        n, m = X.shape[0], A.shape[0]
        D = np.empty((n, m), dtype=np.float32)
        for i in range(n):
            # build paired arrays by repeating X[i]
            Xi = np.repeat(X[i:i+1], m, axis=0)  # (m,d)
            # many SBERT metrics accept (np.ndarray, np.ndarray) and return (m,)
            d = self.distance_metric(Xi, A)
            d = np.asarray(d, dtype=np.float32)
            if d.ndim != 1 or d.shape[0] != m:
                raise ValueError(
                    "Distance_metric fallback expected 1D distances of shape (m,), "
                    f"got {d.shape}. Provide one of SiameseDistanceMetric.EUCLIDEAN/MANHATTAN/COSINE."
                )
            D[i, :] = d
        return D

    def compute_metrics(self, model) -> Dict[str, Any]:
        model.eval()

        anchor_emb = model.encode(
            self.anchors,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=self.show_progress_bar,
        )
        pos_emb = model.encode(
            self.positives,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=self.show_progress_bar,
        )
        neg_emb = model.encode(
            self.negatives,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=self.show_progress_bar,
        )

        # Pairwise distances (n_samples x n_anchors)
        d_pos = self._pairwise(pos_emb, anchor_emb)
        d_neg = self._pairwise(neg_emb, anchor_emb)

        # Loss (reporting)
        loss_pos = float(np.mean(d_pos ** 2))
        if self.margin is None:
            loss_neg = 0.0
        else:
            m = float(self.margin)
            loss_neg = float(np.mean(np.maximum(0.0, m - d_neg) ** 2))
        loss_val = 0.5 * (loss_pos + loss_neg)

        # AUC (+ CI): score = -mean distance to anchors
        pos_scores = -d_pos.mean(axis=1)
        neg_scores = -d_neg.mean(axis=1)

        y_true = np.concatenate([
            np.ones(len(pos_scores), dtype=int),
            np.zeros(len(neg_scores), dtype=int),
        ])
        y_score = np.concatenate([pos_scores, neg_scores]).astype(float)

        try:
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
            auc_val, auc_lo, auc_hi = auc_ci_stratified_bootstrap(
                y_true,
                y_score,
                B=self.ci_bootstrap_B,
                alpha=self.ci_alpha,
                seed=self.ci_seed,
            )
            
            # P-value calculation
            raw_p = self._calculate_p_value(y_true, y_score, auc_val, B=self.permutation_steps)
            # Bonferroni correction (capped at 1.0)
            auc_pval = min(1.0, raw_p * self.bonferroni_correction)
            
        except ValueError:
            fpr = np.array([])
            tpr = np.array([])
            roc_thresholds = np.array([])
            auc_val = float("nan")
            auc_lo = float("nan")
            auc_hi = float("nan")
            auc_pval = float("nan")

        # Voting accuracy (threshold search)
        all_dists = np.concatenate([d_pos.reshape(-1), d_neg.reshape(-1)], axis=0)
        min_d = float(all_dists.min()) if all_dists.size else 0.0
        max_d = float(all_dists.max()) if all_dists.size else 1.0
        thr_grid = np.linspace(min_d, max_d, self.threshold_search_steps)

        n_anchors = len(self.anchors)
        majority_cutoff = n_anchors / 2.0

        best_acc = 0.0
        best_acc_thr = float("nan")
        denom = max(1, (len(self.positives) + len(self.negatives)))

        for t in thr_grid:
            t_val = float(t)
            pos_votes = (d_pos < t_val).sum(axis=1)
            neg_votes = (d_neg < t_val).sum(axis=1)

            tp = int((pos_votes > majority_cutoff).sum())
            tn = int((neg_votes <= majority_cutoff).sum())
            acc = float((tp + tn) / denom)

            if acc > best_acc:
                best_acc = acc
                best_acc_thr = t_val

        # --- Embedding Norms ---
        # Concatenate all embeddings to compute global stats for this eval set
        all_embs = np.concatenate([anchor_emb, pos_emb, neg_emb], axis=0)
        norms = np.linalg.norm(all_embs, axis=1)
        norm_min = float(norms.min())
        norm_mean = float(norms.mean())
        norm_max = float(norms.max())

        return {
            "loss": float(loss_val),
            "accuracy": float(best_acc),
            "accuracy_thr": float(best_acc_thr),
            "auc": float(auc_val),
            "auc_ci_low": float(auc_lo),
            "auc_ci_high": float(auc_hi),
            "auc_pval": float(auc_pval),
            "majority_cutoff": float(majority_cutoff),
            "fpr": fpr,
            "tpr": tpr,
            "roc_thrs": roc_thresholds,
            "norm_min": norm_min,
            "norm_mean": norm_mean,
            "norm_max": norm_max
        }
