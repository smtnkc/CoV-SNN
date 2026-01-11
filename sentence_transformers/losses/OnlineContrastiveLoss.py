from typing import Iterable, Dict
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ..SentenceTransformer import SentenceTransformer
from ..util import SiameseDistanceMetric, get_best_distance_threshold
from sklearn.metrics import roc_auc_score, roc_curve


class OnlineContrastiveLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        distance_metric=SiameseDistanceMetric.EUCLIDEAN,
        margin: float = None,
        size_average: bool = True
    ):
        super().__init__()
        self.distance_metric = distance_metric
        self.margin = margin
        self.model = model
        self.size_average = size_average

    def get_config_dict(self):
        distance_metric_name = self.distance_metric.__name__
        for name, value in vars(SiameseDistanceMetric).items():
            if value == self.distance_metric:
                distance_metric_name = f"SiameseDistanceMetric.{name}"
                break
        return {
            "distance_metric": distance_metric_name,
            "margin": self.margin,
            "size_average": self.size_average,
        }

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sf)["sentence_embedding"] for sf in sentence_features]
        rep_anchor, rep_other = reps

        distances = self.distance_metric(rep_anchor, rep_other)  # [bs]
        labels = labels.view(-1).to(distances.device)

        # Graph-connected zero (prevents "no grad_fn" when mined sets are empty)
        zero = distances.sum() * 0.0

        # Split by class (robust if a class is missing in the batch)
        neg_mask = (labels == 0)
        pos_mask = (labels == 1)
        negs = distances[neg_mask]
        poss = distances[pos_mask]

        # Choose mining thresholds safely
        negative_pairs = distances.new_empty((0,))
        positive_pairs = distances.new_empty((0,))

        # Hard-negative mining
        if negs.numel() > 0:
            if poss.numel() > 0:
                pos_ref = poss.max()  # hard: negatives closer than hardest positive
            else:
                pos_ref = negs.mean()  # fallback when no positives
            negative_pairs = negs[negs < pos_ref]

        # Hard-positive mining
        if poss.numel() > 0:
            if negs.numel() > 0:
                neg_ref = negs.min()  # hard: positives farther than closest negative
            else:
                neg_ref = poss.mean()  # fallback when no negatives
            positive_pairs = poss[poss > neg_ref]

        # Loss terms: MUST stay connected to graph
        positive_loss = positive_pairs.pow(2).sum() if positive_pairs.numel() > 0 else zero

        # If margin is None, treat as "no negative margin term" instead of crashing
        if negative_pairs.numel() > 0 and self.margin is not None:
            negative_loss = F.relu(float(self.margin) - negative_pairs).pow(2).sum()
        else:
            negative_loss = zero

        loss = 0.5 * (positive_loss + negative_loss)

        # --- metrics (detach only here) ---
        d_np = distances.detach().cpu().numpy()
        y_np = labels.detach().cpu().numpy()

        # Accuracy by best distance threshold
        try:
            best_distance_threshold = get_best_distance_threshold(d_np, y_np)
            preds_best = (distances < float(best_distance_threshold)).long()
            acc_by_best = (preds_best == labels.long()).float().mean().item()
        except Exception:
            acc_by_best = float("nan")

        # AUC (can be nan if batch has single class)
        y_true = labels.detach().to(torch.int64).cpu().numpy()
        y_score = (-distances.detach()).cpu().numpy()
        try:
            _fpr, _tpr, _thr = roc_curve(y_true, y_score)
            AUC = float(roc_auc_score(y_true, y_score))
        except ValueError:
            AUC = float("nan")

        # Keep your "3-return-items" contract
        if self.size_average:
            return loss.mean(), acc_by_best, AUC
        return loss.sum(), acc_by_best, AUC
