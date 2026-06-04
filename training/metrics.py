import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

CHALLENGE_WEIGHTS = {0: 5, 1: 1, 2: 3}
CLASS_NAMES       = ["Present", "Absent", "Unknown"]


class CBFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss (Cui et al. 2019) con label smoothing opcional.
    Calcula pesos inversos al volumen efectivo de cada clase para combatir el desbalance.
    """

    def __init__(
        self,
        samples_per_class: list[int],
        gamma: float = 2.0,
        beta: float = 0.9999,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        effective_num = [1.0 - beta ** n for n in samples_per_class]
        weights = [(1.0 - beta) / (e + 1e-8) for e in effective_num]
        total_w = sum(weights)
        weights = [w / total_w * len(weights) for w in weights]

        self.register_buffer("alpha", torch.tensor(weights, dtype=torch.float32))
        self.gamma         = gamma
        self.label_smoothing = label_smoothing

        logger.info(
            f"CBFocalLoss — samples: {samples_per_class} | "
            f"weights: {[f'{w:.3f}' for w in weights]} | "
            f"γ={gamma} | label_smoothing={label_smoothing}"
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce           = F.cross_entropy(logits, targets, reduction="none", label_smoothing=self.label_smoothing)
        p_t          = torch.exp(-ce)
        at           = self.alpha[targets]
        focal_weight = (1.0 - p_t) ** self.gamma
        return (at * focal_weight * ce).mean()


def compute_weighted_accuracy(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    weights: dict = CHALLENGE_WEIGHTS,
) -> float:
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    total_weight   = sum(weights[label] for label in y_true)
    correct_weight = sum(weights[true] for true, pred in zip(y_true, y_pred) if true == pred)
    return correct_weight / total_weight if total_weight > 0 else 0.0


def compute_challenge_score(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
) -> dict:
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    wa  = compute_weighted_accuracy(y_true, y_pred)
    acc = float(np.mean(y_true == y_pred))
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    per_class_recall, per_class_precision = {}, {}
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        per_class_recall[name]    = tp / (tp + fn + 1e-8)
        per_class_precision[name] = tp / (tp + fp + 1e-8)

    return {
        "weighted_accuracy":   wa,
        "standard_accuracy":   acc,
        "confusion_matrix":    cm,
        "per_class_recall":    per_class_recall,
        "per_class_precision": per_class_precision,
    }


def format_metrics(metrics: dict) -> str:
    lines = [
        f"  Weighted Accuracy (challenge): {metrics['weighted_accuracy']:.4f}",
        f"  Standard Accuracy:             {metrics['standard_accuracy']:.4f}",
        "  Per-class Recall:",
    ]
    for name, val in metrics["per_class_recall"].items():
        lines.append(f"    {name:>10}: {val:.3f}")
    return "\n".join(lines)
