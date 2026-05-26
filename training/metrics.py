"""
training/metrics.py — Métricas del PhysioNet Challenge 2022

El challenge evalúa con "Weighted Accuracy" específica para detección de soplos,
con costos asimétricos que penalizan más los Falsos Negativos de soplos presentes:
    Present  → peso 5  (miss crítico: soplo real no detectado)
    Unknown  → peso 3  (incertidumbre diagnóstica)
    Absent   → peso 1  (bajo costo: falsa alarma)

La métrica oficial minimiza el costo clínico total, no la accuracy estándar.

Loss:
    CBFocalLoss — Class-Balanced Focal Loss (Cui et al. 2019) adaptado al MHN.
    Combina Focal Loss (Lin et al. 2017) con re-ponderación por volumen efectivo
    de muestras, eliminando la necesidad de pesos manuales [5.0, 1.0, 7.0].
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

# Pesos del challenge (orden: Present=0, Absent=1, Unknown=2)
CHALLENGE_WEIGHTS = {0: 5, 1: 1, 2: 3}
CLASS_NAMES = ["Present", "Absent", "Unknown"]


# ─────────────────────────────────────────────────────────
# CB-Focal Loss
# Tomado del notebook MZA-PCG v5 (amigo) y adaptado al MHN.
# Se calcula UNA SOLA VEZ con los conteos reales del dataset.
# ─────────────────────────────────────────────────────────

class CBFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss (Cui et al. 2019).

    Combina:
      1. Focal Loss: penaliza muestras fáciles, enfoca las difíciles.
         FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
         Con γ=2: muestras bien clasificadas reciben gradiente ~0.

      2. Class-Balanced weighting: re-pondera por volumen efectivo
         de muestras en lugar de pesos manuales:
         α_i = (1 - β) / (1 - β^n_i)
         donde β=(N-1)/N y n_i = muestras de clase i.

    Ventaja sobre CrossEntropy ponderada manual:
      - Escala automáticamente con el desbalance real del dataset.
      - Con γ=2 el modelo se concentra en los casos difíciles
        (murmurs Present y Unknown) sin ignorar los Absent fáciles.

    Args:
        samples_per_class: Lista [n_Present, n_Absent, n_Unknown]
                           contada del split de entrenamiento real.
        gamma:  Exponent focal (2.0 recomendado para desbalance moderado).
        beta:   Smoothing de volumen efectivo (0.9999 estándar).
    """

    def __init__(
        self,
        samples_per_class: list[int],
        gamma: float = 2.0,
        beta: float = 0.9999,
    ):
        super().__init__()
        # Calcular pesos class-balanced
        effective_num = [1.0 - beta ** n for n in samples_per_class]
        weights = [(1.0 - beta) / (e + 1e-8) for e in effective_num]
        total_w = sum(weights)
        # Normalizar para que los pesos sumen n_classes (igual escala que CrossEntropy)
        weights = [w / total_w * len(weights) for w in weights]

        self.register_buffer("alpha", torch.tensor(weights, dtype=torch.float32))
        self.gamma = gamma

        # Log de pesos para trazabilidad
        import logging
        logging.getLogger(__name__).info(
            f"CBFocalLoss — samples: {samples_per_class} | "
            f"weights: {[f'{w:.3f}' for w in weights]} | γ={gamma}"
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C) — salida cruda del modelo (sin softmax).
            targets: (B,)   — índices de clase [0=Present, 1=Absent, 2=Unknown].
        """
        # Cross-entropy por muestra (sin reducción)
        ce = F.cross_entropy(logits, targets, reduction="none")   # (B,)

        # Probabilidad de la clase correcta
        p_t = torch.exp(-ce)                                      # (B,)

        # Peso class-balanced por muestra
        at = self.alpha[targets]                                   # (B,)

        # Focal weighting: reduce gradiente en muestras bien clasificadas
        focal_weight = (1.0 - p_t) ** self.gamma

        loss = at * focal_weight * ce
        return loss.mean()


# ─────────────────────────────────────────────────────────
# Métricas del challenge
# ─────────────────────────────────────────────────────────

def compute_weighted_accuracy(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    weights: dict = CHALLENGE_WEIGHTS,
) -> float:
    """
    Weighted Accuracy del PhysioNet Challenge 2022.

    Pondera cada clase correctamente predicha por su costo clínico.
    Normaliza por la suma total de pesos posibles.

    Returns:
        Valor entre 0.0 y 1.0 (mayor es mejor).
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    total_weight = sum(weights[label] for label in y_true)
    if total_weight == 0:
        return 0.0

    correct_weight = sum(
        weights[true]
        for true, pred in zip(y_true, y_pred)
        if true == pred
    )
    return correct_weight / total_weight


def compute_challenge_score(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
) -> dict:
    """
    Reporte completo de métricas para el challenge.

    Returns:
        Dict con: weighted_accuracy, standard_accuracy, confusion_matrix,
                  per_class_recall, per_class_precision.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    wa = compute_weighted_accuracy(y_true, y_pred)
    acc = float(np.mean(y_true == y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    per_class_recall = {}
    per_class_precision = {}
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        per_class_recall[name] = tp / (tp + fn + 1e-8)
        per_class_precision[name] = tp / (tp + fp + 1e-8)

    return {
        "weighted_accuracy": wa,
        "standard_accuracy": acc,
        "confusion_matrix": cm,
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
    }


def format_metrics(metrics: dict) -> str:
    """Formatea el diccionario de métricas para logging."""
    lines = [
        f"  Weighted Accuracy (challenge): {metrics['weighted_accuracy']:.4f}",
        f"  Standard Accuracy:             {metrics['standard_accuracy']:.4f}",
        "  Per-class Recall:",
    ]
    for name, val in metrics["per_class_recall"].items():
        lines.append(f"    {name:>10}: {val:.3f}")
    return "\n".join(lines)
