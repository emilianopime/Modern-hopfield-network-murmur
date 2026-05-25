"""
training/metrics.py — Métricas del PhysioNet Challenge 2022

El challenge evalúa con "Weighted Accuracy" específica para detección de soplos,
con costos asimétricos que penalizan más los Falsos Negativos de soplos presentes:
    Present  → peso 5  (miss crítico: soplo real no detectado)
    Unknown  → peso 3  (incertidumbre diagnóstica)
    Absent   → peso 1  (bajo costo: falsa alarma)

La métrica oficial minimiza el costo clínico total, no la accuracy estándar.
"""

from typing import Optional

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

# Pesos del challenge (orden: Present=0, Absent=1, Unknown=2)
CHALLENGE_WEIGHTS = {0: 5, 1: 1, 2: 3}
CLASS_NAMES = ["Present", "Absent", "Unknown"]


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
