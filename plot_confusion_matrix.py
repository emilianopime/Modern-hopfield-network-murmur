"""
Visualiza la matriz de confusión del modelo MHN en el conjunto de validación.
Uso: python plot_confusion_matrix.py [--checkpoint PATH] [--output PATH] [--no-normalize] [--show]
"""
import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("confusion_matrix")

CLASS_NAMES    = ["Present", "Absent", "Unknown"]
CLASS_COLORS   = ["#e74c3c", "#2ecc71", "#f39c12"]   # rojo / verde / naranja
CHALLENGE_WEIGHTS = [5, 1, 3]


# ──────────────────────────────────────────────────────────────
# Inferencia
# ──────────────────────────────────────────────────────────────

def run_inference(checkpoint_path: Path, device: torch.device):
    """Carga modelo + checkpoint y corre inferencia sobre el set de validación."""
    from config import cfg
    from data import build_dataloaders
    from models import MurmurClassifier
    from training.metrics import aggregate_patient_preds

    _, val_loader, _ = build_dataloaders(cfg.data, cfg.preprocess, cfg.training)
    model = MurmurClassifier.from_config(cfg)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    epoch   = checkpoint.get("epoch", "?")
    best_wa = checkpoint.get("best_wa", checkpoint.get("metrics", {}).get("weighted_accuracy", None))
    logger.info(f"Checkpoint cargado — época {epoch} | mejor WA guardado: {best_wa}")

    all_probs, all_labels, all_pids = [], [], []
    with torch.no_grad():
        for spectrograms, tabulars, labels, patient_ids, lengths in val_loader:
            spectrograms = spectrograms.to(device)
            tabulars     = tabulars.to(device)
            probs        = torch.softmax(model(spectrograms, tabulars, lengths.to(device)), dim=-1)
            all_probs.append(probs.cpu())
            all_labels.append(labels)
            all_pids.extend(patient_ids)

    preds, labels = aggregate_patient_preds(
        all_pids, torch.cat(all_probs), torch.cat(all_labels)
    )
    logger.info(f"Pacientes evaluados: {len(preds)} | Distribución real: {np.bincount(labels.numpy())}")
    return preds.numpy(), labels.numpy(), epoch


# ──────────────────────────────────────────────────────────────
# Métricas
# ──────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score
    cm     = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    total_w   = sum(CHALLENGE_WEIGHTS[c] for c in y_true)
    correct_w = sum(CHALLENGE_WEIGHTS[t] for t, p in zip(y_true, y_pred) if t == p)
    wa        = correct_w / total_w if total_w > 0 else 0.0
    acc       = float(np.mean(y_true == y_pred))

    recall, precision, f1 = {}, {}, {}
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        recall[name]    = tp / (tp + fn + 1e-8)
        precision[name] = tp / (tp + fp + 1e-8)
        r, p_           = recall[name], precision[name]
        f1[name]        = 2 * r * p_ / (r + p_ + 1e-8)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0))

    return dict(cm=cm, cm_pct=cm_pct, wa=wa, acc=acc,
                recall=recall, precision=precision, f1=f1, macro_f1=macro_f1)


# ──────────────────────────────────────────────────────────────
# Visualización
# ──────────────────────────────────────────────────────────────

def plot(metrics: dict, epoch, output_path: Path, normalize: bool, show: bool):
    cm     = metrics["cm"]
    cm_pct = metrics["cm_pct"]

    fig = plt.figure(figsize=(14, 7), facecolor="#1a1a2e")
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.1, 0.9], wspace=0.35, figure=fig)

    # ── Panel izquierdo: matriz de confusión ──────────────────
    ax_cm = fig.add_subplot(gs[0])
    display_data = cm_pct if normalize else cm.astype(float)

    im = ax_cm.imshow(display_data, cmap="YlOrRd", vmin=0, vmax=1 if normalize else None, aspect="auto")

    for i in range(3):
        for j in range(3):
            count  = cm[i, j]
            pct    = cm_pct[i, j] * 100
            cell   = display_data[i, j]
            thresh = (display_data.max() * 0.5) if not normalize else 0.5
            color  = "white" if cell > thresh else "#1a1a2e"
            if normalize:
                ax_cm.text(j, i, f"{pct:.1f}%\n({count})", ha="center", va="center",
                           fontsize=11, fontweight="bold", color=color)
            else:
                ax_cm.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center",
                           fontsize=11, fontweight="bold", color=color)

    ax_cm.set_xticks(range(3))
    ax_cm.set_yticks(range(3))
    ax_cm.set_xticklabels(CLASS_NAMES, fontsize=12, color="white")
    ax_cm.set_yticklabels(CLASS_NAMES, fontsize=12, color="white", rotation=0)
    ax_cm.set_xlabel("Predicción", fontsize=13, color="white", labelpad=10)
    ax_cm.set_ylabel("Real", fontsize=13, color="white", labelpad=10)
    ax_cm.set_title(f"Matriz de Confusión — Época {epoch}", fontsize=14, color="white", pad=14)
    ax_cm.tick_params(colors="white")
    for spine in ax_cm.spines.values():
        spine.set_edgecolor("#444466")

    cbar = plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # ── Panel derecho: métricas por clase ────────────────────
    ax_m = fig.add_subplot(gs[1])
    ax_m.set_facecolor("#1a1a2e")
    ax_m.axis("off")

    # Encabezado global
    lines  = []
    styles = []

    lines.append(f"WA (challenge):  {metrics['wa']:.4f}")
    styles.append(dict(fontsize=14, fontweight="bold", color="#f1c40f"))
    lines.append(f"Accuracy std:    {metrics['acc']:.4f}")
    styles.append(dict(fontsize=12, color="#bdc3c7"))
    lines.append(f"Macro F1:        {metrics['macro_f1']:.4f}")
    styles.append(dict(fontsize=12, color="#bdc3c7"))
    lines.append("")
    styles.append(dict(fontsize=6, color="white"))

    # Tabla por clase
    header = f"{'Clase':<10} {'Recall':>7} {'Prec':>7} {'F1':>7}  W"
    lines.append(header)
    styles.append(dict(fontsize=10, color="#95a5a6", family="monospace"))
    lines.append("─" * 44)
    styles.append(dict(fontsize=8, color="#555577", family="monospace"))

    for i, name in enumerate(CLASS_NAMES):
        r  = metrics["recall"][name]
        p  = metrics["precision"][name]
        f  = metrics["f1"][name]
        w  = CHALLENGE_WEIGHTS[i]
        row = f"{name:<10} {r:>7.3f} {p:>7.3f} {f:>7.3f}  ×{w}"
        lines.append(row)
        styles.append(dict(fontsize=11, color=CLASS_COLORS[i], family="monospace", fontweight="bold"))

    lines.append("")
    styles.append(dict(fontsize=6, color="white"))
    lines.append("─" * 44)
    styles.append(dict(fontsize=8, color="#555577", family="monospace"))

    # Distribución real
    dist = np.bincount(np.concatenate([
        np.full(cm[i, :].sum(), i) for i in range(3)
    ]))
    lines.append("Distribución real (val):")
    styles.append(dict(fontsize=10, color="#95a5a6"))
    for i, name in enumerate(CLASS_NAMES):
        n = cm[i, :].sum()
        lines.append(f"  {name:<10}: {n} pacientes")
        styles.append(dict(fontsize=10, color=CLASS_COLORS[i]))

    y = 0.97
    for text, style in zip(lines, styles):
        ax_m.text(0.05, y, text, transform=ax_m.transAxes,
                  va="top", ha="left", **style)
        y -= 0.063 if style.get("fontsize", 11) >= 11 else 0.048

    fig.suptitle(
        "Modern Hopfield Network — Detección de Soplos Cardíacos\nPhysioNet Challenge 2022 · CirCor DigiScope",
        fontsize=11, color="#7f8c8d", y=1.01,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"Figura guardada en: {output_path}")

    if show:
        matplotlib.use("TkAgg")
        plt.show()

    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualiza la matriz de confusión del modelo MHN (nivel paciente)."
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/checkpoint_best.pt",
        help="Ruta al archivo .pt del checkpoint (default: checkpoints/checkpoint_best.pt)",
    )
    parser.add_argument(
        "--output", default="confusion_matrix.png",
        help="Ruta de salida para la imagen PNG (default: confusion_matrix.png)",
    )
    parser.add_argument(
        "--no-normalize", action="store_true",
        help="Muestra conteos absolutos en lugar de porcentajes normalizados.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Abre la figura en una ventana interactiva (requiere display).",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Fuerza el uso de CPU aunque haya GPU disponible.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint no encontrado: {checkpoint_path}")
        sys.exit(1)

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Dispositivo: {device}")

    y_pred, y_true, epoch = run_inference(checkpoint_path, device)
    metrics = compute_metrics(y_true, y_pred)

    logger.info(
        f"WA={metrics['wa']:.4f} | Acc={metrics['acc']:.4f} | MacroF1={metrics['macro_f1']:.4f}"
    )
    for name in CLASS_NAMES:
        logger.info(
            f"  {name:<10} Recall={metrics['recall'][name]:.3f} "
            f"Prec={metrics['precision'][name]:.3f} F1={metrics['f1'][name]:.3f}"
        )

    output_path = Path(args.output)
    plot(metrics, epoch, output_path, normalize=not args.no_normalize, show=args.show)


if __name__ == "__main__":
    main()
