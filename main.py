"""
main.py — Punto de entrada principal del proyecto

Modos de ejecución:
    python main.py train        → Entrenamiento completo (todas las fases)
    python main.py eval         → Evaluación de un checkpoint guardado
    python main.py preprocess   → Pre-calcular y cachear espectrogramas (Fase 1)
    python main.py explain      → Generar visualización de interpretabilidad

Ejemplo de uso rápido:
    python main.py train
    python main.py eval --checkpoint checkpoints/checkpoint_best.pt
    python main.py explain --wav data/physionet_2022/12345_AV.wav
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# ── Configuración de logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def cmd_train(args) -> None:
    """Fase completa de entrenamiento."""
    from config import cfg
    from data import build_dataloaders
    from models import MurmurClassifier
    from training import Trainer

    logger.info(f"Dispositivo: {cfg.training.device}")
    logger.info(f"Backbone CNN: {cfg.cnn.backbone} | Embedding dim: {cfg.cnn.embedding_dim}")
    logger.info(f"HopfieldPooling: β={'auto (1/√d)' if cfg.hopfield.beta is None else cfg.hopfield.beta} | prototipos: {cfg.hopfield.quantity}")

    train_loader, val_loader = build_dataloaders(
        cfg.data, cfg.preprocess, cfg.training
    )
    model = MurmurClassifier.from_config(cfg)
    logger.info(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=cfg.training,
        classifier_cfg=cfg.classifier,
        cnn_cfg=cfg.cnn,
    )
    trainer.train()


def cmd_eval(args) -> None:
    """Evaluación de un checkpoint sobre el conjunto de validación."""
    from config import cfg
    from data import build_dataloaders
    from models import MurmurClassifier
    from training.metrics import compute_challenge_score, format_metrics

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint no encontrado: {checkpoint_path}")
        sys.exit(1)

    _, val_loader = build_dataloaders(cfg.data, cfg.preprocess, cfg.training)
    model = MurmurClassifier.from_config(cfg)

    checkpoint = torch.load(checkpoint_path, map_location=cfg.training.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(cfg.training.device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for spectrograms, labels in val_loader:
            spectrograms = spectrograms.to(cfg.training.device)
            preds = model.predict(spectrograms)
            all_preds.append(preds.cpu())
            all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metrics = compute_challenge_score(all_labels, all_preds)

    logger.info(f"Resultados de evaluación ({checkpoint_path.name}):")
    logger.info(format_metrics(metrics))


def cmd_preprocess(args) -> None:
    """Pre-calcula y cachea todos los espectrogramas de Mel (Fase 1)."""
    from config import cfg
    from data.preprocessing import PCGPreprocessor

    preprocessor = PCGPreprocessor(cfg.preprocess)
    wav_dir = cfg.data.dataset_path
    wav_files = list(wav_dir.glob("*.wav"))

    if not wav_files:
        logger.warning(f"No se encontraron archivos .wav en {wav_dir}")
        return

    logger.info(f"Pre-procesando {len(wav_files)} archivos .wav → caché en {cfg.preprocess.cache_dir}")
    from tqdm import tqdm
    for wav in tqdm(wav_files, desc="Fase 1: Wavelet → S-G → Mel"):
        preprocessor(wav)

    logger.info("Preprocesamiento completado.")


def cmd_explain(args) -> None:
    """Genera visualización de interpretabilidad para un archivo .wav."""
    from config import cfg
    from data.preprocessing import PCGPreprocessor
    from models import MurmurClassifier
    from utils import extract_hopfield_weights, plot_attention_heatmap

    wav_path = Path(args.wav)
    if not wav_path.exists():
        logger.error(f"Archivo no encontrado: {wav_path}")
        sys.exit(1)

    preprocessor = PCGPreprocessor(cfg.preprocess)
    spectrogram = preprocessor(wav_path).unsqueeze(0)  # (1, 1, n_mels, T)

    model = MurmurClassifier.from_config(cfg)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    with torch.no_grad():
        logits, attn_weights = model(spectrogram, return_attention=True)

    probs = torch.softmax(logits, dim=-1).squeeze()
    class_names = cfg.classifier.class_names
    pred_class = class_names[probs.argmax().item()]

    logger.info(f"Predicción: {pred_class}")
    for name, prob in zip(class_names, probs.tolist()):
        logger.info(f"  {name}: {prob:.3f}")

    output_path = Path(f"explain_{wav_path.stem}.png")
    fig = plot_attention_heatmap(
        wav_path=wav_path,
        attention_weights=attn_weights,
        spectrogram=spectrogram.squeeze(0),
        hop_length=cfg.preprocess.hop_length,
        sample_rate=cfg.preprocess.sample_rate,
        output_path=output_path,
        title=f"Interpretabilidad MHN — {wav_path.stem} | Predicción: {pred_class}",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Modern Hopfield Network para detección de soplos cardíacos (PhysioNet 2022)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcomando: train
    subparsers.add_parser("train", help="Entrenar el modelo completo (Fases 2-3-4)")

    # Subcomando: eval
    eval_parser = subparsers.add_parser("eval", help="Evaluar un checkpoint guardado")
    eval_parser.add_argument("--checkpoint", required=True, help="Ruta al archivo .pt")

    # Subcomando: preprocess
    subparsers.add_parser("preprocess", help="Pre-calcular espectrogramas de Mel (Fase 1)")

    # Subcomando: explain
    explain_parser = subparsers.add_parser(
        "explain", help="Visualizar interpretabilidad de atención Hopfield"
    )
    explain_parser.add_argument("--wav", required=True, help="Ruta al archivo .wav")
    explain_parser.add_argument("--checkpoint", default=None, help="Ruta al checkpoint (opcional)")

    args = parser.parse_args()

    commands = {
        "train": cmd_train,
        "eval": cmd_eval,
        "preprocess": cmd_preprocess,
        "explain": cmd_explain,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
