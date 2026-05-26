"""
training/trainer.py — Bucle de entrenamiento y validación

Integra todas las fases en el ciclo de entrenamiento estándar de PyTorch.
Implementa:
    - Descongelado gradual del backbone CNN (freeze_backbone_epochs)
    - Clip de gradientes (estabilidad con MHN y matrices de correlación grandes)
    - Cosine LR con warmup (reemplaza ReduceLROnPlateau)
    - CB-Focal Loss con pesos auto-calculados (reemplaza CrossEntropy manual)
    - ClinicalMixUp en entrenamiento (regularización de espectrogramas)
    - Guardado de checkpoints con el mejor modelo

Técnicas tomadas de arquitecturas de referencia:
    - CBFocalLoss:    MZA-PCG v5 (amigo) — Cui et al. 2019 + Lin et al. 2017
    - ClinicalMixUp: MZA-PCG v5 (amigo) — mezcla de espectrogramas + tabulares
    - Cosine warmup: MZA-PCG v5 (amigo) — get_lr(epoch) con rampa lineal inicial
"""

import logging
import math
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ClassifierConfig, TrainingConfig, CNNConfig
from training.metrics import CBFocalLoss, compute_challenge_score, format_metrics

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# ClinicalMixUp
# Fuente: MZA-PCG v5 notebook (amigo) — adaptado al pipeline MHN.
# Mezcla espectrogramas + features tabulares de dos muestras del batch.
# Usa distribución Beta(α,α) con lam≥0.5 para que la muestra dominante mande.
# ─────────────────────────────────────────────

def clinical_mixup(
    spectrograms: torch.Tensor,
    tabulars: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    ClinicalMixUp: mezcla espectrogramas y features tabulares del batch.

    Se aplica con probabilidad 0.5 para no degradar señales sutiles de murmur.
    lam≥0.5 asegura que la muestra dominante siempre contribuya más.

    Args:
        spectrograms: (B, 1, n_mels, T)
        tabulars:     (B, 7) features clínicas
        labels:       (B,) enteros de clase
        alpha:        parámetro de Beta(α,α); 0.0 desactiva MixUp

    Returns:
        spec_mixed, tab_mixed, labels_a, labels_b, lam
        La loss se calcula como: lam·L(y_a) + (1-lam)·L(y_b)
    """
    if alpha <= 0.0 or random.random() > 0.5:
        # Sin mixup: lam=1.0 → pérdida estándar sobre labels originales
        return spectrograms, tabulars, labels, labels, 1.0

    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)             # garantizar que la muestra dominante manda

    idx = torch.randperm(spectrograms.shape[0], device=spectrograms.device)

    spec_mixed = lam * spectrograms + (1.0 - lam) * spectrograms[idx]
    tab_mixed  = lam * tabulars    + (1.0 - lam) * tabulars[idx]

    return spec_mixed, tab_mixed, labels, labels[idx], lam


# ─────────────────────────────────────────────
# Scheduler: Cosine LR con warmup lineal
# Fuente: MZA-PCG v5 — get_lr(epoch) con warmup
# ─────────────────────────────────────────────

def cosine_lr_with_warmup(
    epoch: int,
    base_lr: float,
    warmup_epochs: int,
    total_epochs: int,
    lr_min_factor: float = 0.01,
) -> float:
    """
    LR schedule con fase de warmup lineal seguida de decaimiento coseno.

    Épocas 1..warmup_epochs: LR crece linealmente 0 → base_lr.
    Épocas warmup+1..total:  LR decae con coseno hasta base_lr * lr_min_factor.

    Ventaja sobre ReduceLROnPlateau:
        - No depende de la métrica de validación (evita el colapso en época 12).
        - Decaimiento suave garantiza exploración prolongada hasta epoch 120.
    """
    if epoch <= warmup_epochs:
        # Rampa lineal
        return base_lr * epoch / max(1, warmup_epochs)
    # Coseno
    p = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * p))
    return base_lr * (lr_min_factor + (1.0 - lr_min_factor) * cosine_decay)


# ─────────────────────────────────────────────
# Trainer principal
# ─────────────────────────────────────────────

class Trainer:
    """
    Encapsula el ciclo completo de entrenamiento y validación.

    Args:
        model:              Instancia de MurmurClassifier.
        train_loader:       DataLoader de entrenamiento.
        val_loader:         DataLoader de validación.
        train_cfg:          Configuración de entrenamiento.
        classifier_cfg:     Configuración del clasificador (num_classes).
        cnn_cfg:            Para saber cuándo descongelar el backbone.
        samples_per_class:  [n_Present, n_Absent, n_Unknown] del split de train.
                            Necesario para CBFocalLoss auto-ponderada.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_cfg: TrainingConfig,
        classifier_cfg: ClassifierConfig,
        cnn_cfg: CNNConfig,
        samples_per_class: list[int],
    ):
        self.model = model.to(train_cfg.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = train_cfg
        self.cnn_cfg = cnn_cfg

        # ── CB-Focal Loss (reemplaza CrossEntropy con pesos manuales) ──────
        # Calcula pesos class-balanced automáticamente desde conteos reales.
        self.criterion = CBFocalLoss(
            samples_per_class=samples_per_class,
            gamma=train_cfg.focal_gamma,
            beta=train_cfg.focal_beta,
        ).to(train_cfg.device)

        # ── Optimizer AdamW ───────────────────────────────────────────────
        # Inicializar SOLO con parámetros no-CNN.
        # Los parámetros del CNN se agregan al descongelar (freeze_backbone_epochs).
        cnn_param_ids = {id(p) for p in model.cnn.parameters()}
        non_cnn_params = [p for p in model.parameters() if id(p) not in cnn_param_ids]

        self.optimizer = AdamW(
            non_cnn_params,
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )

        self.device = train_cfg.device
        self.best_wa = 0.0
        self.checkpoint_dir = train_cfg.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Trainer inicializado — "
            f"epochs={train_cfg.epochs} | "
            f"warmup={train_cfg.warmup_epochs} | "
            f"mixup_α={train_cfg.mixup_alpha} | "
            f"focal_γ={train_cfg.focal_gamma}"
        )

    def train(self) -> None:
        """Ejecuta el entrenamiento completo por `cfg.epochs` épocas."""
        logger.info(f"Iniciando entrenamiento en {self.device} por {self.cfg.epochs} épocas.")

        for epoch in range(1, self.cfg.epochs + 1):
            # Descongelar backbone CNN gradualmente
            self._maybe_unfreeze_backbone(epoch)

            # Actualizar LR con schedule coseno + warmup
            self._update_lr(epoch)

            t0 = time.time()
            train_loss = self._train_epoch(epoch)
            val_metrics = self._validate_epoch(epoch)

            elapsed = time.time() - t0
            wa = val_metrics["weighted_accuracy"]

            # LR actual (del primer grupo del optimizer)
            current_lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                f"Época {epoch:03d}/{self.cfg.epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"WA: {wa:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Tiempo: {elapsed:.1f}s"
            )
            logger.info(format_metrics(val_metrics))

            # Guardar mejor modelo
            if wa > self.best_wa:
                self.best_wa = wa
                self._save_checkpoint(epoch, val_metrics, is_best=True)
                logger.info(f"  ✓ Nuevo mejor modelo (WA={wa:.4f})")

            # Checkpoint periódico
            if epoch % self.cfg.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, val_metrics, is_best=False)

        logger.info(f"Entrenamiento completado. Mejor WA: {self.best_wa:.4f}")

    def _train_epoch(self, epoch: int) -> float:
        """Una época de entrenamiento con ClinicalMixUp. Devuelve la pérdida media."""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Train {epoch}", leave=False)
        for spectrograms, tabulars, labels in pbar:
            spectrograms = spectrograms.to(self.device, non_blocking=True)
            tabulars     = tabulars.to(self.device, non_blocking=True)
            labels       = labels.to(self.device, non_blocking=True)

            # ClinicalMixUp (50% de los batches)
            spec_m, tab_m, labels_a, labels_b, lam = clinical_mixup(
                spectrograms, tabulars, labels, alpha=self.cfg.mixup_alpha
            )

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(spec_m, tab_m)

            # Pérdida mixup: λ·L(y_a) + (1-λ)·L(y_b)
            if lam < 1.0:
                loss = (
                    lam * self.criterion(logits, labels_a)
                    + (1.0 - lam) * self.criterion(logits, labels_b)
                )
            else:
                loss = self.criterion(logits, labels)

            loss.backward()

            # Clip de gradientes: estabiliza el entrenamiento de HopfieldPooling
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> dict:
        """Una época de validación. Devuelve métricas del challenge."""
        self.model.eval()
        all_preds, all_labels = [], []

        for spectrograms, tabulars, labels in tqdm(self.val_loader, desc=f"Val   {epoch}", leave=False):
            spectrograms = spectrograms.to(self.device, non_blocking=True)
            tabulars     = tabulars.to(self.device, non_blocking=True)
            logits = self.model(spectrograms, tabulars)
            preds = torch.argmax(logits, dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(labels)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        return compute_challenge_score(all_labels, all_preds)

    def _update_lr(self, epoch: int) -> None:
        """Aplica el schedule coseno+warmup a todos los grupos del optimizer."""
        # Grupo principal (Hopfield + clasificador + tabular)
        new_lr = cosine_lr_with_warmup(
            epoch=epoch,
            base_lr=self.cfg.learning_rate,
            warmup_epochs=self.cfg.warmup_epochs,
            total_epochs=self.cfg.epochs,
            lr_min_factor=self.cfg.lr_min_factor,
        )
        self.optimizer.param_groups[0]["lr"] = new_lr

        # Grupo del backbone CNN (si ya fue descongelado): LR 10× menor
        if len(self.optimizer.param_groups) > 1:
            self.optimizer.param_groups[1]["lr"] = new_lr * 0.1

    def _maybe_unfreeze_backbone(self, epoch: int) -> None:
        """
        Congela el backbone CNN durante las primeras `freeze_backbone_epochs` épocas
        para entrenar primero el HopfieldPooling y el clasificador, luego hace
        fine-tuning de todo el modelo conjuntamente.
        """
        freeze_until = self.cnn_cfg.freeze_backbone_epochs
        if epoch == 1:
            logger.info(f"Backbone CNN congelado hasta época {freeze_until}.")
            for param in self.model.cnn.parameters():
                param.requires_grad = False
        elif epoch == freeze_until + 1:
            logger.info("Descongelando backbone CNN para fine-tuning completo.")
            for param in self.model.cnn.parameters():
                param.requires_grad = True
            # Agregar CNN como nuevo grupo (LR 10× menor para no destruir ImageNet features)
            self.optimizer.add_param_group({
                "params": list(self.model.cnn.parameters()),
                "lr": self.cfg.learning_rate * 0.1,
                "weight_decay": self.cfg.weight_decay,
            })

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool) -> None:
        tag = "best" if is_best else f"epoch_{epoch:03d}"
        path = self.checkpoint_dir / f"checkpoint_{tag}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "best_wa": self.best_wa,
            },
            path,
        )
