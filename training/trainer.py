"""
training/trainer.py — Bucle de entrenamiento y validación — v3

Mejoras respecto a v2:
    - SGDR (Cosine Annealing con Warm Restarts): reinicia el LR cada T_0=20
      épocas, dando múltiples oportunidades de encontrar mejores mínimos.
      Reinicios en épocas ~26, 46, 66, 86, 106 → mejor checkpoint esperado
      en épocas 40-80 en lugar de época 33.
    - SpecAugment: enmascara franjas de frecuencia y tiempo en espectrogramas.
      Impide memorización de patrones específicos del training set.
    - Label smoothing ε=0.1 en CBFocalLoss: reduce sobreconfianza en Absent.
    - freeze_backbone_epochs 5→12: HopfieldPooling consolida sus patrones
      asociativos antes de que el CNN empiece a perturbarlo.
    - dropout 0.3→0.45 + weight_decay 1e-4→5e-4: más regularización.
"""

import logging
import math
import random
import time
from pathlib import Path

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
# SpecAugment — v3
# ─────────────────────────────────────────────

def spec_augment(
    spec: torch.Tensor,
    freq_mask_p: float = 0.15,
    time_mask_p: float = 0.15,
) -> torch.Tensor:
    """
    SpecAugment (Park et al. 2019) — enmascara franjas de frecuencia y tiempo.

    Aplicado al batch completo con máscaras distintas por muestra.
    Impide que el modelo memorice bins de Mel específicos del training set,
    forzándolo a usar patrones distribuidos (más robusto en validación).

    Args:
        spec:         (B, 1, n_mels, T)
        freq_mask_p:  fracción de bandas mel a enmascarar (0.15 → 19/128 bins)
        time_mask_p:  fracción de frames temporales a enmascarar
    """
    spec   = spec.clone()
    B, C, n_mels, T = spec.shape

    f_size = max(1, int(n_mels * freq_mask_p))
    t_size = max(1, int(T * time_mask_p))

    for b in range(B):
        # Máscara de frecuencia
        f0 = random.randint(0, n_mels - f_size)
        spec[b, :, f0:f0 + f_size, :] = 0.0
        # Máscara de tiempo
        t0 = random.randint(0, max(0, T - t_size))
        spec[b, :, :, t0:t0 + t_size] = 0.0

    return spec


# ─────────────────────────────────────────────
# ClinicalMixUp (igual que v2)
# ─────────────────────────────────────────────

def clinical_mixup(
    spectrograms: torch.Tensor,
    tabulars: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0.0 or random.random() > 0.5:
        return spectrograms, tabulars, labels, labels, 1.0

    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    idx = torch.randperm(spectrograms.shape[0], device=spectrograms.device)

    spec_mixed = lam * spectrograms + (1.0 - lam) * spectrograms[idx]
    tab_mixed  = lam * tabulars    + (1.0 - lam) * tabulars[idx]
    return spec_mixed, tab_mixed, labels, labels[idx], lam


# ─────────────────────────────────────────────
# SGDR — Cosine Annealing con Warm Restarts — v3
# ─────────────────────────────────────────────

def sgdr_lr(
    epoch: int,
    base_lr: float,
    warmup_epochs: int,
    T_0: int,
    lr_min_factor: float = 0.01,
) -> float:
    """
    LR schedule: warmup lineal + Cosine Annealing con Warm Restarts (SGDR).

    Fases:
      Épocas 1..warmup_epochs    : LR sube linealmente 0 → base_lr.
      Épocas warmup+1..warmup+T0 : primer ciclo coseno (base_lr → eta_min).
      Épocas warmup+T0+1..        : reinicio → base_lr, repite ciclos de T_0.

    Con warmup=5, T_0=20 los reinicios ocurren en épocas 26, 46, 66, 86, 106.
    Cada reinicio es una nueva oportunidad de escapar de mínimos locales.

    Ventaja sobre cosine simple:
      El modelo no queda "atrapado" después del primer mínimo (época 33 en v2).
      Cada ciclo puede producir un checkpoint mejor.
    """
    eta_min = base_lr * lr_min_factor

    if epoch <= warmup_epochs:
        return base_lr * epoch / max(1, warmup_epochs)

    # Posición dentro de los ciclos SGDR (después del warmup)
    e_after_warmup = epoch - warmup_epochs            # ≥ 1
    t_in_cycle     = (e_after_warmup - 1) % T_0      # 0..T_0-1
    cos_val = 0.5 * (1.0 + math.cos(math.pi * t_in_cycle / T_0))
    return eta_min + (base_lr - eta_min) * cos_val


# ─────────────────────────────────────────────
# Trainer principal
# ─────────────────────────────────────────────

class Trainer:
    """
    Encapsula el ciclo completo de entrenamiento y validación.

    v3 respecto a v2:
      - SGDR en lugar de cosine simple
      - SpecAugment en _train_epoch
      - label_smoothing en CBFocalLoss
      - freeze_backbone_epochs 5→12
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
        self.model        = model.to(train_cfg.device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = train_cfg
        self.cnn_cfg      = cnn_cfg

        # CB-Focal Loss con label smoothing
        self.criterion = CBFocalLoss(
            samples_per_class=samples_per_class,
            gamma=train_cfg.focal_gamma,
            beta=train_cfg.focal_beta,
            label_smoothing=train_cfg.label_smoothing,
        ).to(train_cfg.device)

        # AdamW — solo parámetros no-CNN inicialmente
        cnn_param_ids  = {id(p) for p in model.cnn.parameters()}
        non_cnn_params = [p for p in model.parameters() if id(p) not in cnn_param_ids]

        self.optimizer = AdamW(
            non_cnn_params,
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )

        self.device         = train_cfg.device
        self.best_wa        = 0.0
        self.checkpoint_dir = train_cfg.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Trainer v3 — epochs={train_cfg.epochs} | "
            f"SGDR T_0={train_cfg.sgdr_T0} | warmup={train_cfg.warmup_epochs} | "
            f"lr={train_cfg.learning_rate} | wd={train_cfg.weight_decay} | "
            f"dropout={train_cfg.label_smoothing} (label_smooth) | "
            f"mixup_α={train_cfg.mixup_alpha} | "
            f"SpecAugment p={train_cfg.spec_augment_prob}"
        )

    def train(self) -> None:
        logger.info(f"Iniciando entrenamiento en {self.device} por {self.cfg.epochs} épocas.")

        for epoch in range(1, self.cfg.epochs + 1):
            self._maybe_unfreeze_backbone(epoch)
            self._update_lr(epoch)

            t0          = time.time()
            train_loss  = self._train_epoch(epoch)
            val_metrics = self._validate_epoch(epoch)
            elapsed     = time.time() - t0

            wa         = val_metrics["weighted_accuracy"]
            current_lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                f"Época {epoch:03d}/{self.cfg.epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"WA: {wa:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Tiempo: {elapsed:.1f}s"
            )
            logger.info(format_metrics(val_metrics))

            if wa > self.best_wa:
                self.best_wa = wa
                self._save_checkpoint(epoch, val_metrics, is_best=True)
                logger.info(f"  ✓ Nuevo mejor modelo (WA={wa:.4f})")

            if epoch % self.cfg.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, val_metrics, is_best=False)

        logger.info(f"Entrenamiento completado. Mejor WA: {self.best_wa:.4f}")

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Train {epoch}", leave=False)
        for spectrograms, tabulars, labels in pbar:
            spectrograms = spectrograms.to(self.device, non_blocking=True)
            tabulars     = tabulars.to(self.device, non_blocking=True)
            labels       = labels.to(self.device, non_blocking=True)

            # SpecAugment — v3
            if random.random() < self.cfg.spec_augment_prob:
                spectrograms = spec_augment(
                    spectrograms,
                    freq_mask_p=self.cfg.spec_freq_mask_p,
                    time_mask_p=self.cfg.spec_time_mask_p,
                )

            # ClinicalMixUp
            spec_m, tab_m, labels_a, labels_b, lam = clinical_mixup(
                spectrograms, tabulars, labels, alpha=self.cfg.mixup_alpha
            )

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(spec_m, tab_m)

            if lam < 1.0:
                loss = (
                    lam * self.criterion(logits, labels_a)
                    + (1.0 - lam) * self.criterion(logits, labels_b)
                )
            else:
                loss = self.criterion(logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> dict:
        self.model.eval()
        all_preds, all_labels = [], []

        for spectrograms, tabulars, labels in tqdm(self.val_loader, desc=f"Val   {epoch}", leave=False):
            spectrograms = spectrograms.to(self.device, non_blocking=True)
            tabulars     = tabulars.to(self.device, non_blocking=True)
            logits       = self.model(spectrograms, tabulars)
            preds        = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(labels)

        return compute_challenge_score(torch.cat(all_labels), torch.cat(all_preds))

    def _update_lr(self, epoch: int) -> None:
        """Aplica SGDR a todos los grupos del optimizer."""
        new_lr = sgdr_lr(
            epoch=epoch,
            base_lr=self.cfg.learning_rate,
            warmup_epochs=self.cfg.warmup_epochs,
            T_0=self.cfg.sgdr_T0,
            lr_min_factor=self.cfg.lr_min_factor,
        )
        self.optimizer.param_groups[0]["lr"] = new_lr
        # Backbone CNN (si ya fue descongelado): LR 10× menor
        if len(self.optimizer.param_groups) > 1:
            self.optimizer.param_groups[1]["lr"] = new_lr * 0.1

    def _maybe_unfreeze_backbone(self, epoch: int) -> None:
        freeze_until = self.cnn_cfg.freeze_backbone_epochs
        if epoch == 1:
            logger.info(f"Backbone CNN congelado hasta época {freeze_until}.")
            for param in self.model.cnn.parameters():
                param.requires_grad = False
        elif epoch == freeze_until + 1:
            logger.info("Descongelando backbone CNN para fine-tuning completo.")
            for param in self.model.cnn.parameters():
                param.requires_grad = True
            self.optimizer.add_param_group({
                "params":       list(self.model.cnn.parameters()),
                "lr":           self.cfg.learning_rate * 0.1,
                "weight_decay": self.cfg.weight_decay,
            })

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool) -> None:
        tag  = "best" if is_best else f"epoch_{epoch:03d}"
        path = self.checkpoint_dir / f"checkpoint_{tag}.pt"
        torch.save({
            "epoch":              epoch,
            "model_state_dict":   self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics":            metrics,
            "best_wa":            self.best_wa,
        }, path)
