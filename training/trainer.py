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
from training.metrics import CBFocalLoss, compute_challenge_score, format_metrics, aggregate_patient_preds

logger = logging.getLogger(__name__)


def spec_augment(spec: torch.Tensor, freq_mask_p: float = 0.15, time_mask_p: float = 0.15) -> torch.Tensor:
    """Enmascara franjas aleatorias de frecuencia y tiempo en el batch."""
    spec   = spec.clone()
    B, C, n_mels, T = spec.shape
    f_size = max(1, int(n_mels * freq_mask_p))
    t_size = max(1, int(T * time_mask_p))

    for b in range(B):
        f0 = random.randint(0, n_mels - f_size)
        spec[b, :, f0:f0 + f_size, :] = 0.0
        t0 = random.randint(0, max(0, T - t_size))
        spec[b, :, :, t0:t0 + t_size] = 0.0

    return spec


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

    return (
        lam * spectrograms + (1.0 - lam) * spectrograms[idx],
        lam * tabulars    + (1.0 - lam) * tabulars[idx],
        labels, labels[idx], lam,
    )


def sgdr_lr(epoch: int, base_lr: float, warmup_epochs: int, T_0: int, lr_min_factor: float = 0.01) -> float:
    """Warmup lineal seguido de Cosine Annealing con Warm Restarts (SGDR)."""
    eta_min = base_lr * lr_min_factor

    if epoch <= warmup_epochs:
        return base_lr * epoch / max(1, warmup_epochs)

    e_after_warmup = epoch - warmup_epochs
    t_in_cycle     = (e_after_warmup - 1) % T_0
    cos_val        = 0.5 * (1.0 + math.cos(math.pi * t_in_cycle / T_0))
    return eta_min + (base_lr - eta_min) * cos_val


class Trainer:
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

        self.criterion = CBFocalLoss(
            samples_per_class=samples_per_class,
            gamma=train_cfg.focal_gamma,
            beta=train_cfg.focal_beta,
            label_smoothing=train_cfg.label_smoothing,
        ).to(train_cfg.device)

        cnn_param_ids  = {id(p) for p in model.cnn.parameters()}
        non_cnn_params = [p for p in model.parameters() if id(p) not in cnn_param_ids]

        self.optimizer = AdamW(non_cnn_params, lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)

        self.device         = train_cfg.device
        self.best_wa        = 0.0
        self.checkpoint_dir = train_cfg.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Trainer — epochs={train_cfg.epochs} | SGDR T_0={train_cfg.sgdr_T0} | "
            f"warmup={train_cfg.warmup_epochs} | lr={train_cfg.learning_rate} | "
            f"wd={train_cfg.weight_decay} | mixup_α={train_cfg.mixup_alpha} | "
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
                f"Loss: {train_loss:.4f} | WA: {wa:.4f} | "
                f"LR: {current_lr:.2e} | Tiempo: {elapsed:.1f}s"
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
        for spectrograms, tabulars, labels, _, _ in pbar:
            spectrograms = spectrograms.to(self.device, non_blocking=True)
            tabulars     = tabulars.to(self.device, non_blocking=True)
            labels       = labels.to(self.device, non_blocking=True)

            if random.random() < self.cfg.spec_augment_prob:
                spectrograms = spec_augment(spectrograms, self.cfg.spec_freq_mask_p, self.cfg.spec_time_mask_p)

            spec_m, tab_m, labels_a, labels_b, lam = clinical_mixup(
                spectrograms, tabulars, labels, alpha=self.cfg.mixup_alpha
            )

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(spec_m, tab_m)

            if lam < 1.0:
                loss = lam * self.criterion(logits, labels_a) + (1.0 - lam) * self.criterion(logits, labels_b)
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
        all_probs, all_labels, all_pids = [], [], []

        for spectrograms, tabulars, labels, patient_ids, lengths in tqdm(self.val_loader, desc=f"Val   {epoch}", leave=False):
            spectrograms = spectrograms.to(self.device, non_blocking=True)
            tabulars     = tabulars.to(self.device, non_blocking=True)
            probs        = torch.softmax(self.model(spectrograms, tabulars, lengths.to(self.device)), dim=-1)
            all_probs.append(probs.cpu())
            all_labels.append(labels)
            all_pids.extend(patient_ids)

        preds, labels = aggregate_patient_preds(
            all_pids, torch.cat(all_probs), torch.cat(all_labels)
        )
        return compute_challenge_score(labels, preds)

    def _update_lr(self, epoch: int) -> None:
        new_lr = sgdr_lr(epoch, self.cfg.learning_rate, self.cfg.warmup_epochs, self.cfg.sgdr_T0, self.cfg.lr_min_factor)
        self.optimizer.param_groups[0]["lr"] = new_lr
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
            "epoch":                epoch,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics":              metrics,
            "best_wa":              self.best_wa,
        }, path)
