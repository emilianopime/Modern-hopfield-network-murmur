"""
training/trainer.py — Bucle de entrenamiento y validación

Integra todas las fases en el ciclo de entrenamiento estándar de PyTorch.
Implementa:
    - Descongelado gradual del backbone CNN (freeze_backbone_epochs)
    - Clip de gradientes (estabilidad con MHN y matrices de correlación grandes)
    - ReduceLROnPlateau sobre Weighted Accuracy del challenge
    - Guardado de checkpoints con el mejor modelo
    - Pesos de clase en CrossEntropyLoss (desbalance Present/Absent/Unknown)
"""

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ClassifierConfig, TrainingConfig, CNNConfig
from training.metrics import compute_challenge_score, format_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """
    Encapsula el ciclo completo de entrenamiento y validación.

    Args:
        model:          Instancia de MurmurClassifier.
        train_loader:   DataLoader de entrenamiento.
        val_loader:     DataLoader de validación.
        train_cfg:      Configuración de entrenamiento.
        classifier_cfg: Para los pesos de clase en la pérdida.
        cnn_cfg:        Para saber cuándo descongelar el backbone.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_cfg: TrainingConfig,
        classifier_cfg: ClassifierConfig,
        cnn_cfg: CNNConfig,
    ):
        self.model = model.to(train_cfg.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = train_cfg
        self.cnn_cfg = cnn_cfg

        # Pesos de clase para CrossEntropyLoss (desbalance del dataset)
        class_weights = torch.tensor(
            classifier_cfg.class_weights, dtype=torch.float32
        ).to(train_cfg.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Inicializar optimizer SOLO con parámetros no-CNN.
        # Los parámetros del CNN se agregan como segundo grupo al descongelar
        # (epoch == freeze_backbone_epochs + 1), con LR reducido para fine-tuning.
        # Esto evita el error "some parameters appear in more than one parameter group"
        # que ocurre si se llama add_param_group con params ya registrados.
        cnn_param_ids = {id(p) for p in model.cnn.parameters()}
        non_cnn_params = [p for p in model.parameters() if id(p) not in cnn_param_ids]

        self.optimizer = AdamW(
            non_cnn_params,
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",                        # maximizar Weighted Accuracy
            patience=train_cfg.lr_patience,
            factor=train_cfg.lr_factor,
        )

        self.device = train_cfg.device
        self.best_wa = 0.0
        self.checkpoint_dir = train_cfg.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> None:
        """Ejecuta el entrenamiento completo por `cfg.epochs` épocas."""
        logger.info(f"Iniciando entrenamiento en {self.device} por {self.cfg.epochs} épocas.")

        for epoch in range(1, self.cfg.epochs + 1):
            # Descongelar backbone CNN gradualmente
            self._maybe_unfreeze_backbone(epoch)

            t0 = time.time()
            train_loss = self._train_epoch(epoch)
            val_metrics = self._validate_epoch(epoch)

            elapsed = time.time() - t0
            wa = val_metrics["weighted_accuracy"]
            self.scheduler.step(wa)

            logger.info(
                f"Época {epoch:03d}/{self.cfg.epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"WA: {wa:.4f} | "
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
        """Una época de entrenamiento. Devuelve la pérdida media."""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Train {epoch}", leave=False)
        for spectrograms, labels in pbar:
            spectrograms = spectrograms.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(spectrograms)
            loss = self.criterion(logits, labels)
            loss.backward()

            # Clip de gradientes: estabiliza el entrenamiento de HopfieldPooling
            # (las matrices de correlación pueden producir gradientes grandes)
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

        for spectrograms, labels in tqdm(self.val_loader, desc=f"Val   {epoch}", leave=False):
            spectrograms = spectrograms.to(self.device, non_blocking=True)
            logits = self.model(spectrograms)
            preds = torch.argmax(logits, dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(labels)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        return compute_challenge_score(all_labels, all_preds)

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
            logger.info(f"Descongelando backbone CNN para fine-tuning completo.")
            for param in self.model.cnn.parameters():
                param.requires_grad = True
            # Agregar CNN como nuevo grupo de parámetros (nunca estuvo en el optimizer)
            # LR reducido 10× para no destruir los features preentrenados de ImageNet
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
