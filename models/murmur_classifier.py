"""
models/murmur_classifier.py — FASE 4: Modelo completo (integración de fases 2-3-4)

Flujo end-to-end:
    (B, 1, n_mels, T)
        │
        ▼  Fase 2: CNN Frontend
    (B, T', embedding_dim)
        │
        ▼  Fase 3: HopfieldPooling
    (B, quantity * output_size)
        │
        ▼  Fase 4: Clasificador lineal
    (B, 3)  →  [Present, Absent, Unknown]

Interpretabilidad:
    Los pesos de atención de HopfieldPooling se almacenan en
    `self.last_attention_weights` para su extracción y visualización
    sobre el fonocardiograma original (ver utils/interpretability.py).
"""

import math

import torch
import torch.nn as nn

from config import CNNConfig, ClassifierConfig, HopfieldConfig
from models.cnn_frontend import CNNFrontend
from models.hopfield_pooling import HopfieldPoolingLayer


class MurmurClassifier(nn.Module):
    """
    Red completa para clasificación de soplos cardíacos.

    Args:
        cnn_cfg:        Configuración del extractor CNN (Fase 2).
        hopfield_cfg:   Configuración de HopfieldPooling (Fase 3).
        classifier_cfg: Configuración del clasificador (Fase 4).
    """

    def __init__(
        self,
        cnn_cfg: CNNConfig,
        hopfield_cfg: HopfieldConfig,
        classifier_cfg: ClassifierConfig,
    ):
        super().__init__()

        # ── Fase 2: CNN Frontend ────────────────────────────────────────────
        self.cnn = CNNFrontend(cnn_cfg)
        embedding_dim = self.cnn.embedding_dim

        # ── Fase 3: Hopfield Pooling ────────────────────────────────────────
        self.hopfield = HopfieldPoolingLayer(embedding_dim, hopfield_cfg)
        pooled_dim = self.hopfield.output_dim

        # ── Fase 4: Clasificador lineal ─────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Dropout(p=classifier_cfg.dropout),
            nn.Linear(pooled_dim, pooled_dim // 2),
            nn.GELU(),
            nn.Dropout(p=classifier_cfg.dropout),
            nn.Linear(pooled_dim // 2, classifier_cfg.num_classes),
        )

        self._init_weights()

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:                (B, 1, n_mels, T_frames).
            return_attention: Si True, devuelve también los pesos de atención
                              del Hopfield para interpretabilidad clínica.

        Returns:
            logits: (B, num_classes) — sin softmax (usar con CrossEntropyLoss).
            attn:   (opcional) pesos de atención de HopfieldPooling.
        """
        # Fase 2
        embeddings = self.cnn(x)              # (B, T', embedding_dim)

        # Fase 3
        pooled = self.hopfield(embeddings)    # (B, quantity * output_size)

        # Fase 4
        logits = self.classifier(pooled)      # (B, num_classes)

        if return_attention:
            attn = self.hopfield.last_attention_weights
            return logits, attn

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predicción de clases (argmax de softmax). Útil en inferencia."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Probabilidades por clase. Útil para agregación multi-focal."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)

    def _init_weights(self):
        """Inicialización Xavier para las capas del clasificador."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def from_config(cfg) -> "MurmurClassifier":
        """Factory method: crea el modelo desde la instancia global de Config."""
        return MurmurClassifier(cfg.cnn, cfg.hopfield, cfg.classifier)
