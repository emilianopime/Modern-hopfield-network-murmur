"""
models/murmur_classifier.py — FASE 4: Modelo completo (integración de fases 2-3-4)

Flujo end-to-end:
    (B, 1, n_mels, T)
        │
        ▼  Fase 2: CNN Frontend
    (B, T', embedding_dim)
        │
        ▼  Fase 3: HopfieldPooling  ← NÚCLEO DEL MODELO (memoria asociativa)
    (B, quantity * output_size)
        │
        ├──── (B, tabular_embed_dim) ← TabularEncoder (fusión clínica)
        │                              [age_group, sex, height, weight, bmi,
        │                               recording_count, pregnancy]
        ▼
    (B, pooled_dim + tabular_embed_dim)
        │
        ▼  Fase 4: Clasificador MLP
    (B, 3)  →  [Present, Absent, Unknown]

TabularEncoder:
    MLP simple inspirado en el feature engineering de PF5367600.py (amigo).
    7 features clínicas → 64d embedding → concatenado con output Hopfield.
    BatchNorm1d como primera capa → normalización automática sin estadísticas externas.

Interpretabilidad:
    Los pesos de atención de HopfieldPooling se almacenan en
    `self.last_attention_weights` para su extracción y visualización
    sobre el fonocardiograma original (ver utils/interpretability.py).
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from config import CNNConfig, ClassifierConfig, HopfieldConfig, TabularConfig
from models.cnn_frontend import CNNFrontend
from models.hopfield_pooling import HopfieldPoolingLayer


# ─────────────────────────────────────────────
# Encoder tabular clínico
# Fuente feature engineering: PF5367600.py (FeatureEngineer)
# ─────────────────────────────────────────────

class TabularEncoder(nn.Module):
    """
    Encoder MLP para features clínicas del paciente.

    Convierte 7 features numéricas (age_group, sex, height, weight,
    bmi, recording_count, pregnancy) en un embedding denso de 64d
    que se fusiona con el output de HopfieldPooling.

    BatchNorm1d como primera capa → normaliza automáticamente sin
    necesitar medias/stds precalculadas del dataset.

    Args:
        n_features: Número de features de entrada (default 7).
        embed_dim:  Dimensión del embedding de salida (default 64).
        dropout:    Tasa de dropout.
    """

    def __init__(self, n_features: int = 7, embed_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(n_features),          # normalización automática
            nn.Linear(n_features, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, embed_dim),
            nn.GELU(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_features) — features clínicas normalizadas.
        Returns:
            (B, embed_dim) — embedding tabular.
        """
        return self.net(x)


# ─────────────────────────────────────────────
# Clasificador completo
# ─────────────────────────────────────────────

class MurmurClassifier(nn.Module):
    """
    Red completa para clasificación de soplos cardíacos.

    El núcleo es HopfieldPooling (Fase 3), que actúa como memoria
    asociativa clínica: compara los embeddings del paciente contra
    prototipos aprendidos de murmur vs. corazón sano.

    La fusión tabular añade contexto clínico (edad, peso, BMI, etc.)
    sin reemplazar ni interferir con la memoria Hopfield.

    Args:
        cnn_cfg:        Configuración del extractor CNN (Fase 2).
        hopfield_cfg:   Configuración de HopfieldPooling (Fase 3).
        classifier_cfg: Configuración del clasificador (Fase 4).
        tabular_cfg:    Configuración del encoder tabular (opcional).
    """

    def __init__(
        self,
        cnn_cfg: CNNConfig,
        hopfield_cfg: HopfieldConfig,
        classifier_cfg: ClassifierConfig,
        tabular_cfg: Optional[TabularConfig] = None,
    ):
        super().__init__()

        # ── Fase 2: CNN Frontend ──────────────────────────────────────────
        self.cnn = CNNFrontend(cnn_cfg)
        embedding_dim = self.cnn.embedding_dim

        # ── Fase 3: Hopfield Pooling (núcleo del modelo) ──────────────────
        self.hopfield = HopfieldPoolingLayer(embedding_dim, hopfield_cfg)
        pooled_dim = self.hopfield.output_dim

        # ── Encoder tabular (fusión clínica) ──────────────────────────────
        self.use_tabular = tabular_cfg is not None and tabular_cfg.enabled
        if self.use_tabular:
            self.tab_encoder = TabularEncoder(
                n_features=tabular_cfg.n_features,
                embed_dim=tabular_cfg.embed_dim,
                dropout=tabular_cfg.dropout,
            )
            fusion_dim = pooled_dim + tabular_cfg.embed_dim
        else:
            self.tab_encoder = None
            fusion_dim = pooled_dim

        # ── Fase 4: Clasificador MLP ──────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(p=classifier_cfg.dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(p=classifier_cfg.dropout),
            nn.Linear(fusion_dim // 2, classifier_cfg.num_classes),
        )

        self._init_weights()

    def forward(
        self,
        x: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:                (B, 1, n_mels, T_frames) — espectrograma Mel.
            tabular:          (B, n_features) — features clínicas (opcional).
                              Si None y use_tabular=True, se usan ceros.
            return_attention: Si True, devuelve también los pesos de atención
                              del Hopfield para interpretabilidad clínica.

        Returns:
            logits: (B, num_classes) — sin softmax (usar con CBFocalLoss).
            attn:   (opcional) pesos de atención de HopfieldPooling.
        """
        # ── Fase 2: CNN → embeddings temporales ──────────────────────────
        embeddings = self.cnn(x)                           # (B, T', embedding_dim)

        # ── Fase 3: HopfieldPooling → representación agregada ────────────
        pooled = self.hopfield(embeddings)                 # (B, quantity * output_size)

        # ── Fusión tabular ────────────────────────────────────────────────
        if self.use_tabular:
            if tabular is None:
                # Fallback para modo explain (sin features tabulares)
                tab_emb = torch.zeros(
                    x.shape[0], self.tab_encoder.net[-2].out_features,
                    device=x.device
                )
            else:
                tab_emb = self.tab_encoder(tabular.to(x.device))   # (B, embed_dim)
            fused = torch.cat([pooled, tab_emb], dim=-1)           # (B, fusion_dim)
        else:
            fused = pooled

        # ── Fase 4: clasificación ─────────────────────────────────────────
        logits = self.classifier(fused)                    # (B, num_classes)

        if return_attention:
            attn = self.hopfield.last_attention_weights
            return logits, attn

        return logits

    def predict(self, x: torch.Tensor, tabular: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predicción de clases (argmax). Útil en inferencia."""
        with torch.no_grad():
            logits = self.forward(x, tabular)
            return torch.argmax(logits, dim=-1)

    def predict_proba(self, x: torch.Tensor, tabular: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Probabilidades por clase. Útil para agregación multi-focal."""
        with torch.no_grad():
            logits = self.forward(x, tabular)
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
        return MurmurClassifier(
            cfg.cnn,
            cfg.hopfield,
            cfg.classifier,
            cfg.tabular if hasattr(cfg, "tabular") else None,
        )
