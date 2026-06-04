from typing import Optional

import torch
import torch.nn as nn

from config import CNNConfig, ClassifierConfig, HopfieldConfig, TabularConfig
from models.cnn_frontend import CNNFrontend
from models.hopfield_pooling import HopfieldPoolingLayer

_STRIDE = CNNFrontend.TEMPORAL_STRIDE


class TabularEncoder(nn.Module):
    """MLP que convierte 7 features clínicas en un embedding de 64d."""

    def __init__(self, n_features: int = 7, embed_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, embed_dim),
            nn.GELU(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MurmurClassifier(nn.Module):
    """
    Clasificador de soplos cardíacos — pipeline completo:
    (B, 1, n_mels, T) → CNNFrontend → HopfieldPooling → [+ TabularEncoder] → MLP → (B, 3)
    """

    def __init__(
        self,
        cnn_cfg: CNNConfig,
        hopfield_cfg: HopfieldConfig,
        classifier_cfg: ClassifierConfig,
        tabular_cfg: Optional[TabularConfig] = None,
    ):
        super().__init__()

        self.cnn      = CNNFrontend(cnn_cfg)
        self.hopfield = HopfieldPoolingLayer(self.cnn.embedding_dim, hopfield_cfg)
        pooled_dim    = self.hopfield.output_dim

        self.use_tabular = tabular_cfg is not None and tabular_cfg.enabled
        if self.use_tabular:
            self.tab_encoder = TabularEncoder(tabular_cfg.n_features, tabular_cfg.embed_dim, tabular_cfg.dropout)
            fusion_dim = pooled_dim + tabular_cfg.embed_dim
        else:
            self.tab_encoder = None
            fusion_dim = pooled_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(p=classifier_cfg.dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(p=classifier_cfg.dropout),
            nn.Linear(fusion_dim // 2, classifier_cfg.num_classes),
        )

        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.cnn(x)                           # (B, T', D)

        # Máscara de padding: True en frames que son solo relleno de ceros.
        # Necesario porque T' varía de 3 a 16 en este dataset — hasta 80% de padding
        # en grabaciones cortas contaminaría la memoria asociativa del Hopfield.
        padding_mask = None
        if lengths is not None:
            T_prime  = embeddings.shape[1]
            t_actual = (lengths.float() / _STRIDE).ceil().long().clamp(1, T_prime)
            padding_mask = (
                torch.arange(T_prime, device=x.device).unsqueeze(0) >= t_actual.unsqueeze(1)
            )
            if not padding_mask.any():
                padding_mask = None

        pooled = self.hopfield(embeddings, padding_mask=padding_mask)

        if self.use_tabular:
            if tabular is None:
                tab_emb = torch.zeros(x.shape[0], self.tab_encoder.net[-2].out_features, device=x.device)
            else:
                tab_emb = self.tab_encoder(tabular.to(x.device))
            fused = torch.cat([pooled, tab_emb], dim=-1)
        else:
            fused = pooled

        logits = self.classifier(fused)

        if return_attention:
            return logits, self.hopfield.last_attention_weights

        return logits

    def predict(
        self,
        x: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            return torch.argmax(self.forward(x, tabular, lengths), dim=-1)

    @staticmethod
    def from_config(cfg) -> "MurmurClassifier":
        return MurmurClassifier(
            cfg.cnn,
            cfg.hopfield,
            cfg.classifier,
            cfg.tabular if hasattr(cfg, "tabular") else None,
        )
