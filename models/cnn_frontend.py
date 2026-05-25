"""
models/cnn_frontend.py — FASE 2: Extractor CNN (Frontend)

Transforma el espectrograma de Mel (salida de Fase 1) en una secuencia
temporal de embeddings vectoriales densos que alimentan la capa MHN.

Input:  (B, 1, n_mels=128, T_frames)    — 1 canal de escala de grises
Output: (B, T', embedding_dim)           — secuencia temporal de embeddings

Arquitectura:
    Se reutiliza ResNet18 (o EfficientNet-B0) con tres modificaciones:
    1. Primer conv: 3 canales → 1 canal (PCG es monocanal).
    2. Se elimina el Global Average Pooling final para preservar el eje temporal.
    3. Se aplica AdaptiveAvgPool2d SOLO sobre la dimensión de frecuencia
       (eje vertical del espectrograma), conservando el eje temporal.
    El resultado es una secuencia (B, T', 512) lista para HopfieldPooling.

Por qué NO se usa Global Average Pooling:
    GAP colapsaría todo el tiempo, diluyendo un soplo breve entre minutos de
    audio normal. HopfieldPooling posterior se encarga de la reducción temporal
    de forma selectiva (Aprendizaje de Múltiples Instancias).
"""

import math
from typing import Literal

import torch
import torch.nn as nn
import torchvision.models as tv_models

from config import CNNConfig


class CNNFrontend(nn.Module):
    """
    Extractor de características CNN con backbone configurable.

    Args:
        config: Instancia de CNNConfig.
    """

    def __init__(self, config: CNNConfig):
        super().__init__()
        self.embedding_dim = config.embedding_dim

        if config.backbone == "resnet18":
            self.backbone, self.embedding_dim = self._build_resnet18(
                pretrained=config.pretrained
            )
        elif config.backbone == "efficientnet_b0":
            self.backbone, self.embedding_dim = self._build_efficientnet_b0(
                pretrained=config.pretrained
            )
        else:
            raise ValueError(
                f"Backbone no soportado: '{config.backbone}'. "
                "Opciones: 'resnet18', 'efficientnet_b0'."
            )

        # Pool adaptativo sobre el eje de frecuencia → conserva eje temporal
        # Input: (B, C, freq, T') → Output: (B, C, 1, T')
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

    # ─────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Espectrograma de Mel con shape (B, 1, n_mels, T_frames).

        Returns:
            Secuencia de embeddings con shape (B, T', embedding_dim).
            T' = T_frames / 32  (tras 5 capas de stride de ResNet18).
        """
        x = self.backbone(x)               # (B, C, freq', T')
        x = self.freq_pool(x)              # (B, C, 1, T')
        x = x.squeeze(2)                   # (B, C, T')
        x = x.permute(0, 2, 1)            # (B, T', C) — formato secuencia para MHN
        return x

    # ─────────────────────────────────────────
    # Constructores de backbone
    # ─────────────────────────────────────────

    @staticmethod
    def _build_resnet18(pretrained: bool) -> tuple[nn.Sequential, int]:
        """
        ResNet18 modificada para entrada monocanal y salida de mapa de características.

        Cambios respecto a la arquitectura original:
        - conv1: 3→1 canal de entrada (escala de grises)
        - Se elimina avgpool y fc (la reducción temporal la hace HopfieldPooling)
        """
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = tv_models.resnet18(weights=weights)

        # Adaptar primer conv a 1 canal de entrada
        resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        if pretrained:
            # Inicializar canal único con el promedio de los 3 canales preentrenados
            with torch.no_grad():
                orig_weight = tv_models.resnet18(
                    weights=tv_models.ResNet18_Weights.DEFAULT
                ).conv1.weight
                resnet.conv1.weight = nn.Parameter(orig_weight.mean(dim=1, keepdim=True))

        # Secuencia sin avgpool ni fc → preserva mapa espacial completo
        backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # (B, 64,  H/4,  T/4)
            resnet.layer2,  # (B, 128, H/8,  T/8)
            resnet.layer3,  # (B, 256, H/16, T/16)
            resnet.layer4,  # (B, 512, H/32, T/32)
        )
        return backbone, 512

    @staticmethod
    def _build_efficientnet_b0(pretrained: bool) -> tuple[nn.Sequential, int]:
        """
        EfficientNet-B0 adaptada a entrada monocanal.
        Embedding dim = 1280 (último bloque de features de EfficientNet-B0).
        """
        weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        effnet = tv_models.efficientnet_b0(weights=weights)

        # Adaptar primer conv a 1 canal
        orig_conv = effnet.features[0][0]
        effnet.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias is not None,
        )
        if pretrained:
            with torch.no_grad():
                effnet.features[0][0].weight = nn.Parameter(
                    orig_conv.weight.mean(dim=1, keepdim=True)
                )

        # Solo el bloque de features (sin avgpool ni classifier)
        return effnet.features, 1280
