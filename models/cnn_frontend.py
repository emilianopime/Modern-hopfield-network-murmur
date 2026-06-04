import torch
import torch.nn as nn
import torchvision.models as tv_models

from config import CNNConfig


class CNNFrontend(nn.Module):
    """
    Extrae embeddings temporales del espectrograma de Mel.
    Entrada:  (B, 1, n_mels, T_frames)
    Salida:   (B, T', embedding_dim) donde T' ≈ T/32
    """

    TEMPORAL_STRIDE = 32  # Downsampling temporal acumulado de ResNet18 (conv1×2 + maxpool×2 + layer2×2 + layer3×2 + layer4×2)

    def __init__(self, config: CNNConfig):
        super().__init__()
        self.embedding_dim = config.embedding_dim

        if config.backbone == "resnet18":
            self.backbone, self.embedding_dim = self._build_resnet18(config.pretrained)
        elif config.backbone == "efficientnet_b0":
            self.backbone, self.embedding_dim = self._build_efficientnet_b0(config.pretrained)
        else:
            raise ValueError(f"Backbone no soportado: '{config.backbone}'. Opciones: 'resnet18', 'efficientnet_b0'.")

        # Pool sobre el eje de frecuencia, preserva el eje temporal para HopfieldPooling
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)    # (B, C, freq', T')
        x = self.freq_pool(x)   # (B, C, 1, T')
        x = x.squeeze(2)        # (B, C, T')
        x = x.permute(0, 2, 1)  # (B, T', C)
        return x

    @staticmethod
    def _build_resnet18(pretrained: bool) -> tuple[nn.Sequential, int]:
        weights  = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet   = tv_models.resnet18(weights=weights)

        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                orig_weight = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT).conv1.weight
                resnet.conv1.weight = nn.Parameter(orig_weight.mean(dim=1, keepdim=True))

        backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        return backbone, 512

    @staticmethod
    def _build_efficientnet_b0(pretrained: bool) -> tuple[nn.Sequential, int]:
        weights  = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        effnet   = tv_models.efficientnet_b0(weights=weights)
        orig_conv = effnet.features[0][0]

        effnet.features[0][0] = nn.Conv2d(
            1, orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias is not None,
        )
        if pretrained:
            with torch.no_grad():
                effnet.features[0][0].weight = nn.Parameter(orig_conv.weight.mean(dim=1, keepdim=True))

        return effnet.features, 1280
