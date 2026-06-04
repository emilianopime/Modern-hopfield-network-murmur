import math
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class PreprocessConfig:
    sample_rate: int = 4000
    wavelet: str = "db6"
    wavelet_level: int = 5
    threshold_mode: str = "soft"
    sg_window_length: int = 51
    sg_polyorder: int = 3
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    f_min: float = 20.0
    f_max: float = 1000.0
    power: float = 2.0
    top_db: float = 80.0
    cache_dir: Path = Path("data/cache")
    cache_enabled: bool = True


@dataclass
class CNNConfig:
    backbone: str = "resnet18"
    pretrained: bool = True
    embedding_dim: int = 512
    # Backbone congelado hasta esta época para que Hopfield consolide primero
    freeze_backbone_epochs: int = 12


@dataclass
class HopfieldConfig:
    hidden_size: int = 512
    output_size: int = 512
    num_heads: int = 8
    quantity: int = 4
    dropout: float = 0.1
    beta: float = None

    @property
    def beta_value(self) -> float:
        if self.beta is not None:
            return self.beta
        return 1.0 / math.sqrt(512)


@dataclass
class ClassifierConfig:
    num_classes: int = 3
    class_names: list = field(default_factory=lambda: ["Present", "Absent", "Unknown"])
    dropout: float = 0.45


@dataclass
class TabularConfig:
    enabled: bool = True
    n_features: int = 7
    embed_dim: int = 64
    dropout: float = 0.2


@dataclass
class DataConfig:
    dataset_path: Path = Path("data/physionet_2022")
    auscultation_positions: list = field(
        default_factory=lambda: ["AV", "PV", "TV", "MV"]
    )
    val_split: float = 0.15
    random_seed: int = 42
    use_weighted_sampler: bool = True
    # Unknown×10, Present×5 para compensar el desbalance de clases
    sampler_weights: list = field(
        default_factory=lambda: [5.0, 1.0, 10.0]
    )


@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_workers: int = 6
    pin_memory: bool = True
    epochs: int = 120
    learning_rate: float = 5e-5
    weight_decay: float = 5e-4
    grad_clip: float = 1.0
    warmup_epochs: int = 5
    lr_min_factor: float = 0.01
    # SGDR: reinicios en épocas ~26, 46, 66, 86, 106 (después del warmup)
    sgdr_T0: int = 20
    focal_gamma: float = 2.0
    focal_beta: float = 0.9999
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.3
    spec_augment_prob: float = 0.5
    spec_freq_mask_p: float = 0.15
    spec_time_mask_p: float = 0.15
    checkpoint_dir: Path = Path("checkpoints")
    save_every_n_epochs: int = 5

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# Configuración global
# ─────────────────────────────────────────────
@dataclass
class Config:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    hopfield: HopfieldConfig = field(default_factory=HopfieldConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    tabular: TabularConfig = field(default_factory=TabularConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


cfg = Config()
