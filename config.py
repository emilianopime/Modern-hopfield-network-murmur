"""
config.py — Configuración centralizada de hiperparámetros y rutas.
Todas las fases leen sus parámetros exclusivamente desde aquí.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch


# ─────────────────────────────────────────────
# Fase 1: Preprocesamiento de señal
# ─────────────────────────────────────────────
@dataclass
class PreprocessConfig:
    sample_rate: int = 4000

    # (1-a) Transformada Wavelet — denoising
    wavelet: str = "db6"
    wavelet_level: int = 5
    threshold_mode: str = "soft"

    # (1-b) Filtro Savitzky-Golay
    sg_window_length: int = 51
    sg_polyorder: int = 3

    # (1-c) Mel Spectrogram
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    f_min: float = 20.0
    f_max: float = 1000.0
    power: float = 2.0
    top_db: float = 80.0

    cache_dir: Path = Path("data/cache")
    cache_enabled: bool = True


# ─────────────────────────────────────────────
# Fase 2: Extractor CNN (Frontend)
# ─────────────────────────────────────────────
@dataclass
class CNNConfig:
    backbone: str = "resnet18"
    pretrained: bool = True
    embedding_dim: int = 512
    # v3: extendido 5→12 para que HopfieldPooling consolide patrones
    # antes de que el CNN empiece a moverse (evita desestabilización época 6)
    freeze_backbone_epochs: int = 12


# ─────────────────────────────────────────────
# Fase 3: Modern Hopfield Network (HopfieldPooling)
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# Fase 4: Clasificador
# ─────────────────────────────────────────────
@dataclass
class ClassifierConfig:
    num_classes: int = 3
    class_names: list = field(default_factory=lambda: ["Present", "Absent", "Unknown"])
    # v3: 0.3→0.45 para frenar memorización (loss train bajó a 0.01 en v2)
    dropout: float = 0.45


# ─────────────────────────────────────────────
# Features tabulares clínicos
# ─────────────────────────────────────────────
@dataclass
class TabularConfig:
    enabled: bool = True
    n_features: int = 7
    embed_dim: int = 64
    dropout: float = 0.2


# ─────────────────────────────────────────────
# Dataset y DataLoader
# ─────────────────────────────────────────────
@dataclass
class DataConfig:
    dataset_path: Path = Path("data/physionet_2022")
    auscultation_positions: list = field(
        default_factory=lambda: ["AV", "PV", "TV", "MV"]
    )
    val_split: float = 0.15
    random_seed: int = 42

    # v3: WeightedRandomSampler — oversampling de Unknown×10 y Present×5
    # Ataca el colapso de Unknown directamente (132 muestras vs 2026 Absent)
    use_weighted_sampler: bool = True
    sampler_weights: list = field(
        default_factory=lambda: [5.0, 1.0, 10.0]  # [Present, Absent, Unknown]
    )


# ─────────────────────────────────────────────
# Entrenamiento
# ─────────────────────────────────────────────
@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_workers: int = 6
    pin_memory: bool = True

    epochs: int = 120
    # v3: 1e-4→5e-5 — convergencia más lenta desplaza el pico a épocas más tardías
    learning_rate: float = 5e-5
    # v3: 1e-4→5e-4 — más regularización L2 para frenar overfitting en Fase 2
    weight_decay: float = 5e-4
    grad_clip: float = 1.0

    # Warmup inicial (épocas 1..warmup_epochs): LR 0 → learning_rate
    warmup_epochs: int = 5
    lr_min_factor: float = 0.01

    # v3: SGDR — Cosine Annealing con Warm Restarts (reemplaza cosine simple)
    # T_0=20: reinicia el LR cada 20 épocas después del warmup.
    # Con warmup=5 y T_0=20, los reinicios ocurren en épocas ~26, 46, 66, 86, 106.
    # Cada reinicio es una nueva oportunidad de explorar — desplaza el mejor
    # checkpoint a épocas 40-80 en lugar de 33.
    sgdr_T0: int = 20

    # v3: CB-Focal Loss con label smoothing ε=0.1
    # Reduce sobreconfianza en Absent (la clase fácil) → mejor generalización
    focal_gamma: float = 2.0
    focal_beta: float = 0.9999
    label_smoothing: float = 0.1

    # ClinicalMixUp
    mixup_alpha: float = 0.3

    # v3: SpecAugment — enmascara franjas de frecuencia y tiempo en espectrogramas
    # Impide que el modelo memorice patrones específicos del training set
    spec_augment_prob: float = 0.5    # probabilidad de aplicar por batch
    spec_freq_mask_p: float = 0.15   # fracción de bandas mel a enmascarar
    spec_time_mask_p: float = 0.15   # fracción de frames temporales a enmascarar

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
