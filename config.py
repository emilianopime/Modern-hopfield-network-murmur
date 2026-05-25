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
    # Características del dataset CirCor DigiScope
    sample_rate: int = 4000            # Hz — frecuencia de muestreo nativa del dataset

    # (1-a) Transformada Wavelet — denoising de ruido clínico
    wavelet: str = "db6"               # Daubechies 6: suave, ideal para señales biomédicas
    wavelet_level: int = 5             # Niveles de descomposición
    threshold_mode: str = "soft"       # 'soft' preserva morfología del ciclo cardíaco

    # (1-b) Filtro Savitzky-Golay — suavizado polinomial post-wavelet
    sg_window_length: int = 51         # ~12.75 ms a 4000 Hz; mantiene transiciones cardíacas
    sg_polyorder: int = 3              # Polinomio cúbico: equilibra suavizado y fidelidad

    # (1-c) Mel Spectrogram
    n_fft: int = 1024                  # Ventana FFT (~256 ms a 4000 Hz)
    hop_length: int = 256              # Salto de ventana (~64 ms → ~62 frames/seg)
    n_mels: int = 128                  # Bandas de Mel
    f_min: float = 20.0                # Hz — inicio del rango de interés PCG
    f_max: float = 1000.0              # Hz — límite superior (soplos: 100-600 Hz)
    power: float = 2.0                 # Potencia del espectrograma (2.0 = potencia de energía)
    top_db: float = 80.0               # Rango dinámico en dB para normalización logarítmica

    # Caching de espectrogramas preprocesados en disco
    cache_dir: Path = Path("data/cache")
    cache_enabled: bool = True


# ─────────────────────────────────────────────
# Fase 2: Extractor CNN (Frontend)
# ─────────────────────────────────────────────
@dataclass
class CNNConfig:
    backbone: str = "resnet18"         # "resnet18" | "efficientnet_b0"
    pretrained: bool = True            # ImageNet pretraining (canal 1→adaptado)
    embedding_dim: int = 512           # Dimensión de embeddings de salida por frame temporal
    freeze_backbone_epochs: int = 5    # Epochs iniciales con backbone congelado (fine-tuning gradual)


# ─────────────────────────────────────────────
# Fase 3: Modern Hopfield Network (HopfieldPooling)
# ─────────────────────────────────────────────
@dataclass
class HopfieldConfig:
    hidden_size: int = 512             # Dimensión del espacio de asociación
    output_size: int = 512             # Dimensión de salida por patrón almacenado
    num_heads: int = 8                 # Cabezas de atención multi-patrón
    quantity: int = 4                  # Prototipos almacenados en memoria asociativa
    dropout: float = 0.1

    # β = 1/√d (inicialización recomendada por el reporte técnico)
    # None → se calcula automáticamente en el modelo desde embedding_dim
    beta: float = None

    @property
    def beta_value(self) -> float:
        """β efectivo: si no se especifica, usa 1/√embedding_dim."""
        if self.beta is not None:
            return self.beta
        # Se delega al modelo; aquí se expone la fórmula de referencia
        return 1.0 / math.sqrt(512)


# ─────────────────────────────────────────────
# Fase 4: Clasificador
# ─────────────────────────────────────────────
@dataclass
class ClassifierConfig:
    num_classes: int = 3               # Present | Absent | Unknown
    class_names: list = field(default_factory=lambda: ["Present", "Absent", "Unknown"])
    dropout: float = 0.3
    # Pesos de clase para CrossEntropyLoss (compensar desbalance del dataset)
    # Orden: [Present, Absent, Unknown]
    class_weights: list = field(default_factory=lambda: [5.0, 1.0, 7.0])


# ─────────────────────────────────────────────
# Dataset y DataLoader
# ─────────────────────────────────────────────
@dataclass
class DataConfig:
    dataset_path: Path = Path("data/physionet_2022")
    auscultation_positions: list = field(
        default_factory=lambda: ["AV", "PV", "TV", "MV"]
    )
    # Partición training/validation
    val_split: float = 0.15
    random_seed: int = 42


# ─────────────────────────────────────────────
# Entrenamiento
# ─────────────────────────────────────────────
@dataclass
class TrainingConfig:
    # Recursos de hardware (RTX 3060 + Ryzen 7 5700x)
    batch_size: int = 32               # 32-64 según VRAM disponible (MHN usa matrices de correlación)
    num_workers: int = 6               # Ryzen 7 5700x (8 núcleos / 16 hilos)
    pin_memory: bool = True

    epochs: int = 60
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0             # Clip de gradientes para estabilidad con MHN

    # Scheduler: ReduceLROnPlateau
    lr_patience: int = 7
    lr_factor: float = 0.5

    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints")
    save_every_n_epochs: int = 5

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# Configuración global (punto de entrada único)
# ─────────────────────────────────────────────
@dataclass
class Config:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    hopfield: HopfieldConfig = field(default_factory=HopfieldConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# Instancia por defecto — importar directamente en otros módulos
cfg = Config()
