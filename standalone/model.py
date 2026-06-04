"""
Modern Hopfield Network — Detección de Soplos Cardíacos
PhysioNet Challenge 2022 · CirCor DigiScope · Clasificación 3-clases (Present / Absent / Unknown)

Archivo único autocontenido. No requiere importar ningún otro módulo local del proyecto.

Uso:
    python standalone/model.py train
    python standalone/model.py preprocess
    python standalone/model.py eval   --checkpoint checkpoints/checkpoint_best.pt
    python standalone/model.py explain --wav data/physionet_2022/12345_AV.wav \
                                        --checkpoint checkpoints/checkpoint_best.pt
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import argparse
import hashlib
import logging
import math
import random
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import torchvision.models as tv_models
from scipy.signal import savgol_filter
from sklearn.metrics import confusion_matrix
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

try:
    from hflayers import HopfieldPooling as _HopfieldPooling
    HOPFIELD_AVAILABLE = True
except ImportError:
    HOPFIELD_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mhn_murmur")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — CONFIGURACIÓN  (modifica aquí antes de entrenar)
# ══════════════════════════════════════════════════════════════════════════════

# ── Rutas ─────────────────────────────────────────────────────────────────────
DATASET_DIR    = Path("data/physionet_2022")
CACHE_DIR      = Path("data/cache")
CHECKPOINT_DIR = Path("checkpoints")

# ── Audio / preprocesamiento ──────────────────────────────────────────────────
SR             = 4000
WAVELET        = "db6"
WAVELET_LEVEL  = 5
SG_WINDOW      = 51
SG_POLYORDER   = 3
N_FFT          = 1024
HOP_LENGTH     = 256
N_MELS         = 128
F_MIN          = 20.0
F_MAX          = 1000.0
TOP_DB         = 80.0
CACHE_ENABLED  = True

# ── Dataset ───────────────────────────────────────────────────────────────────
POSITIONS       = ["AV", "PV", "TV", "MV"]
VAL_SPLIT       = 0.15
RANDOM_SEED     = 42
SAMPLER_WEIGHTS = [5.0, 1.0, 10.0]    # Present ×5 | Absent ×1 | Unknown ×10

# ── CNN Backbone ──────────────────────────────────────────────────────────────
BACKBONE       = "resnet18"            # "resnet18" | "efficientnet_b0"
PRETRAINED     = True
EMBEDDING_DIM  = 512
FREEZE_EPOCHS  = 12                    # épocas con backbone congelado

# ── Hopfield Pooling ──────────────────────────────────────────────────────────
HF_HIDDEN      = 512
HF_OUTPUT      = 512
HF_HEADS       = 8
HF_QUANTITY    = 4                     # prototipos aprendidos
HF_DROPOUT     = 0.1
HF_BETA        = None                  # None → auto: 1/√d

# ── Features tabulares / clínicas ─────────────────────────────────────────────
N_FEATURES     = 11                    # 7 clínicos + 4 one-hot posición AV/PV/TV/MV
TAB_EMB_DIM    = 64
TAB_DROPOUT    = 0.2

# ── Clasificador ──────────────────────────────────────────────────────────────
N_CLASSES      = 3
CLASS_NAMES    = ["Present", "Absent", "Unknown"]
CLS_DROPOUT    = 0.45

# ── Entrenamiento ─────────────────────────────────────────────────────────────
BATCH_SIZE     = 32
NUM_WORKERS    = 6
EPOCHS         = 120
LR             = 5e-5
WEIGHT_DECAY   = 5e-4
GRAD_CLIP      = 1.0
WARMUP_EPOCHS  = 5
LR_MIN_FACTOR  = 0.01
SGDR_T0        = 20                    # ciclo SGDR — reinicios cada T0 épocas tras warmup
FOCAL_GAMMA    = 2.0
FOCAL_BETA     = 0.9999
LABEL_SMOOTH   = 0.1
MIXUP_ALPHA    = 0.3
SPEC_AUG_PROB  = 0.5
SPEC_FREQ_P    = 0.15
SPEC_TIME_P    = 0.15
SAVE_EVERY     = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {DEVICE}")
if torch.cuda.is_available():
    _gpu = torch.cuda.get_device_properties(0)
    print(f"GPU: {_gpu.name} | VRAM: {_gpu.total_memory / 1e9:.1f} GB")


# ── Dataclasses internos — construidos a partir de los globales ───────────────
# No es necesario editar esta parte; solo modifica los globales de arriba.

@dataclass
class PreprocessConfig:
    sample_rate: int      = SR
    wavelet: str          = WAVELET
    wavelet_level: int    = WAVELET_LEVEL
    threshold_mode: str   = "soft"
    sg_window_length: int = SG_WINDOW
    sg_polyorder: int     = SG_POLYORDER
    n_fft: int            = N_FFT
    hop_length: int       = HOP_LENGTH
    n_mels: int           = N_MELS
    f_min: float          = F_MIN
    f_max: float          = F_MAX
    power: float          = 2.0
    top_db: float         = TOP_DB
    cache_dir: Path       = field(default_factory=lambda: CACHE_DIR)
    cache_enabled: bool   = CACHE_ENABLED


@dataclass
class CNNConfig:
    backbone: str               = BACKBONE
    pretrained: bool            = PRETRAINED
    embedding_dim: int          = EMBEDDING_DIM
    freeze_backbone_epochs: int = FREEZE_EPOCHS


@dataclass
class HopfieldConfig:
    hidden_size: int  = HF_HIDDEN
    output_size: int  = HF_OUTPUT
    num_heads: int    = HF_HEADS
    quantity: int     = HF_QUANTITY
    dropout: float    = HF_DROPOUT
    beta: float       = None

    @property
    def beta_value(self) -> float:
        return self.beta if self.beta is not None else 1.0 / math.sqrt(512)


@dataclass
class ClassifierConfig:
    num_classes: int  = N_CLASSES
    class_names: list = field(default_factory=lambda: list(CLASS_NAMES))
    dropout: float    = CLS_DROPOUT


@dataclass
class TabularConfig:
    enabled: bool    = True
    n_features: int  = N_FEATURES
    embed_dim: int   = TAB_EMB_DIM
    dropout: float   = TAB_DROPOUT


@dataclass
class DataConfig:
    dataset_path: Path           = field(default_factory=lambda: DATASET_DIR)
    auscultation_positions: list = field(default_factory=lambda: list(POSITIONS))
    val_split: float             = VAL_SPLIT
    random_seed: int             = RANDOM_SEED
    use_weighted_sampler: bool   = True
    sampler_weights: list        = field(default_factory=lambda: list(SAMPLER_WEIGHTS))


@dataclass
class TrainingConfig:
    batch_size: int          = BATCH_SIZE
    num_workers: int         = NUM_WORKERS
    pin_memory: bool         = True
    epochs: int              = EPOCHS
    learning_rate: float     = LR
    weight_decay: float      = WEIGHT_DECAY
    grad_clip: float         = GRAD_CLIP
    warmup_epochs: int       = WARMUP_EPOCHS
    lr_min_factor: float     = LR_MIN_FACTOR
    sgdr_T0: int             = SGDR_T0
    focal_gamma: float       = FOCAL_GAMMA
    focal_beta: float        = FOCAL_BETA
    label_smoothing: float   = LABEL_SMOOTH
    mixup_alpha: float       = MIXUP_ALPHA
    spec_augment_prob: float = SPEC_AUG_PROB
    spec_freq_mask_p: float  = SPEC_FREQ_P
    spec_time_mask_p: float  = SPEC_TIME_P
    checkpoint_dir: Path     = field(default_factory=lambda: CHECKPOINT_DIR)
    save_every_n_epochs: int = SAVE_EVERY

    @property
    def device(self) -> torch.device:
        return DEVICE


@dataclass
class Config:
    preprocess:  PreprocessConfig  = field(default_factory=PreprocessConfig)
    cnn:         CNNConfig         = field(default_factory=CNNConfig)
    hopfield:    HopfieldConfig    = field(default_factory=HopfieldConfig)
    classifier:  ClassifierConfig  = field(default_factory=ClassifierConfig)
    tabular:     TabularConfig     = field(default_factory=TabularConfig)
    data:        DataConfig        = field(default_factory=DataConfig)
    training:    TrainingConfig    = field(default_factory=TrainingConfig)


cfg = Config()


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

class PCGPreprocessor:
    """
    Transforma grabaciones PCG en espectrogramas de Mel.
    Pipeline: wavelet denoising → Savitzky-Golay → Mel spectrogram → Z-score.
    Devuelve (1, n_mels, T_frames). Cachea resultados en disco.
    """

    def __init__(self, config: PreprocessConfig):
        self.cfg = config
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mel_transform   = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
            power=config.power,
        )
        self._amplitude_to_db = T.AmplitudeToDB(top_db=config.top_db)

    def __call__(self, wav_path: str | Path) -> torch.Tensor:
        wav_path   = Path(wav_path)
        cache_path = self._cache_path(wav_path)
        if self.cfg.cache_enabled and cache_path.exists():
            return torch.load(cache_path, weights_only=True)
        spectrogram = self._process(wav_path)
        if self.cfg.cache_enabled:
            torch.save(spectrogram, cache_path)
        return spectrogram

    def _process(self, wav_path: Path) -> torch.Tensor:
        waveform, sr = torchaudio.load(str(wav_path))
        if sr != self.cfg.sample_rate:
            waveform = T.Resample(orig_freq=sr, new_freq=self.cfg.sample_rate)(waveform)
            logger.warning(f"{wav_path.name}: remuestreado {sr} Hz → {self.cfg.sample_rate} Hz")
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        sig = waveform.squeeze(0).numpy()
        sig = self._wavelet_denoise(sig)
        sig = self._savitzky_golay_smooth(sig)
        mel = self._mel_transform(torch.from_numpy(sig).float().unsqueeze(0))
        return self._normalize(self._amplitude_to_db(mel))

    def _wavelet_denoise(self, signal: np.ndarray) -> np.ndarray:
        coeffs    = pywt.wavedec(signal, wavelet=self.cfg.wavelet, level=self.cfg.wavelet_level, mode="periodization")
        sigma     = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        denoised  = [coeffs[0]] + [pywt.threshold(c, value=threshold, mode=self.cfg.threshold_mode) for c in coeffs[1:]]
        return pywt.waverec(denoised, wavelet=self.cfg.wavelet, mode="periodization")[: len(signal)]

    def _savitzky_golay_smooth(self, signal: np.ndarray) -> np.ndarray:
        wl = self.cfg.sg_window_length
        if wl % 2 == 0:
            wl += 1
        wl = max(wl, self.cfg.sg_polyorder + 2)
        return savgol_filter(signal, window_length=wl, polyorder=self.cfg.sg_polyorder, mode="interp")

    @staticmethod
    def _normalize(t: torch.Tensor) -> torch.Tensor:
        return (t - t.mean()) / (t.std() + 1e-8)

    def _cache_path(self, wav_path: Path) -> Path:
        cfg_str = (
            f"{self.cfg.wavelet}{self.cfg.wavelet_level}"
            f"{self.cfg.sg_window_length}{self.cfg.sg_polyorder}"
            f"{self.cfg.n_fft}{self.cfg.hop_length}{self.cfg.n_mels}"
        )
        key = hashlib.md5(f"{wav_path}{cfg_str}".encode()).hexdigest()
        return self.cfg.cache_dir / f"{wav_path.stem}_{key}.pt"


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

LABEL_MAP    = {"Present": 0, "Absent": 1, "Unknown": 2}
AGE_ORDER    = ["Neonate", "Infant", "Child", "Adolescent", "Young Adult"]
AGE_MAP      = {age: i for i, age in enumerate(AGE_ORDER)}
POSITION_MAP = {"AV": 0, "PV": 1, "TV": 2, "MV": 3}


def _parse_tabular(row: pd.Series, position: str) -> torch.Tensor:
    """
    Devuelve 11 features: 7 clínicos normalizados + 4 one-hot de posición (AV/PV/TV/MV).
    La posición de auscultación es clínicamente relevante para discriminar el tipo de soplo.
    """
    age_group   = AGE_MAP.get(str(row.get("Age", "Child")).strip(), 2) / 4.0
    sex         = 1.0 if str(row.get("Sex", "")).strip() == "Male" else 0.0
    try:
        height_norm = max(0.0, (float(row.get("Height", 0)) - 50.0) / 150.0)
    except (ValueError, TypeError):
        height_norm = 0.5
    try:
        weight_norm = max(0.0, (float(row.get("Weight", 0)) - 3.0) / 97.0)
    except (ValueError, TypeError):
        weight_norm = 0.3
    try:
        h_m      = float(row.get("Height", 0)) / 100.0
        w_kg     = float(row.get("Weight", 0))
        bmi_norm = max(0.0, (w_kg / (h_m ** 2) - 10.0) / 30.0) if h_m > 0.3 else 0.4
    except (ValueError, TypeError, ZeroDivisionError):
        bmi_norm = 0.4
    locs_str             = str(row.get("Recording locations:", "")).strip()
    recording_count_norm = len([l for l in locs_str.split("+") if l.strip()]) / 4.0 if locs_str else 0.0
    pregnancy            = 1.0 if str(row.get("Pregnancy status", "False")).strip().lower() in ("true", "1", "yes") else 0.0
    pos_onehot           = [0.0] * 4
    pos_onehot[POSITION_MAP.get(position, 0)] = 1.0
    return torch.tensor(
        [age_group, sex, height_norm, weight_norm, bmi_norm, recording_count_norm, pregnancy] + pos_onehot,
        dtype=torch.float32,
    )


class PhysioNetDataset(Dataset):
    """Dataset de fonocardiogramas del PhysioNet Challenge 2022."""

    def __init__(
        self,
        csv_path: Path,
        wav_dir: Path,
        preprocessor: PCGPreprocessor,
        positions: list[str] | None = None,
        split_ids: set[str] | None = None,
    ):
        self.preprocessor = preprocessor
        self.positions    = positions or ["AV", "PV", "TV", "MV"]
        self.samples: list[dict] = []

        metadata = pd.read_csv(csv_path)
        if not {"Patient ID", "Murmur"}.issubset(metadata.columns):
            raise ValueError(f"CSV debe contener 'Patient ID' y 'Murmur'. Encontradas: {set(metadata.columns)}")

        for _, row in metadata.iterrows():
            pid       = str(row["Patient ID"])
            label_str = str(row["Murmur"]).strip()
            if split_ids is not None and pid not in split_ids:
                continue
            if label_str not in LABEL_MAP:
                logger.warning(f"Paciente {pid}: etiqueta desconocida '{label_str}', omitido.")
                continue
            for pos in self.positions:
                wav_path = wav_dir / f"{pid}_{pos}.wav"
                if wav_path.exists():
                    self.samples.append({
                        "patient_id": pid,
                        "position":   pos,
                        "wav_path":   wav_path,
                        "label":      LABEL_MAP[label_str],
                        "tabular":    _parse_tabular(row, pos),
                    })

        logger.info(f"Dataset cargado: {len(self.samples)} grabaciones.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, str]:
        s = self.samples[idx]
        return self.preprocessor(s["wav_path"]), s["tabular"], s["label"], s["patient_id"]

    def get_class_counts(self) -> list[int]:
        counts = [0, 0, 0]
        for s in self.samples:
            counts[s["label"]] += 1
        return counts


def collate_fn_pad(batch):
    spectrograms, tabulars, labels, patient_ids = zip(*batch)
    lengths = torch.tensor([s.shape[-1] for s in spectrograms], dtype=torch.long)
    max_t   = lengths.max().item()
    padded  = torch.zeros(len(spectrograms), *spectrograms[0].shape[:-1], max_t)
    for i, spec in enumerate(spectrograms):
        padded[i, ..., : spec.shape[-1]] = spec
    return padded, torch.stack(tabulars), torch.tensor(labels, dtype=torch.long), list(patient_ids), lengths


def build_dataloaders(
    data_cfg: DataConfig,
    preprocess_cfg: PreprocessConfig,
    train_cfg: TrainingConfig,
) -> tuple[DataLoader, DataLoader, list[int]]:
    """División por paciente para evitar data leakage."""
    preprocessor = PCGPreprocessor(preprocess_cfg)
    csv_path     = data_cfg.dataset_path / "training_data.csv"
    wav_dir      = data_cfg.dataset_path

    all_ids  = pd.read_csv(csv_path)["Patient ID"].astype(str).unique().tolist()
    rng      = torch.Generator().manual_seed(data_cfg.random_seed)
    n_val    = int(len(all_ids) * data_cfg.val_split)
    shuffled = torch.randperm(len(all_ids), generator=rng).tolist()
    val_ids   = set(all_ids[i] for i in shuffled[:n_val])
    train_ids = set(all_ids[i] for i in shuffled[n_val:])

    train_ds = PhysioNetDataset(csv_path, wav_dir, preprocessor, data_cfg.auscultation_positions, train_ids)
    val_ds   = PhysioNetDataset(csv_path, wav_dir, preprocessor, data_cfg.auscultation_positions, val_ids)

    samples_per_class = train_ds.get_class_counts()
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | [Present, Absent, Unknown]: {samples_per_class}")

    sampler = None
    shuffle = True
    if data_cfg.use_weighted_sampler:
        weights = [float(data_cfg.sampler_weights[s["label"]]) for s in train_ds.samples]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False
        logger.info(f"WeightedRandomSampler activo — pesos: {data_cfg.sampler_weights}")

    loader_kwargs = dict(
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
        collate_fn=collate_fn_pad,
        persistent_workers=train_cfg.num_workers > 0,
        multiprocessing_context="fork" if train_cfg.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=shuffle, sampler=sampler, drop_last=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False,                                     **loader_kwargs)
    return train_loader, val_loader, samples_per_class


# ══════════════════════════════════════════════════════════════════════════════
# MODELOS
# ══════════════════════════════════════════════════════════════════════════════

class CNNFrontend(nn.Module):
    """
    Extrae embeddings temporales del espectrograma de Mel.
    Entrada:  (B, 1, n_mels, T_frames)
    Salida:   (B, T', embedding_dim) donde T' ≈ T/32
    """

    TEMPORAL_STRIDE = 32

    def __init__(self, config: CNNConfig):
        super().__init__()
        self.embedding_dim = config.embedding_dim
        if config.backbone == "resnet18":
            self.backbone, self.embedding_dim = self._build_resnet18(config.pretrained)
        elif config.backbone == "efficientnet_b0":
            self.backbone, self.embedding_dim = self._build_efficientnet_b0(config.pretrained)
        else:
            raise ValueError(f"Backbone no soportado: '{config.backbone}'.")
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.freq_pool(x).squeeze(2).permute(0, 2, 1)
        return x

    @staticmethod
    def _build_resnet18(pretrained: bool) -> tuple[nn.Sequential, int]:
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet  = tv_models.resnet18(weights=weights)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                orig = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT).conv1.weight
                resnet.conv1.weight = nn.Parameter(orig.mean(dim=1, keepdim=True))
        backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        return backbone, 512

    @staticmethod
    def _build_efficientnet_b0(pretrained: bool) -> tuple[nn.Sequential, int]:
        weights   = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        effnet    = tv_models.efficientnet_b0(weights=weights)
        orig_conv = effnet.features[0][0]
        effnet.features[0][0] = nn.Conv2d(
            1, orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size, stride=orig_conv.stride,
            padding=orig_conv.padding, bias=orig_conv.bias is not None,
        )
        if pretrained:
            with torch.no_grad():
                effnet.features[0][0].weight = nn.Parameter(orig_conv.weight.mean(dim=1, keepdim=True))
        return effnet.features, 1280


class HopfieldPoolingLayer(nn.Module):
    """
    Memoria asociativa continua sobre la secuencia temporal de embeddings CNN.
    Entrada:  (B, T', embedding_dim)
    Salida:   (B, quantity * output_size)

    Usa hflayers.HopfieldPooling si está instalado; cae a MultiheadAttention en caso contrario.
    """

    def __init__(self, embedding_dim: int, config: HopfieldConfig):
        super().__init__()
        self.cfg = config
        beta     = config.beta if config.beta is not None else 1.0 / math.sqrt(embedding_dim)

        if HOPFIELD_AVAILABLE:
            self.pool = _HopfieldPooling(
                input_size=embedding_dim,
                hidden_size=config.hidden_size,
                output_size=config.output_size,
                num_heads=config.num_heads,
                quantity=config.quantity,
                scaling=beta,
                dropout=config.dropout,
                pattern_projection_as_static=False,
                normalize_stored_pattern=True,
                normalize_state_pattern=True,
            )
            self.output_dim = config.quantity * config.output_size
        else:
            warnings.warn("hopfield-layers no encontrado. Usando MultiheadAttention como fallback.")
            self.pool            = None
            self.learned_queries = nn.Parameter(torch.randn(config.quantity, embedding_dim) / math.sqrt(embedding_dim))
            self.mha             = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=config.num_heads,
                                                         dropout=config.dropout, batch_first=True)
            self.beta_param      = nn.Parameter(torch.tensor(beta))
            self.output_dim      = config.quantity * embedding_dim

        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if HOPFIELD_AVAILABLE:
            out = self.pool(x, stored_pattern_padding_mask=padding_mask)
            self.last_attn_weights = self.pool.get_association_matrix(x).detach()
        else:
            B       = x.shape[0]
            queries = self.learned_queries.unsqueeze(0).expand(B, -1, -1) * self.beta_param
            out, self.last_attn_weights = self.mha(
                query=queries, key=x, value=x,
                key_padding_mask=padding_mask, need_weights=True, average_attn_weights=False,
            )
        return out.reshape(out.shape[0], -1)

    @property
    def last_attention_weights(self) -> torch.Tensor | None:
        return getattr(self, "last_attn_weights", None)


class TabularEncoder(nn.Module):
    """MLP que convierte features clínicas en un embedding de 64d."""

    def __init__(self, n_features: int = 11, embed_dim: int = 64, dropout: float = 0.2):
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


_TEMPORAL_STRIDE = CNNFrontend.TEMPORAL_STRIDE


class MurmurClassifier(nn.Module):
    """
    Pipeline completo: (B, 1, n_mels, T) → CNNFrontend → HopfieldPooling → [+ Tabular] → MLP → (B, 3)
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
            fusion_dim       = pooled_dim + tabular_cfg.embed_dim
        else:
            self.tab_encoder = None
            fusion_dim       = pooled_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(p=classifier_cfg.dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(p=classifier_cfg.dropout),
            nn.Linear(fusion_dim // 2, classifier_cfg.num_classes),
        )
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.cnn(x)

        padding_mask = None
        if lengths is not None:
            T_prime  = embeddings.shape[1]
            t_actual = (lengths.float() / _TEMPORAL_STRIDE).ceil().long().clamp(1, T_prime)
            padding_mask = torch.arange(T_prime, device=x.device).unsqueeze(0) >= t_actual.unsqueeze(1)
            if not padding_mask.any():
                padding_mask = None

        pooled = self.hopfield(embeddings, padding_mask=padding_mask)

        if self.use_tabular:
            tab_emb = (
                self.tab_encoder(tabular.to(x.device))
                if tabular is not None
                else torch.zeros(x.shape[0], self.tab_encoder.net[-2].out_features, device=x.device)
            )
            fused = torch.cat([pooled, tab_emb], dim=-1)
        else:
            fused = pooled

        logits = self.classifier(fused)
        return (logits, self.hopfield.last_attention_weights) if return_attention else logits

    def predict(self, x, tabular=None, lengths=None) -> torch.Tensor:
        with torch.no_grad():
            return torch.argmax(self.forward(x, tabular, lengths), dim=-1)

    @staticmethod
    def from_config(config: Config) -> "MurmurClassifier":
        return MurmurClassifier(
            config.cnn, config.hopfield, config.classifier,
            config.tabular if hasattr(config, "tabular") else None,
        )


# ══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS Y PÉRDIDA
# ══════════════════════════════════════════════════════════════════════════════

CHALLENGE_WEIGHTS = {0: 5, 1: 1, 2: 3}
CLASS_NAMES       = ["Present", "Absent", "Unknown"]
_MAX_ENTROPY      = math.log(3)


class CBFocalLoss(nn.Module):
    """Class-Balanced Focal Loss (Cui et al. 2019) con label smoothing opcional."""

    def __init__(self, samples_per_class: list[int], gamma: float = 2.0,
                 beta: float = 0.9999, label_smoothing: float = 0.0):
        super().__init__()
        eff    = [1.0 - beta ** n for n in samples_per_class]
        w      = [(1.0 - beta) / (e + 1e-8) for e in eff]
        total  = sum(w)
        w      = [x / total * len(w) for x in w]
        self.register_buffer("alpha", torch.tensor(w, dtype=torch.float32))
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        logger.info(f"CBFocalLoss — weights: {[f'{x:.3f}' for x in w]} | γ={gamma} | ls={label_smoothing}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce           = F.cross_entropy(logits, targets, reduction="none", label_smoothing=self.label_smoothing)
        p_t          = torch.exp(-ce)
        focal_weight = (1.0 - p_t) ** self.gamma
        return (self.alpha[targets] * focal_weight * ce).mean()


def compute_weighted_accuracy(y_true, y_pred) -> float:
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    total   = sum(CHALLENGE_WEIGHTS[l] for l in y_true)
    correct = sum(CHALLENGE_WEIGHTS[t] for t, p in zip(y_true, y_pred) if t == p)
    return correct / total if total > 0 else 0.0


def compute_challenge_score(y_true, y_pred) -> dict:
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    wa  = compute_weighted_accuracy(y_true, y_pred)
    acc = float(np.mean(y_true == y_pred))
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    recall, precision = {}, {}
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]; fn = cm[i, :].sum() - tp; fp = cm[:, i].sum() - tp
        recall[name]    = tp / (tp + fn + 1e-8)
        precision[name] = tp / (tp + fp + 1e-8)
    return {"weighted_accuracy": wa, "standard_accuracy": acc,
            "confusion_matrix": cm, "per_class_recall": recall, "per_class_precision": precision}


def aggregate_patient_preds(
    patient_ids: list[str], probs: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Agrega N grabaciones en 1 predicción por paciente usando ponderación por confianza."""
    patient_probs: dict[str, list] = {}
    patient_label: dict[str, int]  = {}
    for i, pid in enumerate(patient_ids):
        if pid not in patient_probs:
            patient_probs[pid] = []
            patient_label[pid] = labels[i].item()
        patient_probs[pid].append(probs[i])
    agg_preds, agg_labels = [], []
    for pid, prob_list in patient_probs.items():
        stacked    = torch.stack(prob_list)
        entropy    = -(stacked * (stacked + 1e-8).log()).sum(dim=1)
        confidence = 1.0 - entropy / _MAX_ENTROPY
        weights    = torch.softmax(confidence * 3.0, dim=0).unsqueeze(1)
        agg_preds.append((stacked * weights).sum(dim=0).argmax().item())
        agg_labels.append(patient_label[pid])
    return torch.tensor(agg_preds), torch.tensor(agg_labels)


def format_metrics(metrics: dict) -> str:
    lines = [
        f"  Weighted Accuracy (challenge): {metrics['weighted_accuracy']:.4f}",
        f"  Standard Accuracy:             {metrics['standard_accuracy']:.4f}",
        "  Per-class Recall:",
    ]
    for name, val in metrics["per_class_recall"].items():
        lines.append(f"    {name:>10}: {val:.3f}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def spec_augment(spec: torch.Tensor, freq_mask_p: float = 0.15, time_mask_p: float = 0.15) -> torch.Tensor:
    spec   = spec.clone()
    B, C, n_mels, T = spec.shape
    f_size = max(1, int(n_mels * freq_mask_p))
    t_size = max(1, int(T * time_mask_p))
    for b in range(B):
        f0 = random.randint(0, n_mels - f_size)
        t0 = random.randint(0, max(0, T - t_size))
        spec[b, :, f0:f0 + f_size, :] = 0.0
        spec[b, :, :, t0:t0 + t_size] = 0.0
    return spec


def clinical_mixup(
    spectrograms: torch.Tensor, tabulars: torch.Tensor, labels: torch.Tensor, alpha: float = 0.3
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0.0 or random.random() > 0.5:
        return spectrograms, tabulars, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    idx = torch.randperm(spectrograms.shape[0], device=spectrograms.device)
    return (
        lam * spectrograms + (1.0 - lam) * spectrograms[idx],
        lam * tabulars     + (1.0 - lam) * tabulars[idx],
        labels, labels[idx], lam,
    )


def sgdr_lr(epoch: int, base_lr: float, warmup_epochs: int, T_0: int, lr_min_factor: float = 0.01) -> float:
    eta_min      = base_lr * lr_min_factor
    if epoch <= warmup_epochs:
        return base_lr * epoch / max(1, warmup_epochs)
    t_in_cycle   = (epoch - warmup_epochs - 1) % T_0
    cos_val      = 0.5 * (1.0 + math.cos(math.pi * t_in_cycle / T_0))
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
        self.device       = train_cfg.device
        self.best_wa      = 0.0
        self.checkpoint_dir = train_cfg.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = CBFocalLoss(
            samples_per_class=samples_per_class,
            gamma=train_cfg.focal_gamma,
            beta=train_cfg.focal_beta,
            label_smoothing=train_cfg.label_smoothing,
        ).to(train_cfg.device)

        cnn_ids    = {id(p) for p in model.cnn.parameters()}
        non_cnn    = [p for p in model.parameters() if id(p) not in cnn_ids]
        self.optimizer = AdamW(non_cnn, lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)

        logger.info(
            f"Trainer — epochs={train_cfg.epochs} | SGDR T_0={train_cfg.sgdr_T0} | "
            f"warmup={train_cfg.warmup_epochs} | lr={train_cfg.learning_rate} | "
            f"wd={train_cfg.weight_decay} | mixup_α={train_cfg.mixup_alpha}"
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
            wa          = val_metrics["weighted_accuracy"]
            logger.info(
                f"Época {epoch:03d}/{self.cfg.epochs} | Loss: {train_loss:.4f} | WA: {wa:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | Tiempo: {elapsed:.1f}s"
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
            loss   = (
                lam * self.criterion(logits, labels_a) + (1.0 - lam) * self.criterion(logits, labels_b)
                if lam < 1.0 else self.criterion(logits, labels)
            )
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
        preds, labels = aggregate_patient_preds(all_pids, torch.cat(all_probs), torch.cat(all_labels))
        return compute_challenge_score(labels, preds)

    def _update_lr(self, epoch: int) -> None:
        new_lr = sgdr_lr(epoch, self.cfg.learning_rate, self.cfg.warmup_epochs, self.cfg.sgdr_T0, self.cfg.lr_min_factor)
        self.optimizer.param_groups[0]["lr"] = new_lr
        if len(self.optimizer.param_groups) > 1:
            self.optimizer.param_groups[1]["lr"] = new_lr * 0.1

    def _maybe_unfreeze_backbone(self, epoch: int) -> None:
        if epoch == 1:
            logger.info(f"Backbone CNN congelado hasta época {self.cnn_cfg.freeze_backbone_epochs}.")
            for p in self.model.cnn.parameters():
                p.requires_grad = False
        elif epoch == self.cnn_cfg.freeze_backbone_epochs + 1:
            logger.info("Descongelando backbone CNN para fine-tuning completo.")
            for p in self.model.cnn.parameters():
                p.requires_grad = True
            self.optimizer.add_param_group({
                "params": list(self.model.cnn.parameters()),
                "lr": self.cfg.learning_rate * 0.1,
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


# ══════════════════════════════════════════════════════════════════════════════
# COMANDOS CLI
# ══════════════════════════════════════════════════════════════════════════════

def cmd_train(args) -> None:
    logger.info(f"Dispositivo: {cfg.training.device}")
    logger.info(f"Backbone CNN: {cfg.cnn.backbone} | Embedding dim: {cfg.cnn.embedding_dim}")
    logger.info(f"HopfieldPooling: β={'auto (1/√d)' if cfg.hopfield.beta is None else cfg.hopfield.beta} | prototipos: {cfg.hopfield.quantity}")
    logger.info(f"Épocas: {cfg.training.epochs} | Warmup: {cfg.training.warmup_epochs} | MixUp α={cfg.training.mixup_alpha}")

    train_loader, val_loader, samples_per_class = build_dataloaders(cfg.data, cfg.preprocess, cfg.training)
    logger.info(f"Conteos [Present, Absent, Unknown]: {samples_per_class}")

    model = MurmurClassifier.from_config(cfg)
    logger.info(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")

    Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=cfg.training,
        classifier_cfg=cfg.classifier,
        cnn_cfg=cfg.cnn,
        samples_per_class=samples_per_class,
    ).train()


def cmd_eval(args) -> None:
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint no encontrado: {checkpoint_path}")
        sys.exit(1)

    _, val_loader, _ = build_dataloaders(cfg.data, cfg.preprocess, cfg.training)
    model = MurmurClassifier.from_config(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=cfg.training.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(cfg.training.device)
    model.eval()

    all_probs, all_labels, all_pids = [], [], []
    with torch.no_grad():
        for spectrograms, tabulars, labels, patient_ids, lengths in val_loader:
            spectrograms = spectrograms.to(cfg.training.device)
            tabulars     = tabulars.to(cfg.training.device)
            probs        = torch.softmax(model(spectrograms, tabulars, lengths.to(cfg.training.device)), dim=-1)
            all_probs.append(probs.cpu())
            all_labels.append(labels)
            all_pids.extend(patient_ids)

    preds, labels = aggregate_patient_preds(all_pids, torch.cat(all_probs), torch.cat(all_labels))
    logger.info(f"Resultados ({checkpoint_path.name}):")
    logger.info(format_metrics(compute_challenge_score(labels, preds)))


def cmd_preprocess(args) -> None:
    preprocessor = PCGPreprocessor(cfg.preprocess)
    wav_files    = list(cfg.data.dataset_path.glob("*.wav"))
    if not wav_files:
        logger.warning(f"No se encontraron archivos .wav en {cfg.data.dataset_path}")
        return
    logger.info(f"Pre-procesando {len(wav_files)} archivos .wav → caché en {cfg.preprocess.cache_dir}")
    for wav in tqdm(wav_files, desc="Wavelet → S-G → Mel"):
        preprocessor(wav)
    logger.info("Preprocesamiento completado.")


def cmd_explain(args) -> None:
    from utils import extract_hopfield_weights, plot_attention_heatmap

    wav_path = Path(args.wav)
    if not wav_path.exists():
        logger.error(f"Archivo no encontrado: {wav_path}")
        sys.exit(1)

    preprocessor = PCGPreprocessor(cfg.preprocess)
    spectrogram  = preprocessor(wav_path).unsqueeze(0)
    model        = MurmurClassifier.from_config(cfg)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    with torch.no_grad():
        logits, attn_weights = model(spectrogram, return_attention=True)

    probs      = torch.softmax(logits, dim=-1).squeeze()
    pred_class = cfg.classifier.class_names[probs.argmax().item()]
    logger.info(f"Predicción: {pred_class}")
    for name, prob in zip(cfg.classifier.class_names, probs.tolist()):
        logger.info(f"  {name}: {prob:.3f}")

    output_path = Path(f"explain_{wav_path.stem}.png")
    plot_attention_heatmap(
        wav_path=wav_path,
        attention_weights=attn_weights,
        spectrogram=spectrogram.squeeze(0),
        hop_length=cfg.preprocess.hop_length,
        sample_rate=cfg.preprocess.sample_rate,
        output_path=output_path,
        title=f"Interpretabilidad MHN — {wav_path.stem} | Predicción: {pred_class}",
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Modern Hopfield Network para detección de soplos cardíacos (PhysioNet 2022)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train",      help="Entrenar el modelo completo")
    subparsers.add_parser("preprocess", help="Pre-calcular espectrogramas de Mel")

    eval_p = subparsers.add_parser("eval", help="Evaluar un checkpoint guardado")
    eval_p.add_argument("--checkpoint", required=True, help="Ruta al archivo .pt")

    exp_p = subparsers.add_parser("explain", help="Visualizar interpretabilidad Hopfield")
    exp_p.add_argument("--wav",        required=True, help="Ruta al archivo .wav")
    exp_p.add_argument("--checkpoint", default=None,  help="Ruta al checkpoint (opcional)")

    args = parser.parse_args()
    {"train": cmd_train, "eval": cmd_eval, "preprocess": cmd_preprocess, "explain": cmd_explain}[args.command](args)


if __name__ == "__main__":
    main()
