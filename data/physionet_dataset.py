"""
data/physionet_dataset.py — Dataset PhysioNet Challenge 2022 (CirCor DigiScope)

Estructura esperada del directorio del dataset:
    data/physionet_2022/
    ├── training_data.csv          # Metadatos de pacientes + etiquetas
    ├── {patient_id}_{pos}.wav     # Grabaciones (pos: AV, PV, TV, MV)
    ├── {patient_id}.txt           # Metadata del paciente (Age, Sex, Height, Weight...)
    └── {patient_id}_{pos}.hea    # Cabeceras WFDB (opcional)

Etiqueta objetivo (nivel paciente):
    "Murmur": "Present" | "Absent" | "Unknown"

Estrategia multi-focal:
    Cada grabación de una posición auscultoria (AV/PV/TV/MV) se trata como
    una instancia independiente durante el entrenamiento (comparten la etiqueta
    del paciente). En inferencia, las predicciones por posición se agregan
    (probabilidades promedio) para obtener la predicción a nivel paciente.

Features tabulares:
    Se extraen del CSV (Age, Sex, Height, Weight, Pregnancy status, Recording locations).
    Derivan 7 features numéricas (incluyendo BMI y recording_count).
    Inspirado en FeatureEngineer de PF5367600.py (script del amigo).
    Se fusionan con el output de HopfieldPooling en MurmurClassifier.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from config import DataConfig, PreprocessConfig, TrainingConfig, TabularConfig
from data.preprocessing import PCGPreprocessor

logger = logging.getLogger(__name__)

# Mapeo de etiqueta textual → índice de clase
LABEL_MAP = {"Present": 0, "Absent": 1, "Unknown": 2}

# Mapeo ordinal de grupos de edad (de PF5367600.py / MZA-PCG v5)
AGE_ORDER = ["Neonate", "Infant", "Child", "Adolescent", "Young Adult"]
AGE_MAP = {age: i for i, age in enumerate(AGE_ORDER)}

# Número de features tabulares (debe coincidir con TabularConfig.n_features)
# [age_group, sex, height, weight, bmi, recording_count, pregnancy]
TABULAR_N_FEATURES = 7


# ─────────────────────────────────────────────
# Extracción de features tabulares clínicas
# Fuente: FeatureEngineer (PF5367600.py) + MZA-PCG v5 parser
# ─────────────────────────────────────────────

def _parse_tabular(row: pd.Series) -> torch.Tensor:
    """
    Extrae 7 features clínicos de una fila del CSV.

    Features (en orden):
        0. age_group       Ordinal 0-4 (Neonate→Young Adult), normalizado [0,1]
        1. sex             0=Female, 1=Male
        2. height          cm, normalizado por rango pediátrico típico
        3. weight          kg, normalizado por rango pediátrico típico
        4. bmi             kg/m², normalizado (de PF5367600.py FeatureEngineer)
        5. recording_count Número de zonas grabadas (1-4), normalizado [0,1]
        6. pregnancy       0=False, 1=True

    La normalización usa BatchNorm1d en TabularEncoder → no necesita
    estadísticas externas. Aquí solo escalamos a rangos razonables.
    """
    # ── Age group ────────────────────────────────────────────────
    age_str = str(row.get("Age", "Child")).strip()
    age_group = AGE_MAP.get(age_str, 2) / 4.0          # → [0.0, 1.0]

    # ── Sex ──────────────────────────────────────────────────────
    sex_str = str(row.get("Sex", "Unknown")).strip()
    sex = 1.0 if sex_str == "Male" else 0.0

    # ── Height (cm) ───────────────────────────────────────────────
    try:
        height = float(row.get("Height", 0))
        height_norm = max(0.0, (height - 50.0) / 150.0)  # rango ~[0,1] para 50-200 cm
    except (ValueError, TypeError):
        height_norm = 0.5

    # ── Weight (kg) ───────────────────────────────────────────────
    try:
        weight = float(row.get("Weight", 0))
        weight_norm = max(0.0, (weight - 3.0) / 97.0)    # rango ~[0,1] para 3-100 kg
    except (ValueError, TypeError):
        weight_norm = 0.3

    # ── BMI (de FeatureEngineer en PF5367600.py) ─────────────────
    try:
        h_m = float(row.get("Height", 0)) / 100.0
        w_kg = float(row.get("Weight", 0))
        if h_m > 0.3:
            bmi = w_kg / (h_m ** 2)
            bmi_norm = max(0.0, (bmi - 10.0) / 30.0)     # rango ~[0,1] para BMI 10-40
        else:
            bmi_norm = 0.4
    except (ValueError, TypeError, ZeroDivisionError):
        bmi_norm = 0.4

    # ── Recording count (de FeatureEngineer en PF5367600.py) ──────
    locs_str = str(row.get("Recording locations:", "")).strip()
    n_locs = len([l for l in locs_str.split("+") if l.strip()]) if locs_str else 0
    recording_count_norm = n_locs / 4.0                   # → [0.0, 1.0] (max 4 zonas)

    # ── Pregnancy status ──────────────────────────────────────────
    preg_str = str(row.get("Pregnancy status", "False")).strip().lower()
    pregnancy = 1.0 if preg_str in ("true", "1", "yes") else 0.0

    features = [
        age_group,
        sex,
        height_norm,
        weight_norm,
        bmi_norm,
        recording_count_norm,
        pregnancy,
    ]
    return torch.tensor(features, dtype=torch.float32)


# ─────────────────────────────────────────────
# Dataset principal
# ─────────────────────────────────────────────

class PhysioNetDataset(Dataset):
    """
    Dataset de fonocardiogramas del PhysioNet Challenge 2022.

    Cada ítem corresponde a UNA grabación de UNA posición de UN paciente.
    La etiqueta es la del paciente (nivel paciente → etiqueta de instancia).

    Devuelve: (spectrogram, tabular, label)
        spectrogram: (1, n_mels, T)  — espectrograma Mel preprocesado
        tabular:     (7,)            — features clínicas del paciente
        label:       int             — 0=Present, 1=Absent, 2=Unknown

    Args:
        csv_path:    Ruta al archivo training_data.csv del challenge.
        wav_dir:     Directorio que contiene los archivos .wav.
        preprocessor: Instancia de PCGPreprocessor (Fase 1).
        positions:   Lista de posiciones a incluir (ej: ["AV", "PV"]).
        split_ids:   Conjunto de patient_ids a incluir (para train/val split).
    """

    def __init__(
        self,
        csv_path: Path,
        wav_dir: Path,
        preprocessor: PCGPreprocessor,
        positions: list[str] | None = None,
        split_ids: set[str] | None = None,
    ):
        self.preprocessor = preprocessor
        self.positions = positions or ["AV", "PV", "TV", "MV"]
        self.samples: list[dict] = []

        metadata = pd.read_csv(csv_path)
        # El challenge usa columna "Patient ID" y "Murmur"
        required_cols = {"Patient ID", "Murmur"}
        if not required_cols.issubset(metadata.columns):
            raise ValueError(
                f"CSV debe contener columnas {required_cols}. "
                f"Encontradas: {set(metadata.columns)}"
            )

        for _, row in metadata.iterrows():
            patient_id = str(row["Patient ID"])
            label_str = str(row["Murmur"]).strip()

            if split_ids is not None and patient_id not in split_ids:
                continue
            if label_str not in LABEL_MAP:
                logger.warning(
                    f"Paciente {patient_id}: etiqueta desconocida '{label_str}', omitido."
                )
                continue

            label = LABEL_MAP[label_str]
            tabular = _parse_tabular(row)                  # (7,) features clínicas

            for pos in self.positions:
                wav_path = wav_dir / f"{patient_id}_{pos}.wav"
                if wav_path.exists():
                    self.samples.append(
                        {
                            "patient_id": patient_id,
                            "position": pos,
                            "wav_path": wav_path,
                            "label": label,
                            "tabular": tabular,            # ← nuevo
                        }
                    )

        logger.info(
            f"Dataset cargado: {len(self.samples)} grabaciones "
            f"de {len(metadata)} pacientes."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.samples[idx]
        spectrogram = self.preprocessor(sample["wav_path"])  # (1, n_mels, T)
        return spectrogram, sample["tabular"], sample["label"]

    def get_patient_ids(self) -> list[str]:
        return list({s["patient_id"] for s in self.samples})

    def get_class_counts(self) -> list[int]:
        """
        Devuelve [n_Present, n_Absent, n_Unknown] a nivel de grabación.
        Usado por CBFocalLoss para calcular pesos class-balanced.
        """
        counts = [0, 0, 0]
        for s in self.samples:
            counts[s["label"]] += 1
        return counts


# ─────────────────────────────────────────────
# Collate con padding dinámico
# ─────────────────────────────────────────────

def collate_fn_pad(
    batch: list[tuple[torch.Tensor, torch.Tensor, int]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate con padding dinámico en el eje temporal.

    Las grabaciones PCG tienen duraciones variables. Se hace padding con ceros
    hasta la longitud del elemento más largo del batch (padding a la derecha).
    La CNN y HopfieldPooling son agnósticos a la longitud temporal exacta.

    Returns:
        spectrograms: (B, 1, n_mels, T_max) — con padding
        tabulars:     (B, 7)                 — features clínicas
        labels:       (B,)                   — enteros de clase
    """
    spectrograms, tabulars, labels = zip(*batch)
    max_t = max(s.shape[-1] for s in spectrograms)

    padded = torch.zeros(len(spectrograms), *spectrograms[0].shape[:-1], max_t)
    for i, spec in enumerate(spectrograms):
        padded[i, ..., : spec.shape[-1]] = spec

    return (
        padded,
        torch.stack(tabulars),                            # (B, 7)
        torch.tensor(labels, dtype=torch.long),
    )


# ─────────────────────────────────────────────
# Builder de DataLoaders
# ─────────────────────────────────────────────

def build_dataloaders(
    data_cfg: DataConfig,
    preprocess_cfg: PreprocessConfig,
    train_cfg: TrainingConfig,
) -> tuple[DataLoader, DataLoader, list[int]]:
    """
    Construye los DataLoaders de entrenamiento y validación.

    La división train/val se hace a nivel de PACIENTE para evitar data leakage
    (todas las posiciones de un paciente van al mismo split).

    Returns:
        train_loader, val_loader, samples_per_class
        donde samples_per_class = [n_Present, n_Absent, n_Unknown]
        del split de entrenamiento (para CBFocalLoss).
    """
    preprocessor = PCGPreprocessor(preprocess_cfg)
    csv_path = data_cfg.dataset_path / "training_data.csv"
    wav_dir = data_cfg.dataset_path

    # División por paciente (evita leakage entre splits)
    all_ids = pd.read_csv(csv_path)["Patient ID"].astype(str).unique().tolist()

    rng = torch.Generator().manual_seed(data_cfg.random_seed)
    n_val = int(len(all_ids) * data_cfg.val_split)
    shuffled = torch.randperm(len(all_ids), generator=rng).tolist()
    val_ids = set([all_ids[i] for i in shuffled[:n_val]])
    train_ids = set([all_ids[i] for i in shuffled[n_val:]])

    train_ds = PhysioNetDataset(
        csv_path, wav_dir, preprocessor,
        positions=data_cfg.auscultation_positions,
        split_ids=train_ids,
    )
    val_ds = PhysioNetDataset(
        csv_path, wav_dir, preprocessor,
        positions=data_cfg.auscultation_positions,
        split_ids=val_ids,
    )

    # Conteos reales del split de entrenamiento → para CBFocalLoss
    samples_per_class = train_ds.get_class_counts()
    logger.info(
        f"Train: {len(train_ds)} muestras | "
        f"Val: {len(val_ds)} muestras | "
        f"Conteos [Present, Absent, Unknown]: {samples_per_class}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
        collate_fn=collate_fn_pad,
        drop_last=True,
        persistent_workers=train_cfg.num_workers > 0,
        multiprocessing_context="fork" if train_cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
        collate_fn=collate_fn_pad,
        persistent_workers=train_cfg.num_workers > 0,
        multiprocessing_context="fork" if train_cfg.num_workers > 0 else None,
    )

    return train_loader, val_loader, samples_per_class
