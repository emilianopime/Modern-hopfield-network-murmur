"""
data/physionet_dataset.py — Dataset PhysioNet Challenge 2022 (CirCor DigiScope)

v3: añade WeightedRandomSampler para oversampling de clases minoritarias.
    Unknown × 10, Present × 5, Absent × 1 → el modelo ve Unknown casi
    tan seguido como Absent, atacando el colapso de Unknown directamente.
"""

import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from config import DataConfig, PreprocessConfig, TrainingConfig
from data.preprocessing import PCGPreprocessor

logger = logging.getLogger(__name__)

LABEL_MAP  = {"Present": 0, "Absent": 1, "Unknown": 2}
AGE_ORDER  = ["Neonate", "Infant", "Child", "Adolescent", "Young Adult"]
AGE_MAP    = {age: i for i, age in enumerate(AGE_ORDER)}
TABULAR_N_FEATURES = 7


# ─────────────────────────────────────────────
# Extracción de features tabulares
# ─────────────────────────────────────────────

def _parse_tabular(row: pd.Series) -> torch.Tensor:
    """
    Extrae 7 features clínicos de una fila del CSV.
    [age_group, sex, height, weight, bmi, recording_count, pregnancy]
    Fuente feature engineering: PF5367600.py (FeatureEngineer del amigo).
    """
    age_str   = str(row.get("Age", "Child")).strip()
    age_group = AGE_MAP.get(age_str, 2) / 4.0

    sex_str = str(row.get("Sex", "Unknown")).strip()
    sex = 1.0 if sex_str == "Male" else 0.0

    try:
        height      = float(row.get("Height", 0))
        height_norm = max(0.0, (height - 50.0) / 150.0)
    except (ValueError, TypeError):
        height_norm = 0.5

    try:
        weight      = float(row.get("Weight", 0))
        weight_norm = max(0.0, (weight - 3.0) / 97.0)
    except (ValueError, TypeError):
        weight_norm = 0.3

    try:
        h_m = float(row.get("Height", 0)) / 100.0
        w_kg = float(row.get("Weight", 0))
        bmi_norm = max(0.0, (w_kg / (h_m ** 2) - 10.0) / 30.0) if h_m > 0.3 else 0.4
    except (ValueError, TypeError, ZeroDivisionError):
        bmi_norm = 0.4

    locs_str = str(row.get("Recording locations:", "")).strip()
    n_locs   = len([l for l in locs_str.split("+") if l.strip()]) if locs_str else 0
    recording_count_norm = n_locs / 4.0

    preg_str   = str(row.get("Pregnancy status", "False")).strip().lower()
    pregnancy  = 1.0 if preg_str in ("true", "1", "yes") else 0.0

    return torch.tensor(
        [age_group, sex, height_norm, weight_norm, bmi_norm, recording_count_norm, pregnancy],
        dtype=torch.float32,
    )


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class PhysioNetDataset(Dataset):
    """
    Dataset de fonocardiogramas del PhysioNet Challenge 2022.
    Devuelve: (spectrogram, tabular, label)
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
        self.positions    = positions or ["AV", "PV", "TV", "MV"]
        self.samples: list[dict] = []

        metadata = pd.read_csv(csv_path)
        required_cols = {"Patient ID", "Murmur"}
        if not required_cols.issubset(metadata.columns):
            raise ValueError(f"CSV debe contener {required_cols}. Encontradas: {set(metadata.columns)}")

        for _, row in metadata.iterrows():
            patient_id = str(row["Patient ID"])
            label_str  = str(row["Murmur"]).strip()

            if split_ids is not None and patient_id not in split_ids:
                continue
            if label_str not in LABEL_MAP:
                logger.warning(f"Paciente {patient_id}: etiqueta desconocida '{label_str}', omitido.")
                continue

            label   = LABEL_MAP[label_str]
            tabular = _parse_tabular(row)

            for pos in self.positions:
                wav_path = wav_dir / f"{patient_id}_{pos}.wav"
                if wav_path.exists():
                    self.samples.append({
                        "patient_id": patient_id,
                        "position":   pos,
                        "wav_path":   wav_path,
                        "label":      label,
                        "tabular":    tabular,
                    })

        logger.info(f"Dataset cargado: {len(self.samples)} grabaciones.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        sample      = self.samples[idx]
        spectrogram = self.preprocessor(sample["wav_path"])
        return spectrogram, sample["tabular"], sample["label"]

    def get_patient_ids(self) -> list[str]:
        return list({s["patient_id"] for s in self.samples})

    def get_class_counts(self) -> list[int]:
        """[n_Present, n_Absent, n_Unknown] — para CBFocalLoss."""
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
    spectrograms, tabulars, labels = zip(*batch)
    max_t  = max(s.shape[-1] for s in spectrograms)
    padded = torch.zeros(len(spectrograms), *spectrograms[0].shape[:-1], max_t)
    for i, spec in enumerate(spectrograms):
        padded[i, ..., : spec.shape[-1]] = spec
    return (
        padded,
        torch.stack(tabulars),
        torch.tensor(labels, dtype=torch.long),
    )


# ─────────────────────────────────────────────
# WeightedRandomSampler — v3
# ─────────────────────────────────────────────

def build_weighted_sampler(
    dataset: PhysioNetDataset,
    class_sample_weights: list[float],
) -> WeightedRandomSampler:
    """
    Crea un WeightedRandomSampler que oversamples clases minoritarias.

    class_sample_weights = [w_Present, w_Absent, w_Unknown]
    Valores recomendados: [5.0, 1.0, 10.0]
      → Unknown aparece 10× más que Absent por unidad de muestreo
      → Present aparece 5× más que Absent por unidad de muestreo

    Usa replacement=True (oversampling con reemplazo) — el número total
    de muestras por época sigue siendo len(dataset) pero con distribución
    re-balanceada entre clases.
    """
    weights = [float(class_sample_weights[s["label"]]) for s in dataset.samples]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
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
    División por paciente para evitar data leakage.

    Returns:
        train_loader, val_loader, samples_per_class
        donde samples_per_class = [n_Present, n_Absent, n_Unknown] reales
        del split de entrenamiento (para CBFocalLoss).
    """
    preprocessor = PCGPreprocessor(preprocess_cfg)
    csv_path     = data_cfg.dataset_path / "training_data.csv"
    wav_dir      = data_cfg.dataset_path

    all_ids  = pd.read_csv(csv_path)["Patient ID"].astype(str).unique().tolist()
    rng      = torch.Generator().manual_seed(data_cfg.random_seed)
    n_val    = int(len(all_ids) * data_cfg.val_split)
    shuffled = torch.randperm(len(all_ids), generator=rng).tolist()
    val_ids  = set(all_ids[i] for i in shuffled[:n_val])
    train_ids = set(all_ids[i] for i in shuffled[n_val:])

    train_ds = PhysioNetDataset(csv_path, wav_dir, preprocessor,
                                positions=data_cfg.auscultation_positions,
                                split_ids=train_ids)
    val_ds   = PhysioNetDataset(csv_path, wav_dir, preprocessor,
                                positions=data_cfg.auscultation_positions,
                                split_ids=val_ids)

    samples_per_class = train_ds.get_class_counts()
    logger.info(
        f"Train: {len(train_ds)} muestras | Val: {len(val_ds)} muestras | "
        f"Conteos [Present, Absent, Unknown]: {samples_per_class}"
    )

    # v3: WeightedRandomSampler — oversampling Unknown×10, Present×5
    if data_cfg.use_weighted_sampler:
        sampler  = build_weighted_sampler(train_ds, data_cfg.sampler_weights)
        shuffle  = False     # sampler y shuffle son mutuamente excluyentes
        logger.info(
            f"WeightedRandomSampler activo — "
            f"pesos [Present, Absent, Unknown]: {data_cfg.sampler_weights}"
        )
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
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
