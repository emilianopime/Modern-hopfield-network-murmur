"""
data/physionet_dataset.py — Dataset PhysioNet Challenge 2022 (CirCor DigiScope)

Estructura esperada del directorio del dataset:
    data/physionet_2022/
    ├── training_data.csv          # Metadatos de pacientes + etiquetas
    ├── {patient_id}_{pos}.wav     # Grabaciones (pos: AV, PV, TV, MV)
    └── {patient_id}_{pos}.hea    # Cabeceras WFDB (opcional)

Etiqueta objetivo (nivel paciente):
    "Murmur": "Present" | "Absent" | "Unknown"

Estrategia multi-focal:
    Cada grabación de una posición auscultoria (AV/PV/TV/MV) se trata como
    una instancia independiente durante el entrenamiento (comparten la etiqueta
    del paciente). En inferencia, las predicciones por posición se agregan
    (probabilidades promedio) para obtener la predicción a nivel paciente.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from config import DataConfig, PreprocessConfig, TrainingConfig
from data.preprocessing import PCGPreprocessor

logger = logging.getLogger(__name__)

# Mapeo de etiqueta textual → índice de clase
LABEL_MAP = {"Present": 0, "Absent": 1, "Unknown": 2}


class PhysioNetDataset(Dataset):
    """
    Dataset de fonocardiogramas del PhysioNet Challenge 2022.

    Cada ítem corresponde a UNA grabación de UNA posición de UN paciente.
    La etiqueta es la del paciente (nivel paciente → etiqueta de instancia).

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

            for pos in self.positions:
                wav_path = wav_dir / f"{patient_id}_{pos}.wav"
                if wav_path.exists():
                    self.samples.append(
                        {
                            "patient_id": patient_id,
                            "position": pos,
                            "wav_path": wav_path,
                            "label": label,
                        }
                    )

        logger.info(
            f"Dataset cargado: {len(self.samples)} grabaciones "
            f"de {len(metadata)} pacientes."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        spectrogram = self.preprocessor(sample["wav_path"])  # (1, n_mels, T)
        return spectrogram, sample["label"]

    def get_patient_ids(self) -> list[str]:
        return list({s["patient_id"] for s in self.samples})


def collate_fn_pad(
    batch: list[tuple[torch.Tensor, int]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate con padding dinámico en el eje temporal.

    Las grabaciones PCG tienen duraciones variables. Se hace padding con ceros
    hasta la longitud del elemento más largo del batch (padding a la derecha).
    La CNN y HopfieldPooling son agnósticos a la longitud temporal exacta.
    """
    spectrograms, labels = zip(*batch)
    max_t = max(s.shape[-1] for s in spectrograms)

    padded = torch.zeros(len(spectrograms), *spectrograms[0].shape[:-1], max_t)
    for i, spec in enumerate(spectrograms):
        padded[i, ..., : spec.shape[-1]] = spec

    return padded, torch.tensor(labels, dtype=torch.long)


def build_dataloaders(
    data_cfg: DataConfig,
    preprocess_cfg: PreprocessConfig,
    train_cfg: TrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    """
    Construye los DataLoaders de entrenamiento y validación.

    La división train/val se hace a nivel de PACIENTE para evitar data leakage
    (todas las posiciones de un paciente van al mismo split).
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

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
        collate_fn=collate_fn_pad,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
        collate_fn=collate_fn_pad,
    )

    logger.info(f"Train: {len(train_ds)} muestras | Val: {len(val_ds)} muestras")
    return train_loader, val_loader
