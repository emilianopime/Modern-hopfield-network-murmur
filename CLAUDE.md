# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## claude response
responde en espanol para cualquier prompt que te haga.

## Project Goal

Automatic cardiac murmur classification using Modern Hopfield Networks (MHN) applied to the **George B. Moody PhysioNet Challenge 2022** (CirCor DigiScope dataset). 3-class patient-level classification: **Present / Absent / Unknown**, using multi-focal phonocardiogram (PCG) recordings (`.wav` files from aortic, pulmonary, tricuspid, and mitral positions).

Challenge scoring weights: Present=5, Absent=1, Unknown=3. The model optimizes Weighted Accuracy, not standard accuracy.

## Environment Setup

> **CachyOS CUDA isolation protocol**: Do NOT install the CUDA toolkit system-wide. Use PyTorch's bundled CUDA 12.1 binaries to avoid library conflicts with the OS.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install hopfield-layers librosa
```

The existing `.venv/` uses Python 3.14 (`/usr/bin/python3.14`). `hopfield-layers` is optional — if missing, `HopfieldPoolingLayer` automatically falls back to `nn.MultiheadAttention` with learned queries.

## Dataset Layout

```
data/physionet_2022/
├── training_data.csv       ← requiere columnas: Patient ID, Murmur, Age, Sex, Height, Weight, ...
├── 12345_AV.wav
├── 12345_PV.wav
└── ...                     ← nombre de archivo: {PatientID}_{Position}.wav
```

## Commands

```bash
# Entrenamiento completo
python main.py train

# Pre-calcular y cachear espectrogramas solamente (Fase 1)
python main.py preprocess

# Evaluar un checkpoint guardado
python main.py eval --checkpoint checkpoints/checkpoint_best.pt

# Visualizar interpretabilidad (pesos de atención Hopfield sobre el audio)
python main.py explain --wav data/physionet_2022/12345_AV.wav --checkpoint checkpoints/checkpoint_best.pt
```

## Module Map

```
config.py                   ← todos los hiperparámetros (único lugar para modificarlos)
data/preprocessing.py       ← PCGPreprocessor: wavelet → Savitzky-Golay → Mel spectrogram + caché
data/physionet_dataset.py   ← PhysioNetDataset, collate_fn_pad, WeightedRandomSampler, build_dataloaders
models/cnn_frontend.py      ← CNNFrontend: ResNet18/EfficientNet-B0 modificado para PCG monocanal
models/hopfield_pooling.py  ← HopfieldPoolingLayer: wrapper MHN con fallback a MHA
models/murmur_classifier.py ← MurmurClassifier: integra CNN + Hopfield + TabularEncoder + clasificador
training/trainer.py         ← Trainer: bucle completo con SpecAugment, ClinicalMixUp, SGDR
training/metrics.py         ← CBFocalLoss, compute_challenge_score, format_metrics
utils/interpretability.py   ← extract_hopfield_weights, plot_attention_heatmap
```

## Architecture: 4-Stage Pipeline

```
.wav → (1, n_mels=128, T) → (B, T', 512) → (B, quantity*512) → [Present, Absent, Unknown]
        Preprocessing        CNNFrontend     HopfieldPooling      MLP Classifier
                                                    ↑
                                        + (B, 64) TabularEncoder
                                          (age, sex, height, weight, bmi, n_locs, pregnancy)
```

### Stage 1 — Preprocessing (`data/preprocessing.py`)
`PCGPreprocessor.__call__(wav_path)` returns `(1, 128, T_frames)`. Pipeline: Daubechies-6 wavelet denoising → Savitzky-Golay smoothing → Mel spectrogram (4000 Hz, 20–1000 Hz, 128 bins) → Z-score normalization. Results are cached in `data/cache/` keyed by MD5 of file path + config params.

### Stage 2 — CNN Frontend (`models/cnn_frontend.py`)
ResNet18 modified: (1) first conv adapted from 3→1 channel using averaged pretrained weights; (2) `avgpool` and `fc` removed; (3) `AdaptiveAvgPool2d((1, None))` collapses only the frequency axis, preserving the temporal axis. Output: `(B, T', 512)` where `T' = T/32`.

### Stage 3 — Modern Hopfield Network (`models/hopfield_pooling.py`)
`HopfieldPoolingLayer` wraps `hflayers.HopfieldPooling`. Key parameter `quantity=4` sets the number of learned prototype queries. β initialized as `1/√embedding_dim`. Output: `(B, quantity * output_size)`. Attention weights stored in `self.last_attention_weights` for interpretability.

### Stage 4 — Classification Head (`models/murmur_classifier.py`)
`TabularEncoder` maps 7 clinical features → 64d embedding (BatchNorm1d → Linear → LayerNorm → GELU → Dropout → Linear → GELU). Concatenated with Hopfield output → `LayerNorm → Dropout → Linear → GELU → Dropout → Linear(3)`.

## Critical Design Decisions

**Patient-level train/val split** (`data/physionet_dataset.py:build_dataloaders`): split is done by patient ID before building datasets to prevent data leakage. Multiple recordings from the same patient (up to 4 positions) always land in the same split.

**Variable-length recordings**: `collate_fn_pad` zero-pads spectrograms in each batch to the longest recording. No fixed truncation — the Hopfield layer handles arbitrary-length sequences.

**Two-phase training** (`training/trainer.py:_maybe_unfreeze_backbone`):
- Epochs 1–12: CNN frozen, only HopfieldPooling + Tabular + Classifier train. LR warms up linearly.
- Epoch 13+: full fine-tuning. CNN added as a second param group at `lr × 0.1` (differential learning rate).

**SGDR schedule**: after warmup, cosine annealing with warm restarts every T₀=20 epochs. Restarts at epochs ~26, 46, 66, 86, 106. Best checkpoint in v3 was epoch 117 (WA=0.7197).

**WeightedRandomSampler**: Unknown×10, Present×5 oversampling addresses the extreme class imbalance (132 Unknown vs 2026 Absent samples). This raised Unknown recall from 0.208 to 0.500.

## Critical Hyperparameter: Beta (β)

β controls retrieval selectivity in HopfieldPooling. Initialized at `1/√d`. High β → fixates on single frame; low β → degenerates to average pooling. Tune carefully; it is the most impactful hyperparameter in this architecture.

## Hardware Targets

| Resource | Spec | Usage |
|---|---|---|
| GPU | NVIDIA RTX 3060 12GB | `batch_size=32`; MHN correlation matrices in VRAM |
| CPU | AMD Ryzen 7 5700x (8C/16T) | `DataLoader(num_workers=6)` for parallel spectrogram prep |
| RAM | 32GB DDR4 | Pre-computed spectrograms cached to avoid OOM |
