# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
## claude response
responde en espanol para cualquier prompt que te haga.
## Project Goal

Automatic cardiac murmur classification using Modern Hopfield Networks (MHN) applied to the **George B. Moody PhysioNet Challenge 2022** (CirCor DigiScope dataset). The task is a 3-class patient-level classification: **Present / Absent / Unknown**, using multi-focal phonocardiogram (PCG) recordings (`.wav` files from aortic, pulmonary, tricuspid, and mitral positions).

## Environment Setup

> **CachyOS CUDA isolation protocol**: Do NOT install the CUDA toolkit system-wide. Use PyTorch's bundled CUDA 12.1 binaries to avoid library conflicts with the OS.

```bash
# Create and activate virtualenv (project uses .venv/)
python -m venv .venv
source .venv/bin/activate

# Install PyTorch with embedded CUDA 12.1 binaries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install MHN layer library and audio processing
pip install hopfield-layers librosa
```

The existing `.venv/` uses Python 3.14 (`/usr/bin/python3.14`).

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

## Structure by Phase

```
data/preprocessing.py       ← Fase 1: wavelet denoising → Savitzky-Golay → Mel spectrogram
models/cnn_frontend.py      ← Fase 2: CNN feature extractor (ResNet18/EfficientNet-B0)
models/hopfield_pooling.py  ← Fase 3: HopfieldPooling (memoria asociativa continua)
models/murmur_classifier.py ← Fase 4: modelo completo (integra fases 2-3-4)
training/trainer.py         ← bucle de entrenamiento con freeze gradual del backbone
training/metrics.py         ← weighted accuracy del PhysioNet Challenge
utils/interpretability.py   ← extracción y visualización de pesos de atención
config.py                   ← configuración centralizada (todos los hiperparámetros)
```

Todos los hiperparámetros se modifican exclusivamente en `config.py` — ningún módulo tiene "magic numbers" propios.

## Architecture: 4-Stage Pipeline

The full pipeline is defined in `Reporte_Tecnico_Hopfield_Moderno_PhysioNet.pdf` and implemented end-to-end:

```
.wav audio → Mel Spectrogram → CNN Embeddings → HopfieldPooling → Linear Classifier
```

### Stage 1 — Preprocessing
Convert raw `.wav` recordings to **Mel spectrograms** using `torchaudio`. This produces a 2D time-frequency matrix that captures harmonic content and subtle murmur transitions. Use `DataLoader(num_workers=6)` to saturate the 8-core Ryzen CPU during parallel spectrogram computation.

### Stage 2 — Feature Extractor (CNN Frontend)
A lightweight CNN (ResNet18 or EfficientNet-B0, modified) processes spectrogram blocks and outputs a **temporal sequence of dense embedding vectors**. This is the `Q` (query) fed into the Hopfield layer.

### Stage 3 — Modern Hopfield Network (`HopfieldPooling`)
The core of the architecture. Imported from `hopfield-layers`:

```python
from hflayers import HopfieldPooling
```

The MHN update rule is mathematically equivalent to Softmax attention:
`M(Q, K, V) = softmax(β · Q · Kᵀ) · V`

where **`K` and `V` are learned stored patterns** (the associative memory, not dynamic tokens). This means the layer acts as a **fixed clinical pattern memory** — comparing the patient's embeddings against internalized prototypes of murmur vs. normal heartbeats.

**HopfieldPooling** collapses the temporal dimension by selectively extracting only frames that match a stored "murmur" pattern, discarding irrelevant audio (breathing, friction, ambient noise). This solves the **Multiple Instance Learning** problem: a murmur may appear in only a fraction-of-a-second of a multi-minute recording.

### Stage 4 — Classification Head
A linear layer maps the pooled representation to 3-class probabilities. Loss: `CrossEntropyLoss`.

## Critical Hyperparameter: Beta (β)

β controls the selectivity ("temperature") of the Hopfield memory retrieval:

- **Initialize**: `β = 1 / √d` where `d` is the embedding dimension
- **β too high** → model fixates on only the single most active frame (ignores context)
- **β too low** → degenerates to simple average pooling (loses the associative memory advantage)

Tune β dynamically during training; it is the most impactful hyperparameter in this architecture.

## Hardware Targets

| Resource | Spec | Usage |
|---|---|---|
| GPU | NVIDIA RTX 3060 12GB | `batch_size=32` or `64`; MHN stores correlation matrices in VRAM |
| CPU | AMD Ryzen 7 5700x (8C/16T) | `DataLoader(num_workers=6)` for parallel spectrogram prep |
| RAM | 32GB DDR4 | Pre-computed spectrograms cached in memory to avoid OOM |

## Interpretability

The attention weights from `HopfieldPooling` directly indicate **which milliseconds of audio activated the murmur memory**. Extract these weights during inference to overlay a visual alert on the original phonocardiogram — this is the clinical explainability feature (avoids black-box behavior).

## Key Design Rationale

- **Why MHN over Transformer?** Standard Transformers compute dynamic self-attention between all input tokens. The MHN layer uses **fixed learned patterns as keys/values**, acting as a clinical prototype memory rather than dynamic routing. This improves robustness to variable-length recordings.
- **Why avoid Global Average Pooling?** A brief murmur diluted across minutes of normal audio would vanish under average pooling. HopfieldPooling is selective — it surfaces anomalies.
- **Multi-focal inputs**: Each patient has recordings from multiple auscultation positions. The model must handle this; patient-level (not recording-level) labels are the target.
