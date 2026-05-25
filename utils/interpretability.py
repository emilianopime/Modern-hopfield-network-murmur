"""
utils/interpretability.py — Visualización clínica de pesos de atención

Extrae los pesos de atención de HopfieldPooling e indica exactamente
qué milisegundos de la grabación PCG activaron la "memoria de soplo".

Esto convierte el modelo en una herramienta auditoriable para el médico:
    - Entrada:  grabación .wav + pesos de atención del Hopfield
    - Salida:   heatmap superpuesto sobre el espectrograma / forma de onda

Flujo de interpretabilidad:
    1. Inferencia con `return_attention=True`
    2. `extract_hopfield_weights()` mapea pesos al eje temporal del audio
    3. `plot_attention_heatmap()` genera la visualización clínica
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio


def extract_hopfield_weights(
    attention_weights: torch.Tensor,
    hop_length: int = 256,
    sample_rate: int = 4000,
    prototype_idx: int = 0,
) -> np.ndarray:
    """
    Convierte los pesos de atención del Hopfield a una curva de relevancia temporal.

    Args:
        attention_weights: Tensor de shape (B, num_heads, quantity, T').
                           Obtenido de `model(x, return_attention=True)`.
        hop_length:        Salto de la STFT (en muestras). Conecta T' con el audio.
        sample_rate:       Frecuencia de muestreo del audio original.
        prototype_idx:     Índice del prototipo Hopfield a visualizar
                           (0 = prototipo más discriminativo, generalmente soplo).

    Returns:
        Array 1D de shape (T',) con la relevancia por frame temporal [0, 1].
        Cada valor indica qué tan fuertemente ese frame activó el prototipo de soplo.
    """
    # Promedio sobre batch y cabezas de atención para un prototipo dado
    # attn: (B, heads, quantity, T') → tomar batch=0, promedio de heads, prototipo dado
    if attention_weights.dim() == 4:
        weights = attention_weights[0, :, prototype_idx, :].mean(dim=0)  # (T',)
    elif attention_weights.dim() == 3:
        weights = attention_weights[0, prototype_idx, :]  # (T',)
    else:
        weights = attention_weights.squeeze()

    weights = weights.cpu().float().numpy()

    # Normalizar a [0, 1]
    w_min, w_max = weights.min(), weights.max()
    if w_max > w_min:
        weights = (weights - w_min) / (w_max - w_min)

    return weights


def plot_attention_heatmap(
    wav_path: str | Path,
    attention_weights: torch.Tensor,
    spectrogram: torch.Tensor,
    hop_length: int = 256,
    sample_rate: int = 4000,
    output_path: Optional[Path] = None,
    title: str = "Activación de Memoria Hopfield sobre PCG",
) -> plt.Figure:
    """
    Genera la visualización clínica de interpretabilidad.

    Produce una figura de 3 paneles:
        Panel 1: Forma de onda PCG preprocesada con heatmap de atención.
        Panel 2: Espectrograma de Mel con heatmap superpuesto.
        Panel 3: Curva de relevancia temporal del prototipo Hopfield.

    Args:
        wav_path:           Ruta al archivo .wav original.
        attention_weights:  Tensor de pesos del Hopfield (B, heads, quantity, T').
        spectrogram:        Tensor (1, n_mels, T') — salida de Fase 1.
        hop_length:         Salto de STFT en muestras.
        sample_rate:        Frecuencia de muestreo.
        output_path:        Si se proporciona, guarda la figura en disco.
        title:              Título de la figura.

    Returns:
        Figura matplotlib.
    """
    # Cargar forma de onda original
    waveform, sr = torchaudio.load(str(wav_path))
    waveform_np = waveform.squeeze().numpy()
    time_audio = np.arange(len(waveform_np)) / sr

    # Extraer pesos de atención
    attn = extract_hopfield_weights(attention_weights, hop_length, sample_rate)
    T_frames = attn.shape[0]
    time_frames = np.arange(T_frames) * hop_length / sample_rate

    # Interpolar pesos al eje temporal del audio
    attn_interp = np.interp(time_audio, time_frames, attn)

    # Espectrograma en dB (convertir desde tensor)
    mel_db = spectrogram.squeeze(0).cpu().numpy()  # (n_mels, T')

    # ── Figura ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Panel 1: Forma de onda + atención como color de fondo
    ax1 = axes[0]
    ax1.plot(time_audio, waveform_np, color="steelblue", lw=0.6, alpha=0.8)
    # Superponer heatmap de atención como área coloreada
    ax1.fill_between(time_audio, waveform_np.min(), waveform_np.max(),
                     alpha=attn_interp * 0.4, color="crimson")
    ax1.set_ylabel("Amplitud")
    ax1.set_title("Señal PCG con activación Hopfield (rojo = soplo detectado)")
    ax1.set_xlim([0, time_audio[-1]])

    # Panel 2: Espectrograma de Mel con heatmap superpuesto
    ax2 = axes[1]
    im = ax2.imshow(
        mel_db,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[0, time_frames[-1], 0, mel_db.shape[0]],
    )
    # Superponer atención como curva sobre el espectrograma
    ax2.plot(time_frames, attn * mel_db.shape[0], color="cyan", lw=2,
             label="Activación Hopfield")
    ax2.set_ylabel("Bandas Mel")
    ax2.set_title("Espectrograma de Mel (Fase 1) + Curva de Atención")
    ax2.legend(loc="upper right")
    plt.colorbar(im, ax=ax2, label="dB")

    # Panel 3: Curva de relevancia temporal
    ax3 = axes[2]
    ax3.fill_between(time_frames, 0, attn, alpha=0.7, color="crimson", label="Relevancia")
    ax3.set_xlabel("Tiempo (s)")
    ax3.set_ylabel("Relevancia [0-1]")
    ax3.set_title("Pesos de Atención HopfieldPooling — Prototipo 'Soplo'")
    ax3.set_xlim([0, time_frames[-1]])
    ax3.set_ylim([0, 1.05])
    ax3.legend()

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Figura guardada en: {output_path}")

    return fig
