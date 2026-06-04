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
    Convierte pesos de atención (B, heads, quantity, T') en una curva de relevancia temporal [0, 1].
    prototype_idx selecciona qué prototipo Hopfield visualizar.
    """
    if attention_weights.dim() == 4:
        weights = attention_weights[0, :, prototype_idx, :].mean(dim=0)
    elif attention_weights.dim() == 3:
        weights = attention_weights[0, prototype_idx, :]
    else:
        weights = attention_weights.squeeze()

    weights = weights.cpu().float().numpy()
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
    Figura de 3 paneles: forma de onda + heatmap, espectrograma + curva de atención, relevancia temporal.
    Guarda en disco si se provee output_path.
    """
    waveform, sr = torchaudio.load(str(wav_path))
    waveform_np  = waveform.squeeze().numpy()
    time_audio   = np.arange(len(waveform_np)) / sr

    attn      = extract_hopfield_weights(attention_weights, hop_length, sample_rate)
    T_frames  = attn.shape[0]
    time_frames = np.arange(T_frames) * hop_length / sample_rate
    attn_interp = np.interp(time_audio, time_frames, attn)

    mel_db = spectrogram.squeeze(0).cpu().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax1 = axes[0]
    ax1.plot(time_audio, waveform_np, color="steelblue", lw=0.6, alpha=0.8)
    ax1.fill_between(time_audio, waveform_np.min(), waveform_np.max(),
                     alpha=attn_interp * 0.4, color="crimson")
    ax1.set_ylabel("Amplitud")
    ax1.set_title("Señal PCG con activación Hopfield (rojo = soplo detectado)")
    ax1.set_xlim([0, time_audio[-1]])

    ax2 = axes[1]
    im = ax2.imshow(mel_db, aspect="auto", origin="lower", cmap="magma",
                    extent=[0, time_frames[-1], 0, mel_db.shape[0]])
    ax2.plot(time_frames, attn * mel_db.shape[0], color="cyan", lw=2, label="Activación Hopfield")
    ax2.set_ylabel("Bandas Mel")
    ax2.set_title("Espectrograma de Mel + Curva de Atención")
    ax2.legend(loc="upper right")
    plt.colorbar(im, ax=ax2, label="dB")

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
