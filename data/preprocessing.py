"""
data/preprocessing.py — FASE 1: Preprocesamiento de señal PCG

Pipeline completo:
    .wav  →  [1-a] Wavelet Denoising
          →  [1-b] Savitzky-Golay Smoothing
          →  [1-c] Mel Spectrogram (torchaudio)
          →  Tensor normalizado listo para la CNN

Justificación de cada paso:
    - Wavelet (db6): elimina ruido no estacionario (fricción del estetoscopio,
      llanto, respiración). Umbral suave preserva la morfología del soplo.
    - Savitzky-Golay: suavizado polinomial post-reconstrucción. Elimina
      artefactos residuales de la reconstrucción wavelet sin perder los picos
      sistólicos/diastólicos donde se manifiesta el soplo.
    - Mel Spectrogram: transforma la señal 1D temporal en una representación
      2D tiempo-frecuencia que captura los armónicos del soplo (20–1000 Hz).
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pywt
import torch
import torchaudio
import torchaudio.transforms as T
from scipy.signal import savgol_filter

from config import PreprocessConfig

logger = logging.getLogger(__name__)


class PCGPreprocessor:
    """
    Transforma grabaciones PCG en espectrogramas de Mel preprocesados.

    Uso:
        preprocessor = PCGPreprocessor(cfg.preprocess)
        spectrogram = preprocessor("data/physionet_2022/12345_AV.wav")
        # → Tensor shape: (1, n_mels, T_frames)
    """

    def __init__(self, config: PreprocessConfig):
        self.cfg = config
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)

        # Transformación Mel Spectrogram (reutilizable, en CPU)
        self._mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
            power=config.power,
        )
        self._amplitude_to_db = T.AmplitudeToDB(top_db=config.top_db)

    # ─────────────────────────────────────────
    # API pública
    # ─────────────────────────────────────────

    def __call__(self, wav_path: str | Path) -> torch.Tensor:
        """
        Procesa un archivo .wav y devuelve el espectrograma normalizado.

        Returns:
            Tensor de shape (1, n_mels, T_frames), dtype=float32.
            La dimensión 1 actúa como canal único para la CNN.
        """
        wav_path = Path(wav_path)
        cache_path = self._cache_path(wav_path)

        if self.cfg.cache_enabled and cache_path.exists():
            return torch.load(cache_path, weights_only=True)

        spectrogram = self._process(wav_path)

        if self.cfg.cache_enabled:
            torch.save(spectrogram, cache_path)

        return spectrogram

    def process_batch(
        self, wav_paths: list[Path]
    ) -> list[torch.Tensor]:
        """Procesa una lista de archivos (útil para preparar el dataset completo)."""
        return [self(p) for p in wav_paths]

    # ─────────────────────────────────────────
    # Pipeline interno
    # ─────────────────────────────────────────

    def _process(self, wav_path: Path) -> torch.Tensor:
        # Carga audio con torchaudio (devuelve Tensor)
        waveform, sr = torchaudio.load(str(wav_path))

        # Remuestrear si la frecuencia no coincide con la esperada
        if sr != self.cfg.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.cfg.sample_rate)
            waveform = resampler(waveform)
            logger.warning(
                f"{wav_path.name}: remuestreado de {sr} Hz → {self.cfg.sample_rate} Hz"
            )

        # Convertir a mono si es multicanal
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Extraer señal 1D como numpy para procesado DSP
        signal_np = waveform.squeeze(0).numpy()

        # ── Paso 1-a: Wavelet Denoising ─────────────────────────────────
        signal_np = self._wavelet_denoise(signal_np)

        # ── Paso 1-b: Savitzky-Golay Smoothing ──────────────────────────
        signal_np = self._savitzky_golay_smooth(signal_np)

        # Reconvertir a Tensor (shape: 1 × N)
        waveform_clean = torch.from_numpy(signal_np).float().unsqueeze(0)

        # ── Paso 1-c: Mel Spectrogram ────────────────────────────────────
        mel_spec = self._mel_transform(waveform_clean)       # (1, n_mels, T)
        mel_db = self._amplitude_to_db(mel_spec)             # escala logarítmica en dB

        # Normalización Z-score por espectrograma (media=0, std=1)
        mel_db = self._normalize(mel_db)

        return mel_db  # shape: (1, n_mels, T_frames)

    # ─────────────────────────────────────────
    # Paso 1-a: Wavelet Denoising
    # ─────────────────────────────────────────

    def _wavelet_denoise(self, signal: np.ndarray) -> np.ndarray:
        """
        Descomposición wavelet multinivel + umbralización suave.

        Método:
            - Wavelet Daubechies 6 (db6): forma suave que encaja con la
              morfología del fonocardiograma.
            - Umbral universal de Donoho-Johnstone:
                  λ = σ̂ · √(2 · ln N)
              donde σ̂ = MAD(coeficientes detalle nivel 1) / 0.6745.
            - Umbralización 'soft': reduce los coeficientes hacia cero
              preservando continuidad (evita artefactos de Gibbs).
            - Solo los coeficientes de detalle son umbralados; la
              aproximación gruesa (nivel máximo) se conserva intacta.
        """
        coeffs = pywt.wavedec(
            signal,
            wavelet=self.cfg.wavelet,
            level=self.cfg.wavelet_level,
            mode="periodization",
        )

        # Estimación robusta de σ a partir de los detalles más finos (nivel 1)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))

        # Umbralizar todos los niveles de detalle (índices 1 en adelante)
        coeffs_denoised = [coeffs[0]] + [
            pywt.threshold(c, value=threshold, mode=self.cfg.threshold_mode)
            for c in coeffs[1:]
        ]

        # Reconstrucción de la señal
        denoised = pywt.waverec(coeffs_denoised, wavelet=self.cfg.wavelet, mode="periodization")

        # Asegurar misma longitud que la entrada (waverec puede agregar una muestra)
        return denoised[: len(signal)]

    # ─────────────────────────────────────────
    # Paso 1-b: Savitzky-Golay Smoothing
    # ─────────────────────────────────────────

    def _savitzky_golay_smooth(self, signal: np.ndarray) -> np.ndarray:
        """
        Filtro Savitzky-Golay sobre la señal reconstruida post-wavelet.

        Ajusta un polinomio local de grado `polyorder` a cada ventana de
        `window_length` muestras. A diferencia del promedio móvil, preserva
        los máximos y mínimos del ciclo cardíaco (S1, S2, soplo).

        window_length=51 a 4000 Hz → ventana de ~12.75 ms, suficientemente
        estrecha para no diluir eventos cardíacos pero amplia para eliminar
        oscilaciones de alta frecuencia residuales.
        """
        # window_length debe ser impar y > polyorder
        wl = self.cfg.sg_window_length
        if wl % 2 == 0:
            wl += 1
        wl = max(wl, self.cfg.sg_polyorder + 2)

        return savgol_filter(
            signal,
            window_length=wl,
            polyorder=self.cfg.sg_polyorder,
            mode="interp",
        )

    # ─────────────────────────────────────────
    # Utilidades
    # ─────────────────────────────────────────

    @staticmethod
    def _normalize(tensor: torch.Tensor) -> torch.Tensor:
        """Normalización Z-score: (x - μ) / (σ + ε)."""
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / (std + 1e-8)

    def _cache_path(self, wav_path: Path) -> Path:
        """Genera una ruta de caché única basada en el hash del archivo y la config."""
        cfg_str = (
            f"{self.cfg.wavelet}{self.cfg.wavelet_level}"
            f"{self.cfg.sg_window_length}{self.cfg.sg_polyorder}"
            f"{self.cfg.n_fft}{self.cfg.hop_length}{self.cfg.n_mels}"
        )
        key = hashlib.md5(f"{wav_path}{cfg_str}".encode()).hexdigest()
        return self.cfg.cache_dir / f"{wav_path.stem}_{key}.pt"

    @property
    def output_n_mels(self) -> int:
        return self.cfg.n_mels
