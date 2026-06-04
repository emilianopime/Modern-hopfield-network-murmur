import hashlib
import logging
from pathlib import Path

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
    Transforma grabaciones PCG en espectrogramas de Mel.
    Pipeline: wavelet denoising → Savitzky-Golay → Mel spectrogram → Z-score.
    Devuelve (1, n_mels, T_frames). Cachea resultados en disco.
    """

    def __init__(self, config: PreprocessConfig):
        self.cfg = config
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)

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

    def __call__(self, wav_path: str | Path) -> torch.Tensor:
        wav_path = Path(wav_path)
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
            resampler = T.Resample(orig_freq=sr, new_freq=self.cfg.sample_rate)
            waveform = resampler(waveform)
            logger.warning(f"{wav_path.name}: remuestreado de {sr} Hz → {self.cfg.sample_rate} Hz")

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        signal_np = waveform.squeeze(0).numpy()
        signal_np = self._wavelet_denoise(signal_np)
        signal_np = self._savitzky_golay_smooth(signal_np)

        waveform_clean = torch.from_numpy(signal_np).float().unsqueeze(0)
        mel_spec = self._mel_transform(waveform_clean)
        mel_db = self._amplitude_to_db(mel_spec)

        return self._normalize(mel_db)

    def _wavelet_denoise(self, signal: np.ndarray) -> np.ndarray:
        """Umbralización wavelet de Donoho-Johnstone con db6 soft-thresholding."""
        coeffs = pywt.wavedec(signal, wavelet=self.cfg.wavelet, level=self.cfg.wavelet_level, mode="periodization")
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        coeffs_denoised = [coeffs[0]] + [
            pywt.threshold(c, value=threshold, mode=self.cfg.threshold_mode)
            for c in coeffs[1:]
        ]
        denoised = pywt.waverec(coeffs_denoised, wavelet=self.cfg.wavelet, mode="periodization")
        return denoised[: len(signal)]

    def _savitzky_golay_smooth(self, signal: np.ndarray) -> np.ndarray:
        wl = self.cfg.sg_window_length
        if wl % 2 == 0:
            wl += 1
        wl = max(wl, self.cfg.sg_polyorder + 2)
        return savgol_filter(signal, window_length=wl, polyorder=self.cfg.sg_polyorder, mode="interp")

    @staticmethod
    def _normalize(tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / (std + 1e-8)

    def _cache_path(self, wav_path: Path) -> Path:
        cfg_str = (
            f"{self.cfg.wavelet}{self.cfg.wavelet_level}"
            f"{self.cfg.sg_window_length}{self.cfg.sg_polyorder}"
            f"{self.cfg.n_fft}{self.cfg.hop_length}{self.cfg.n_mels}"
        )
        key = hashlib.md5(f"{wav_path}{cfg_str}".encode()).hexdigest()
        return self.cfg.cache_dir / f"{wav_path.stem}_{key}.pt"
