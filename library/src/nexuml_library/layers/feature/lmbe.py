"""Log Mel Band Energy (LMBE) feature extractor layer."""

from __future__ import annotations
from nexuml.core.discovery import layer

import logging

import torch

from nexuml.core.base_layer import PipelineLayer

logger = logging.getLogger(__name__)


@layer("LMBE")
class LMBE(PipelineLayer):
    """Compute Log Mel Band Energies from a raw waveform.

    Uses torchaudio's MelSpectrogram + AmplitudeToDB pipeline.
    Optionally falls back to librosa for compatibility.

    Args:
        n_mels: Number of Mel filterbanks.
        n_fft: FFT size.
        hop_length: Number of samples between frames.
        win_length: Window length in samples.
        power: Spectrogram power (1=magnitude, 2=power).
        fmin: Minimum frequency.
        fmax: Maximum frequency.
        sr: Sample rate.
        mel_scale: Mel scale type ("slaney" or "htk").
        pad_mode: Padding mode for STFT.
        use_librosa: Use librosa backend instead of torchaudio.
    """

    def __init__(
        self,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 512,
        win_length: int | None = None,
        power: int = 2,
        fmin: int = 0,
        fmax: int = 8000,
        sr: int | None = None,
        sample_rate: int | None = None,
        # Common alias used by scenarios/configs; previously silently dropped.
        sampling_rate: int | None = None,
        mel_scale: str = "slaney",
        pad_mode: str = "constant",
        use_librosa: bool = False,
        to_db: bool = True,
        normalize: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        resolved_sr = (
            sample_rate
            if sample_rate is not None
            else (sr if sr is not None else (sampling_rate if sampling_rate is not None else 16000))
        )
        self.sr = resolved_sr
        self.power = power
        self.use_librosa = use_librosa
        self.mel_scale = mel_scale
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.win_length = win_length if win_length is not None else n_fft
        self.fmin = fmin
        self.fmax = fmax
        self.to_db = to_db
        self.normalize = normalize

        self.mel_spectrogram = None
        self.amplitude_to_db = None
        if not use_librosa:
            import torchaudio

            self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr,
                n_fft=n_fft,
                win_length=self.win_length,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=fmin,
                f_max=fmax,
                power=float(power),
                normalized=False,
                mel_scale=mel_scale,
                norm=mel_scale,
                pad_mode=pad_mode,
            )
            self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
                stype="power" if power == 2 else "magnitude",
            )

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_librosa:
            mel = self._forward_librosa(x)
        else:
            assert self.mel_spectrogram is not None
            mel = self.mel_spectrogram(x)
            if self.to_db:
                assert self.amplitude_to_db is not None
                mel = self.amplitude_to_db(mel)

        if self.normalize:
            dims = tuple(range(1, mel.ndim))
            mean = mel.mean(dim=dims, keepdim=True)
            std = mel.std(dim=dims, keepdim=True).clamp_min(1e-6)
            mel = (mel - mean) / std

        # PatchEmbedding expects (B, C, H, W), so expose spectrograms with an
        # explicit channel dimension.
        if mel.ndim == 3:
            mel = mel.unsqueeze(1)
        elif mel.ndim == 2:
            mel = mel.unsqueeze(0).unsqueeze(0)
        return mel

    def _forward_librosa(self, x: torch.Tensor) -> torch.Tensor:
        import librosa
        import numpy as np

        x_np = x.squeeze(1).cpu().numpy() if x.dim() > 1 else x.cpu().numpy()
        mel = librosa.feature.melspectrogram(
            y=x_np,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=self.power,
            fmax=self.fmax,
            fmin=self.fmin,
            win_length=self.win_length,
            htk=self.mel_scale == "htk",
            norm=self.mel_scale if self.mel_scale in ("slaney",) else None,
        )
        log_mel = 20.0 / self.power * np.log10(np.maximum(mel, np.finfo(np.float64).eps))
        return torch.from_numpy(log_mel).to(device=x.device, dtype=x.dtype)
