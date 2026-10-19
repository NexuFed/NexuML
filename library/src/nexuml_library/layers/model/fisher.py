"""FISHER industrial-signal encoder for NexuML's generic nn_module.

Implements the public checkpoint's waveform-to-log-STFT preprocessing
and exposes one embedding tensor per observation. The checkpoint and
its model implementation remain owned by the original FISHER authors.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Literal, cast

import torch
from torch import nn
from torch.nn import functional as F

_FISHER_LOG_MEAN = 3.017344307886898
_FISHER_LOG_STD = 2.1531635155379805 * 2


class FISHEREncoder(nn.Module):
    """Wrap the pretrained FISHER model as a waveform-to-embedding module.

    The checkpoint expects log-magnitude STFTs (25-ms window, 10-ms hop,
    no centering, published normalization), not log-mel spectrograms.
    Inputs are [batch, samples], or [batch, channels, samples]; multiple
    channels are averaged before encoding.

    Args:
        model_name: Official Hugging Face checkpoint or compatible local path.
        sample_rate: Sampling rate of incoming waveforms; no resampling.
        pooling: Native FISHER extract_features output or pooled feature-only
            tokens, using mean or mean_std.
        frozen: Disable backbone gradients and keep it in evaluation mode.
        trust_remote_code: Explicit permission to run the checkpoint's Python
            code. Only enable for trusted repositories.
        revision: Optional model revision or immutable commit hash.
        local_files_only: Load only model files already cached locally.

    Raises:
        ImportError: If optional Transformers is not installed.
        ValueError: If the sampling rate or loaded FISHER model is invalid.
    """

    window: torch.Tensor

    def __init__(
        self,
        model_name: str = "jiangab/FISHER-small-0723",
        sample_rate: int = 16000,
        pooling: Literal["native", "mean", "mean_std"] = "native",
        frozen: bool = True,
        trust_remote_code: bool = True,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        if sample_rate < 1000:
            raise ValueError("sample_rate must be at least 1000 Hz")
        if pooling not in {"native", "mean", "mean_std"}:
            raise ValueError(f"Unsupported FISHER pooling method: {pooling}")
        self.sample_rate = sample_rate
        self.pooling = pooling
        self.frozen = frozen
        self.n_fft = 25 * sample_rate // 1000
        self.hop_length = 10 * sample_rate // 1000
        self.max_frames = 1024
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)

        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "FISHEREncoder requires Transformers; install nexuml-library[pretrained]"
            ) from exc

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
        )
        configuration = getattr(self.model, "cfg", None)
        self.band_width = int(getattr(configuration, "band_width", 0))
        if self.band_width <= 0:
            raise ValueError(
                "Expected a FISHER checkpoint exposing model.cfg.band_width"
            )

        if frozen:
            self.model.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True) -> FISHEREncoder:
        """Set train/eval mode while preserving a frozen backbone's eval mode.

        Returns:
            This module with the requested train/eval mode.
        """
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    def spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform samples to the official FISHER log-STFT input.

        Args:
            waveform: Float waveforms [B, T] or [B, C, T].

        Returns:
            Log-STFT tensor [B, 1, frames, frequency_bins].

        Raises:
            ValueError: If input has invalid shape, length or nonfinite values.
            TypeError: If waveform samples are not floating-point.
        """
        if waveform.ndim == 3:
            waveform = waveform.mean(dim=1)
        if waveform.ndim != 2 or not waveform.shape[0]:
            raise ValueError("FISHEREncoder expects [B, T] or [B, C, T] waveforms")
        if not waveform.is_floating_point():
            raise TypeError("FISHEREncoder requires floating-point waveforms")
        if waveform.shape[1] < self.n_fft:
            raise ValueError(
                f"Waveform has {waveform.shape[1]} samples; "
                f"FISHER requires at least {self.n_fft}"
            )
        if not torch.isfinite(waveform).all():
            raise ValueError("FISHEREncoder requires finite waveform samples")

        samples = waveform.to(torch.float32)
        samples = samples - samples.mean(dim=-1, keepdim=True)
        stft = torch.stft(
            samples,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(device=samples.device, dtype=samples.dtype),
            center=False,
            return_complex=True,
        )
        features = torch.log(stft.abs() + 1e-10).transpose(-2, -1)
        features = (features + _FISHER_LOG_MEAN) / _FISHER_LOG_STD
        features = features[:, : self.max_frames, :]
        if features.shape[-1] < self.band_width:
            features = F.pad(features, (0, self.band_width - features.shape[-1]))
        return features.unsqueeze(1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract one FISHER embedding for each waveform.

        Args:
            waveform: Float waveforms [B, T] or [B, C, T].

        Returns:
            Feature tensor [B, D]. D depends on pooling and sampling rate.

        Raises:
            TypeError: If a model output is not a tensor or mapping.
            ValueError: If feature shape disagrees with the batch dimension.
        """
        spec = self.spectrogram(waveform)
        context = torch.no_grad() if self.frozen else nullcontext()
        with context:
            if self.pooling == "native":
                output = cast(Any, self.model).extract_features(spec)
            else:
                result = self.model(spec, mode="IMAGE", mask=False, features_only=True)
                if not isinstance(result, dict):
                    raise TypeError("FISHER feature-only forward must return a dictionary")
                tokens = result.get("x")
                if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
                    raise ValueError("FISHER feature-only output must contain [B*bands, N, D]")
                batch = spec.shape[0]
                if tokens.shape[0] % batch:
                    raise ValueError("FISHER token batch is incompatible with waveform batch")
                tokens = tokens.reshape(batch, -1, tokens.shape[-1])
                mean = tokens.mean(dim=1)
                if self.pooling == "mean_std":
                    deviation = tokens.std(dim=1, unbiased=False)
                    output = torch.cat((mean, deviation), dim=-1)
                else:
                    output = mean
        if not isinstance(output, torch.Tensor):
            raise TypeError("FISHEREncoder must return a tensor")
        if output.ndim != 2 or output.shape[0] != waveform.shape[0]:
            raise ValueError("FISHEREncoder must return [B, D] embeddings")
        return output
