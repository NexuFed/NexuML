"""Reconstruction visualizer."""

from __future__ import annotations
from nexuml.core.discovery import eval_algorithm

import logging
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from nexuml.evaluation.algorithm import EvalAlgorithm
from nexuml.evaluation.storage import ReservoirTensorDictBuffer
from nexuml_library.evaluation.visualizers._plotting import format_label, log_figure

logger = logging.getLogger(__name__)


@eval_algorithm("reconstruction_visualizer")
class ReconstructionVisualizer(EvalAlgorithm):
    """Shows side-by-side original vs reconstructed samples."""

    type_key = "reconstruction_visualizer"

    def __init__(
        self,
        feature_key: str | None = None,
        reconstructed_key: str | None = None,
        mask_key: str | None = None,
        label_keys: list[str] | None = None,
        n_samples: int = 8,
        patch_size: int | tuple[int, int] | None = None,
        storage_backend: str = "memory",
        storage_path: str | None = None,
    ) -> None:
        if feature_key is None or reconstructed_key is None:
            raise ValueError(
                "ReconstructionVisualizer requires explicit feature_key and reconstructed_key."
            )
        self.feature_key = feature_key
        self.reconstructed_key = reconstructed_key
        self.mask_key = mask_key
        self.label_keys = list(label_keys or [])
        self.n_samples = n_samples
        self.patch_size = patch_size
        self._resolved_patch_size: tuple[int, int] | None = None
        self._storage = ReservoirTensorDictBuffer(
            max_samples=n_samples,
            storage_backend=storage_backend,
            storage_path=Path(storage_path) if storage_path is not None else None,
        )

    def fit_batch(self, x: TensorDict, y: TensorDict | None) -> None:
        pass

    def fit_end(self) -> None:
        pass

    def eval_batch(self, x: TensorDict, y: TensorDict | None) -> None:
        if self.feature_key not in x.keys() or self.reconstructed_key not in x.keys():
            return

        original = cast(torch.Tensor, x[self.feature_key]).detach().cpu()
        reconstructed = cast(torch.Tensor, x[self.reconstructed_key]).detach().cpu()
        payload: dict[str, torch.Tensor] = {
            "feature": original,
            "reconstructed": reconstructed,
        }
        if self.mask_key is not None and self.mask_key in x.keys():
            payload["mask"] = cast(torch.Tensor, x[self.mask_key]).detach().cpu()
        for label_key in self.label_keys:
            if y is not None and label_key in y.keys():
                payload[f"label__{label_key}"] = (
                    cast(torch.Tensor, y[label_key]).detach().cpu().reshape(-1).float()
                )
            else:
                payload[f"label__{label_key}"] = torch.full((original.shape[0],), float("nan"))

        self._storage.add_batch(TensorDict(payload, batch_size=[original.shape[0]]))  # ty: ignore[invalid-argument-type]

    def eval_end(self) -> None:
        pass

    def results(self) -> dict[str, float]:
        data = self._storage.get()
        if data is None:
            return {}
        orig, recon, _, _ = self._prepared_samples(data)
        if orig is None or recon is None:
            return {}
        mse = float(((orig - recon) ** 2).mean())
        return {"reconstruction_mse_sample": mse}

    def visualize(self, logger_obj: Any) -> None:  # ty: ignore[invalid-method-override]
        data = self._storage.get()
        if data is None:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            original, reconstructed, errors, masked = self._prepared_samples(data)
            if original is None or reconstructed is None or errors is None:
                return
            n = len(original)
            fig, axes = plt.subplots(3, n, figsize=(max(10, n * 2.4), 7.2))
            if n == 1:
                axes = np.asarray(axes).reshape(3, 1)

            for idx in range(n):
                orig = self._to_display_image(original[idx]).numpy()
                recon = self._to_display_image(reconstructed[idx]).numpy()
                err = self._to_display_image(errors[idx]).numpy()
                vmin = float(min(orig.min(), recon.min()))
                vmax = float(max(orig.max(), recon.max()))
                axes[0, idx].imshow(
                    orig,
                    aspect="auto",
                    origin="lower",
                    vmin=vmin,
                    vmax=vmax,
                    cmap="magma",
                    interpolation="nearest",
                )
                axes[1, idx].imshow(
                    recon,
                    aspect="auto",
                    origin="lower",
                    vmin=vmin,
                    vmax=vmax,
                    cmap="magma",
                    interpolation="nearest",
                )
                axes[2, idx].imshow(
                    err,
                    aspect="auto",
                    origin="lower",
                    cmap="viridis",
                    interpolation="nearest",
                )
                for row in range(3):
                    axes[row, idx].set_xticks([])
                    axes[row, idx].set_yticks([])
                sample_title = self._sample_title(data, idx)
                if sample_title is not None:
                    axes[0, idx].set_title(sample_title, fontsize=9, pad=8)

            axes[0, 0].set_ylabel("Original")
            axes[1, 0].set_ylabel("Reconstructed")
            axes[2, 0].set_ylabel("|Error|")
            fig.suptitle("Reconstruction Overview", y=0.98)
            fig.subplots_adjust(top=0.86, bottom=0.06, wspace=0.08, hspace=0.08)
            log_figure(logger_obj, "eval/reconstruction/overview", fig)
            plt.close(fig)

            if masked:
                masked_fig, masked_axes = plt.subplots(1, n, figsize=(max(10, n * 2.4), 3.3))
                if n == 1:
                    masked_axes = [masked_axes]
                for idx in range(n):
                    if idx >= len(masked):
                        masked_axes[idx].axis("off")
                        continue
                    masked_sample = self._to_display_image(masked[idx]).numpy()
                    masked_axes[idx].imshow(
                        masked_sample,
                        aspect="auto",
                        origin="lower",
                        cmap="magma",
                        interpolation="nearest",
                    )
                    masked_axes[idx].set_xticks([])
                    masked_axes[idx].set_yticks([])
                    sample_title = self._sample_title(data, idx)
                    if sample_title is not None:
                        masked_axes[idx].set_title(sample_title, fontsize=9, pad=8)
                masked_fig.suptitle("Masked Reconstruction Input", y=0.98)
                masked_fig.subplots_adjust(top=0.78, bottom=0.08, wspace=0.08)
                log_figure(logger_obj, "eval/reconstruction/masked_input", masked_fig)
                plt.close(masked_fig)
        except Exception as e:
            logger.warning(f"ReconstructionVisualizer.visualize() failed: {e}")

    def _align_for_plot(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if original.ndim == 3 and reconstructed.ndim == 3:
            unpatchified = self._unpatchify(reconstructed, original)
            if unpatchified is not None:
                return unpatchified
        if original.ndim == 4 and reconstructed.ndim == 3:
            unpatchified = self._unpatchify(reconstructed, original)
            if unpatchified is not None:
                cropped_original, reconstructed_full = unpatchified
                return cropped_original, reconstructed_full
        if original.ndim == 4 and reconstructed.ndim == 3:
            if original.shape[0] == reconstructed.shape[0] and original.shape[1] == 1:
                if original.shape[2:] == reconstructed.shape[1:]:
                    return original, reconstructed.unsqueeze(1)
        if original.ndim == 3 and reconstructed.ndim == 4:
            if original.shape[0] == reconstructed.shape[0] and reconstructed.shape[1] == 1:
                if original.shape[1:] == reconstructed.shape[2:]:
                    return original.unsqueeze(1), reconstructed
        if original.ndim == 4 and reconstructed.ndim == 4:
            if original.shape[0] == reconstructed.shape[0]:
                if original.shape[1] == 1 and original.shape[2:] == reconstructed.shape[1:3]:
                    return original, reconstructed.permute(0, 3, 1, 2)
                if reconstructed.shape[1] == 1 and original.shape[1:3] == reconstructed.shape[2:]:
                    return original.permute(0, 3, 1, 2), reconstructed
        reshaped = self._reshape_to_match(original, reconstructed)
        if reshaped is not None:
            return reshaped
        if original.shape != reconstructed.shape:
            min_len = min(
                original.reshape(original.shape[0], -1).shape[1],
                reconstructed.reshape(reconstructed.shape[0], -1).shape[1],
            )
            original = original.reshape(original.shape[0], -1)[:, :min_len]
            reconstructed = reconstructed.reshape(reconstructed.shape[0], -1)[:, :min_len]
        return original, reconstructed

    def _reshape_to_match(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if original.shape == reconstructed.shape:
            return original, reconstructed
        if original.shape[0] != reconstructed.shape[0]:
            return None
        if original[0].numel() == reconstructed[0].numel():
            return original, reconstructed.reshape(original.shape)
        return None

    def _unpatchify(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if original.ndim == 3:
            original = original.unsqueeze(1)
        if reconstructed.ndim == 2:
            reconstructed = reconstructed.unsqueeze(0)

        if original.ndim != 4 or reconstructed.ndim != 3:
            return None

        patch_size = self.patch_size
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if patch_size is None:
            channels = original.shape[1]
            patch_area = reconstructed.shape[-1] // max(channels, 1)
            side = int(round(patch_area**0.5))
            if side <= 0 or side * side * channels != reconstructed.shape[-1]:
                return None
            patch_size = (side, side)
        self._resolved_patch_size = patch_size

        patch_h, patch_w = patch_size
        batch, channels, height, width = original.shape
        n_patches_h, n_patches_w = self._resolve_patch_grid(
            height=height,
            width=width,
            patch_h=patch_h,
            patch_w=patch_w,
            expected_tokens=reconstructed.shape[1],
        )
        expected_tokens = n_patches_h * n_patches_w
        if expected_tokens == 0 or reconstructed.shape[1] != expected_tokens:
            return None

        padded = self._pad_to_patch_grid(original, n_patches_h, n_patches_w, patch_h, patch_w)
        restored = (
            reconstructed.reshape(batch, n_patches_h, n_patches_w, channels, patch_h, patch_w)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch, channels, n_patches_h * patch_h, n_patches_w * patch_w)
        )
        return padded, restored

    def _apply_mask(self, original: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
        if original.ndim != 3 or mask.ndim != 1:
            return None
        channels, height, width = original.shape
        patch_size = self.patch_size
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if patch_size is None:
            patch_size = self._resolved_patch_size
        if patch_size is None:
            return None
        patch_h, patch_w = patch_size
        n_patches_h = height // patch_h
        n_patches_w = width // patch_w
        if mask.numel() not in {n_patches_h * n_patches_w, n_patches_h * n_patches_w + 1}:
            return None
        patch_mask = mask[-(n_patches_h * n_patches_w) :].reshape(n_patches_h, n_patches_w)
        expanded = patch_mask.repeat_interleave(patch_h, dim=0).repeat_interleave(patch_w, dim=1)
        expanded = expanded[:height, :width].unsqueeze(0).expand(channels, -1, -1)
        return original * expanded.float()

    def _prepared_samples(
        self,
        data: TensorDict,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, list[torch.Tensor]]:
        original = cast(torch.Tensor, data["feature"]).detach().cpu()
        reconstructed = cast(torch.Tensor, data["reconstructed"]).detach().cpu()
        original, reconstructed = self._align_for_plot(original, reconstructed)
        errors = (reconstructed - original).abs()
        masked_samples: list[torch.Tensor] = []
        if "mask" in data.keys():
            masks = cast(torch.Tensor, data["mask"]).detach().cpu()
            for idx in range(original.shape[0]):
                masked = self._apply_mask(original[idx], masks[idx])
                if masked is not None:
                    masked_samples.append(masked)
        return original, reconstructed, errors, masked_samples

    def _to_display_image(self, sample: torch.Tensor) -> torch.Tensor:
        sample = sample.detach().cpu().squeeze()
        if sample.ndim == 0:
            sample = sample.unsqueeze(0).unsqueeze(0)
        elif sample.ndim == 1:
            sample = sample.unsqueeze(0)
        elif sample.ndim == 3:
            sample = sample.permute(1, 0, 2).reshape(sample.shape[1], -1)
        elif sample.ndim > 3:
            sample = sample.reshape(sample.shape[-2], -1)
        return sample

    def _sample_title(self, data: TensorDict, idx: int) -> str | None:
        parts: list[str] = []
        for label_key in self.label_keys:
            td_key = f"label__{label_key}"
            if td_key not in data.keys():
                continue
            value = float(cast(torch.Tensor, data[td_key])[idx].item())
            if not np.isfinite(value):
                continue
            parts.append(f"{label_key}={format_label(value)}")
        if not parts:
            return None
        return " | ".join(parts)

    def _resolve_patch_grid(
        self,
        *,
        height: int,
        width: int,
        patch_h: int,
        patch_w: int,
        expected_tokens: int,
    ) -> tuple[int, int]:
        floor_h = height // patch_h
        floor_w = width // patch_w
        ceil_h = math.ceil(height / patch_h)
        ceil_w = math.ceil(width / patch_w)
        candidates = [
            (ceil_h, ceil_w),
            (floor_h, floor_w),
        ]
        for n_patches_h, n_patches_w in candidates:
            if n_patches_h > 0 and n_patches_w > 0 and n_patches_h * n_patches_w == expected_tokens:
                return n_patches_h, n_patches_w
        if ceil_h > 0 and expected_tokens % ceil_h == 0:
            return ceil_h, expected_tokens // ceil_h
        if floor_h > 0 and expected_tokens % floor_h == 0:
            return floor_h, expected_tokens // floor_h
        return floor_h, floor_w

    def _pad_to_patch_grid(
        self,
        original: torch.Tensor,
        n_patches_h: int,
        n_patches_w: int,
        patch_h: int,
        patch_w: int,
    ) -> torch.Tensor:
        target_height = n_patches_h * patch_h
        target_width = n_patches_w * patch_w
        pad_h = max(0, target_height - original.shape[-2])
        pad_w = max(0, target_width - original.shape[-1])
        if pad_h == 0 and pad_w == 0:
            return original[:, :, :target_height, :target_width]
        padded = F.pad(original, (0, pad_w, 0, pad_h))
        return padded[:, :, :target_height, :target_width]
