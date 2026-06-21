"""Torch-file export backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from nexuml.data.export.backend import ExportBackend, register_export_backend


def _sample_path(root: Path, index: int) -> Path:
    return root / "data" / f"{index:08d}.pt"


@register_export_backend("torch")
class TorchBackend(ExportBackend):
    """Store one `.pt` file per sample containing all exported keys."""

    def __init__(self, **_kwargs) -> None:
        self._export_dir: Path | None = None
        self._feature_shapes: dict[str, tuple[int, ...]] = {}
        self._dtype: torch.dtype | None = None
        self._num_samples: int = 0
        self._saved: int = 0

    def initialize(
        self,
        export_dir: Path,
        num_samples: int,
        feature_shapes: dict[str, tuple[int, ...]],
        dtype: torch.dtype | str | None = None,
    ) -> None:
        self._export_dir = export_dir
        self._feature_shapes = dict(feature_shapes)
        self._num_samples = int(num_samples)
        (export_dir / "data").mkdir(parents=True, exist_ok=True)
        if dtype is None:
            self._dtype = None
        else:
            self._dtype = getattr(torch, str(dtype)) if isinstance(dtype, str) else dtype
        self._saved = 0

    def save_sample(self, index: int, features: dict[str, torch.Tensor]) -> None:
        if self._export_dir is None:
            raise RuntimeError("Backend has not been initialized")
        if not 0 <= index < self._num_samples:
            raise IndexError(f"Sample index {index} out of bounds for {self._num_samples} samples")

        payload: dict[str, torch.Tensor] = {}
        for key, expected_shape in self._feature_shapes.items():
            if key not in features:
                raise KeyError(f"Missing feature key for export: {key}")
            tensor = features[key].detach().cpu()
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"Feature '{key}' has shape {tuple(tensor.shape)}, expected {expected_shape}"
                )
            if self._dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(self._dtype)
            payload[key] = tensor

        torch.save(payload, _sample_path(self._export_dir, index))
        self._saved += 1

    def finalize(self) -> dict[str, Any]:
        dtype_name = None if self._dtype is None else str(self._dtype).split(".")[-1]
        return {
            "format": "torch",
            "dtype": dtype_name,
            "samples_saved": self._saved,
            "key_specs": {
                key: {
                    "encoding": "pt",
                    "storage": {
                        "type": "file",
                        "path": "data",
                        "pattern": "{index:08d}.pt",
                    },
                }
                for key in self._feature_shapes
            },
        }

    @staticmethod
    def load_sample(export_dir: Path, index: int) -> dict[str, torch.Tensor]:
        payload = torch.load(
            _sample_path(export_dir, index), map_location="cpu", weights_only=False
        )
        return {key: value.detach().cpu() for key, value in payload.items()}
