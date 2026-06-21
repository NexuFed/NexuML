"""File-per-sample NumPy export backend."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nexuml.data.export.backend import ExportBackend, register_export_backend

logger = logging.getLogger(__name__)


def _sample_path(root: Path, key: str, index: int) -> Path:
    return root / "data" / key / f"{index:08d}.npy"


@register_export_backend("numpy")
class NumpyBackend(ExportBackend):
    """Store each key as one `.npy` file per sample."""

    def __init__(self, **_kwargs) -> None:
        self._export_dir: Path | None = None
        self._feature_shapes: dict[str, tuple[int, ...]] = {}
        self._dtype: np.dtype[Any] = np.dtype(np.float32)
        self._num_samples: int = 0
        self._saved: int = 0

    def initialize(
        self,
        export_dir: Path,
        num_samples: int,
        feature_shapes: dict[str, tuple[int, ...]],
        dtype: np.dtype[Any] | str | None = None,
    ) -> None:
        self._export_dir = export_dir
        self._feature_shapes = dict(feature_shapes)
        self._dtype = np.dtype(np.float32 if dtype is None else dtype)
        self._num_samples = int(num_samples)
        for key in feature_shapes:
            (export_dir / "data" / key).mkdir(parents=True, exist_ok=True)
        self._saved = 0

    def save_sample(self, index: int, features: dict[str, torch.Tensor]) -> None:
        if self._export_dir is None:
            raise RuntimeError("Backend has not been initialized")
        if not 0 <= index < self._num_samples:
            raise IndexError(f"Sample index {index} out of bounds for {self._num_samples} samples")

        expected_keys = set(self._feature_shapes)
        actual_keys = set(features)
        missing = expected_keys - actual_keys
        if missing:
            raise KeyError(f"Missing feature keys for export: {sorted(missing)}")
        unexpected = actual_keys - expected_keys
        if unexpected:
            raise KeyError(f"Unexpected feature keys for export: {sorted(unexpected)}")

        for key, tensor in features.items():
            array = tensor.detach().cpu().numpy().astype(self._dtype, copy=False)
            if array.shape != self._feature_shapes[key]:
                raise ValueError(
                    f"Feature '{key}' has shape {array.shape}, expected {self._feature_shapes[key]}"
                )
            np.save(_sample_path(self._export_dir, key, index), array, allow_pickle=False)
        self._saved += 1

    def finalize(self) -> dict[str, Any]:
        return {
            "format": "numpy",
            "dtype": self._dtype.name,
            "samples_saved": self._saved,
            "key_specs": {
                key: {
                    "encoding": "npy",
                    "storage": {
                        "type": "directory",
                        "path": f"data/{key}",
                        "pattern": "{index:08d}.npy",
                    },
                }
                for key in self._feature_shapes
            },
        }

    @staticmethod
    def load_sample(export_dir: Path, index: int) -> dict[str, torch.Tensor]:
        data_dir = export_dir / "data"
        result: dict[str, torch.Tensor] = {}
        for key_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
            sample_path = _sample_path(export_dir, key_dir.name, index)
            if not sample_path.exists():
                continue
            result[key_dir.name] = torch.from_numpy(np.load(sample_path, allow_pickle=False))
        return result
