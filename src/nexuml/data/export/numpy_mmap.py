"""NumPy memory-mapped export backend."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nexuml.data.export.backend import ExportBackend, register_export_backend

logger = logging.getLogger(__name__)


@register_export_backend("numpy_mmap")
class NumpyMmapBackend(ExportBackend):
    """Memory-mapped NumPy backend for large-scale datasets.

    Pre-allocates a single `.npy` file per feature key with shape
    `(num_samples, *feature_shape)` for efficient sequential writes and
    memory-mapped reads. Requires all samples to have uniform feature shapes.
    """

    def __init__(self, **_kwargs) -> None:
        self._export_dir: Path | None = None
        self._mmaps: dict[str, np.memmap] = {}
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
        self._num_samples = num_samples
        self._feature_shapes = dict(feature_shapes)
        data_dir = export_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        self._dtype = np.dtype(np.float32 if dtype is None else dtype)

        self._mmaps: dict[str, np.memmap[tuple[Any, ...], np.dtype[Any]]] = {}
        for key, shape in feature_shapes.items():
            full_shape = (num_samples, *shape)
            mmap_path = data_dir / f"{key}.npy"
            self._mmaps[key] = np.lib.format.open_memmap(
                str(mmap_path),
                dtype=self._dtype,
                mode="w+",
                shape=full_shape,
            )

        self._saved = 0
        logger.info(
            "NumpyMmapBackend: initialized %d samples, keys=%s in %s",
            num_samples,
            list(feature_shapes.keys()),
            data_dir,
        )

    def save_sample(self, index: int, features: dict[str, torch.Tensor]) -> None:
        if not 0 <= index < self._num_samples:
            raise IndexError(f"Sample index {index} out of bounds for {self._num_samples} samples")

        expected_keys = set(self._mmaps)
        feature_keys = set(features)

        missing_keys = expected_keys - feature_keys
        if missing_keys:
            raise KeyError(f"Missing feature keys for export: {sorted(missing_keys)}")

        unexpected_keys = feature_keys - expected_keys
        if unexpected_keys:
            raise KeyError(f"Unexpected feature keys for export: {sorted(unexpected_keys)}")

        for key, tensor in features.items():
            expected_shape = self._feature_shapes[key]
            tensor_array = tensor.detach().cpu().numpy()
            if tensor_array.shape != expected_shape:
                raise ValueError(
                    f"Feature '{key}' has shape {tensor_array.shape}, expected {expected_shape}"
                )
            self._mmaps[key][index] = tensor_array
        self._saved += 1

    def save_batch(self, start_index: int, features: dict[str, torch.Tensor]) -> None:
        expected_keys = set(self._mmaps)
        feature_keys = set(features)

        missing_keys = expected_keys - feature_keys
        if missing_keys:
            raise KeyError(f"Missing feature keys for export: {sorted(missing_keys)}")

        unexpected_keys = feature_keys - expected_keys
        if unexpected_keys:
            raise KeyError(f"Unexpected feature keys for export: {sorted(unexpected_keys)}")

        batch_sizes = {int(tensor.shape[0]) for tensor in features.values()}
        if len(batch_sizes) != 1:
            raise ValueError(f"Batch size mismatch for export: {sorted(batch_sizes)}")

        batch_size = next(iter(batch_sizes))
        end_index = start_index + batch_size
        if not 0 <= start_index < self._num_samples or end_index > self._num_samples:
            raise IndexError(
                f"Batch slice [{start_index}:{end_index}) out of bounds "
                f"for {self._num_samples} samples"
            )

        for key, tensor in features.items():
            expected_shape = self._feature_shapes[key]
            tensor_array = tensor.detach().cpu().numpy()
            if tensor_array.shape[1:] != expected_shape:
                raise ValueError(
                    f"Feature '{key}' batch has shape {tensor_array.shape[1:]}, "
                    f"expected {expected_shape}"
                )
            self._mmaps[key][start_index:end_index] = tensor_array

        self._saved = max(self._saved, end_index)

    def finalize(self) -> dict[str, Any]:
        # Flush all mmaps
        for mmap in self._mmaps.values():
            mmap.flush()
        self._mmaps.clear()
        logger.info(
            "NumpyMmapBackend: saved %d / %d samples",
            self._saved,
            self._num_samples,
        )
        return {
            "format": "numpy_mmap",
            "dtype": self._dtype.name,
            "samples_saved": self._saved,
            "key_specs": {
                key: {
                    "encoding": "npy",
                    "storage": {
                        "type": "file",
                        "path": f"data/{key}.npy",
                    },
                }
                for key in self._feature_shapes
            },
        }

    @staticmethod
    def load_sample(export_dir: Path, index: int) -> dict[str, torch.Tensor]:
        data_dir = export_dir / "data"
        result = {}
        for npy_file in data_dir.glob("*.npy"):
            key = npy_file.stem
            mmap = np.load(str(npy_file), mmap_mode="r")
            if not 0 <= index < mmap.shape[0]:
                raise IndexError(
                    f"Sample index {index} out of bounds for '{key}' with {mmap.shape[0]} samples"
                )
            value = mmap[index].copy()
            if isinstance(value, np.ndarray):
                result[key] = torch.from_numpy(value)
            else:
                result[key] = torch.as_tensor(value)
        return result
