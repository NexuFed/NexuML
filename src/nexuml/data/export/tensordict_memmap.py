"""TensorDict memmap export backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from nexuml.data.export.backend import ExportBackend, register_export_backend


@register_export_backend("tensordict_memmap")
class TensorDictMemmapBackend(ExportBackend):
    """Persist the exported dataset as one memmapped TensorDict."""

    def __init__(self, **_kwargs) -> None:
        self._export_dir: Path | None = None
        self._feature_shapes: dict[str, tuple[int, ...]] = {}
        self._dtype: torch.dtype = torch.float32
        self._num_samples: int = 0
        self._saved: int = 0
        self._storage: TensorDict | None = None

    def initialize(
        self,
        export_dir: Path,
        num_samples: int,
        feature_shapes: dict[str, tuple[int, ...]],
        dtype: np.dtype[Any] | str | None = None,
    ) -> None:
        self._export_dir = export_dir
        self._feature_shapes = dict(feature_shapes)
        self._num_samples = int(num_samples)
        if dtype is None:
            self._dtype = torch.float32
        elif isinstance(dtype, str):
            self._dtype = getattr(torch, dtype)
        else:
            np_dtype = np.dtype(dtype)
            self._dtype = torch.from_numpy(np.empty((), dtype=np_dtype)).dtype

        storage = TensorDict(
            {
                key: torch.empty((num_samples, *shape), dtype=self._dtype)
                for key, shape in feature_shapes.items()
            },
            batch_size=[num_samples],
        )
        self._storage = storage.memmap(prefix=str(export_dir / "data"))
        self._saved = 0

    def save_sample(self, index: int, features: dict[str, torch.Tensor]) -> None:
        if self._storage is None:
            raise RuntimeError("Backend has not been initialized")
        if not 0 <= index < self._num_samples:
            raise IndexError(f"Sample index {index} out of bounds for {self._num_samples} samples")

        for key, expected_shape in self._feature_shapes.items():
            if key not in features:
                raise KeyError(f"Missing feature key for export: {key}")
            tensor = features[key].detach().cpu()
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"Feature '{key}' has shape {tuple(tensor.shape)}, expected {expected_shape}"
                )
            self._storage[key][index] = tensor.to(self._dtype)
        self._saved += 1

    def save_batch(self, start_index: int, features: dict[str, torch.Tensor]) -> None:
        if self._storage is None:
            raise RuntimeError("Backend has not been initialized")
        batch_sizes = {int(tensor.shape[0]) for tensor in features.values()}
        if len(batch_sizes) != 1:
            raise ValueError(f"Batch size mismatch for export: {sorted(batch_sizes)}")
        batch_size = next(iter(batch_sizes))
        end_index = start_index + batch_size
        for key, expected_shape in self._feature_shapes.items():
            if key not in features:
                raise KeyError(f"Missing feature key for export: {key}")
            tensor = features[key].detach().cpu()
            if tuple(tensor.shape[1:]) != expected_shape:
                raise ValueError(
                    f"Feature '{key}' batch has shape {tuple(tensor.shape[1:])}, "
                    f"expected {expected_shape}"
                )
            self._storage[key][start_index:end_index] = tensor.to(self._dtype)
        self._saved = max(self._saved, end_index)

    def finalize(self) -> dict[str, Any]:
        return {
            "format": "tensordict_memmap",
            "dtype": str(self._dtype).split(".")[-1],
            "samples_saved": self._saved,
            "key_specs": {
                key: {
                    "encoding": "memmap",
                    "storage": {
                        "type": "tensordict_memmap",
                        "path": "data",
                    },
                }
                for key in self._feature_shapes
            },
        }

    @staticmethod
    def load_sample(export_dir: Path, index: int) -> dict[str, torch.Tensor]:
        storage = TensorDict.load_memmap(export_dir / "data")
        sample: TensorDict = storage[index]  # ty: ignore[invalid-assignment]
        return {str(key): torch.as_tensor(value) for key, value in sample.items()}
