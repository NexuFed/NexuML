"""Utilities for evaluation algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import random
import tempfile
import uuid

import numpy as np

import torch


class ReservoirSampler:
    """Online reservoir sampling (Vitter's Algorithm R).

    Maintains a fixed-size random sample of items seen so far,
    suitable for streaming large datasets without storing everything.
    """

    def __init__(self, max_samples: int) -> None:
        self.max_samples = max_samples
        self._buffer: list[torch.Tensor] = []
        self._n_seen: int = 0

    def add(self, items: torch.Tensor) -> None:
        """Add a batch of items (first dim = batch size)."""
        for item in items:
            item = item.detach().cpu()
            self._n_seen += 1
            if len(self._buffer) < self.max_samples:
                self._buffer.append(item)
            else:
                j = random.randint(0, self._n_seen - 1)
                if j < self.max_samples:
                    self._buffer[j] = item

    def get(self) -> torch.Tensor | None:
        """Return sampled items as a tensor, or None if empty."""
        if not self._buffer:
            return None
        return torch.stack(self._buffer)

    @property
    def n_seen(self) -> int:
        return self._n_seen

    @property
    def n_sampled(self) -> int:
        return len(self._buffer)


class FeatureStore(ABC):
    """Collect fit features for batch-fit distance estimators."""

    @abstractmethod
    def append(self, features: torch.Tensor) -> None:
        """Append a feature batch."""

    @abstractmethod
    def finalize(self) -> None:
        """Finalize storage for reading."""

    @abstractmethod
    def as_array(self) -> torch.Tensor | np.ndarray | None:
        """Return stored features as a CPU-compatible matrix."""

    @abstractmethod
    def cleanup(self) -> None:
        """Close handles and remove temporary files when not retained."""


class RAMFeatureStore(FeatureStore):
    """In-memory feature storage with optional reservoir sampling."""

    def __init__(self, max_samples: int | None = None) -> None:
        self.max_samples = max_samples
        self._sampler = ReservoirSampler(max_samples) if max_samples is not None else None
        self._chunks: list[torch.Tensor] = []
        self._data: torch.Tensor | None = None

    def append(self, features: torch.Tensor) -> None:
        features = features.detach().float().flatten(1).cpu()
        if self._sampler is not None:
            self._sampler.add(features)
        else:
            self._chunks.append(features)

    def finalize(self) -> None:
        if self._sampler is not None:
            self._data = self._sampler.get()
        elif self._chunks:
            self._data = torch.cat(self._chunks, dim=0)
        else:
            self._data = None

    def as_array(self) -> torch.Tensor | None:
        if self._data is None:
            self.finalize()
        return self._data

    def cleanup(self) -> None:
        self._chunks.clear()
        self._data = None


class MemmapFeatureStore(FeatureStore):
    """Disk-backed feature storage that streams directly to a numpy memmap.

    Features are written to disk during ``append()`` without buffering in RAM.
    The memmap is pre-allocated to ``(max_samples, D)`` when ``max_samples`` is
    provided; otherwise capacity starts at 1024 rows and doubles as needed.
    ``finalize()`` trims the file to the actual row count written.
    """

    _INITIAL_CAPACITY = 1024

    def __init__(
        self,
        max_samples: int | None = None,
        storage_path: str | Path | None = None,
        retain_storage: bool = False,
    ) -> None:
        self.max_samples = max_samples
        self.storage_path = (
            Path(storage_path)
            if storage_path is not None
            else Path(tempfile.mkdtemp(prefix="nexuml-features-"))
        )
        self.retain_storage = retain_storage or storage_path is not None
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.filename = self.storage_path / f"features-{uuid.uuid4().hex}.dat"
        self._memmap: np.memmap | None = None
        self._capacity: int = 0
        self._n_written: int = 0
        self._dim: int | None = None

    def _init_memmap(self, dim: int, dtype: np.dtype) -> None:
        self._dim = dim
        self._capacity = (
            self.max_samples if self.max_samples is not None else self._INITIAL_CAPACITY
        )
        self._memmap = np.memmap(self.filename, dtype=dtype, mode="w+", shape=(self._capacity, dim))

    def _grow(self) -> None:
        assert self._memmap is not None and self._dim is not None
        new_capacity = self._capacity * 2
        old_data = np.array(self._memmap[: self._n_written])
        dtype = self._memmap.dtype
        del self._memmap
        self._memmap = np.memmap(
            self.filename, dtype=dtype, mode="w+", shape=(new_capacity, self._dim)
        )
        self._memmap[: self._n_written] = old_data
        self._capacity = new_capacity

    def append(self, features: torch.Tensor) -> None:
        arr = features.detach().float().flatten(1).cpu().numpy()
        n, dim = arr.shape
        if self._memmap is None:
            self._init_memmap(dim, arr.dtype)
        end = self._n_written + n
        while end > self._capacity:
            self._grow()
        memmap = self._memmap
        assert memmap is not None
        memmap[self._n_written : end] = arr
        self._n_written = end

    def finalize(self) -> None:
        if self._memmap is None:
            return
        dtype = self._memmap.dtype
        dim = self._dim
        assert dim is not None
        trimmed = np.array(self._memmap[: self._n_written])
        del self._memmap
        self._memmap = np.memmap(
            self.filename, dtype=dtype, mode="w+", shape=(self._n_written, dim)
        )
        self._memmap[:] = trimmed
        self._memmap.flush()
        self._capacity = self._n_written

    def as_array(self) -> np.ndarray | None:
        if self._memmap is None:
            return None
        if self._capacity != self._n_written:
            self.finalize()
        return self._memmap

    def cleanup(self) -> None:
        if self._memmap is not None:
            self._memmap.flush()
            self._memmap = None
        if not self.retain_storage and self.filename.exists():
            self.filename.unlink(missing_ok=True)


def create_feature_store(
    backend: str = "ram",
    *,
    max_samples: int | None = None,
    storage_path: str | Path | None = None,
    retain_storage: bool = False,
) -> FeatureStore:
    """Create a feature store backend.

    Returns:
        The requested ``FeatureStore`` instance.

    Raises:
        ValueError: If *backend* is not ``"ram"`` or ``"memmap"``.
    """
    if backend == "ram":
        return RAMFeatureStore(max_samples=max_samples)
    if backend == "memmap":
        return MemmapFeatureStore(
            max_samples=max_samples,
            storage_path=storage_path,
            retain_storage=retain_storage,
        )
    raise ValueError(f"Unknown feature storage backend: {backend}")
