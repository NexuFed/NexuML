"""Temporary storage helpers for evaluation algorithms."""

from __future__ import annotations

import logging
import importlib
import random
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import cast

from tensordict import TensorDict

logger = logging.getLogger(__name__)


def _normalize_tensordict(data: TensorDict) -> TensorDict:
    """Move a TensorDict to CPU and ensure it has a leading batch dimension.

    Returns:
        The normalized TensorDict.
    """
    td = cast(TensorDict, data.apply(lambda tensor: tensor.detach().cpu()))
    if len(td.batch_size) == 0:
        td = td.unsqueeze(0)
    return td


class _BaseTensorDictStorage(ABC):
    """Backend interface for temporary TensorDict storage."""

    def __init__(self, max_samples: int, storage_path: str | Path | None = None) -> None:
        self.max_samples = max_samples
        self.storage_path = Path(storage_path) if storage_path is not None else None
        self._size = 0

    @abstractmethod
    def set(self, index: int, item: TensorDict) -> None:
        """Store a single TensorDict row at the given index."""

    @abstractmethod
    def get(self) -> TensorDict | None:
        """Return the stored TensorDict rows."""

    def finalize(self) -> None:
        """Finalize storage if the backend requires it."""

    def __len__(self) -> int:
        return self._size


class _MemoryTensorDictStorage(_BaseTensorDictStorage):
    """In-memory list-backed TensorDict storage."""

    def __init__(self, max_samples: int, storage_path: str | Path | None = None) -> None:
        super().__init__(max_samples=max_samples, storage_path=storage_path)
        self._items: list[TensorDict] = []

    def set(self, index: int, item: TensorDict) -> None:
        row = _normalize_tensordict(item)
        if row.batch_size[0] != 1:
            raise ValueError("Temporary storage expects single-row TensorDict items.")
        if index == len(self._items):
            self._items.append(row)
        elif 0 <= index < len(self._items):
            self._items[index] = row
        else:
            raise IndexError(f"Storage index {index} out of range for size {len(self._items)}.")
        self._size = len(self._items)

    def get(self) -> TensorDict | None:
        if not self._items:
            return None
        return TensorDict.cat(self._items, dim=0)


class _MemmapTensorDictStorage(_BaseTensorDictStorage):
    """Lazy memmap-backed TensorDict storage."""

    def __init__(self, max_samples: int, storage_path: str | Path | None = None) -> None:
        super().__init__(max_samples=max_samples, storage_path=storage_path)
        LazyMemmapStorage = importlib.import_module("torchrl.data").LazyMemmapStorage

        base_dir = (
            self.storage_path
            if self.storage_path is not None
            else Path(tempfile.mkdtemp(prefix="nexuml_eval_")) / "storage"
        )
        base_dir.mkdir(parents=True, exist_ok=True)
        self.storage_path = base_dir
        self._storage = LazyMemmapStorage(
            max_size=max_samples,
            scratch_dir=base_dir,
            existsok=True,
        )

    def set(self, index: int, item: TensorDict) -> None:
        row = _normalize_tensordict(item)
        if row.batch_size[0] != 1:
            raise ValueError("Temporary storage expects single-row TensorDict items.")
        self._storage.set(range(index, index + 1), row)
        self._size = max(self._size, index + 1)

    def get(self) -> TensorDict | None:
        if self._size == 0:
            return None
        return self._storage[: self._size]

    def finalize(self) -> None:
        dump = getattr(self._storage, "dump", None)
        if callable(dump) and self.storage_path is not None:
            dump(self.storage_path)


def create_temporary_storage(
    backend: str = "memory",
    max_samples: int = 2000,
    storage_path: str | Path | None = None,
) -> _BaseTensorDictStorage:
    """Create a temporary TensorDict storage backend.

    Returns:
        The requested ``_BaseTensorDictStorage`` instance.

    Raises:
        ValueError: If *backend* is not ``"memory"`` or ``"memmap"``.
    """
    if backend == "memory":
        return _MemoryTensorDictStorage(max_samples=max_samples, storage_path=storage_path)
    if backend == "memmap":
        return _MemmapTensorDictStorage(max_samples=max_samples, storage_path=storage_path)
    raise ValueError(f"Unknown temporary storage backend '{backend}'.")


class AppendableTensorDictBuffer:
    """Append TensorDict batches into temporary storage up to a fixed capacity."""

    def __init__(
        self,
        max_samples: int,
        storage_backend: str = "memory",
        storage_path: str | Path | None = None,
    ) -> None:
        self.max_samples = max_samples
        self._storage = create_temporary_storage(
            backend=storage_backend,
            max_samples=max_samples,
            storage_path=storage_path,
        )
        self.truncated = False

    def add_batch(self, data: TensorDict) -> None:
        td = _normalize_tensordict(data)
        for idx in range(td.batch_size[0]):
            if len(self._storage) >= self.max_samples:
                self.truncated = True
                return
            row = td[idx]
            assert isinstance(row, TensorDict)
            self._storage.set(len(self._storage), row)

    def get(self) -> TensorDict | None:
        return self._storage.get()

    def finalize(self) -> None:
        self._storage.finalize()

    def __len__(self) -> int:
        return len(self._storage)


class ReservoirTensorDictBuffer:
    """Reservoir-sample TensorDict rows into temporary storage."""

    def __init__(
        self,
        max_samples: int,
        storage_backend: str = "memory",
        storage_path: str | Path | None = None,
    ) -> None:
        self.max_samples = max_samples
        self._storage = create_temporary_storage(
            backend=storage_backend,
            max_samples=max_samples,
            storage_path=storage_path,
        )
        self.n_seen = 0

    def add_batch(self, data: TensorDict) -> None:
        td = _normalize_tensordict(data)
        for idx in range(td.batch_size[0]):
            self.n_seen += 1
            row = td[idx]
            assert isinstance(row, TensorDict)
            if len(self._storage) < self.max_samples:
                self._storage.set(len(self._storage), row)
                continue

            position = random.randint(0, self.n_seen - 1)
            if position < self.max_samples:
                self._storage.set(position, row)

    def get(self) -> TensorDict | None:
        return self._storage.get()

    def finalize(self) -> None:
        self._storage.finalize()

    def __len__(self) -> int:
        return len(self._storage)
