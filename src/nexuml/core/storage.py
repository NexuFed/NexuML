"""Shared storage for inter-layer communication."""

from __future__ import annotations
from pathlib import Path

import torch
from tensordict import MemoryMappedTensor, TensorDict, cat as tensordict_cat


def _concatenate_values(
    existing: torch.Tensor | TensorDict, value: torch.Tensor | TensorDict
) -> torch.Tensor | TensorDict:
    """Concatenate compatible tensors or TensorDict batches.

    Returns:
        Concatenated values of the same type as the two inputs.

    Raises:
        TypeError: If tensor and TensorDict storage types are mixed.
    """
    if isinstance(existing, TensorDict) and isinstance(value, TensorDict):
        return tensordict_cat([existing, value], dim=0)
    if isinstance(existing, torch.Tensor) and isinstance(value, torch.Tensor):
        return torch.cat([existing, value], dim=0)
    raise TypeError("Cannot append a TensorDict to a Tensor or vice versa")


class SharedStorage:
    """Manages shared memory for inter-layer communication in the pipeline."""

    def __init__(
        self,
        memory_mapped: bool = True,
        map_location: Path = Path("/tmp/shared_storage"),
    ):
        self.memory_mapped = memory_mapped
        self.map_location = map_location

        self.data = TensorDict({}, batch_size=[])

    def set(self, key: str, value: torch.Tensor | TensorDict):
        if key in self.data and self.data[key].shape == value.shape:
            self.data[key].copy_(value)  # ty: ignore[invalid-argument-type]
        elif self.memory_mapped:
            memmap_path = self.map_location / key
            memmap_path.parent.mkdir(parents=True, exist_ok=True)
            memmap_path = memmap_path.as_posix()
            if not isinstance(value, TensorDict):
                value = MemoryMappedTensor.from_tensor(
                    value, filename=memmap_path, copy_existing=True, existsok=True
                )
            self.data[key] = value.memmap_(prefix=memmap_path)
        else:
            self.data[key] = value.share_memory_()

    def append(self, key: str, value: torch.Tensor | TensorDict):
        if key not in self.data:
            self.set(key, value)
        elif self.memory_mapped:
            self.set(key, _concatenate_values(self.data[key], value))
        else:
            self.data[key] = _concatenate_values(self.data[key], value)

    def set_ring_buffer(self, key: str, value: torch.Tensor | TensorDict, counter: int):
        B = value.shape[0]
        if key not in self.data:
            raise ValueError(f"Key {key} must be initialized before using ring buffer.")

        data_size = self.data[key].shape[0]
        if counter + B < data_size:
            self.data[key][counter : counter + B].copy_(value)  # ty: ignore[invalid-argument-type]
        else:
            remaining = data_size - counter
            if remaining > 0:
                self.data[key][counter : counter + remaining].copy_(value[:remaining])  # ty: ignore[invalid-argument-type]
            overflow = B - remaining
            if overflow > 0:
                self.data[key][0:overflow].copy_(value[remaining : remaining + overflow])  # ty: ignore[invalid-argument-type]

        return (counter + B) % data_size

    def get(self, key: str):
        return self.data.get(key, None)

    def clear(self, key: str):
        if key in self.data:
            del self.data[key]
