"""Export backend abstraction for SuperDataset data serialization."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Schema for config.yaml written alongside exported data."""

    format_version: int
    backend: str
    writer: str
    num_samples: int
    label_names: list[str]
    num_classes: dict[str, int]
    modality: str
    x_keys: list[str]
    y_keys: list[str]
    label_prefix: str
    feature_shapes: dict[str, list[int]]
    key_specs: dict[str, dict[str, Any]] = field(default_factory=dict)
    source_datasets: list[str] = field(default_factory=list)
    merge_labels_config: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class ExportBackend(ABC):
    """Abstract base class for dataset export backends.

    Each backend handles serialization of feature tensors to a specific
    storage format. The metadata (labels, splits, source info) is always
    stored as parquet alongside the backend-specific data directory.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Backends accept format-specific keyword arguments."""

    @abstractmethod
    def initialize(
        self,
        export_dir: Path,
        num_samples: int,
        feature_shapes: dict[str, tuple[int, ...]],
        dtype: Any | None = None,
    ) -> None:
        """Prepare storage (create files, allocate mmap, etc.)."""

    @abstractmethod
    def save_sample(self, index: int, features: dict[str, torch.Tensor]) -> None:
        """Save one sample's feature tensors."""

    def save_batch(self, start_index: int, features: dict[str, torch.Tensor]) -> None:
        """Save a batch of feature tensors.

        Backends can override this for more efficient contiguous writes.

        Raises:
            ValueError: If batch sizes across feature tensors are inconsistent.
        """
        batch_sizes = {int(tensor.shape[0]) for tensor in features.values()}
        if len(batch_sizes) != 1:
            raise ValueError(f"Export batch size mismatch: {sorted(batch_sizes)}")

        batch_size = next(iter(batch_sizes))
        for offset in range(batch_size):
            sample = {key: tensor[offset] for key, tensor in features.items()}
            self.save_sample(start_index + offset, sample)

    @abstractmethod
    def finalize(self) -> dict[str, Any]:
        """Flush/close files. Return backend-specific metadata for config.yaml."""

    @staticmethod
    @abstractmethod
    def load_sample(export_dir: Path, index: int) -> dict[str, torch.Tensor]:
        """Load one sample's feature tensors from an exported directory."""


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_BACKEND_REGISTRY: dict[str, type[ExportBackend]] = {}


def register_export_backend(name: str):
    """Decorator to register an export backend.

    Returns:
        A decorator that registers the backend class and returns it.
    """

    def decorator(cls: type[ExportBackend]) -> type[ExportBackend]:
        if name in _BACKEND_REGISTRY:
            logger.warning("Overwriting export backend '%s'", name)
        _BACKEND_REGISTRY[name] = cls
        return cls

    return decorator


def get_export_backend(name: str) -> type[ExportBackend]:
    """Retrieve a registered export backend by name.

    Returns:
        The backend class for the given name.

    Raises:
        KeyError: If no backend is registered under that name.
    """
    if name not in _BACKEND_REGISTRY:
        available = ", ".join(sorted(_BACKEND_REGISTRY.keys())) or "(none)"
        raise KeyError(f"Unknown export backend '{name}'. Available: {available}")
    return _BACKEND_REGISTRY[name]


def list_export_backends() -> list[str]:
    """Return names of all registered export backends."""
    return sorted(_BACKEND_REGISTRY.keys())
