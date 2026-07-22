"""Loader backend protocol and registry."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from torch.utils.data import WeightedRandomSampler

logger = logging.getLogger(__name__)


class LoaderBackend(Protocol):
    """Backend interface for creating split-specific loaders."""

    def create_loader(
        self,
        module: Any,
        dataset: Any,
        *,
        split: str,
        shuffle: bool = False,
        sampler: WeightedRandomSampler | None = None,
    ) -> Any: ...


_LOADER_BACKENDS: dict[str, LoaderBackend] = {}


def register_loader_backend(name: str, backend: LoaderBackend) -> None:
    """Register a dataloader backend implementation."""
    _LOADER_BACKENDS[name] = backend


def get_loader_backend(name: str) -> LoaderBackend:
    """Retrieve a registered loader backend by name.

    Returns:
        The loader backend registered under ``name``.

    Raises:
        KeyError: If no backend is registered under that name.
    """
    _ensure_default_backends()
    if name not in _LOADER_BACKENDS:
        available = ", ".join(sorted(_LOADER_BACKENDS))
        raise KeyError(f"Loader backend '{name}' is not registered. Available: [{available}]")
    return _LOADER_BACKENDS[name]


def list_loader_backends() -> list[str]:
    """Return names of all registered loader backends."""
    _ensure_default_backends()
    return sorted(_LOADER_BACKENDS.keys())


def _ensure_default_backends() -> None:
    """Lazily register built-in backends without overwriting existing entries."""
    if "torch" not in _LOADER_BACKENDS:
        from nexuml.data.loaders.torch_backend import TorchLoaderBackend

        register_loader_backend("torch", TorchLoaderBackend())

    if "dali" not in _LOADER_BACKENDS:
        try:
            from nexuml.data.loaders.dali_backend import DaliLoaderBackend

            register_loader_backend("dali", DaliLoaderBackend())
        except ImportError:
            logger.debug("DALI not available; 'dali' backend not registered")

    if "tensor_shards" not in _LOADER_BACKENDS:
        from nexuml.data.loaders.tensor_shards_backend import (
            TensorShardsLoaderBackend,
        )

        register_loader_backend(
            "tensor_shards",
            TensorShardsLoaderBackend(),
        )
