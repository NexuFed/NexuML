"""Loader backend protocol and typed materialization helpers."""

from __future__ import annotations

from typing import Any, Protocol

from torch.utils.data import WeightedRandomSampler

from nexuml.core.components import LoaderBackendDefinition
from nexuml.core.registry import get_component_registry


class LoaderBackend(Protocol):
    """Runtime protocol implemented by loader backends."""

    def create_loader(
        self,
        module: Any,
        dataset: Any,
        *,
        split: str,
        shuffle: bool = False,
        sampler: WeightedRandomSampler | None = None,
    ) -> Any: ...


def get_loader_backend(definition: LoaderBackendDefinition) -> LoaderBackend:
    """Materialize a loader backend definition.

    Returns:
        Runtime loader backend.
    """
    return definition.build()


def list_loader_backends() -> list[str]:
    """List registered loader backend identities.

    Returns:
        Stable loader backend names.
    """
    return [entry.name for entry in get_component_registry().entries(kind="loader_backend")]
