"""Dataset registry with automatic discovery."""

from __future__ import annotations

import builtins
import logging
from typing import Any

from nexuml.core.discovery import (
    DiscoveryError,
    Scanner,
    discover_library_packages,
    register_items,
)
from nexuml.data.dataset import NexuDataset

logger = logging.getLogger(__name__)


class DatasetRegistry:
    """Registry for NexuDataset subclasses with automatic discovery.

    Also consumes decorated discovery results from the Scanner.
    """

    def __init__(self) -> None:
        self._registry: dict[str, type] = {}
        self._errors: builtins.list[DiscoveryError] = []
        self._loaded = False

    @property
    def errors(self) -> builtins.list[DiscoveryError]:
        """Discovery failures from the last scan (import + registration)."""
        self.ensure_loaded()
        return list(self._errors)

    def register(self, type_key: str, cls: type) -> None:
        if type_key in self._registry:
            existing = self._registry[type_key]
            if existing is not cls:
                raise ValueError(
                    f"Registry conflict: '{type_key}' already registered to "
                    f"{existing.__module__}.{existing.__name__}, "
                    f"cannot register {cls.__module__}.{cls.__name__}"
                )
        self._registry[type_key] = cls

    def get(self, type_key: str) -> type:
        self.ensure_loaded()
        if type_key not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"Dataset type '{type_key}' not found in registry. Available: [{available}]"
            )
        return self._registry[type_key]

    def list(self) -> dict[str, type]:
        self.ensure_loaded()
        return dict(self._registry)

    def instantiate(self, type_key: str, **params: Any) -> NexuDataset:
        cls = self.get(type_key)
        return cls(**params)

    def scan(self, package_paths: builtins.list[str] | None = None) -> None:
        """Scan packages for NexuDataset subclasses and register them.

        Also consumes decorated discovery results from the Scanner.
        """
        if package_paths is None:
            package_paths = discover_library_packages()

        # Consume decorated discovery results. Import/registration failures are
        # collected (not raised) so one broken module cannot hide every dataset.
        scanner = Scanner()
        for package_path in package_paths:
            scanner.scan_package(package_path)

        self._errors = list(scanner.errors)
        register_items(scanner.by_kind("data_source"), self.register, self._errors)

        self._loaded = True
        logger.info(
            "Dataset registry loaded: %d dataset(s)%s - %s",
            len(self._registry),
            f", {len(self._errors)} error(s)" if self._errors else "",
            ", ".join(sorted(self._registry.keys())),
        )

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.scan()


_default_registry: DatasetRegistry | None = None


def get_dataset_registry() -> DatasetRegistry:
    """Return the default global DatasetRegistry, creating it if needed."""
    global _default_registry
    if _default_registry is None:
        _default_registry = DatasetRegistry()
    return _default_registry
