"""Scenario registry with decorator-based discovery."""

from __future__ import annotations

import builtins
import logging
from typing import Callable

from nexuml.core.discovery import (
    DiscoveryError,
    Scanner,
    discover_library_packages,
    register_items,
)
from nexuml.core.types import ScenarioSpec

logger = logging.getLogger(__name__)


class ScenarioRegistry:
    """Registry for scenario functions with automatic discovery.

    Supports only explicitly decorated scenario functions.
    """

    def __init__(self) -> None:
        self._registry: dict[str, Callable[[], ScenarioSpec]] = {}
        self._errors: builtins.list[DiscoveryError] = []
        self._loaded = False

    @property
    def errors(self) -> builtins.list[DiscoveryError]:
        """Discovery failures from the last scan (import + registration)."""
        self.ensure_loaded()
        return list(self._errors)

    def register(self, key: str, fn: Callable[[], ScenarioSpec]) -> None:
        if key in self._registry:
            existing = self._registry[key]
            if existing is not fn:
                existing_name = getattr(existing, "__name__", repr(existing))
                fn_name = getattr(fn, "__name__", repr(fn))
                raise ValueError(
                    f"Scenario registry conflict: '{key}' already registered to "
                    f"{getattr(existing, '__module__', '?')}.{existing_name}, "
                    f"cannot register {getattr(fn, '__module__', '?')}.{fn_name}"
                )
        self._registry[key] = fn

    def get(self, key: str) -> Callable[[], ScenarioSpec]:
        self.ensure_loaded()
        if key not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(f"Scenario '{key}' not found in registry. Available: [{available}]")
        return self._registry[key]

    def list(self) -> dict[str, Callable[[], ScenarioSpec]]:
        self.ensure_loaded()
        return dict(self._registry)

    def scan(self, package_paths: builtins.list[str] | None = None) -> None:
        """Scan packages for scenario functions.

        Collects only explicitly decorated scenarios via Scanner.
        """
        if package_paths is None:
            package_paths = discover_library_packages()

        # Decorated discovery. Scanner records per-module import failures instead
        # of raising, so one broken module cannot hide every other scenario.
        scanner = Scanner()
        for package_path in package_paths:
            scanner.scan_package(package_path)

        self._errors = list(scanner.errors)
        register_items(scanner.by_kind("scenario"), self.register, self._errors)

        self._loaded = True
        logger.info(
            "Scenario registry loaded: %d scenario(s)%s - %s",
            len(self._registry),
            f", {len(self._errors)} error(s)" if self._errors else "",
            ", ".join(sorted(self._registry.keys())),
        )

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.scan()


_default_scenario_registry: ScenarioRegistry | None = None


def get_scenario_registry() -> ScenarioRegistry:
    """Return the default global ScenarioRegistry, creating it if needed."""
    global _default_scenario_registry
    if _default_scenario_registry is None:
        _default_scenario_registry = ScenarioRegistry()
    return _default_scenario_registry
