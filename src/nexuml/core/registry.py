"""Layer registry with deterministic dynamic discovery."""

from __future__ import annotations

import builtins
import inspect
import logging
from typing import Any

import torch.nn as nn

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.discovery import (
    DiscoveryError,
    Scanner,
    discover_library_packages,
    register_items,
)

logger = logging.getLogger(__name__)


class LayerRegistry:
    """Registry for pipeline layers with automatic discovery.

    Discovers PipelineLayer subclasses in specified packages and supports
    manual registration. Validates constructor params against signatures.
    Also consumes decorated discovery results.
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
                f"Layer type '{type_key}' not found in registry. Available: [{available}]"
            )
        return self._registry[type_key]

    def list(self) -> dict[str, type]:
        self.ensure_loaded()
        return dict(self._registry)

    def validate_params(self, type_key: str, params: dict[str, Any]) -> dict[str, Any]:
        """Validate and cast params against the constructor signature.

        Returns:
            Validated and type-cast parameter dictionary.

        Raises:
            ValueError: If a required parameter is missing from *params*.
        """
        cls = self.get(type_key)
        sig = inspect.signature(cls.__init__)
        validated: dict[str, Any] = {}

        # Collect parameter info (skip self and known pipeline params)
        pipeline_params = {
            "input_sizes",
            "keys_in",
            "keys_out",
            "label_key",
            "label_in_x",
            "num_classes",
            "kwargs",
            "output_sizes",
            "shared_memory",
            "shared_outputs",
            "shared_inputs",
            "delay_epochs",
            "update_every_n_epochs",
        }

        for name, param in sig.parameters.items():
            if name in ("self",) or name in pipeline_params:
                continue
            if name in params:
                value = params[name]
                # Try to cast to annotated type if available
                if param.annotation is not inspect.Parameter.empty:
                    try:
                        annotation = param.annotation
                        if annotation in (int, float, str, bool):
                            value = annotation(value)
                    except (ValueError, TypeError):
                        pass
                validated[name] = value
            elif param.default is inspect.Parameter.empty:
                raise ValueError(
                    f"Required parameter '{name}' missing for layer type '{type_key}'. "
                    f"Signature: {sig}"
                )

        # Pass through extra kwargs if **kwargs accepted
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if has_var_keyword:
            for k, v in params.items():
                if k not in validated:
                    validated[k] = v

        return validated

    def instantiate(
        self,
        type_key: str,
        *,
        input_sizes: dict[str, tuple],
        keys_in: builtins.list[str],
        keys_out: builtins.list[str],
        **params: Any,
    ) -> nn.Module:
        """Instantiate a layer from the registry.

        Returns:
            Instantiated ``nn.Module`` (or ``TorchModuleAdapter`` wrapper).
        """
        cls = self.get(type_key)
        validated = self.validate_params(type_key, params)

        if issubclass(cls, PipelineLayer):
            return cls(
                input_sizes=input_sizes,
                keys_in=keys_in,
                keys_out=keys_out,
                **validated,
            )
        else:
            # Wrap plain nn.Module in TorchModuleAdapter
            from nexuml.core.torch_adapter import TorchModuleAdapter

            module = cls(**validated)
            return TorchModuleAdapter(
                module=module,
                input_sizes=input_sizes,
                keys_in=keys_in,
                keys_out=keys_out,
            )

    def scan(self, package_paths: builtins.list[str] | None = None) -> None:
        """Scan packages for PipelineLayer subclasses and register them.

        Also consumes decorated discovery results from the Scanner.
        """
        if package_paths is None:
            package_paths = discover_library_packages()

        # Consume decorated discovery results. Import/registration failures are
        # collected (not raised) so one broken module cannot hide every layer.
        scanner = Scanner()
        for package_path in package_paths:
            scanner.scan_package(package_path)

        self._errors = list(scanner.errors)
        register_items(scanner.by_kind("layer"), self.register, self._errors)

        self._loaded = True
        logger.info(
            "Registry loaded: %d layer(s)%s - %s",
            len(self._registry),
            f", {len(self._errors)} error(s)" if self._errors else "",
            ", ".join(sorted(self._registry.keys())),
        )

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.scan()


# Module-level singleton — lazy initialized
_default_registry: LayerRegistry | None = None


def get_registry() -> LayerRegistry:
    """Get the default layer registry (lazy-initialized singleton).

    Returns:
        The module-level singleton ``LayerRegistry`` instance.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = LayerRegistry()
    return _default_registry
