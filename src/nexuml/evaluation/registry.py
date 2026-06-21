"""Registry for evaluation algorithms with deterministic dynamic discovery."""

from __future__ import annotations

import builtins
import inspect
import logging
from typing import Any

from nexuml.core.discovery import (
    DiscoveryError,
    Scanner,
    discover_library_packages,
    register_items,
)
from nexuml.core.log_paths import resolve_logs_root_str
from nexuml.core.types import EvalAlgorithmSpec
from nexuml.evaluation.algorithm import EvalAlgorithm

logger = logging.getLogger(__name__)


class EvalAlgorithmRegistry:
    """Registry for evaluation algorithms with automatic discovery.

    Also consumes decorated discovery results from the Scanner.
    """

    def __init__(self) -> None:
        self._registry: dict[str, type[EvalAlgorithm]] = {}
        self._errors: builtins.list[DiscoveryError] = []
        self._loaded = False

    @property
    def errors(self) -> builtins.list[DiscoveryError]:
        """Discovery failures from the last scan (import + registration)."""
        self.ensure_loaded()
        return list(self._errors)

    def register(self, type_key: str, cls: type[EvalAlgorithm]) -> None:
        if type_key in self._registry:
            existing = self._registry[type_key]
            if existing is not cls:
                raise ValueError(
                    f"Registry conflict: '{type_key}' already registered to "
                    f"{existing.__module__}.{existing.__name__}, "
                    f"cannot register {cls.__module__}.{cls.__name__}"
                )
        self._registry[type_key] = cls

    def get(self, type_key: str) -> type[EvalAlgorithm]:
        self.ensure_loaded()
        if type_key not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(f"Evaluation algorithm '{type_key}' not found. Available: [{available}]")
        return self._registry[type_key]

    def list(self) -> dict[str, type[EvalAlgorithm]]:
        self.ensure_loaded()
        return dict(self._registry)

    def validate_params(self, type_key: str, params: dict[str, Any]) -> dict[str, Any]:
        cls = self.get(type_key)
        sig = inspect.signature(cls.__init__)
        validated: dict[str, Any] = {}

        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if name in params:
                validated[name] = params[name]
            elif param.default is inspect.Parameter.empty:
                raise ValueError(
                    f"Required parameter '{name}' missing for eval algorithm '{type_key}'. "
                    f"Signature: {sig}"
                )

        has_var_keyword = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in sig.parameters.values()
        )
        if has_var_keyword:
            for key, value in params.items():
                if key not in validated:
                    validated[key] = value
        else:
            unknown = set(params) - set(validated)
            if unknown:
                raise TypeError(
                    f"Unknown parameter(s) {sorted(unknown)} for algorithm '{type_key}'. "
                    f"Valid params: {sorted(validated)}"
                )

        return self._resolve_output_params(validated)

    def _resolve_output_params(self, params: dict[str, Any]) -> dict[str, Any]:
        resolved = dict(params)
        for key in ("output_dir", "output"):
            value = resolved.get(key)
            if isinstance(value, str):
                resolved[key] = resolve_logs_root_str(value)
        return resolved

    def create(self, spec: EvalAlgorithmSpec) -> EvalAlgorithm:
        cls = self.get(spec.type)
        merged = dict(spec.params)
        if spec.feature_key is not None:
            merged["feature_key"] = spec.feature_key
        if spec.label_key is not None:
            merged["label_key"] = spec.label_key
        validated = self.validate_params(spec.type, merged)
        return cls(**validated)

    def scan(self, package_paths: builtins.list[str] | None = None) -> None:
        """Scan packages for EvalAlgorithm subclasses and register them.

        Also consumes decorated discovery results from the Scanner.
        """
        if package_paths is None:
            package_paths = discover_library_packages()

        # Consume decorated discovery results. Import/registration failures are
        # collected (not raised) so one broken module cannot hide every algorithm.
        scanner = Scanner()
        for package_path in package_paths:
            scanner.scan_package(package_path)

        self._errors = list(scanner.errors)
        register_items(scanner.by_kind("eval_algorithm"), self.register, self._errors)

        self._loaded = True
        logger.info(
            "Evaluation registry loaded: %d algorithm(s)%s - %s",
            len(self._registry),
            f", {len(self._errors)} error(s)" if self._errors else "",
            ", ".join(sorted(self._registry.keys())),
        )

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.scan()


_default_eval_registry: EvalAlgorithmRegistry | None = None


def get_eval_registry() -> EvalAlgorithmRegistry:
    """Get the default evaluation registry singleton.

    Returns:
        The default ``EvalAlgorithmRegistry`` instance.
    """
    global _default_eval_registry
    if _default_eval_registry is None:
        _default_eval_registry = EvalAlgorithmRegistry()
    return _default_eval_registry


_registry = get_eval_registry()
_registry.ensure_loaded()
EVAL_ALGORITHM_REGISTRY: dict[str, type[EvalAlgorithm]] = _registry._registry


def create_algorithm(spec: EvalAlgorithmSpec) -> EvalAlgorithm:
    """Instantiate an evaluation algorithm from a spec.

    Returns:
        The created ``EvalAlgorithm`` instance.
    """
    return get_eval_registry().create(spec)
