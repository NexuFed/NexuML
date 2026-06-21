"""Callback factory and registry for NexuML training."""

from __future__ import annotations

import importlib
import logging
from typing import Any

from nexuml.core.log_paths import resolve_logs_root_str

logger = logging.getLogger(__name__)

# Registry mapping aliases to fully qualified class paths.
_CALLBACK_REGISTRY: dict[str, str] = {
    "checkpoint": "lightning.pytorch.callbacks.ModelCheckpoint",
    "lr_monitor": "lightning.pytorch.callbacks.LearningRateMonitor",
    "early_stopping": "lightning.pytorch.callbacks.EarlyStopping",
    "rich_progress": "lightning.pytorch.callbacks.RichProgressBar",
    "device_stats": "lightning.pytorch.callbacks.DeviceStatsMonitor",
}


def register_callback(alias: str, dotted_path: str) -> None:
    """Register a callback alias to a fully qualified class path."""
    _CALLBACK_REGISTRY[alias] = dotted_path


def get_callback_path(alias: str) -> str | None:
    """Look up the dotted path for a callback alias.

    Returns:
        Dotted import path for the alias, or ``None`` if not registered.
    """
    return _CALLBACK_REGISTRY.get(alias)


def list_callbacks() -> dict[str, str]:
    """Return a copy of the current callback registry."""
    return dict(_CALLBACK_REGISTRY)


def build_callbacks(callback_specs: list[Any]) -> list[Any]:
    """Instantiate Lightning callbacks from a list of CallbackSpec objects.

    Returns:
        List of instantiated Lightning callback objects.
    """
    callbacks = []
    for spec in callback_specs:
        cb = _build_one(spec.type, spec.params)
        if cb is not None:
            callbacks.append(cb)
    return callbacks


def _build_one(type_str: str, params: dict[str, Any]) -> Any | None:
    dotted_path = _CALLBACK_REGISTRY.get(type_str, type_str)
    resolved_params = _resolve_callback_path_params(params)
    parts = dotted_path.rsplit(".", 1)
    if len(parts) != 2:
        logger.warning(f"Invalid callback path '{dotted_path}'. Skipping.")
        return None

    module_path, class_name = parts
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(**resolved_params)
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not load callback '{dotted_path}': {e}")
        return None


def _resolve_callback_path_params(params: dict[str, Any]) -> dict[str, Any]:
    """Resolve relative callback output directories under NEXUML_LOGS_ROOT.

    Returns:
        Copy of *params* with relative ``dirpath``/``filepath`` values resolved
        to absolute paths under ``NEXUML_LOGS_ROOT``.
    """
    resolved = dict(params)
    for key in ("dirpath", "filepath"):
        value = resolved.get(key)
        if isinstance(value, str):
            resolved[key] = resolve_logs_root_str(value)
    return resolved
