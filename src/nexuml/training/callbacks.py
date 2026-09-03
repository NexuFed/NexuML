"""Build configured Lightning callbacks."""

from __future__ import annotations

import logging
from typing import Any

from lightning.pytorch.callbacks import Callback

from nexuml.core.log_paths import resolve_logs_root_str
from nexuml.core.types import CallbackSpec

logger = logging.getLogger(__name__)


def build_callbacks(callback_specs: list[CallbackSpec]) -> list[Callback]:
    """Instantiate Lightning callbacks from portable factory specs.

    Returns:
        List of instantiated Lightning callback objects.
    """
    callbacks: list[Callback] = []
    for spec in callback_specs:
        try:
            callback = spec.build(**_resolve_callback_path_params(spec.kwargs))
        except (TypeError, ValueError) as exc:
            logger.warning("Could not build callback %r: %s", spec.factory, exc)
            continue
        if not isinstance(callback, Callback):
            logger.warning(
                "Callback factory %r returned %s instead of Callback",
                spec.factory,
                type(callback).__name__,
            )
            continue
        callbacks.append(callback)
    return callbacks


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
