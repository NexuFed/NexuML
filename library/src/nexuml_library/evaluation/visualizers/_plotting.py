"""Shared plotting helpers for evaluation visualizers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nexuml.tracking.logger import (
    log_artifact,
    staged_artifact_path,
)

logger = logging.getLogger(__name__)
_UNSUPPORTED_FIGURE_LOGGERS: set[str] = set()


def iter_loggers(logger_obj: Any) -> list[Any]:
    """Normalize Lightning logger containers into a flat list.

    Returns:
        Flat list of non-None logger instances.
    """
    if logger_obj is None:
        return []
    if isinstance(logger_obj, (list, tuple, set)):
        return [logger for logger in logger_obj if logger is not None]

    nested = getattr(logger_obj, "loggers", None)
    if nested is not None:
        return [logger for logger in nested if logger is not None]

    return [logger_obj]


def log_figure(logger_obj: Any, tag: str, fig: Any, global_step: int = 0) -> None:
    """Log a matplotlib figure to supported Lightning loggers."""
    artifact_name = Path(f"{tag}.png").as_posix()
    artifact_dir = (
        str(Path(artifact_name).parent) if Path(artifact_name).parent != Path(".") else None
    )
    file_artifact_backends: list[Any] = []
    for backend in iter_loggers(logger_obj):
        experiment = getattr(backend, "experiment", None)
        handled_directly = False
        if experiment is not None and hasattr(experiment, "add_figure"):
            experiment.add_figure(tag, fig, global_step=global_step)
            handled_directly = True

        run_id = getattr(backend, "run_id", None)
        if experiment is not None and run_id is not None and hasattr(experiment, "log_figure"):
            experiment.log_figure(run_id, fig, artifact_name)
            handled_directly = True

        if not handled_directly:
            file_artifact_backends.append(backend)

    if file_artifact_backends:
        with staged_artifact_path(artifact_name, prefix="nexuml_fig_") as temp_path:
            fig.savefig(temp_path, bbox_inches="tight", dpi=200)

            for backend in file_artifact_backends:
                before = len(_UNSUPPORTED_FIGURE_LOGGERS)
                log_artifact(backend, temp_path, artifact_path=artifact_dir)
                after = len(_UNSUPPORTED_FIGURE_LOGGERS)
                if after == before:
                    continue

    for backend in iter_loggers(logger_obj):
        backend_name = type(backend).__name__
        experiment = getattr(backend, "experiment", None)
        run_id = getattr(backend, "run_id", None)
        is_supported = (
            (experiment is not None and hasattr(experiment, "add_figure"))
            or (experiment is not None and run_id is not None and hasattr(experiment, "log_figure"))
            or backend_name in {"TensorBoardLogger", "DVCLiveLogger"}
            or hasattr(backend, "log_dir")
            or hasattr(backend, "dir")
        )
        if not is_supported and backend_name not in _UNSUPPORTED_FIGURE_LOGGERS:
            logger.info(
                "Skipping figure logging for unsupported logger backend '%s' (tag=%s).",
                backend_name,
                tag,
            )
            _UNSUPPORTED_FIGURE_LOGGERS.add(backend_name)


def apply_axis_style(ax: Any) -> None:
    """Apply a consistent lightweight style to an axis."""
    ax.grid(True, alpha=0.2, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def percentile_range(values: np.ndarray, upper_q: float = 0.99) -> tuple[float, float] | None:
    """Return a robust plotting range from finite values."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    low = float(finite.min())
    high = float(np.quantile(finite, upper_q))
    if not np.isfinite(high) or high <= low:
        high = float(finite.max())
    if high <= low:
        high = low + 1e-6
    return low, high


def format_label(value: Any) -> str:
    """Render numeric labels compactly while keeping strings untouched.

    Returns:
        String representation of *value*.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            value = value.item()
        else:
            value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            value = value.item()
        else:
            value = value.tolist()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def to_flat_numpy(values: torch.Tensor | None) -> np.ndarray | None:
    """Convert a tensor into a flat numpy array.

    Returns:
        Flat numpy array, or None if *values* is None.
    """
    if values is None:
        return None
    return values.detach().cpu().reshape(-1).numpy()
