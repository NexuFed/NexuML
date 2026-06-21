"""Shared data-root resolution for scenario data builders."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_data_root(data_root: str, env_var: str = "NEXUML_DATA_ROOT") -> Path:
    """Resolve a dataset path against the optional global data root.

    Absolute paths are used as-is. Relative paths are resolved under
    ``NEXUML_DATA_ROOT`` when set. Otherwise, relative paths are returned
    unchanged and remain relative to the current working directory.

    Returns:
        Path: Resolved dataset path.
    """
    requested = Path(data_root)
    if requested.is_absolute():
        return requested

    env_root = os.getenv(env_var)
    if not env_root:
        return requested

    return Path(env_root) / requested
