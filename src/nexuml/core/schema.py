"""Schema utilities for resolved config YAML I/O."""

from __future__ import annotations

from pathlib import Path

from nexuml.core.config import ResolvedConfig


def dump_resolved_yaml(config: ResolvedConfig, path: Path) -> None:
    """Serialize a resolved config to YAML file."""
    config.save(path)


def load_resolved_yaml(path: Path) -> ResolvedConfig:
    """Load a resolved config from YAML file.

    Returns:
        Deserialized ``ResolvedConfig`` instance.
    """
    return ResolvedConfig.load(path)
