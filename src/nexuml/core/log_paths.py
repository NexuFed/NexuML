"""Utilities for resolving log and artifact output paths."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

LOGS_ROOT_ENV = "NEXUML_LOGS_ROOT"


def resolve_logs_root(path: str | Path, *, env_var: str = LOGS_ROOT_ENV) -> Path:
    """Prefix relative log paths with ``NEXUML_LOGS_ROOT`` when configured.

    Returns:
        Resolved absolute or relative path.
    """
    candidate = Path(path).expanduser()
    root = os.environ.get(env_var)
    if not root or candidate.is_absolute():
        return candidate
    return Path(root).expanduser() / candidate


def resolve_logs_root_str(path: str | Path, *, env_var: str = LOGS_ROOT_ENV) -> str:
    """Return the logs root path resolved to a string."""
    return str(resolve_logs_root(path, env_var=env_var))


def resolve_logs_file_uri(uri: str, *, env_var: str = LOGS_ROOT_ENV) -> str:
    """Resolve relative file URIs under logs root; preserve absolute/remote URIs.

    Returns:
        Resolved ``file://`` URI with absolute path, or the original *uri*
        unchanged if it is absolute or remote.
    """
    if not uri.startswith("file:"):
        return uri
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc or parsed.path.startswith("/"):
        return uri
    return resolve_logs_root(parsed.path or ".", env_var=env_var).resolve().as_uri()
