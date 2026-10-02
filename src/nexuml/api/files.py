"""Authorized transport file access, separate from NexuML dataset path policies."""

import hashlib
import os
import tempfile
import threading
from pathlib import Path

from nexuml.api.errors import ApiError


class Files:
    """One selected directory boundary and serialized revision-checked writes."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.lock = threading.Lock()

    def path(self, value: str) -> Path:
        """Resolve a path and reject traversal/symlink escapes.

        Returns:
            Canonical path within the selected working directory.

        Raises:
            ApiError: When a path escapes the authorized directory.
        """
        candidate = Path(value).expanduser()
        if ".." in candidate.parts:
            raise ApiError(403, "forbidden_path", "Path traversal is not allowed.")
        candidate = (self.directory / candidate).resolve()
        if not candidate.is_relative_to(self.directory):
            raise ApiError(403, "forbidden_path", "Path is outside the working directory.")
        return candidate

    @staticmethod
    def revision(path: Path) -> str | None:
        """Hash file bytes for conflict detection.

        Returns:
            Current revision, or None for a new file.
        """
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

    def save(self, path: Path, text: str, base_revision: str | None) -> str:
        """Atomically save only against the supplied current revision.

        Returns:
            Revision of the saved file.

        Raises:
            ApiError: When the source changed after load.
        """
        with self.lock:
            path = self.path(str(path))
            if self.revision(path) != base_revision:
                raise ApiError(409, "conflict", "File changed externally. Reload before saving.")
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=".studio-", dir=path.parent)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    stream.write(text)
                    stream.flush()
                    os.fsync(stream.fileno())
                # Check again after validation/temp write; do not silently replace an external edit.
                if self.revision(path) != base_revision:
                    raise ApiError(
                        409, "conflict", "File changed externally. Reload before saving."
                    )
                os.replace(temporary, self.path(str(path)))
            finally:
                Path(temporary).unlink(missing_ok=True)
            return hashlib.sha256(text.encode()).hexdigest()
