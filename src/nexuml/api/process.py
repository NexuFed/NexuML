"""Fresh selected-interpreter calls for discovery and configuration validation."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from nexuml.api.errors import ApiError


def worker_environment() -> dict[str, str]:
    """Inherit the selected environment without forwarding API credentials.

    Returns:
        Worker environment, with private connection settings removed.
    """
    return {
        key: value
        for key, value in os.environ.items()
        if key not in {"NEXUML_API_TOKEN", "NEXUML_STUDIO_TOKEN"}
    }


def call_worker(action: str, payload: dict, directory: Path) -> dict:
    """Run one existing NexuML call in a fresh selected-interpreter process.

    Returns:
        Plain result data from the worker's private response file.

    Raises:
        ApiError: If validation, initialization or the bounded call fails.
    """
    with tempfile.TemporaryDirectory(prefix="nexuml-api-") as temporary:
        root = Path(temporary)
        request, response = root / "request.json", root / "response.json"
        try:
            request.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        except ValueError as exc:
            raise ApiError(
                422, "validation", "Configuration must contain finite JSON values."
            ) from exc
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "nexuml.api.worker", action, str(request), str(response)],
                cwd=directory,
                env=worker_environment(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
            )
        except subprocess.TimeoutExpired as exc:
            raise ApiError(
                408, "timeout", "NexuML discovery/configuration call timed out."
            ) from exc
        if completed.returncode != 0 or not response.is_file():
            raise ApiError(500, "runtime_error", "Selected NexuML worker could not initialize.")
        result = json.loads(response.read_text(encoding="utf-8"))
        if "error" in result:
            error = result["error"]
            raise ApiError(error["status"], error["code"], error["message"], error.get("fields"))
        return result
