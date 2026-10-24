"""Load project-owned Ray Jobs settings independently of a training scenario."""

from __future__ import annotations

from pathlib import Path

from ruamel.yaml import YAML

from nexuml.core.serialization import restore_component
from nexuml.execution.definitions import RayJobsSDKExecution


def load_ray_job_profile(path: Path) -> RayJobsSDKExecution:
    """Load project overrides for the Ray Jobs execution defaults.

    Relative paths retain their meaning from the submission working directory,
    not the profile's directory. The backend resolves ``{run_id}`` per submission.
    Profiles contain executable setup commands; only use trusted project files.

    Returns:
        Validated execution settings.

    Raises:
        ValueError: If the document is not a valid Ray Jobs execution profile.
    """
    data = YAML(typ="safe").load(path.read_text())
    schema_error = (
        "Execution profile requires type: ray-jobs-sdk, version: '1', params: {...}; "
        "replace legacy kind syntax and use the SDK backend, not KubeRay ray-job."
    )
    if not isinstance(data, dict) or set(data) != {"type", "version", "params"}:
        raise ValueError(schema_error)
    try:
        execution = restore_component(kind="execution_backend", value=data)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{schema_error} {error}") from error
    if not isinstance(execution, RayJobsSDKExecution):
        raise ValueError(schema_error)
    return execution
