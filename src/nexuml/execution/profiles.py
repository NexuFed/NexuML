"""Load project-owned Ray Jobs settings independently of a training scenario."""

from __future__ import annotations

from pathlib import Path

from ruamel.yaml import YAML

from nexuml.core.types import RayJobExecutionSpec


def load_ray_job_profile(path: Path) -> RayJobExecutionSpec:
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
    if not isinstance(data, dict) or data.get("kind") != "ray_job":
        raise ValueError("Execution profile must be a YAML mapping with kind: ray_job")
    return RayJobExecutionSpec.model_validate(data)
