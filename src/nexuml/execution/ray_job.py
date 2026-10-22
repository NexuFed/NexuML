"""Asynchronous Ray Jobs submission of the canonical local CLI lifecycle."""

from __future__ import annotations

import shlex

from nexuml.core.config import ResolvedConfig
from nexuml.core.types import LocalExecutionSpec, RayJobExecutionSpec, ScenarioSpec


def submit_ray_job(scenario: ScenarioSpec) -> str:
    """Submit a resolved scenario without constructing a local training runtime.

    Returns:
        Ray's submission ID, not a training result.

    Raises:
        ValueError: If execution is not ray_job or the Python command is invalid.
        RuntimeError: If the optional Ray Jobs dependency is unavailable or submission fails.
    """
    execution = scenario.execution
    if not isinstance(execution, RayJobExecutionSpec):
        raise ValueError("submit_ray_job requires scenario.execution.kind='ray_job'")
    execution = RayJobExecutionSpec.model_validate(execution.model_dump())
    command = shlex.split(execution.py_executable)
    if not command:
        raise ValueError("ray_job py_executable must contain a Python command")

    try:
        from ray.job_submission import JobSubmissionClient
    except ImportError as error:
        raise RuntimeError("Ray Jobs execution requires the nexuml[ray] extra") from error

    # Only transport the resolved local scenario, never submission credentials or recursive jobs.
    remote = scenario.model_copy(update={"execution": LocalExecutionSpec()})
    config = ResolvedConfig.from_scenario(remote).to_yaml()
    bootstrap = "\n".join(
        [
            "from pathlib import Path",
            "from tempfile import TemporaryDirectory",
            "from nexuml.cli.main import app",
            "with TemporaryDirectory(prefix='nexuml-ray-job-') as directory:",
            "    config = Path(directory) / 'scenario.yaml'",
            f"    config.write_text({config!r}, encoding='utf-8')",
            "    app(['train', '--config', str(config)])",
        ]
    )
    entrypoint = shlex.join([*command, "-c", bootstrap])
    if execution.setup_command is not None:
        entrypoint = f"{shlex.join(execution.setup_command)} && exec {entrypoint}"
    client = JobSubmissionClient(execution.address, verify=execution.verify)
    return client.submit_job(
        entrypoint=entrypoint,
        runtime_env=execution.runtime_env,
        entrypoint_num_cpus=execution.entrypoint_num_cpus,
        entrypoint_num_gpus=execution.entrypoint_num_gpus,
        entrypoint_memory=execution.entrypoint_memory,
        entrypoint_resources=execution.entrypoint_resources,
    )
