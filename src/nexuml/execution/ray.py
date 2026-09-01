"""Thin Ray Train execution for the canonical NexuML Lightning session."""

from __future__ import annotations

from typing import Any

import lightning as L

from nexuml.core.types import RayExecutionSpec, ScenarioSpec


class RayExecutionError(RuntimeError):
    """Raised when a Ray execution request cannot be represented cleanly."""


def _ray_strategy(name: str, params: dict[str, Any]) -> Any:
    """Map NexuML's training strategy to Ray's official Lightning strategy."""
    try:
        from ray.train.lightning import RayDDPStrategy, RayDeepSpeedStrategy, RayFSDPStrategy
    except ImportError as error:
        raise RayExecutionError("Ray execution requires the nexuml[ray] extra") from error

    if name in {"auto", "ddp"}:
        return RayDDPStrategy(**params)
    if name == "fsdp":
        return RayFSDPStrategy(**params)
    if name == "deepspeed":
        return RayDeepSpeedStrategy(**params)
    raise RayExecutionError(
        "Ray supports training.strategy values 'auto', 'ddp', 'fsdp', or 'deepspeed'; "
        f"got {name!r}"
    )


def _prepare_session_trainer(session: Any) -> L.Trainer:
    """Build the Ray-aware Lightning Trainer and attach it to one NexuSession."""
    try:
        from ray.train.lightning import (
            RayLightningEnvironment,
            RayTrainReportCallback,
            prepare_trainer,
        )
    except ImportError as error:
        raise RayExecutionError("Ray execution requires the nexuml[ray] extra") from error

    training = session.scenario.training
    callbacks = [*session.trainer_callbacks, RayTrainReportCallback()]
    resolved_accelerator = (
        session.accelerator if session.accelerator != "auto" else training.accelerator
    )

    trainer = L.Trainer(
        max_epochs=training.max_epochs,
        accelerator=resolved_accelerator,
        devices="auto",
        strategy=_ray_strategy(training.strategy, training.strategy_params),
        plugins=[RayLightningEnvironment()],
        precision=training.precision,  # ty: ignore[invalid-argument-type]
        default_root_dir=str(session.log_dir),
        enable_progress_bar=session.enable_progress_bar,
        enable_model_summary=False,
        logger=session.trainer_loggers,
        callbacks=callbacks,
        num_sanity_val_steps=0 if training.max_epochs == 0 else 2,
        log_every_n_steps=1,
    )
    prepared = prepare_trainer(trainer)
    session._trainer = prepared
    return prepared


def train_loop_per_worker(config: dict[str, Any]) -> None:
    """Run one Ray worker through the normal NexuML session lifecycle."""
    from nexuml.training.lightning import NexuSession

    scenario = ScenarioSpec.model_validate(config["scenario"])
    session = NexuSession.from_scenario(scenario)
    _prepare_session_trainer(session)
    session.run()


def _scaling_config(execution: RayExecutionSpec) -> Any:
    """Build Ray's native ScalingConfig from placement-only NexuML settings."""
    try:
        from ray.train import ScalingConfig
    except ImportError as error:
        raise RayExecutionError("Ray execution requires the nexuml[ray] extra") from error

    resources = dict(execution.resources_per_worker)
    return ScalingConfig(
        num_workers=execution.workers,
        use_gpu=resources.get("GPU", 0) > 0,
        resources_per_worker=resources,
    )


def _run_config(scenario: ScenarioSpec, execution: RayExecutionSpec) -> Any:
    """Build Ray's native RunConfig and keep Ray responsible for run storage."""
    try:
        from ray.train import RunConfig
    except ImportError as error:
        raise RayExecutionError("Ray execution requires the nexuml[ray] extra") from error

    return RunConfig(name=scenario.name, storage_path=execution.storage_path)


def _connect(execution: RayExecutionSpec) -> None:
    """Connect to the configured existing cluster without wrapping Ray Jobs."""
    try:
        import ray
    except ImportError as error:
        raise RayExecutionError("Ray execution requires the nexuml[ray] extra") from error

    if ray.is_initialized():
        return

    runtime_env: dict[str, Any] = {"env_vars": {"RAY_TRAIN_V2_ENABLED": "1"}}
    working_dir = execution.target.working_dir
    if working_dir:
        runtime_env.update(
            {
                "working_dir": working_dir,
                "py_executable": "uv run",
            }
        )
    ray.init(address=execution.target.address, runtime_env=runtime_env)


def run_ray(scenario: ScenarioSpec) -> Any:
    """Execute one resolved scenario with Ray Train and return Ray's native Result.

    Returns:
        The ``ray.train.Result`` returned by ``TorchTrainer.fit``.

    Raises:
        RayExecutionError: If the scenario does not select Ray execution or Ray
            dependencies are unavailable.
    """
    execution = scenario.execution
    if not isinstance(execution, RayExecutionSpec):
        raise RayExecutionError("run_ray requires scenario.execution.kind='ray'")

    _connect(execution)
    try:
        from ray.train.torch import TorchTrainer
    except ImportError as error:
        raise RayExecutionError("Ray execution requires the nexuml[ray] extra") from error

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={"scenario": scenario.model_dump(mode="json")},
        scaling_config=_scaling_config(execution),
        run_config=_run_config(scenario, execution),
    )
    return trainer.fit()


__all__ = [
    "RayExecutionError",
    "run_ray",
    "train_loop_per_worker",
]
