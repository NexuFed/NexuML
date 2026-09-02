"""Thin Ray Train execution for the canonical NexuML Lightning session."""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, cast
from uuid import uuid4

import lightning as L

from nexuml.core.types import RayExecutionSpec, ScenarioSpec


class RayExecutionError(RuntimeError):
    """Raised when a Ray execution request cannot be represented cleanly."""


def _ray_strategy(name: str, params: dict[str, Any]) -> Any:
    """Map the NexuML strategy to Ray's official Lightning strategy.

    Returns:
        Ray Lightning strategy instance.

    Raises:
        RayExecutionError: If Ray is unavailable or the strategy is unsupported.
    """
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
        f"Ray supports training.strategy values 'auto', 'ddp', 'fsdp', or 'deepspeed'; got {name!r}"
    )


def _prepare_session_trainer(session: Any) -> L.Trainer:
    """Build and attach the Ray-aware Lightning Trainer.

    Returns:
        Trainer prepared by Ray for the current worker.

    Raises:
        RayExecutionError: If the Ray Lightning integration is unavailable.
    """
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
        precision=training.precision,
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


def _final_metrics(result: Any) -> dict[str, float | int]:
    """Flatten final NexuML metrics into Ray's scalar result payload.

    Returns:
        Scalar validation, test, and evaluation metrics.
    """
    metrics: dict[str, float | int] = {}
    for prefix, rows in (
        ("val", getattr(result, "validation_results", ())),
        ("test", getattr(result, "test_results", ())),
    ):
        if not rows:
            continue
        row = rows[-1]
        if not isinstance(row, Mapping):
            continue
        for key, value in row.items():
            name = str(key)
            metric_key = name if name.startswith(f"{prefix}/") else f"{prefix}/{name}"
            if isinstance(value, bool):
                metrics[metric_key] = int(value)
            elif isinstance(value, (int, float)):
                metrics[metric_key] = value

    evaluation = getattr(result, "eval_algorithm_results", {})
    if isinstance(evaluation, Mapping):
        for key, value in evaluation.items():
            if isinstance(value, bool):
                metrics[str(key)] = int(value)
            elif isinstance(value, (int, float)):
                metrics[str(key)] = value
    metrics.setdefault("nexuml/completed", 1)
    return metrics


def _ensure_distributed_semantics(scenario: ScenarioSpec) -> None:
    """Reject pipeline phases whose distributed semantics are not defined yet.

    Raises:
        RayExecutionError: If the scenario contains stateful post-training work
            that cannot yet be aggregated globally across Ray workers.
    """
    if scenario.evaluation.algorithms:
        configured = ", ".join(spec.name or spec.type for spec in scenario.evaluation.algorithms)
        raise RayExecutionError(
            "Ray execution does not yet support evaluation.algorithms with rank-sharded data: "
            "evaluation algorithms accumulate state independently on each worker, so reducing "
            "their final scalars would not reproduce global evaluation semantics. "
            f"Configured algorithms: {configured}. Keep them disabled for Ray until global "
            "evaluation-state aggregation is implemented."
        )

    from nexuml.core.post_train_layer import PostTrainFitLayer
    from nexuml.core.registry import get_registry

    registry = get_registry()
    for stage in scenario.pipeline.stages.values():
        for layer_spec in stage:
            layer_type = registry.get(layer_spec.type_key)
            if issubclass(layer_type, PostTrainFitLayer):
                raise RayExecutionError(
                    "Ray execution does not yet support PostTrainFitLayer semantics: "
                    "the post-train fit pass must see the full training set rather than one "
                    "DALI rank shard. Keep this scenario local until distributed post-train "
                    "finalization is implemented."
                )


def train_loop_per_worker(config: dict[str, Any]) -> None:
    """Run one Ray worker through the normal NexuML session lifecycle.

    Raises:
        RayExecutionError: If Ray Train is unavailable in the worker environment.
    """
    try:
        import ray.train as train
    except ImportError as error:
        raise RayExecutionError("Ray execution requires the nexuml[ray] extra") from error

    from nexuml.training.lightning import NexuSession

    scenario = ScenarioSpec.model_validate(config["scenario"])
    session = NexuSession.from_scenario(
        scenario,
        enable_loggers=train.get_context().get_world_rank() == 0,
    )
    _prepare_session_trainer(session)
    result = session.run()
    train.report(_final_metrics(result))


def _scaling_config(execution: RayExecutionSpec) -> Any:
    """Build Ray's native placement configuration.

    Returns:
        Ray ``ScalingConfig`` matching the execution specification.

    Raises:
        RayExecutionError: If Ray Train is unavailable.
    """
    try:
        from ray.train import ScalingConfig
    except ImportError as error:
        raise RayExecutionError("Ray execution requires the nexuml[ray] extra") from error

    resources = dict(execution.resources_per_worker)
    return ScalingConfig(
        num_workers=cast(Any, execution.workers),
        use_gpu=resources.get("GPU", 0) > 0,
        resources_per_worker=resources,
    )


def _run_config(
    scenario: ScenarioSpec,
    execution: RayExecutionSpec,
    worker_runtime_env: dict[str, Any],
    run_name: str | None = None,
) -> Any:
    """Build Ray's native run configuration.

    Returns:
        Ray ``RunConfig`` with NexuML's run name and storage path.

    Raises:
        RayExecutionError: If Ray Train is unavailable.
    """
    try:
        from ray.train import RunConfig
    except ImportError as error:
        raise RayExecutionError("Ray execution requires the nexuml[ray] extra") from error

    storage_path = execution.storage_path
    endpoint = os.getenv("AWS_ENDPOINT_URL_S3") or os.getenv("AWS_ENDPOINT_URL")
    storage_filesystem = None
    if storage_path and storage_path.startswith("s3://") and endpoint:
        from pyarrow.fs import S3FileSystem

        scheme, separator, endpoint_override = endpoint.partition("://")
        if not separator:
            scheme = "https"
            endpoint_override = endpoint
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        anonymous = access_key == secret_key == "anonymous"
        storage_filesystem = S3FileSystem(
            access_key=None if anonymous else access_key,
            secret_key=None if anonymous else secret_key,
            session_token=os.getenv("AWS_SESSION_TOKEN"),
            anonymous=anonymous,
            region=os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION")),
            scheme=scheme,
            endpoint_override=endpoint_override.rstrip("/"),
            background_writes=False,
        )
        storage_path = storage_path.removeprefix("s3://").partition("?")[0]

    return cast(Any, RunConfig)(
        name=run_name or scenario.name,
        storage_path=storage_path,
        storage_filesystem=storage_filesystem,
        worker_runtime_env=worker_runtime_env,
    )


def _connect(execution: RayExecutionSpec) -> dict[str, Any]:
    """Connect to the configured existing cluster without wrapping Ray Jobs.

    Returns:
        Runtime environment shared by the Ray Client server and Train workers.

    Raises:
        RayExecutionError: If Ray is unavailable.
    """
    try:
        import ray
    except ImportError as error:
        raise RayExecutionError("Ray execution requires the nexuml[ray] extra") from error

    if os.getenv("AWS_ENDPOINT_URL_S3") or os.getenv("AWS_ENDPOINT_URL"):
        os.environ.setdefault("AWS_REQUEST_CHECKSUM_CALCULATION", "WHEN_REQUIRED")
        os.environ.setdefault("AWS_RESPONSE_CHECKSUM_VALIDATION", "WHEN_REQUIRED")

    env_vars = {
        name: os.environ[name]
        for name in (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_DEFAULT_REGION",
            "AWS_REGION",
            "AWS_ENDPOINT_URL",
            "AWS_ENDPOINT_URL_S3",
            "AWS_CA_BUNDLE",
            "AWS_REQUEST_CHECKSUM_CALCULATION",
            "AWS_RESPONSE_CHECKSUM_VALIDATION",
        )
        if name in os.environ
    }
    runtime_env: dict[str, Any] = {
        "env_vars": {
            "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
            "RAY_TRAIN_V2_ENABLED": "1",
            "RAY_TRAIN_WORKER_GROUP_START_TIMEOUT_S": "600",
            "TIMEOUT_FOR_SPECIFIC_SERVER_S": "600",
            **env_vars,
        }
    }
    working_dir = execution.target.working_dir
    if working_dir:
        runtime_env["working_dir"] = working_dir
    if execution.target.py_executable:
        runtime_env["py_executable"] = execution.target.py_executable
    if not ray.is_initialized():
        ray.init(address=execution.target.address, runtime_env=runtime_env)
    return dict(ray.get_runtime_context().runtime_env)


def run_ray(scenario: ScenarioSpec) -> Any:
    """Execute one resolved scenario with Ray Train and return Ray's native Result.

    Returns:
        The ``ray.train.Result`` returned by ``TorchTrainer.fit``.

    Raises:
        RayExecutionError: If the scenario cannot be executed safely with Ray or
            Ray dependencies are unavailable.
    """
    execution = scenario.execution
    if not isinstance(execution, RayExecutionSpec):
        raise RayExecutionError("run_ray requires scenario.execution.kind='ray'")

    _ensure_distributed_semantics(scenario)
    runtime_env = _connect(execution)
    try:
        from ray.train.torch import TorchTrainer
    except ImportError as error:
        raise RayExecutionError("Ray execution requires the nexuml[ray] extra") from error

    run_name = f"{scenario.name}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={"scenario": scenario.model_dump(mode="json")},
        scaling_config=_scaling_config(execution),
        run_config=_run_config(scenario, execution, runtime_env, run_name),
    )
    return trainer.fit()


__all__ = [
    "RayExecutionError",
    "run_ray",
    "train_loop_per_worker",
]
