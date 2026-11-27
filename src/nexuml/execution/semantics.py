"""Shared guard against plausible but incorrect rank-local finalization."""

from typing import Any

from nexuml.core.types import ReinforcementLearningSpec


class ExecutionError(RuntimeError):
    """An execution request cannot preserve NexuML semantics or ownership."""


def ensure_distributed_semantics(
    scenario: Any, *, world_size: int | None = None, accelerator: str | None = None
) -> None:
    """Reject stateful phases until global distributed finalization exists.

    Raises:
        ExecutionError: If rank-sharded execution would change scenario semantics.
    """
    if scenario.evaluation.algorithms:
        raise ExecutionError(
            "Distributed execution does not yet support evaluation.algorithms with rank-sharded "
            "data: rank-local state is not global evaluation. Keep this scenario local until "
            "evaluation-state aggregation is implemented."
        )
    if any(
        layer.component.requires_post_train_fit
        for stage in scenario.pipeline.stages.values()
        for layer in stage
    ):
        raise ExecutionError(
            "Distributed execution does not yet support PostTrainFitLayer semantics: "
            "the post-train pass must see the full dataset. Keep this scenario local until "
            "global distributed finalization is implemented."
        )
    training = scenario.training
    if not isinstance(training, ReinforcementLearningSpec) or world_size in (None, 1):
        return
    assert world_size is not None
    collector = training.collector
    accelerator = accelerator or training.accelerator
    if accelerator not in ("cpu", "gpu", "cuda"):
        raise ExecutionError("DDP RL requires an explicit CPU or CUDA accelerator")
    name = "cpu" if accelerator == "cpu" else "cuda"
    if (
        collector.backend != "direct"
        or not collector.sync
        or collector.policy_sync_interval_batches != 1
        or (collector.env_device, collector.storing_device) != (name, name)
        or collector.policy_device not in ("auto", name)
    ):
        raise ExecutionError(
            f"DDP RL requires synchronous direct collection on each rank's {name} device "
            f"(env_device and storing_device='{name}', policy_device='auto' or '{name}')"
        )
    if training.strategy not in ("auto", "ddp") or str(training.precision) != "32-true":
        raise ExecutionError("DDP RL supports DDP with 32-true precision")
    if training.seed is None:
        raise ExecutionError("DDP RL requires an explicit base seed")
    if training.frames_per_batch % (world_size * collector.num_envs_per_collector):
        raise ExecutionError(
            "Global frames_per_batch must divide evenly into rank environment steps"
        )
    if training.total_frames % training.frames_per_batch:
        raise ExecutionError("DDP RL total_frames must be a multiple of global frames_per_batch")
    spec = scenario.policy_evaluation or training.evaluation
    if spec.episodes and spec.process is None and spec.ray is None:
        raise ExecutionError("DDP RL fit evaluation requires independent process or Ray placement")
    assert scenario.interaction is not None
    environment = scenario.interaction.environment
    if environment.describe().groups:
        raise ExecutionError("DDP RL does not yet support grouped environment contracts")
    try:
        training.algorithm.validate_distributed_training(training.frames_per_batch, world_size)
        environment.validate_distributed_training(world_size)
    except ValueError as error:
        raise ExecutionError(str(error)) from error
