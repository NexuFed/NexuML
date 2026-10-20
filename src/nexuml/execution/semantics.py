"""Shared guard against plausible but incorrect rank-local finalization."""

from typing import Any


class ExecutionError(RuntimeError):
    """An execution request cannot preserve NexuML semantics or ownership."""


def ensure_distributed_semantics(scenario: Any) -> None:
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
