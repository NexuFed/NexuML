"""Supervise inferred cluster gates without oracle conditioning or validation leak."""

from __future__ import annotations

from typing import Any, cast

from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torch.nn import functional as F

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext
from nexuml.core.discovery import layer
from nexuml_library.layers.model.rul_trajectory_loss import (
    RULTrajectoryConsistencyLoss,
    _RULTrajectoryConsistencyRuntime,
)


@layer("RULClusterAuxiliaryLoss")
class RULClusterAuxiliaryLoss(RULTrajectoryConsistencyLoss):
    """Add TRAIN-only failure/working cluster pseudo-label classification."""

    working_weight: float = Field(default=0.0, ge=0.0, le=100.0)
    failure_weight: float = Field(default=0.0, ge=0.0, le=100.0)

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        """Create the native RUL+auxiliary objective.

        Returns:
            Model loss with train-only masked auxiliary terms.
        """
        return _RULClusterAuxiliaryRuntime(
            consistency_weight=self.consistency_weight,
            previous_key=self.previous_key,
            working_weight=self.working_weight,
            failure_weight=self.failure_weight,
            **context.runtime_kwargs(),
        )


class _RULClusterAuxiliaryRuntime(_RULTrajectoryConsistencyRuntime):
    """Ignore cluster labels outside TRAIN and keep validation loss pure RUL."""

    def __init__(
        self, *, working_weight: float, failure_weight: float, **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.working_weight = working_weight
        self.failure_weight = failure_weight

    @staticmethod
    def _auxiliary_loss(
        x: TensorDict, y: TensorDict, kind: str,
    ) -> Tensor:
        """Compute only labeled training cross-entropy with ignore_index=-1.

        Returns:
            Scalar differentiable cross-entropy, or zero for empty masks.

        Raises:
            ValueError: For missing or malformed pseudo-labels/logits.
        """
        logits_key = f"{kind}_cluster_logits"
        label_key = f"{kind}_cluster"
        if logits_key not in x or label_key not in y:
            raise ValueError(f"Missing {kind} cluster logits or target")
        logits = cast(Tensor, x[logits_key])
        targets = cast(Tensor, y[label_key]).view(-1).long()
        if (
            logits.ndim != 2 or logits.shape[0] != targets.numel()
            or not bool((targets >= -1).all())
            or not bool((targets < logits.shape[1]).all())
        ):
            raise ValueError("Cluster pseudo-labels are incompatible with gate outputs")
        selected = targets >= 0
        if not bool(selected.any()):
            return logits.sum() * 0.0
        return F.cross_entropy(logits[selected], targets[selected])

    def forward(
        self, x: TensorDict | Tensor, y: TensorDict | None = None,
    ) -> tuple[TensorDict | Tensor, TensorDict | None]:
        """Use predicted probabilities in the CfC; labels only in TRAIN loss.

        Returns:
            TensorDict and labels, with native scalar `rul_loss`.

        Raises:
            TypeError: If input lacks native TensorDict routing.
        """
        processed, labels = super().forward(x, y)
        if not isinstance(processed, TensorDict):
            raise TypeError("Conditional RUL loss requires TensorDict")
        if self.training and isinstance(labels, TensorDict):
            key = self.keys_out[0]
            result = cast(Tensor, processed[key])
            for kind, weight in (
                ("working", self.working_weight),
                ("failure", self.failure_weight),
            ):
                if weight > 0.0:
                    result = result + weight * self._auxiliary_loss(
                        processed, labels, kind
                    ).expand_as(result)
            processed[key] = result
        return processed, labels
