"""RUL supervision with optional same-engine one-cycle trajectory consistency."""

from __future__ import annotations

from typing import Any, cast

from pydantic import Field
from tensordict import TensorDict
from torch import Tensor

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer


@layer("RULTrajectoryConsistencyLoss")
class RULTrajectoryConsistencyLoss(LayerDefinition):
    """Add a training-only consecutive-cycle penalty to the native RUL MSE."""

    consistency_weight: float = Field(default=0.0, ge=0.0, le=10.0)
    previous_key: str = "previous_predicted_rul"

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        """Build supervised sequence-consistency objective.

        Returns:
            NexuML per-batch scalar RUL regression loss.
        """
        return _RULTrajectoryConsistencyRuntime(
            consistency_weight=self.consistency_weight,
            previous_key=self.previous_key,
            **context.runtime_kwargs(),
        )


class _RULTrajectoryConsistencyRuntime(PipelineLayer):
    """Use exact predecessor windows; no synthetic shifted token embeddings."""

    def __init__(
        self,
        *,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        consistency_weight: float,
        previous_key: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out, **kwargs)
        self.consistency_weight = consistency_weight
        self.previous_key = previous_key

    def forward(
        self,
        x: TensorDict | Tensor,
        y: TensorDict | None = None,
    ) -> tuple[TensorDict | Tensor, TensorDict | None]:
        """Use comparable validation MSE and train-only adjacent-cycle penalty.

        Returns:
            Predictions plus the native scalar `rul_loss` required by NexuML.

        Raises:
            TypeError: If the model does not supply TensorDict predictions.
            ValueError: If model outputs or predecessor flags are misaligned.
        """
        if not isinstance(x, TensorDict):
            raise TypeError("Trajectory consistency requires TensorDict")
        current = cast(Tensor, x[cast(list[str], self.keys_in)[0]])
        if y is None:
            loss = current.square().mean() * 0.0
        else:
            label = cast(Tensor, y["rul"])
            if current.shape != label.shape or current.ndim != 2:
                raise ValueError("Trajectory loss requires aligned [batch,1] RUL")
            loss = (current - label).square().mean()
            if self.training and self.consistency_weight > 0:
                previous = cast(Tensor, x[self.previous_key])
                mask = cast(Tensor, x["pair_mask"])
                if previous.shape != current.shape or mask.shape != current.shape:
                    raise ValueError("Trajectory loss requires aligned real predecessor pairs")
                # RUL decrements by one for each true preceding cycle, except
                # when labels are capped and both true targets equal 130.
                true_step = (label < 130.0).to(label.dtype)
                penalty = (previous - current - true_step).square()
                weight = mask.clamp(0, 1)
                consistency = (weight * penalty).sum() / weight.sum().clamp_min(1.0)
                loss = loss + self.consistency_weight * consistency
        x[self.keys_out[0]] = loss.expand(x.batch_size)
        return x, y

    def forward_tensor(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Require TensorDict routing for paired predictions and labels.

        Raises:
            NotImplementedError: Because a tensor alone lacks the real pair mask.
        """
        raise NotImplementedError("Use TensorDict RUL trajectory loss")
