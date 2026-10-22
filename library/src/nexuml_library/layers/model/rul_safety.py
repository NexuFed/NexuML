"""Cost-sensitive RUL regression without evaluation-target leakage."""

from __future__ import annotations

from typing import Any, cast

import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer


@layer("RULAsymmetricSquaredLoss")
class RULAsymmetricSquaredLoss(LayerDefinition):
    """Expectile-style loss increasing the cost of optimistic RUL predictions.

    A positive error means RUL is overestimated: predicted > actual.
    Unlike the exponential NASA score, weighted squared error has stable
    gradients and a transparent relative penalty.
    """

    overestimate_weight: float = Field(default=2.0, ge=1.0, le=20.0)

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        """Instantiate a regular native NexuML RUL objective.

        Returns:
            Per-batch asymmetric squared-error loss layer.
        """
        return _RULAsymmetricRuntime(
            **context.runtime_kwargs(),
            overestimate_weight=self.overestimate_weight,
        )


class _RULAsymmetricRuntime(PipelineLayer):
    """Compute asymmetric squared errors using supervised target TensorDicts."""

    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        overestimate_weight: float,
        label_key: str = "rul",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            input_sizes=input_sizes,
            keys_in=keys_in,
            keys_out=keys_out,
            label_key=label_key,
            **kwargs,
        )
        self.overestimate_weight = overestimate_weight

    def forward(
        self, x: TensorDict | Tensor, y: TensorDict | None = None
    ) -> tuple[TensorDict | Tensor, TensorDict | None]:
        """Write a native scalar loss with no access to held-out RUL values.

        Returns:
            Features with one loss value per batch observation.

        Raises:
            TypeError: If features are not supplied as a TensorDict.
            ValueError: If predictions and labels have incompatible shapes.
        """
        if not isinstance(x, TensorDict):
            raise TypeError("RUL safety loss expects a NexuML TensorDict")
        predicted = cast(Tensor, x[cast(list[str], self.keys_in)[0]])
        target = self.get_label(x, y)
        if target is None:
            loss = predicted.square().mean() * 0.0
        else:
            if predicted.ndim != 2 or predicted.shape != target.shape:
                raise ValueError("RUL safety loss needs matching [batch, 1] shapes")
            error = predicted - target
            weights = torch.where(
                error > 0,
                error.new_tensor(self.overestimate_weight),
                error.new_tensor(1.0),
            )
            loss = (weights * error.square()).mean()
        x[self.keys_out[0]] = loss.expand(x.batch_size)
        return x, y

    def forward_tensor(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Unused: this supervised loss requires TensorDict target labels."""
        raise NotImplementedError
