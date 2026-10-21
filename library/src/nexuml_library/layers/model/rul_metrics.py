"""RUL regression metrics in physical target units for NexuML training/test."""

from __future__ import annotations

from typing import cast

import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer


@layer("RULRegressionMetrics")
class RULRegressionMetrics(LayerDefinition):
    """Log MAE/RMSE and asymmetric late-prediction penalty in original units."""

    target_scale: float = Field(gt=0)
    prediction_index: int = Field(default=0, ge=0)

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        """Build scalar metrics from a configured regression output.

        Returns:
            NexuML-compatible loss-stage metric layer.
        """
        return _RULMetricsRuntime(**context.runtime_kwargs(), **self.model_dump())


class _RULMetricsRuntime(PipelineLayer):
    """Compute descriptive metrics without contributing gradients to objective."""

    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        target_scale: float,
        prediction_index: int = 0,
        label_key: str = "rul",
        **kwargs: object,
    ) -> None:
        super().__init__(
            input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out,
            label_key=label_key, **kwargs,
        )
        self.target_scale = target_scale
        self.prediction_index = prediction_index

    def forward(
        self, x: TensorDict, y: TensorDict | None = None,
    ) -> tuple[TensorDict, TensorDict | None]:
        """Emit MAE, RMSE and positive late-prediction excess per batch.

        Returns:
            Existing feature/label pair with metric outputs. Errors have
            the original dataset's target unit (cycles or recordings).
        """
        preds = cast(Tensor, x[cast(list[str], self.keys_in)[0]])
        truth = self.get_label(x, y)
        if truth is None:
            values = (preds.sum() * 0,) * 3
        else:
            if preds.ndim != 2 or self.prediction_index >= preds.shape[-1]:
                raise ValueError("Invalid RUL prediction channel for metrics")
            error = (preds[:, self.prediction_index] - truth[:, 0]) * self.target_scale
            values = (
                error.abs().mean(),
                error.square().mean().sqrt(),
                error.clamp_min(0).mean(),
            )
        if len(self.keys_out) != len(values):
            raise ValueError("RUL metric layer requires exactly three output keys")
        for name, value in zip(self.keys_out, values):
            x[name] = value.detach().expand(x.batch_size)
        return x, y

    def forward_tensor(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Unused: this metric requires separate label TensorDicts."""
        raise NotImplementedError
