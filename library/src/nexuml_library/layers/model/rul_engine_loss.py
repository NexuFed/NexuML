"""Inverse-engine-window-frequency supervised RUL loss for frozen-feature studies.

Weights equal N_train / (N_engines * N_windows_engine), so the expected
sample-weighted training loss is an equally weighted mean of engine MSEs.
All validation and test weights remain exactly one.
"""

from __future__ import annotations

from typing import Any, cast

from tensordict import TensorDict
from torch import Tensor

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer


@layer("RULInverseEngineFrequencyLoss")
class RULInverseEngineFrequencyLoss(LayerDefinition):
    """Apply train-only engine-frequency reweighting to squared RUL error."""

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        """Build a batch-size-safe weighted MSE loss.

        Returns:
            Native NexuML loss layer.
        """
        return _RULInverseEngineFrequencyRuntime(**context.runtime_kwargs())


class _RULInverseEngineFrequencyRuntime(PipelineLayer):
    """Use prepared label weights; do not depend on minibatch engine counts."""

    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        **kwargs: Any,
    ) -> None:
        super().__init__(
            input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out,
            **kwargs,
        )

    def forward(
        self, x: TensorDict | Tensor, y: TensorDict | None = None
    ) -> tuple[TensorDict | Tensor, TensorDict | None]:
        """Calculate mean weighted RUL squared error over all batch windows.

        Returns:
            Features with a scalar-derived loss vector per batch sample.

        Raises:
            TypeError: If predictions are not stored in a TensorDict.
            ValueError: If targets or engine weights have incompatible shapes.
        """
        if not isinstance(x, TensorDict):
            raise TypeError("Engine-weighted RUL loss expects a TensorDict")
        predicted = cast(Tensor, x[cast(list[str], self.keys_in)[0]])
        if y is None:
            # Native compiler propagates shapes without supervised targets.
            loss = predicted.square().mean() * 0.0
        else:
            target = cast(Tensor, y["rul"])
            weights = cast(Tensor, y["rul_weight"])
            if (
                predicted.ndim != 2
                or predicted.shape != target.shape
                or weights.shape != target.shape
            ):
                raise ValueError(
                    "Engine-weighted RUL loss requires matching [batch,1] tensors"
                )
            if not bool((weights > 0).all()):
                raise ValueError("Engine weights must be strictly positive")
            loss = (weights * (predicted - target).square()).mean()
        x[self.keys_out[0]] = loss.expand(x.batch_size)
        return x, y

    def forward_tensor(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Unused because the loss must inspect paired engine-weight labels."""
        raise NotImplementedError
