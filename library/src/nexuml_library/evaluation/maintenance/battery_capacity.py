"""Battery capacity SMAPE training and held-out forecasting evaluation.

Following PatchFormer (2025) and RUL-Mamba (2025), targets are observed
future capacity, not inferred remaining life. SMAPE is unitless, MAE and
RMSE are in ampere-hours. EOL-RUL requires a separate trajectory rollout.
"""

from __future__ import annotations

import math
from typing import Any, cast

from pydantic import Field
from tensordict import TensorDict
from torch import Tensor

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import (
    EvalAlgorithmDefinition,
    EvalBuildContext,
    LayerBuildContext,
    LayerDefinition,
)
from nexuml.core.discovery import eval_algorithm, layer
from nexuml.evaluation.algorithm import EvalAlgorithm


@layer("BatteryCapacitySMAPE")
class BatteryCapacitySMAPE(LayerDefinition):
    """Paper-style symmetric percentage error on real capacity targets."""

    epsilon: float = Field(default=1e-6, gt=0)

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        """Instantiate a regular native NexuML capacity loss layer.

        Returns:
            TensorDict-based SMAPE loss layer.
        """
        return _BatterySMAPERuntime(
            **context.runtime_kwargs(), epsilon=self.epsilon
        )


class _BatterySMAPERuntime(PipelineLayer):
    """Gradient-carrying capacity SMAPE without hidden test-target access."""

    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        epsilon: float,
        label_key: str = "future_capacity",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out,
            label_key=label_key, **kwargs
        )
        self.epsilon = epsilon

    def forward(
        self, x: TensorDict | Tensor, y: TensorDict | None = None,
    ) -> tuple[TensorDict | Tensor, TensorDict | None]:
        """Compute 2*|prediction-target|/(|prediction|+|target|).

        Returns:
            Feature TensorDict containing a native scalar capacity loss.

        Raises:
            TypeError: If features are not presented as a TensorDict.
            ValueError: If prediction and label dimensions differ.
        """
        if not isinstance(x, TensorDict):
            raise TypeError("Capacity SMAPE requires tensor-dictionary features")
        prediction = cast(Tensor, x[cast(list[str], self.keys_in)[0]])
        target = self.get_label(x, y)
        if target is None:
            loss = prediction.square().mean() * 0.0
        else:
            if prediction.shape != target.shape:
                raise ValueError("Capacity prediction and target shapes disagree")
            denominator = (prediction.abs() + target.abs()).clamp_min(self.epsilon)
            loss = (2.0 * (prediction - target).abs() / denominator).mean()
        x[self.keys_out[0]] = loss.expand(x.batch_size)
        return x, y

    def forward_tensor(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Unused: labels are provided by the native NexuML data pipeline."""
        raise NotImplementedError


@eval_algorithm("BatteryCapacityEvaluator")
class BatteryCapacityEvaluator(EvalAlgorithmDefinition):
    """Evaluate one-step future battery capacity without calling it RUL."""

    def build(self, context: EvalBuildContext) -> EvalAlgorithm:
        """Instantiate capacity forecasting metrics.

        Returns:
            Sample-weighted MAE, RMSE and SMAPE evaluation algorithm.
        """
        return _BatteryCapacityEval(
            prediction_key=context.feature_key or "predicted_capacity",
            label_key=context.label_key or "future_capacity",
        )


class _BatteryCapacityEval(EvalAlgorithm):
    """Accumulate held-out physical Ah forecast errors per observation."""

    def __init__(self, *, prediction_key: str, label_key: str) -> None:
        self.prediction_key = prediction_key
        self.label_key = label_key
        self.count = 0
        self.mae = 0.0
        self.mse = 0.0
        self.smape = 0.0

    def eval_batch(self, x: TensorDict, y: TensorDict | None) -> None:
        """Accumulate next-cycle capacity errors over all held-out windows.

        Raises:
            ValueError: If labels are missing or prediction dimensions mismatch.
        """
        if y is None or self.label_key not in y.keys():
            raise ValueError("Capacity evaluation requires held-out capacity labels")
        prediction = cast(Tensor, x[self.prediction_key]).detach()
        target = cast(Tensor, y[self.label_key]).detach()
        if prediction.ndim != 2 or prediction.shape != target.shape:
            raise ValueError("Capacity evaluation requires matching [batch,1] shapes")
        error = prediction - target
        self.count += error.numel()
        self.mae += float(error.abs().sum())
        self.mse += float(error.square().sum())
        self.smape += float(
            (2.0 * error.abs() / (prediction.abs() + target.abs()).clamp_min(1e-6))
            .sum()
        )

    def results(self) -> dict[str, float]:
        """Return physical-unit forecast errors, not remaining useful life.

        Returns:
            Mean absolute Ah error, RMSE in Ah and unitless capacity SMAPE.

        Raises:
            ValueError: If the held-out capacity evaluation was empty.
        """
        if self.count == 0:
            raise ValueError("No held-out battery capacity predictions")
        return {
            "capacity_test_mae_ah": self.mae / self.count,
            "capacity_test_rmse_ah": math.sqrt(self.mse / self.count),
            "capacity_test_smape": self.smape / self.count,
            "capacity_test_count": float(self.count),
        }
