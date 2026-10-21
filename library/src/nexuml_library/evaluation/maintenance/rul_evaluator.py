"""Exact, sample-weighted held-out RUL metrics via NexuML evaluation algorithms."""

from __future__ import annotations

import math
from typing import cast

from pydantic import Field
from tensordict import TensorDict
from torch import Tensor

from nexuml.core.components import EvalAlgorithmDefinition, EvalBuildContext
from nexuml.core.discovery import eval_algorithm
from nexuml.evaluation.algorithm import EvalAlgorithm


@eval_algorithm("RULBenchmarkEvaluator")
class RULBenchmarkEvaluator(EvalAlgorithmDefinition):
    """Evaluate normalized point/quantile RUL predictions in original units."""

    target_scale: float = Field(gt=0)
    point_index: int = Field(default=0, ge=0)
    lower_index: int | None = Field(default=None, ge=0)
    upper_index: int | None = Field(default=None, ge=0)

    def build(self, context: EvalBuildContext) -> EvalAlgorithm:
        """Create a streaming test metric accumulator.

        Returns:
            Unbiased sample-weighted MAE/RMSE and optional interval statistics.
        """
        return _RULBenchmarkRuntime(
            prediction_key=context.feature_key or "predicted_rul",
            label_key=context.label_key or "rul",
            target_scale=self.target_scale,
            point_index=self.point_index,
            lower_index=self.lower_index,
            upper_index=self.upper_index,
        )


class _RULBenchmarkRuntime(EvalAlgorithm):
    """Stream residual statistics without storing per-sample evaluation data."""

    def __init__(
        self,
        *,
        prediction_key: str,
        label_key: str,
        target_scale: float,
        point_index: int,
        lower_index: int | None,
        upper_index: int | None,
    ) -> None:
        self.prediction_key = prediction_key
        self.label_key = label_key
        self.target_scale = target_scale
        self.point_index = point_index
        self.lower_index = lower_index
        self.upper_index = upper_index
        self.count = 0
        self.absolute = 0.0
        self.square = 0.0
        self.late = 0.0
        self.covered = 0
        self.interval_width = 0.0

    def eval_batch(self, x: TensorDict, y: TensorDict | None) -> None:
        """Accumulate exact errors from a held-out batch without fitting.

        Raises:
            ValueError: If required prediction/label columns are missing.
        """
        if y is None or self.label_key not in y.keys():
            raise ValueError("RUL test evaluator requires labeled held-out observations")
        predicted = cast(Tensor, x[self.prediction_key]).detach().float()
        truth = cast(Tensor, y[self.label_key]).detach().float()
        if (
            predicted.ndim != 2
            or truth.shape != (len(predicted), 1)
            or self.point_index >= predicted.shape[1]
        ):
            raise ValueError("RUL predicted/target test shapes disagree")
        error = (
            predicted[:, self.point_index] - truth[:, 0]
        ).mul(self.target_scale)
        self.count += len(error)
        self.absolute += float(error.abs().sum())
        self.square += float(error.square().sum())
        self.late += float(error.clamp_min(0).sum())
        if self.lower_index is not None and self.upper_index is not None:
            if max(self.lower_index, self.upper_index) >= predicted.shape[1]:
                raise ValueError("RUL prediction interval indices exceed output channels")
            low = predicted[:, self.lower_index]
            high = predicted[:, self.upper_index]
            if bool((low > high).any()):
                # Quantile crossing is an evaluation failure, not a valid interval.
                raise ValueError("Predicted RUL lower quantile exceeds upper quantile")
            self.covered += int(((truth[:, 0] >= low) & (truth[:, 0] <= high)).sum())
            self.interval_width += float(((high - low) * self.target_scale).sum())

    def results(self) -> dict[str, float]:
        """Report held-out MAE, RMSE and optionally interval calibration.

        Returns:
            Exact asset-window-weighted metrics in cycles or recordings.
        """
        if self.count == 0:
            raise ValueError("RUL benchmark evaluator received zero test samples")
        result = {
            "rul_test_mae": self.absolute / self.count,
            "rul_test_rmse": math.sqrt(self.square / self.count),
            "rul_test_late_bias": self.late / self.count,
            "rul_test_count": float(self.count),
        }
        if self.lower_index is not None and self.upper_index is not None:
            result["rul_test_interval_coverage"] = self.covered / self.count
            result["rul_test_interval_width"] = self.interval_width / self.count
        return result
