"""Decision rules: fit thresholds from train scores, emit binary decisions."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import ParamSpec

import torch
import torch.nn as nn

from nexuml.core.types import FactorySpec


class DecisionRule(nn.Module):
    """Abstract decision rule.

    Lifecycle:
      1. fit(train_scores) — fit threshold from normal train scores
      2. forward(scores) -> decisions — apply threshold to produce binary decisions
    """

    def fit(self, train_scores: torch.Tensor) -> None:
        """Fit the decision threshold from train-only scores."""
        raise NotImplementedError

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """Return 1 (anomalous) / 0 (normal) binary decisions."""
        raise NotImplementedError

    @property
    def threshold(self) -> float | None:
        buf = getattr(self, "_threshold", None)
        if isinstance(buf, torch.Tensor):
            v = float(buf.item())
            return None if v != v else v  # nan → None
        return buf


class DecisionRuleSpec(FactorySpec):
    """Portable constructor configuration for a decision rule."""


class QuantileThresholdRule(DecisionRule):
    """Threshold at a given quantile of the train score distribution.

    ``quantile`` in [0, 1]; e.g. 0.95 means the top-5% of train scores.
    """

    def __init__(self, quantile: float = 0.95) -> None:
        super().__init__()
        if not (0.0 <= quantile <= 1.0):
            raise ValueError(f"quantile must be in [0, 1], got {quantile}")
        self.quantile = quantile
        self.register_buffer("_threshold", torch.tensor(float("nan")))

    def fit(self, train_scores: torch.Tensor) -> None:
        scores = train_scores.float().cpu().flatten()
        self._threshold = torch.tensor(float(torch.quantile(scores, self.quantile)))

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        if self._threshold is None or (
            isinstance(self._threshold, torch.Tensor) and self._threshold.isnan()
        ):
            raise RuntimeError("QuantileThresholdRule.fit() must be called before forward().")
        threshold = self._threshold.to(device=scores.device, dtype=scores.dtype)
        return (scores.float().flatten() >= threshold).long()


class PercentileThresholdRule(DecisionRule):
    """Threshold at a given percentile of the train score distribution.

    ``percentile`` in [0, 100]; equivalent to QuantileThresholdRule with
    quantile = percentile / 100.
    """

    def __init__(self, percentile: float = 95.0) -> None:
        super().__init__()
        if not (0.0 <= percentile <= 100.0):
            raise ValueError(f"percentile must be in [0, 100], got {percentile}")
        self.percentile = percentile
        self.register_buffer("_threshold", torch.tensor(float("nan")))

    def fit(self, train_scores: torch.Tensor) -> None:
        scores = train_scores.float().cpu().flatten()
        self._threshold = torch.tensor(float(torch.quantile(scores, self.percentile / 100.0)))

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        if self._threshold is None or (
            isinstance(self._threshold, torch.Tensor) and self._threshold.isnan()
        ):
            raise RuntimeError("PercentileThresholdRule.fit() must be called before forward().")
        threshold = self._threshold.to(device=scores.device, dtype=scores.dtype)
        return (scores.float().flatten() >= threshold).long()


def _fit_gamma_percentile_threshold(train_scores: torch.Tensor, percentile: float) -> float:
    scores = train_scores.detach().float().cpu().flatten()
    if scores.numel() == 0:
        raise ValueError("GammaPercentileRule.fit() requires at least one score.")

    if scores.numel() == 1:
        return float(scores.item())

    offset = 0.0
    min_score = float(scores.min().item())
    if min_score <= 0.0:
        offset = -min_score + 1e-6

    shifted = (scores + offset).numpy()

    try:
        from scipy.stats import gamma as gamma_dist

        shape, loc, scale = gamma_dist.fit(shifted, floc=0.0)
        threshold = float(gamma_dist.ppf(percentile / 100.0, shape, loc=loc, scale=scale))
    except Exception:
        threshold = float(torch.quantile(scores, percentile / 100.0).item())
    else:
        threshold -= offset

    if not math.isfinite(threshold):
        threshold = float(torch.quantile(scores, percentile / 100.0).item())
    return threshold


class GammaPercentileRule(DecisionRule):
    """Gamma-fit threshold at a given percentile of train scores."""

    def __init__(self, percentile: float = 90.0) -> None:
        super().__init__()
        if not (0.0 <= percentile <= 100.0):
            raise ValueError(f"percentile must be in [0, 100], got {percentile}")
        self.percentile = percentile
        self.register_buffer("_threshold", torch.tensor(float("nan")))

    def fit(self, train_scores: torch.Tensor) -> None:
        threshold = _fit_gamma_percentile_threshold(train_scores, self.percentile)
        self._threshold = torch.tensor(float(threshold))

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        if self._threshold is None or (
            isinstance(self._threshold, torch.Tensor) and self._threshold.isnan()
        ):
            raise RuntimeError("GammaPercentileRule.fit() must be called before forward().")
        threshold = self._threshold.to(device=scores.device, dtype=scores.dtype)
        return (scores.float().flatten() >= threshold).long()


P = ParamSpec("P")


def decision_rule(
    factory: Callable[P, DecisionRule], *args: P.args, **kwargs: P.kwargs
) -> DecisionRuleSpec:
    """Configure an importable decision-rule factory.

    Returns:
        Portable decision-rule specification.
    """
    return DecisionRuleSpec.from_factory(factory, *args, **kwargs)
