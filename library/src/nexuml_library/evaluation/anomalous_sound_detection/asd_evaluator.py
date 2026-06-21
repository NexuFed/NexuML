"""Composable ASD reporting eval algorithms: AnomalyEvaluator and AnomalyVisualizer.

Score-producing components (GroupDistancePipelineLayer, ScoreCalibrationPipelineLayer,
ScoreReductionPipelineLayer, DecisionRulePipelineLayer) are now in
nexuml_library.layers.head as PostTrainFitLayer subclasses.
"""

from __future__ import annotations

import logging
from numbers import Integral, Real
import re
from typing import Any, cast

import torch
from tensordict import NonTensorData, TensorDict

from nexuml.core.discovery import eval_algorithm
from nexuml.evaluation.algorithm import EvalAlgorithm
from nexuml_library.evaluation.anomalous_sound_detection._metrics import (
    _binary_f1,
    compute_anomaly_metrics,
)

logger = logging.getLogger(__name__)

# Backward-compat alias for code that imports _compute_anomaly_metrics from here
_compute_anomaly_metrics = compute_anomaly_metrics


def _canonical_group_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[+-]?(?:0|[1-9]\d*)(?:\.\d+)?", text):
            numeric = float(text)
            return int(numeric) if numeric.is_integer() else numeric
        return text
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        return int(numeric) if numeric.is_integer() else numeric
    return value


def _axis_values_to_list(val: Any) -> list[Any]:
    """Coerce an axis-value (tensor / NonTensorData / list) to a flat Python list.

    Returns:
        Flat Python list with canonical group values.
    """
    if isinstance(val, list):
        return [_canonical_group_value(v) for v in val]
    if isinstance(val, NonTensorData):
        return [_canonical_group_value(v) for v in val.data]
    flat = val.detach().cpu().reshape(-1)
    return [_canonical_group_value(v.item() if hasattr(v, "item") else v) for v in flat]


# ---------------------------------------------------------------------------
# AnomalyEvaluator — reporting only, no score production
# ---------------------------------------------------------------------------


@eval_algorithm("anomaly_evaluator")
class AnomalyEvaluator(EvalAlgorithm):
    """Reporting algorithm: computes AUC, pAUC, and DCASE metrics.

    Reads anomaly_score from the pipeline output TensorDict. Does not produce
    any score keys — score production is handled by pipeline layers.
    """

    def __init__(
        self,
        score_key: str = "anomaly_score",
        label_key: str = "y_true",
        decision_key: str | None = None,
        group_keys: list[str] | None = None,
        dcase_metric_axes: dict[str, str] | None = None,
        max_fpr: float = 0.1,
    ) -> None:
        self.score_key = score_key
        self.label_key = label_key
        self.decision_key = decision_key
        self.group_keys = list(group_keys or [])
        self.dcase_metric_axes = dict(dcase_metric_axes or {})
        self.max_fpr = max_fpr
        self._scores: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []
        self._decisions: list[torch.Tensor] = []
        self._group_vals: dict[str, list[Any]] = {k: [] for k in self.group_keys}
        self._metrics: dict[str, Any] = {}

    def eval_batch(self, x: TensorDict, y: TensorDict | None) -> None:
        if self.score_key not in x.keys():
            return
        self._scores.append(x[self.score_key].float().cpu().flatten())
        if y is not None and self.label_key in y.keys():
            self._labels.append(y[self.label_key].float().cpu().flatten())
        if self.decision_key and self.decision_key in x.keys():
            self._decisions.append(cast(torch.Tensor, x[self.decision_key]).long().cpu().flatten())
        for key in set(self.group_keys) | set(self.dcase_metric_axes.values()):
            src = y if (y is not None and key in y.keys()) else x
            if key in src.keys():
                self._group_vals.setdefault(key, []).extend(_axis_values_to_list(src[key]))

    def eval_end(self) -> None:
        if not self._scores:
            return
        scores = torch.cat(self._scores)
        if not self._labels:
            self._metrics = {"score_mean": float(scores.mean()), "score_std": float(scores.std())}
            return
        labels = torch.cat(self._labels)
        valid_label_mask = torch.isfinite(labels) & ((labels == 0) | (labels == 1))
        if not valid_label_mask.any():
            self._metrics = {"score_mean": float(scores.mean()), "score_std": float(scores.std())}
            return

        self._metrics = _compute_anomaly_metrics(
            scores[valid_label_mask], labels[valid_label_mask], self.max_fpr
        )
        if self._decisions:
            decisions = torch.cat(self._decisions)
            if decisions.numel() == labels.numel():
                self._metrics["f1_decision"] = _binary_f1(
                    decisions[valid_label_mask], labels[valid_label_mask].long()
                )
        if {"machine", "section", "domain"}.issubset(self.dcase_metric_axes):
            try:
                from nexuml_library.evaluation.metrics.dcase2026 import compute_dcase_task2_metrics

                machines = torch.tensor(self._group_vals[self.dcase_metric_axes["machine"]])
                sections = torch.tensor(self._group_vals[self.dcase_metric_axes["section"]])
                domains = torch.tensor(self._group_vals[self.dcase_metric_axes["domain"]])
                if (
                    machines.shape[0]
                    == sections.shape[0]
                    == domains.shape[0]
                    == scores.shape[0]
                    == labels.shape[0]
                ):
                    dcase_m = compute_dcase_task2_metrics(
                        scores[valid_label_mask].numpy(),
                        labels[valid_label_mask].long().numpy(),
                        machines[valid_label_mask].numpy(),
                        sections[valid_label_mask].numpy(),
                        domains[valid_label_mask].numpy(),
                        max_fpr=self.max_fpr,
                    )
                    self._metrics.update(dcase_m)
            except Exception as exc:
                logger.warning("AnomalyEvaluator: DCASE metrics failed: %s", exc)

    def results(self) -> dict[str, Any]:
        return dict(self._metrics)


# ---------------------------------------------------------------------------
# AnomalyVisualizer — diagnostic plots, reporting only
# ---------------------------------------------------------------------------


@eval_algorithm("anomaly_visualizer")
class AnomalyVisualizer(EvalAlgorithm):
    """Diagnostic visualization consuming declared score, feature, and grouping axes."""

    def __init__(
        self,
        score_key: str = "anomaly_score",
        feature_key: str | None = "latent",
        label_key: str = "y_true",
        group_keys: list[str] | None = None,
        max_plot_samples: int = 2000,
    ) -> None:
        self.score_key = score_key
        self.feature_key = feature_key
        self.label_key = label_key
        self.group_keys = list(group_keys or [])
        self.max_plot_samples = max_plot_samples
        self._scores: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []
        self._features: list[torch.Tensor] = []

    def eval_batch(self, x: TensorDict, y: TensorDict | None) -> None:
        if self.score_key in x.keys():
            self._scores.append(x[self.score_key].float().cpu().flatten())
        if y is not None and self.label_key in y.keys():
            self._labels.append(y[self.label_key].float().cpu().flatten())
        if self.feature_key and self.feature_key in x.keys():
            feats = x[self.feature_key].float().cpu()
            if feats.dim() > 2:
                feats = feats.flatten(1)
            self._features.append(feats)

    def visualize(self, logger: Any) -> None:
        if not self._scores or logger is None:
            return
        scores = torch.cat(self._scores).numpy()
        labels = torch.cat(self._labels).long().numpy() if self._labels else None
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            if labels is not None:
                for cls, color, name in [(0, "blue", "normal"), (1, "red", "anomalous")]:
                    mask = labels == cls
                    ax.hist(scores[mask], bins=40, alpha=0.5, label=name, color=color)
                ax.legend()
            else:
                ax.hist(scores, bins=40)
            ax.set_xlabel(self.score_key)
            ax.set_title("Anomaly Score Distribution")
            log_img = getattr(logger, "log_image", None)
            if callable(log_img):
                log_img("anomaly_score_hist", fig)
            plt.close(fig)
        except Exception as exc:
            logger.warning("AnomalyVisualizer: plot failed: %s", exc) if hasattr(
                logger, "warning"
            ) else None

    def results(self) -> dict[str, float]:
        return {}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
