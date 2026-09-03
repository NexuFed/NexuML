"""Evaluation spec builders for anomaly detection scenarios."""

from __future__ import annotations

from typing import Literal, cast

from nexuml.core.types import AxisKeySpec, EvalAlgorithmSpec, EvaluationSpec, LayerSpec
from nexuml_library.evaluation.anomalous_sound_detection.asd_evaluator import AnomalyEvaluator
from nexuml_library.layers.head.decision_rule import DecisionRulePipelineLayer
from nexuml_library.layers.loss.classification_metrics import ClassificationMetrics


def _axis_spec_for_dcase2026_eval(key: str) -> AxisKeySpec:
    source = {
        "view": "x",
        "sample_index": "x",
        "basename": "metadata",
        "machine": "metadata",
        "domain": "metadata",
        "target": "metadata",
        "section": "metadata",
    }.get(key, "y")
    return AxisKeySpec(key=key, source=cast(Literal["x", "y", "metadata"], source))


def decision_rule_spec(
    score_key: str = "anomaly_score",
    decision_key: str = "decision",
    fit_mask_key: str | None = None,
    fit_label_key: str | None = None,
) -> list[LayerSpec]:
    """Return a decision-rule layer that produces *decision_key* from *score_key*."""
    return [
        LayerSpec(
            component=DecisionRulePipelineLayer(
                fit_mask_key=fit_mask_key,
                fit_label_key=fit_label_key,
            ),
            keys_in=[score_key],
            keys_out=[decision_key],
        ),
    ]


def classification_metrics_spec(
    score_key: str = "anomaly_score",
    decision_key: str = "decision",
    label_key: str = "anomaly",
    metrics: list[str] | None = None,
    fit_mask_key: str | None = None,
    fit_label_key: str | None = None,
) -> dict[str, list[LayerSpec]]:
    """Return a pipeline stage dict containing a decision rule and ``ClassificationMetrics``.

    A ``DecisionRulePipelineLayer`` is prepended so that the *decision* key is
    produced from *score_key* before ``ClassificationMetrics`` consumes it.
    """
    metric_names = metrics or ["accuracy", "f1"]
    layers: list[LayerSpec] = [
        *decision_rule_spec(
            score_key=score_key,
            decision_key=decision_key,
            fit_mask_key=fit_mask_key,
            fit_label_key=fit_label_key,
        ),
        LayerSpec(
            component=ClassificationMetrics(metrics=metric_names),
            keys_in=[decision_key],
            keys_out=metric_names,
            label_key=label_key,
        ),
    ]
    return {"Loss": layers}


def anomaly_evaluation_spec(
    label_key: str = "anomaly",
    group_keys: list[str] | None = None,
    output_score_key: str = "anomaly_score",
    decision_key: str | None = "decision",
    dcase_metric_axes: dict[str, str] | None = None,
    max_fpr: float = 0.1,
) -> EvaluationSpec:
    """Evaluation-only spec for anomalous sound detection.

    Returns only the reporting algorithm (anomaly_evaluator). Score-producing
    components (group_distance, calibration, reduction) are pipeline layers.

    Returns:
        EvaluationSpec: Evaluation specification with anomaly evaluator algorithm.
    """
    group_keys = group_keys or ["machine"]
    axis_key_specs = [_axis_spec_for_dcase2026_eval(k) for k in group_keys]
    axis_key_specs_plus_label: list[AxisKeySpec | str] = [
        *axis_key_specs,
        AxisKeySpec(key=label_key, source="y"),
    ]

    algorithms = [
        EvalAlgorithmSpec(
            algorithm=AnomalyEvaluator(
                score_key=output_score_key,
                decision_key=decision_key,
                group_keys=group_keys,
                dcase_metric_axes=dcase_metric_axes,
                max_fpr=max_fpr,
            ),
            name="anomaly_eval",
            axis_keys=axis_key_specs_plus_label,
            label_key=label_key,
        )
    ]
    return EvaluationSpec(
        metrics=["auc", "pauc_10"],
        test_result_metrics=["auc", "pauc_10", "omega"],
        algorithms=algorithms,
    )
