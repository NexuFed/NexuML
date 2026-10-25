# nexuml_library.scenarios.evaluation.anomaly

*module*

Evaluation spec builders for anomaly detection scenarios.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/evaluation/anomaly.py#L1-L1)

## nexuml_library.scenarios.evaluation.anomaly.anomaly_evaluation_spec

*function*

```python
def anomaly_evaluation_spec(label_key: str = 'anomaly', group_keys: list[str] | None = None, output_score_key: str = 'anomaly_score', decision_key: str | None = 'decision', dcase_metric_axes: dict[str, str] | None = None, max_fpr: float = 0.1) -> EvaluationSpec
```

Evaluation-only spec for anomalous sound detection.

Returns only the reporting algorithm (anomaly_evaluator). Score-producing
components (group_distance, calibration, reduction) are pipeline layers.

**Returns**

- `EvaluationSpec` (`EvaluationSpec`) — Evaluation specification with anomaly evaluator algorithm.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/evaluation/anomaly.py#L76-L117)

## nexuml_library.scenarios.evaluation.anomaly.classification_metrics_spec

*function*

```python
def classification_metrics_spec(score_key: str = 'anomaly_score', decision_key: str = 'decision', label_key: str = 'anomaly', metrics: list[str] | None = None, fit_mask_key: str | None = None, fit_label_key: str | None = None) -> dict[str, list[LayerSpec]]
```

Return a pipeline stage dict containing a decision rule and ``ClassificationMetrics``.

A ``DecisionRulePipelineLayer`` is prepended so that the *decision* key is
produced from *score_key* before ``ClassificationMetrics`` consumes it.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/evaluation/anomaly.py#L45-L73)

## nexuml_library.scenarios.evaluation.anomaly.decision_rule_spec

*function*

```python
def decision_rule_spec(score_key: str = 'anomaly_score', decision_key: str = 'decision', fit_mask_key: str | None = None, fit_label_key: str | None = None) -> list[LayerSpec]
```

Return a decision-rule layer that produces *decision_key* from *score_key*.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/evaluation/anomaly.py#L26-L42)
