# nexuml_library.scenarios.evaluation.base

*module*

Evaluation scenario fragments.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/evaluation/base.py#L1-L1)

## nexuml_library.scenarios.evaluation.base.classification_evaluation

*function*

```python
def classification_evaluation(label_key: str = 'class', feature_key: str = 'latent', max_samples: int = 2000, metrics: list[str] = ['accuracy', 'f1']) -> EvaluationSpec
```

Evaluation spec for classification metrics.

**Returns**

- `EvaluationSpec` (`EvaluationSpec`) — Evaluation specification with classification metrics.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/evaluation/base.py#L60-L90)

## nexuml_library.scenarios.evaluation.base.merge_eval_specs

*function*

```python
def merge_eval_specs(eval_specs: list[EvaluationSpec]) -> EvaluationSpec
```

Merge a list of evaluation specs into a single combined spec.

**Returns**

- `EvaluationSpec` (`EvaluationSpec`) — Merged evaluation specification.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/evaluation/base.py#L11-L27)

## nexuml_library.scenarios.evaluation.base.reconstruction_evaluation

*function*

```python
def reconstruction_evaluation(reconstruction_feature_key: str = 'features', reconstruction_key: str = 'reconstructed', reconstruction_mask_key: str | None = None, reconstruction_patch_size: int | tuple[int, int] | None = None, reconstruction_label_keys: list[str] | None = None, metrics: list[str] = ['mse', 'mae']) -> EvaluationSpec
```

Evaluation spec for reconstruction metrics.

**Returns**

- `EvaluationSpec` (`EvaluationSpec`) — Evaluation specification with reconstruction metrics.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/evaluation/base.py#L30-L57)

## nexuml_library.scenarios.evaluation.base.regression_evaluation

*function*

```python
def regression_evaluation() -> EvaluationSpec
```

Evaluation spec for regression metrics.

**Returns**

- `EvaluationSpec` (`EvaluationSpec`) — Evaluation specification with regression metrics.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/evaluation/base.py#L93-L99)
