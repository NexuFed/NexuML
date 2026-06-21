# Evaluation

Evaluation in NexuML runs automatically at the end of `nexuml train`. Post-train pipeline layers are fitted on the training split, then evaluation algorithms score the test split.

## Prerequisites

- NexuML installed (`uv sync`)
- A scenario with evaluation algorithms configured

## Configuring evaluation

```python
from nexuml.core.types import ScenarioSpec, EvaluationSpec, EvalAlgorithmSpec

ScenarioSpec(
    name="my_scenario",
    evaluation=EvaluationSpec(
        metrics=["mse", "mae"],
        algorithms=[
            EvalAlgorithmSpec(
                type="knn1",
                feature_key="z",
                label_key="target",
            ),
        ],
        test_result_metrics="none",   # "none", "all", or list of metric names
    ),
    ...
)
```

## How evaluation works

1. Training completes and the best checkpoint is restored.
2. Layers implementing `PostTrainPipelineLayer` are fitted on the full training split (e.g. kNN fitting, GMM fitting).
3. The pipeline runs on the test split with `forward_until` semantics to extract features.
4. Each registered `EvalAlgorithm` in `evaluation.algorithms` scores the test samples.
5. Results are logged to MLflow and written to the configured eval storage backend.

## `EvalAlgorithmSpec` fields

| Field | Type | Description |
|---|---|---|
| `type` | `str` | Registered eval algorithm key (e.g. `"knn1"`, `"gmm"`) |
| `name` | `str \| None` | Display name in results (defaults to `type`) |
| `params` | `dict` | Parameters forwarded to the algorithm constructor |
| `enabled` | `bool` | Set `False` to skip this algorithm |
| `axis_keys` | `list[AxisKeySpec]` | Keys for grouping evaluation results |
| `feature_key` | `str \| None` | TensorDict key for input features |
| `label_key` | `str \| None` | TensorDict key for ground-truth labels |

## Eval storage backends

Results are stored using the eval-storage backend specified in `DistanceEstimatorSpec.storage_backend`:

| Backend | Description |
|---|---|
| `memory` | In-memory (fast, lost on exit) |
| `memmap` | Memory-mapped file (persists to disk) |

```python
from nexuml.core.types import DistanceEstimatorSpec

DistanceEstimatorSpec(
    storage_backend="memmap",
    storage_path=".experiments/eval_storage/",
)
```

## Surfacing eval metrics to tuning

By default, evaluation algorithm results are not logged as MLflow metrics accessible to Optuna. To surface specific metrics:

```python
EvaluationSpec(
    test_result_metrics=["omega", "auc"],   # logs these as MLflow metrics
    algorithms=[...],
)
```

Or to surface all evaluation metrics:

```python
EvaluationSpec(
    test_result_metrics="all",
    algorithms=[...],
)
```

This is required when `nexuml tune --metric` references an evaluation metric such as `omega`.

## Custom eval algorithms

Register a custom algorithm with `@eval_algorithm`:

```python
from nexuml.core.discovery import eval_algorithm

@eval_algorithm("my_eval")
class MyEval:
    def fit(self, features, labels):
        self.mean_ = features.mean(0)

    def score(self, features):
        return ((features - self.mean_) ** 2).sum(-1)
```

Use in a scenario:

```python
EvalAlgorithmSpec(type="my_eval", feature_key="z")
```

## Inspect available algorithms

```bash
nexuml registry list eval
```

## Implementation map

- `src/nexuml/evaluation/` — evaluation orchestration, storage, algorithm base
- `src/nexuml/core/types.py` — `EvaluationSpec`, `EvalAlgorithmSpec`, `DistanceEstimatorSpec`
- `src/nexuml/core/post_train_layer.py` — `PostTrainPipelineLayer` base class
- `src/nexuml/core/discovery.py` — `@eval_algorithm` decorator

## See also

- [Discovery decorators](../reference/decorators.md)
- [Optuna tuning](tune.md) — surfacing eval metrics via `test_result_metrics`
- [Tracking and logging](tracking.md)
