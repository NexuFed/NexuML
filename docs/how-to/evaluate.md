# Evaluate a model

Evaluation is part of the normal `nexuml train` lifecycle; there is no separate top-level `nexuml evaluate` command in the current CLI.

NexuML separates three concerns that are easy to mix together.

## 1. Pipeline losses and metrics

Pipeline layers can emit named TensorDict values such as `classification_loss`, `accuracy`, or `f1`. `TrainingSpec.loss_keys` selects/weights loss values; `TrainingSpec.metric_keys` selects pipeline metrics to log during train/validation/test.

These are part of the model pipeline itself.

## 2. Post-train fitted pipeline layers

Some score-producing components need the completed training set after gradient training. Such components use the `PostTrainFitLayer` lifecycle.

The canonical local session runs:

```text
fit
→ validate
→ fit each unfitted PostTrainFitLayer over the training loader
→ test
```

Fitted pipeline state can then produce values (for example anomaly scores) during the test pipeline pass.

## 3. Evaluation algorithms

`EvaluationSpec.algorithms` contains typed `EvalAlgorithmDefinition` values. They are reporting/analysis consumers of the test pipeline output rather than hidden score-producing model stages.

Example using the base-library anomaly evaluator:

```python
from nexuml.core.types import EvalAlgorithmSpec, EvaluationSpec
from nexuml_library.evaluation.anomalous_sound_detection.asd_evaluator import AnomalyEvaluator

evaluation = EvaluationSpec(
    algorithms=[
        EvalAlgorithmSpec(
            algorithm=AnomalyEvaluator(
                score_key="anomaly_score",
                max_fpr=0.1,
            ),
            label_key="y_true",
        )
    ],
    test_result_metrics=["auc", "pauc"],
)
```

During test, NexuML:

1. runs the compiled pipeline;
2. attaches declared/available evaluation metadata where required;
3. calls `eval_batch(x, y)` on each algorithm;
4. calls `eval_end()` after the test epoch;
5. calls `visualize(logger)`;
6. collects scalar values from `results()`.

The public definition contains immutable semantic configuration. Its private runtime owns mutable accumulators.

## Route inputs explicitly

`EvalAlgorithmSpec` owns placement/routing fields such as:

- `name` and `enabled`;
- `feature_key` and `label_key`;
- `axis_keys` for grouped evaluation.

Algorithm-specific values stay on the typed algorithm definition.

## Surface selected results

`EvaluationSpec.test_result_metrics` controls which evaluation scalars are mirrored into test results (`"none"`, `"all"`, or a list). This is useful when another workflow, such as tuning, needs a metric from the evaluation result set.

## Distributed execution

Ray currently rejects `evaluation.algorithms` because each worker would otherwise accumulate independent rank-local state. See [Ray execution](training-backends/ray.md).

## Custom algorithms

Use [Add a custom eval algorithm](custom-eval-algorithm.md) for the definition/runtime contract.
