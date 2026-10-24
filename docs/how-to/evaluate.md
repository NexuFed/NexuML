# Evaluate a model

Use one command for trusted exported models and policies:

```bash
nexuml evaluate /approved/export
```

The export decides which existing evaluation path runs:

- **Model pipeline:** evaluate the `NexuDataset` test split using the saved data
  definition, pipeline losses/metrics and `EvaluationSpec.algorithms`.
- **Complete policy:** run episodes in the selected environment. This includes
  RL-trained and offline imitation policies; evaluation does not build a learner.

Neither path trains, creates an optimizer or replaces the exported weights. The
model path uses the packaged computation directly, including fitted pipeline state,
through the existing Lightning test hooks. Raw weights without a complete model
definition are not sufficient. Packages and Python scenario files are trusted
executable local code, not a sandbox.

## Select evaluation inputs with a Python file

Training and evaluation share the same `scenario() -> ScenarioSpec` file contract:

```bash
nexuml train --scenario-file experiment.py
nexuml evaluate /approved/export --scenario-file experiment.py
```

The evaluation file supplies **data/evaluation/logging settings** for a model, or
the **evaluation environment, optional `policy_evaluation`, and logging** for a
policy. Its pipeline, policy and training settings do not replace the exported
model. Without the flag, evaluation uses the definitions embedded in the export.
The selected environment is `interaction.evaluation_environment` when supplied,
otherwise `interaction.environment`.

For models, existing `test` metadata assignments are respected. Datasets without
split metadata use the existing seeded data-module split. A source that generates
new partitions when rebuilt must preserve its partition or use deterministic
splitting; evaluation cannot recover an unrecorded random training partition.
An empty test split is an error, not permission to use train/validation samples.
Cached preprocessing must already exist and is never overwritten or rebuilt by
evaluation; unfitted pipeline components are not fitted on test data.

## Overrides and device placement

```bash
nexuml evaluate /approved/model -O data.loader.batch_size=128
nexuml evaluate /approved/model -O evaluation.test_result_metrics=all
nexuml evaluate /approved/policy --device cuda:0 \
  -O training.evaluation.episodes=20 \
  -O training.evaluation.seed=123
```

Model overrides are limited to `data`, `evaluation` and `logging`. Policy overrides
retain the existing `policy_evaluation` or `training.evaluation` path, logging and
`interaction.evaluation_environment`; see [policy evaluation](reinforcement-learning.md).
In particular, native training recipes may save zero in-fit evaluation episodes:
request the desired episode count explicitly for standalone evaluation. Do not mix
the two policy-evaluation override namespaces. Both paths default to CPU; `--device`
selects evaluation placement without changing the saved training device.

Evaluation is also part of the normal `nexuml train` lifecycle. The following
sections describe the same model evaluation components used by standalone testing.

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
