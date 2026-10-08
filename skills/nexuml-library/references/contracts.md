# Public component contracts (typed NexuML 0.2)

Inspect the selected installation before using these signatures. The bundled
starter contains runnable examples of datasets, loss layers, evaluators and
scenario composition without importing `nexuml_library`.

## Minimal package boundary

```text
user-project/
  pyproject.toml
  src/my_library/
    __init__.py
    data/
    layers/
    evaluation/
    scenarios/
```

This is an example, not a required reorganization. Entry-point values name an
importable package. Keep normal Python imports navigable; decorators supply
discovery/YAML identity, not string-based Python construction.

## Definitions and build methods

| Definition | Decorator | Build method / runtime |
| --- | --- | --- |
| `LayerDefinition` | `layer(name, version="1")` | `build(LayerBuildContext)` → `PipelineLayer` |
| `DataSourceDefinition` | `data_source(name, version="1")` | `build()` → `NexuDataset` |
| `EvalAlgorithmDefinition` | `eval_algorithm(name, version="1")` | `build(EvalBuildContext)` → `EvalAlgorithm` |
| `LoaderBackendDefinition` | `loader_backend(name, version="1")` | `build()` → installed loader runtime contract |

Definitions live in `nexuml.core.components`; decorators in
`nexuml.core.discovery`. Fields reject unknown parameters, are immutable and
cannot contain arbitrary live runtime objects. Use Pydantic constraints for
meaningful bounds; create a replacement definition instead of mutating one.

### Layers

Subclass `nexuml.core.base_layer.PipelineLayer` for NexuML-specific behavior.
Forward `**context.runtime_kwargs()` into the runtime constructor. The basic
`forward_tensor(x, y=None)` seam receives a routed tensor and optional label.
It is not a generic multi-input forward: for richer TensorDict routing, inspect
the actual base/runtime contract and existing user components first.

The compiler may call a layer without real labels during dummy shape
propagation. In this version it sets `_shape_propagation_mode`; a label-dependent
loss can return a shape-only placeholder in that mode. Keep missing-label errors
for real execution, and test both compilation and the real-label numerical path.
Verify this version-specific behavior before copying the starter's guard.

Ordinary modules can use `from nexuml import nn_module`. Its factory must be a
top-level importable class/function and arguments portable values. Live modules,
lambdas, closures and tensor/device objects are not portable constructor specs.

### Data

Subclass `nexuml.data.dataset.NexuDataset`. A sample returns
`(x: TensorDict, y: TensorDict | None)`, normally with `batch_size=[]`. For a
vector sample of width four, `x["features"].shape == (4,)`; batching adds the
leading sample dimension. Put label routing/targets and input shapes on `DataSpec`.

Partition membership needs actual evidence. `DataSpec` has `train_split`,
`val_split`, `test_split`, defaulting to 0.7/0.15/0.15. Ratios alone do not prove
official or group-aware splits. Metadata-backed datasets may supply pre-existing
assignments, so inspect the resulting loaders even when `test_split=0`.

### Evaluation and fitting

`nexuml.evaluation.algorithm.EvalAlgorithm` has optional `fit_batch`, `fit_end`,
`eval_batch`, `eval_end` and required `results() -> dict[str, float]`.
`EvalBuildContext` supplies `feature_key`/`label_key`; keep routing on
`EvalAlgorithmSpec`. These algorithms consume outputs, not generate new keys.

Score production is supported, but has a different home:

| Behavior | Placement |
| --- | --- |
| Stateless score from current tensors | Ordinary pipeline layer; write a distinct score key before its consumers. |
| Train-fitted score/calibration/decision | `LayerDefinition` building a `PostTrainFitLayer` runtime. |
| Dataset-wide AUC/RMSE/reporting | `EvalAlgorithmDefinition` in `EvaluationSpec.algorithms`; consume existing keys. |

Import `PostTrainFitLayer` from `nexuml.core.post_train_layer`. Declare
`requires_post_train_fit = True` on its definition so execution backends can
check support. Implement `collect_batch(x, y)`, `finalize_fit()` and
`_transform_forward(x, y)`; the last writes output keys and returns `(x, y)`.
Implement `_get_fit_state()` / `_set_fit_state(state)` for fitted checkpoint state.

The native local `NexuSession.run()` order is fit → validate → train-loader
post-fit passes → test. It arms each unfitted layer in pipeline order, collects
training batches through a predict pass, and finalizes on `on_predict_end()`.
Collection passes through without new keys; once fitted, transformation emits
the score/decision for downstream pipeline layers and then reporting evaluators.
Do not arm/refit on final-test data or invent an evaluation-algorithm producer chain.

Placement matters: a stage named `evaluation` is not an execution-phase switch.
An unfitted post-fit layer raises if training/validation reaches it; use a
supported frozen scoring scenario (often `max_epochs=0`) or an existing lifecycle
recipe, rather than appending it blindly to gradient training. Compiler dummy
passes also lack unfitted score keys: check downstream shape contracts and the
installed recipe instead of pretending compilation proves a fitted chain works.
Distributed backends currently reject these stateful phases; inspect capability
checks. A core-only install can implement the runtime in the user's library.

When installed, the base library's `AnomalyScore` is a stateless producer and
`DecisionRulePipelineLayer` is a post-fit producer; `AnomalyEvaluator` reports
their outputs. Verify those imports exist rather than assuming every scoring
class mentioned by older docs is still supplied.

Version-matched implementation evidence: `nexuml.core.post_train_layer`,
`NexuSession.run` / `_fit_post_train_layers`, and the installed decision-rule
runtime. Public explanation: https://nexufed.github.io/NexuML/how-to/evaluate/.

Define aggregation precisely: macro vs micro, per-group vs pooled, number of
elements vs samples, thresholds, score direction and fit/evaluation partitions.
Test unequal batch sizes and do not fit normalization/thresholds on final-test data.

## Scenarios and persistence

`@scenario(name)` marks an importable recipe returning `ScenarioSpec`. A trusted
file used with `--scenario-file` exposes `scenario() -> ScenarioSpec` without
requiring registration. Keep reusable factory modules in the importable library;
scenario-file loader module names are not stable import targets for reusable code.

Compose `LayerSpec(component=Definition(...), keys_in=[...], keys_out=[...])`
inside ordered `PipelineSpec(stages={...})`. `TrainingSpec.loss_keys` weights
produced loss tensors; `metric_keys` selects actual produced metrics.
`EvalAlgorithmSpec(algorithm=Definition(...), ...)` places an evaluator.

Persist using `ResolvedConfig.from_scenario(spec).to_yaml()`; restore through
`ResolvedConfig.from_yaml(text).to_scenario()`. Test in a fresh process where
the external library is installed/discoverable, without a source-checkout path.
