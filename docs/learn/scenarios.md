# Scenarios

A scenario is the central abstraction in NexuML. It declares everything about an experiment — data, model architecture, training configuration, evaluation, and exports — in one composable Python object.

## What is a scenario?

A scenario is a Python function decorated with `@scenario("key")` that returns a `ScenarioSpec`:

```python
from nexuml.core.discovery import scenario
from nexuml.core.types import ScenarioSpec, DataSpec, PipelineSpec, LayerSpec, TrainingSpec

@scenario("my-classifier")
def my_classifier() -> ScenarioSpec:
    return ScenarioSpec(
        name="my_classifier",
        data=DataSpec(source_type="my-dataset"),
        pipeline=PipelineSpec(
            stages=[
                LayerSpec(
                    type_key="my-backbone",
                    keys_in={"x": "image"},
                    keys_out=["features"],
                ),
                LayerSpec(
                    type_key="classification-head",
                    keys_in={"x": "features"},
                    keys_out=["logits", "classification_loss"],
                ),
            ]
        ),
        training=TrainingSpec(max_epochs=10),
    )
```

After defining this function in an installed library, `nexuml registry list scenarios` will show `my-classifier`, and `nexuml train my-classifier` will run it.

## `ScenarioSpec` anatomy

### `data` — `DataSpec`

Declares the data source and split configuration.

```python
DataSpec(
    source_type="cifar10",   # resolves to a registered @data_source
    # Additional fields depend on the registered source
)
```

`source_type` is a registry key. The corresponding registered class or function provides the actual dataset.

### `pipeline` — `PipelineSpec`

An ordered list of `LayerSpec` entries. Each layer is resolved from the registry and executed in sequence.

```python
PipelineSpec(
    stages=[
        LayerSpec(
            type_key="resnet18",       # resolves to a registered @layer
            keys_in={"x": "image"},    # TensorDict key mapping: layer input ← pipeline key
            keys_out=["features"],     # TensorDict keys this layer writes
        ),
        LayerSpec(
            type_key="classification-head",
            keys_in={"x": "features"},
            keys_out=["logits", "classification_loss"],
        ),
    ]
)
```

`type_key` is a registry key. `keys_in` and `keys_out` define the TensorDict key contracts — NexuML validates these at `build` time.

### `training` — `TrainingSpec`

Optimizer, scheduler, max epochs, loss keys, and metric keys.

```python
TrainingSpec(
    max_epochs=10,
    loss_keys={"classification_loss": 1.0},  # weighted loss terms from TensorDict
    metric_keys=["accuracy", "f1"],
)
```

### `evaluation` — `EvaluationSpec`

Which evaluation algorithm to run at test time.

```python
EvaluationSpec(type="classification")
```

### `exports` — `list[ExportSpec]`

Optional list of export actions that run after training.

### `logging` — `LoggingSpec`

Diagram output, TensorBoard, and other logging configuration.

### `checkpoint` — `CheckpointLoadSpec`

Optional checkpoint to load before training (for resuming or fine-tuning).

### `tuning` — `TuningSpec`

Optuna search space for hyperparameter tuning via `nexuml tune`.

## Registered scenarios vs scenario files

### Registered scenario

A scenario registered with `@scenario("key")` is available by name to all CLI commands:

```bash
nexuml resolve my-classifier
nexuml train my-classifier --max-epochs=5
```

Use this when you want a reusable, discoverable scenario that lives in a library.

### Trusted Python scenario file (`--scenario-file`)

A local `.py` file that exposes a `scenario() -> ScenarioSpec` function can be passed directly to CLI commands:

```bash
nexuml train --scenario-file my_experiment.py
```

The file does not need to be in an installed package. Use this for local experiments and agent-authored scenarios. See [Trusted scenario files](../how-to/scenario-file.md) for details and security considerations.

## Lifecycle summary

```
@scenario("key")   →   nexuml resolve   →   configs/<key>.yaml
                   →   nexuml build     →   compiled pipeline (validation)
                   →   nexuml train     →   trained model + checkpoints
                   →   nexuml evaluate  →   metrics
                   →   nexuml export    →   model package
```

## Next steps

- [Define a scenario](../how-to/define-scenario.md) — step-by-step how-to
- [Trusted scenario files](../how-to/scenario-file.md) — local Python file workflow
- [Write a custom composed scenario](../how-to/custom-scenario.md) — full tutorial
- [Decorators and discovery](decorators-and-discovery.md) — how `@scenario` registers components
- [`ScenarioSpec` reference](../reference/scenario-spec.md) — all fields
- Generated API: [`nexuml.core.types`](../reference/api/nexuml/core/types.md)
