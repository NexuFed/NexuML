# Mental model

NexuML organises deep learning experiments around a central object called a `ScenarioSpec`. Understanding the lifecycle of a `ScenarioSpec` explains why every CLI command exists and what it does.

## The lifecycle

```
ScenarioSpec
    │
    │ nexuml resolve
    ▼
config YAML          ← reproducible, version-controllable
    │
    │ nexuml build
    ▼
compiled pipeline    ← layers instantiated, tensor contracts validated
    │
    │ nexuml train
    ▼
trained model        ← Lightning runs the loop, checkpoints written
    │
    │ nexuml evaluate (or inline during train)
    ▼
evaluation results   ← metrics, confusion matrix, etc.
    │
    │ nexuml export
    ▼
model package        ← portable, reloadable for inference
```

## What each step does

### `ScenarioSpec` — declare everything

A `ScenarioSpec` is a Python dataclass that holds every decision about an experiment:

- **`data`** (`DataSpec`) — where data comes from and how it is split.
- **`pipeline`** (`PipelineSpec`) — ordered sequence of `LayerSpec` entries.
- **`training`** (`TrainingSpec`) — optimizer, scheduler, max epochs, loss keys.
- **`evaluation`** (`EvaluationSpec`) — which evaluation algorithm to run.
- **`logging`** (`LoggingSpec`) — diagram output, TensorBoard, etc.
- **`exports`** (`list[ExportSpec]`) — what to export after training.
- **`checkpoint`** (`CheckpointLoadSpec`) — optional checkpoint to resume from.
- **`tuning`** (`TuningSpec`) — Optuna search space, if used.

A scenario is just a Python function decorated with `@scenario("key")` that returns a `ScenarioSpec`.

### `resolve` — compile to YAML

`nexuml resolve cifar-resnet` calls the scenario function, validates the resulting `ScenarioSpec`, and writes a YAML config to `configs/cifar-resnet.yaml`.

The YAML is the reproducible record of the experiment. Checking it into version control lets you reproduce training exactly.

### `build` — compile the pipeline

`nexuml build configs/cifar-resnet.yaml` reads the YAML, instantiates each `LayerSpec` into a concrete `PipelineLayer`, and validates tensor key contracts. This step catches shape mismatches and missing registry keys before any GPU time is spent.

### `train` — run the Lightning loop

`nexuml train cifar-resnet --max-epochs=2` compiles the pipeline, wraps it in a Lightning `LightningModule`, and hands it to a Lightning `Trainer`. NexuML configures data loaders from `DataSpec`, loss from `TrainingSpec`, and metrics from `EvaluationSpec`.

### `evaluate` — measure performance

Evaluation can run inline at the end of training (configured in `EvaluationSpec`) or separately. The evaluation algorithm is a registered component resolved by `EvalAlgorithmSpec.type`.

### `export` — package for inference

`nexuml export cifar-resnet --checkpoint <path>` loads the checkpoint, reconstructs the pipeline, and writes a self-contained package to `exported_model/`. The package can be reloaded without the original scenario code.

## Why this design?

- **Reproducibility:** The resolved YAML captures every hyperparameter, so experiments can be re-run from the file alone.
- **Composability:** `PipelineSpec` layers are independent, tested units. You can swap layers by changing `type_key`.
- **Separation of concerns:** Data, model, training, and evaluation are each their own spec. Changing the optimizer does not touch the layer definitions.
- **Framework integration:** The Lightning training loop is unchanged. NexuML adds the layer-contract system and discovery on top.

## Next

- [Scenarios](scenarios.md) — anatomy of `ScenarioSpec` fields and types
- [Coming from Lightning](from-lightning.md) — mapping NexuML concepts to Lightning
- [Train CIFAR ResNet](../start/train-cifar-resnet.md) — run the lifecycle yourself
