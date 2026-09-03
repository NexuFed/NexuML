# Scenario and configuration reference

`ScenarioSpec` is the Pydantic composition root for one NexuML experiment.

## Top-level sections

| Field | Responsibility |
| --- | --- |
| `name` | scenario identifier |
| `pipeline` | ordered `PipelineSpec` stages of `LayerSpec` entries |
| `training` | optimizer/scheduler, losses/metrics, epochs, batch size, accelerator/devices/strategy/precision |
| `data` | typed source(s), split policy, loader, shapes/classes, preprocessing/materialization |
| `evaluation` | post-training evaluation algorithms and result selection |
| `logging` | TensorBoard/MLflow/DVCLive/diagram configuration or `None` |
| `callbacks` | Lightning callback specs |
| `tuning` | Optuna run defaults or `None` |
| `checkpoint` | selective/pretrained weight-loading policy or `None` |
| `exports` | requested model artifacts |
| `execution` | local or Ray placement |

Use the generated [`nexuml.core.types` API](api/nexuml/core/types.md) for exact field types/defaults. The table above explains ownership rather than duplicating a second hand-maintained schema.

## Pipeline and components

Python scenarios store typed definitions directly:

```python
LayerSpec(
    component=LinearEncoder(hidden_dims=[32], output_dim=8),
    keys_in=["features"],
    keys_out=["embedding"],
)
```

`LayerSpec` owns graph wiring/routing/scheduling. `LinearEncoder(...)` owns component-specific semantic configuration.

For an ordinary importable one-input/one-output tensor module, use the generic definition:

```python
from nexuml import nn_module
import torch

LayerSpec(
    component=nn_module(torch.nn.Dropout, p=0.5),
    keys_in=["embedding"],
    keys_out=["regularized"],
)
```

Use a registered `LayerDefinition` when the component needs richer NexuML build context, labels/metadata, multiple inputs/outputs, lifecycle behavior, or other semantic integration.

## Data and loaders

`DataSpec.source` and each `DatasetSpec.source` are typed `DataSourceDefinition` values. `LoaderSpec.backend` is a typed `LoaderBackendDefinition`.

```python
from nexuml.core.types import DataSpec, LoaderSpec
from nexuml.data.loaders.definitions import TorchLoader

DataSpec(
    source=MyDataset(...),
    loader=LoaderSpec(backend=TorchLoader()),
)
```

!!! note "Current 0.2 loader default"
    The core `LoaderSpec` default factory creates `TorchLoader()`. Select `DaliLoader()` or `TensorShardsLoader()` explicitly when a scenario requires either specialized backend.

An explicit `LoaderSpec.batch_size` overrides `TrainingSpec.batch_size`. Leave it `None` to defer to the training batch size (including automatic batch-size probing).

## Evaluation

`EvalAlgorithmSpec.algorithm` contains a typed `EvalAlgorithmDefinition`. Placement/routing (`name`, `enabled`, axis keys, feature/label keys) stays on the surrounding spec.

Evaluation algorithms consume test outputs; score-producing stateful model behavior belongs in the pipeline (for example a `PostTrainFitLayer`).

## Checkpoints

`CheckpointLoadSpec` is **selective weight reuse**, not the Lightning resume mechanism. Full trainer resume is requested with `nexuml train --trainer-checkpoint PATH`.

See [Checkpoints](../how-to/checkpoints.md).

## Execution

`ScenarioSpec.execution` is a discriminated union:

- `LocalExecutionSpec` — current-process execution (default);
- `RayExecutionSpec` — existing-Ray-cluster placement/resources.

Training semantics remain in `TrainingSpec` in both cases.

## Persistence

`ResolvedConfig.from_scenario(...).to_yaml()` lowers registered definitions to stable component identity + validated parameter data. Restoration discovers the definition and validates it again.

Registered semantic definitions do not persist mutable runtime objects. `NnModule` is the explicit external-factory exception and stores an importable factory target plus JSON-safe constructor values.

Removed legacy selector fields are rejected rather than silently translated.
