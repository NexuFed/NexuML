# ScenarioSpec Reference

`ScenarioSpec` is the top-level Pydantic model describing a NexuML experiment.

| Field | Type | Purpose |
|---|---|---|
| `name` | `str` | Scenario identifier |
| `pipeline` | `PipelineSpec` | Staged typed layers and graph wiring |
| `data` | `DataSpec` | Typed sources, splits, loader, targets, and preprocessing |
| `training` | `TrainingSpec` | Optimizer, scheduler, epochs, batch size, losses, and metrics |
| `evaluation` | `EvaluationSpec` | Typed post-training algorithms and result selection |
| `logging` | `LoggingSpec \| None` | Tracking and diagrams |
| `callbacks` | `list[CallbackSpec]` | Lightning callbacks |
| `tuning` | `TuningSpec \| None` | Tuning defaults |
| `checkpoint` | `CheckpointLoadSpec \| None` | Resume and fine-tune policy |
| `exports` | `list[ExportSpec]` | Requested artifacts |

## PipelineSpec And LayerSpec

`PipelineSpec.stages` is an ordered `dict[str, list[LayerSpec]]`.

```python
from nexuml.core.types import LayerSpec, PipelineSpec
from nexuml_library.layers.model.linear_encoder import LinearEncoder

pipeline = PipelineSpec(
    stages={
        "encode": [
            LayerSpec(
                component=LinearEncoder(hidden_dims=[32], output_dim=8),
                keys_in=["features"],
                keys_out=["latent"],
            )
        ]
    }
)
```

| `LayerSpec` field | Type | Purpose |
|---|---|---|
| `component` | `LayerDefinition` | Typed semantic layer configuration |
| `keys_in` | `list[str] \| dict[str, str]` | TensorDict inputs |
| `keys_out` | `list[str]` | TensorDict outputs |
| `label_key` | `str \| list[str] \| None` | Label routing |
| `label_in_x` | `bool` | Read labels from the input TensorDict |
| `meta_in`, `meta_out` | `dict[str, str] \| None` | Metadata wiring |
| `delay_epochs` | `int` | Runtime scheduling |
| `update_every_n_epochs` | `int` | Runtime scheduling |

Graph wiring and compiler values do not belong on the component definition.

An ordinary one-input/one-output PyTorch module can use the universal direct definition:

```python
import torch

from nexuml import nn_module

layer = LayerSpec(
    component=nn_module(torch.nn.Flatten, start_dim=1, end_dim=-1),
    keys_in=["image"],
    keys_out=["features"],
)
```

This path accepts exactly one input key, one output key, no label routing, and one tensor result. Use a registered `LayerDefinition` for richer NexuML semantics.

## DataSpec

```python
from nexuml.core.types import DataSpec, DatasetSpec, LoaderSpec
from nexuml.data.loaders.definitions import TorchLoader
from nexuml_library.data.synthetic import SyntheticDataset

data = DataSpec(
    source=SyntheticDataset(feature_shape=(64,), num_samples=1000),
    datasets=[DatasetSpec(source=SyntheticDataset(feature_shape=(64,), seed=7))],
    loader=LoaderSpec(backend=TorchLoader(), batch_size=64, num_workers=4),
    input_shapes={"features": [64]},
)
```

`source` and each `DatasetSpec.source` are `DataSourceDefinition` values. `LoaderSpec.backend` is a `LoaderBackendDefinition`; common policy such as batch size, worker count, shuffling, and weighted sampling remains on `LoaderSpec`.

`DataSpec` also preserves train/validation/test splits, targets, modalities, preprocessing, label merging, stage skipping, and class/input metadata.

## EvaluationSpec

```python
from nexuml.core.types import EvalAlgorithmSpec, EvaluationSpec
from nexuml_library.evaluation.anomalous_sound_detection.asd_evaluator import AnomalyEvaluator

evaluation = EvaluationSpec(
    algorithms=[
        EvalAlgorithmSpec(
            algorithm=AnomalyEvaluator(max_fpr=0.1),
            label_key="y_true",
        )
    ],
    test_result_metrics="all",
)
```

Algorithm semantics belong on `algorithm`. Placement fields such as `enabled`, `name`, `axis_keys`, `feature_key`, and `label_key` remain on `EvalAlgorithmSpec`.

## Persistence

`ResolvedConfig.from_scenario(spec).to_yaml()` lowers every component to stable `type`, `version`, and validated `params`. `ResolvedConfig.from_yaml()` restores concrete definitions through discovery and exact registry lookup. Registered semantic components do not persist Python module paths or runtime objects.

The universal `NnModule` component stores an external `module:name` factory target and JSON-safe constructor values in its `params`. Loading and compiling that trusted config imports and invokes the factory, so its package must be available. Live instances, lambdas, closures, local definitions, and nonportable constructor values are unsupported.

Removed selector fields are rejected rather than translated.

## Other Specs

- `TrainingSpec` configures optimization, scheduling, devices, precision, and logged loss/metric keys.
- `CallbackSpec` retains callback `type` and `params`; callbacks are not component-definition roles.
- `LoggingSpec` configures TensorBoard, DVCLive, MLflow, and diagrams.
- `CheckpointLoadSpec` configures selective loading and freezing.
- `ExportSpec` requests train packages, ONNX, or safetensors output.

## See Also

- [Scenarios](../learn/scenarios.md)
- [Architecture](../explanation/architecture.md)
- [API reference](api/nexuml/core/types.md)
