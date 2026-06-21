# ScenarioSpec reference

`ScenarioSpec` is the top-level dataclass that describes a complete NexuML pipeline. It is a Pydantic model (`SpecModel`) and can be validated, dumped to YAML, and reloaded.

## Top-level fields

```python
from nexuml.core.types import ScenarioSpec

ScenarioSpec(
    name="my-scenario",
    pipeline=PipelineSpec(...),
    data=DataSpec(...),
    training=TrainingSpec(...),
    evaluation=EvaluationSpec(...),
    logging=LoggingSpec(...),
    callbacks=[CallbackSpec(...)],
    tuning=TuningSpec(...),
    checkpoint=CheckpointLoadSpec(...),
    exports=[ExportSpec(...)],
)
```

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | `str` | yes | Scenario identifier |
| `pipeline` | `PipelineSpec` | no | Staged `LayerSpec` list |
| `data` | `DataSpec` | no | Data source, splits, loader, targets |
| `training` | `TrainingSpec` | no | Optimizer, scheduler, epochs, batch size |
| `evaluation` | `EvaluationSpec` | no | Metrics and eval algorithms |
| `logging` | `LoggingSpec` | no | TensorBoard / DVCLive / MLflow / diagrams |
| `callbacks` | `list[CallbackSpec]` | no | Lightning callbacks |
| `tuning` | `TuningSpec` | no | Defaults for `nexuml tune` |
| `checkpoint` | `CheckpointLoadSpec` | no | Resume / fine-tune policy |
| `exports` | `list[ExportSpec]` | no | Artifacts to produce after training |

## `PipelineSpec`

```python
from nexuml.core.types import PipelineSpec, LayerSpec

PipelineSpec(stages={
    "encode": [
        LayerSpec(
            type_key="linear_encoder",
            keys_in=["features"],
            keys_out=["z"],
            params={"input_dim": 64, "output_dim": 8},
        ),
    ],
    "loss": [
        LayerSpec(
            type_key="reconstruction_loss",
            keys_in=["z", "features"],
            keys_out=["reconstruction_loss"],
            params={"input_dim": 8},
        ),
    ],
})
```

| Field | Type | Description |
|---|---|---|
| `stages` | `dict[str, list[LayerSpec]]` | Ordered stages, each a list of layers |

## `LayerSpec`

```python
LayerSpec(
    type_key="linear_encoder",
    keys_in=["features"],
    keys_out=["z"],
    params={"input_dim": 64, "output_dim": 8},
    meta_in=None,
    meta_out=None,
)
```

| Field | Type | Required | Description |
|---|---|---|---|
| `type_key` | `str` | yes | Registered layer key |
| `keys_in` | `list[str] \| dict[str, str]` | yes | Input TensorDict keys |
| `keys_out` | `list[str]` | yes | Output TensorDict keys |
| `params` | `dict[str, Any]` | no | Constructor arguments for the layer |
| `meta_in` | `dict[str, str] \| None` | no | Metadata input mapping |
| `meta_out` | `dict[str, str] \| None` | no | Metadata output mapping |

## `DataSpec`

```python
from nexuml.core.types import DataSpec, DatasetSpec, LoaderSpec, TargetSpec

DataSpec(
    source_type="synthetic",
    params={"feature_shape": [64], "num_samples": 1000},
    targets=[TargetSpec(type="regression", key="target")],
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    datasets=[DatasetSpec(type_key="my_dataset", params={"root": "/data"})],
    loader=LoaderSpec(backend="dali", batch_size=64, num_workers=4),
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `source_type` | `str` | `"synthetic"` | Registered data-source key |
| `params` | `dict[str, Any]` | `{}` | Data-source constructor args |
| `targets` | `list[TargetSpec]` | `[]` | Target definitions |
| `train_split` | `float` | `0.7` | Training fraction |
| `val_split` | `float` | `0.15` | Validation fraction |
| `test_split` | `float` | `0.15` | Test fraction |
| `datasets` | `list[DatasetSpec]` | `[]` | Additional dataset specs |
| `loader` | `LoaderSpec` | `LoaderSpec()` | DataLoader configuration |
| `input_shapes` | `dict[str, list[int]]` | `{}` | Named input shapes |
| `num_classes` | `int \| None` | `None` | Number of classes |
| `merge_labels` | `dict[str, dict] \| None` | `None` | Label merging rules |
| `feature_key` | `str` | `"features"` | Default feature key |
| `skip_pipeline_stages` | `list[str]` | `[]` | Stages to skip |
| `preprocessing` | `PreprocessingSpec` | `PreprocessingSpec()` | Optional preprocessing stage |

## `TrainingSpec`

```python
from nexuml.core.types import TrainingSpec, OptimizerSpec, SchedulerSpec

TrainingSpec(
    optimizer=OptimizerSpec(type="torch.optim.Adam", params={"lr": 1e-3}),
    scheduler=SchedulerSpec(type="torch.optim.lr_scheduler.ConstantLR"),
    loss_keys={"reconstruction_loss": 1.0},
    metric_keys=["train/loss", "val/loss"],
    progress_bar_keys=["loss"],
    max_epochs=10,
    batch_size=64,
    lr=1e-3,
    accelerator="auto",
    devices="auto",
    strategy="auto",
    precision="32-true",
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `optimizer` | `OptimizerSpec` | `OptimizerSpec()` | Optimizer config |
| `scheduler` | `SchedulerSpec` | `SchedulerSpec()` | LR scheduler config |
| `loss_keys` | `dict[str, float]` | `{}` | TensorDict loss keys and weights |
| `metric_keys` | `list[str]` | `[]` | Metrics to log |
| `progress_bar_keys` | `list[str] \| "all" \| "none"` | `["loss"]` | Keys shown in progress bar |
| `max_epochs` | `int` | `10` | Training epochs |
| `batch_size` | `int \| AutoBatchSizeSpec` | `64` | Batch size or auto batch-size spec |
| `lr` | `float` | `1e-3` | Learning rate (convenience field) |
| `accelerator` | `str` | `"auto"` | Lightning accelerator |
| `devices` | `str \| int` | `"auto"` | Lightning devices |
| `strategy` | `str` | `"auto"` | Lightning strategy |
| `precision` | `str \| int` | `"32-true"` | Lightning precision |

## `EvaluationSpec`

```python
from nexuml.core.types import EvaluationSpec, EvalAlgorithmSpec

EvaluationSpec(
    metrics=["mse", "mae"],
    algorithms=[EvalAlgorithmSpec(type="knn1", feature_key="z")],
    test_result_metrics=["omega"],
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `metrics` | `list[str]` | `["mse", "mae"]` | Built-in metric names |
| `algorithms` | `list[EvalAlgorithmSpec]` | `[]` | Post-training eval algorithms |
| `test_result_metrics` | `list[str] \| "all" \| "none"` | `"none"` | Eval metrics to surface as logged metrics |

## `LoggingSpec`

```python
from nexuml.core.types import LoggingSpec, TensorBoardSpec, MLflowSpec, DVCLiveSpec, DiagramSpec

LoggingSpec(
    tensorboard=TensorBoardSpec(log_dir=".experiments/tensorboard"),
    mlflow=MLflowSpec(tracking_uri="file:./.experiments/mlflow"),
    diagram=DiagramSpec(enabled=True, depth=2),
    experiment_name="NexuML",
    run_name="my-run",
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `tensorboard` | `TensorBoardSpec \| None` | `None` | TensorBoard logger |
| `dvclive` | `DVCLiveSpec \| None` | `None` | DVCLive logger |
| `mlflow` | `MLflowSpec \| None` | `None` | MLflow logger |
| `diagram` | `DiagramSpec \| None` | `DiagramSpec()` | Mermaid diagram output |
| `temp_artifact_dir` | `str` | `"/tmp/mlruns"` | Temp artifact directory |
| `experiment_name` | `str` | `"NexuML"` | Experiment name |
| `run_name` | `str \| None` | `None` | Run name |
| `log_system_metrics` | `bool` | `False` | Log system metrics |

## `CallbackSpec`

```python
from nexuml.core.types import CallbackSpec

CallbackSpec(type="checkpoint", params={"monitor": "val/loss", "mode": "min"})
```

| Field | Type | Required | Description |
|---|---|---|---|
| `type` | `str` | yes | Callback alias or dotted import path |
| `params` | `dict[str, Any]` | `{}` | Constructor arguments |

### Callback aliases

| Alias | Typical Lightning callback |
|---|---|
| `checkpoint` | `lightning.pytorch.callbacks.ModelCheckpoint` |
| `lr_monitor` | `lightning.pytorch.callbacks.LearningRateMonitor` |
| `early_stopping` | `lightning.pytorch.callbacks.EarlyStopping` |
| `rich_progress` | `lightning.pytorch.callbacks.RichProgressBar` |
| `device_stats` | `lightning.pytorch.callbacks.DeviceStatsMonitor` |

Dotted paths are also accepted, for example `my_package.callbacks.MyCallback`.

## `TuningSpec`

```python
from nexuml.core.types import TuningSpec

TuningSpec(
    n_trials=50,
    directions=["minimize"],
    metric_key="val/loss",
    storage=".experiments/optuna/optuna.log",
    prune=False,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `n_trials` | `int` | `50` | Number of trials |
| `directions` | `list[str]` | `["minimize"]` | `minimize` or `maximize` per objective |
| `metric_key` | `str` | `"val/loss"` | Metric to optimize |
| `storage` | `str` | `".experiments/optuna/optuna.log"` | Optuna storage path |
| `prune` | `bool` | `False` | Enable pruning |

## `CheckpointLoadSpec`

```python
from nexuml.core.types import CheckpointLoadSpec

CheckpointLoadSpec(
    source=".experiments/checkpoints/my-scenario/last.ckpt",
    include=["encoder.*"],
    exclude=[],
    allow_missing=True,
    allow_shape_mismatch=True,
    freeze_loaded=False,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `source` | `str \| None` | `None` | Checkpoint path |
| `include` | `list[str]` | `[]` | Patterns to include |
| `exclude` | `list[str]` | `[]` | Patterns to exclude |
| `allow_missing` | `bool` | `True` | Allow missing keys |
| `allow_shape_mismatch` | `bool` | `True` | Allow shape mismatches |
| `freeze_loaded` | `bool` | `False` | Freeze loaded parameters |

## `ExportSpec`

```python
from nexuml.core.types import ExportSpec

ExportSpec(kind="train_package", output="exported_model")
```

| Field | Type | Default | Description |
|---|---|---|---|
| `kind` | `"train_package" \| "onnx" \| "safetensors"` | `"train_package"` | Export kind |
| `output` | `str \| None` | `None` | Output path |
| `include` | `list[str]` | `[]` | Include patterns |
| `exclude` | `list[str]` | `[]` | Exclude patterns |

## See also

- [Define a scenario](../how-to/define-scenario.md)
- [Run scenarios](../how-to/run-scenarios.md)
- [Add a custom layer](../how-to/custom-layer.md)
- [Optuna tuning](../how-to/tune.md)
- [API reference](api/nexuml/core/types.md)
