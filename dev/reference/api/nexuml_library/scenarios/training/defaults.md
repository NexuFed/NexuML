# nexuml_library.scenarios.training.defaults

*module*

Default training scenario fragments.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/scenarios/training/defaults.py#L1-L1)

## nexuml_library.scenarios.training.defaults.DEFAULT_AUTO_BATCH_SIZE

*attribute* · *module attribute*

```python
DEFAULT_AUTO_BATCH_SIZE = AutoBatchSizeSpec(min=8, max=128, candidates='power_of_two', safety='margin', margin=0.8)
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/scenarios/training/defaults.py#L31-L37)

## nexuml_library.scenarios.training.defaults.default_callbacks

*function*

```python
def default_callbacks() -> list[CallbackSpec]
```

Create a default list of CallbackSpec.

**Returns**

- (`list[CallbackSpec]`) — list[CallbackSpec]: Default callbacks for training.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/scenarios/training/defaults.py#L105-L124)

## nexuml_library.scenarios.training.defaults.default_checkpoint

*function*

```python
def default_checkpoint(path: str | None = None) -> CheckpointLoadSpec
```

Create a default CheckpointLoadSpec.

**Returns**

- `CheckpointLoadSpec` (`CheckpointLoadSpec`) — Default checkpoint loading configuration.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/scenarios/training/defaults.py#L127-L137)

## nexuml_library.scenarios.training.defaults.default_exports

*function*

```python
def default_exports(path: str | None = None) -> list[ExportSpec]
```

Create a default exports configuration.

**Returns**

- (`list[ExportSpec]`) — list[ExportSpec]: Default exports configuration.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/scenarios/training/defaults.py#L155-L163)

## nexuml_library.scenarios.training.defaults.default_logging

*function*

```python
def default_logging(experiment_name: str = 'NexuML', run_name: str | None = None, log_system_metrics: bool = False, use_tensorboard: bool = True, use_mlflow: bool = True, use_dvclive: bool = False) -> LoggingSpec
```

Create a default LoggingSpec.

**Returns**

- `LoggingSpec` (`LoggingSpec`) — Default logging configuration with optional TensorBoard, MLflow and DVC Live backends.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/scenarios/training/defaults.py#L66-L102)

## nexuml_library.scenarios.training.defaults.default_training

*function*

```python
def default_training(lr: float = 0.001, batch_size: BatchSizeSpec | None = 64, max_epochs: int = 10, loss_keys: dict[str, float] | None = None, metric_keys: list[str] | None = None, optimizer_factory: Callable[..., torch.optim.Optimizer] = torch.optim.Adam) -> TrainingSpec
```

Create a default TrainingSpec.

**Returns**

- `TrainingSpec` (`TrainingSpec`) — Default training configuration with optimizer, scheduler and loss keys.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/scenarios/training/defaults.py#L40-L63)

## nexuml_library.scenarios.training.defaults.default_tuning

*function*

```python
def default_tuning() -> TuningSpec
```

Create a default tuning configuration.

**Returns**

- `TuningSpec` (`TuningSpec`) — Default tuning configuration.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/scenarios/training/defaults.py#L140-L152)
