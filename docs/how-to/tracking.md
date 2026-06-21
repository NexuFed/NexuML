# Tracking and logging

NexuML integrates with TensorBoard, DVCLive, and MLflow for experiment tracking. Configure loggers in `LoggingSpec` inside a `ScenarioSpec`.

## Prerequisites

- NexuML installed (`uv sync`)
- Optional: MLflow UI for visualization (`pip install mlflow`)
- Optional: DVCLive (`pip install dvclive`)

## Configure loggers

```python
from nexuml.core.types import (
    ScenarioSpec, LoggingSpec,
    TensorBoardSpec, DVCLiveSpec, MLflowSpec,
)

ScenarioSpec(
    name="my_scenario",
    logging=LoggingSpec(
        experiment_name="MyProject",
        run_name="experiment_v1",
        log_system_metrics=False,
        temp_artifact_dir="/tmp/mlruns",

        tensorboard=TensorBoardSpec(
            log_dir=".experiments/tensorboard",
        ),
        mlflow=MLflowSpec(
            tracking_uri="sqlite:///.experiments/mlflow.db",
            experiment_name="MyProject",        # defaults to LoggingSpec.experiment_name
            run_id=None,                         # resume an existing run by ID
            parent_run_id=None,                  # nest under a parent run
            auto_nested_runs=True,               # auto-create nested runs for tuning
            log_model=False,
            tags={},
            synchronous=None,
        ),
        dvclive=DVCLiveSpec(
            dir=".experiments/dvclive",
        ),
    ),
    ...
)
```

## `LoggingSpec` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `experiment_name` | `str` | `"NexuML"` | Experiment name for all backends |
| `run_name` | `str \| None` | `None` | Run name (auto-generated if `None`) |
| `log_system_metrics` | `bool` | `False` | Log CPU/GPU/memory metrics to MLflow |
| `temp_artifact_dir` | `str` | `"/tmp/mlruns"` | Temp directory for MLflow artifact staging |
| `tensorboard` | `TensorBoardSpec \| None` | `None` | TensorBoard config (`None` disables) |
| `mlflow` | `MLflowSpec \| None` | `None` | MLflow config (`None` disables) |
| `dvclive` | `DVCLiveSpec \| None` | `None` | DVCLive config (`None` disables) |
| `diagram` | `DiagramSpec \| None` | `DiagramSpec()` | Mermaid diagram config |

## TensorBoard

```python
TensorBoardSpec(log_dir=".experiments/tensorboard")
```

View logs:

```bash
tensorboard --logdir .experiments/tensorboard
```

## MLflow

```python
MLflowSpec(
    tracking_uri="sqlite:///.experiments/mlflow.db",
    experiment_name="MyProject",
    log_model=False,
)
```

Supported tracking URIs:

| Format | Example |
|---|---|
| Local SQLite | `sqlite:///.experiments/mlflow.db` |
| Local file store | `file:./.experiments/mlflow` |
| Remote server | `http://mlflow.example.com` |

### Service info

```bash
# Start local MLflow UI
mlflow ui --backend-store-uri sqlite:///.experiments/mlflow.db
# Open http://127.0.0.1:5000
```

### Run metadata artifacts

Each training run logs:

- Hyperparameters from `ScenarioSpec`
- Scalar metrics at each epoch (loss, val/loss, custom metric keys)
- Model artifact (if `log_model=True`)
- Scenario config YAML

### Nested runs for tuning

When `auto_nested_runs=True` (default), each Optuna trial is a nested MLflow run under the parent study run. The best trial's parameters are promoted to the parent run.

## DVCLive

```python
DVCLiveSpec(dir=".experiments/dvclive")
```

DVCLive logs metrics as JSON/CSV files and optionally generates HTML reports. Useful for DVC-based experiment tracking.

## Disable all logging

```python
LoggingSpec()  # tensorboard=None, mlflow=None, dvclive=None by default
```

Or set the entire `logging` field to `None` in the scenario:

```python
ScenarioSpec(name="my_scenario", logging=None, ...)
```

## Logging without a scenario file

The default logging (when `ScenarioSpec.logging` is `None`) uses no external loggers. Add a `LoggingSpec` to enable them.

The library `default_logging()` helper creates a standard `LoggingSpec` with TensorBoard and MLflow enabled:

```python
from nexuml_library.scenarios.training.defaults import default_logging
from nexuml.core.types import ScenarioSpec

ScenarioSpec(
    name="my_scenario",
    logging=default_logging(
        experiment_name="MyProject",
        use_tensorboard=True,
        use_mlflow=True,
        use_dvclive=False,
    ),
    ...
)
```

## Implementation map

- `src/nexuml/tracking/logger.py` — TensorBoard, DVCLive, MLflow backend implementations
- `src/nexuml/core/types.py` — `LoggingSpec`, `TensorBoardSpec`, `DVCLiveSpec`, `MLflowSpec`
- `src/nexuml/training/lightning.py` — logger attachment to Lightning `Trainer`
- `library/src/nexuml_library/scenarios/training/defaults.py` — `default_logging()`

## See also

- [Environment roots](../reference/environment.md) — `NEXUML_LOGS_ROOT`
- [Pipeline diagrams](../explanation/diagrams.md)
- [Checkpoints](checkpoints.md)
- [Optuna tuning](tune.md) — MLflow nested runs
