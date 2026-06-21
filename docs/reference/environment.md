# Environment roots

Two environment variables control where NexuML reads data from and writes experiment artifacts to.

## Variables

### `NEXUML_DATA_ROOT`

Root directory for dataset files. Data sources that accept a `root` parameter use this as the base when no explicit path is provided.

```bash
export NEXUML_DATA_ROOT=/mnt/datasets
```

Effects:

- Dataset `root` resolution for registered data sources
- Preprocessing output paths (when `preprocessing.path` is `None`)
- Downloaded dataset storage

### `NEXUML_LOGS_ROOT`

Root directory for experiment artifacts. Defaults to `.experiments/` in the current working directory when not set.

```bash
export NEXUML_LOGS_ROOT=/mnt/experiments
```

Effects:

| Artifact | Path |
|---|---|
| Lightning checkpoints | `$NEXUML_LOGS_ROOT/checkpoints/<scenario>/` |
| TensorBoard logs | `$NEXUML_LOGS_ROOT/tensorboard/` (or `logging.tensorboard.log_dir`) |
| MLflow tracking | `$NEXUML_LOGS_ROOT/mlflow.db` (or `logging.mlflow.tracking_uri`) |
| DVCLive | `$NEXUML_LOGS_ROOT/dvclive/` (or `logging.dvclive.dir`) |
| Optuna storage | `$NEXUML_LOGS_ROOT/optuna/optuna.log` (or `TuningSpec.storage`) |
| Mermaid diagrams | `$NEXUML_LOGS_ROOT/diagrams/<scenario>.md` (or `logging.diagram.output_dir`) |
| Preprocessing cache | `$NEXUML_LOGS_ROOT/preprocessing/` |
| Temp MLflow artifacts | `logging.temp_artifact_dir` (default: `/tmp/mlruns`) |

## Overriding per-artifact paths

Individual paths can be overridden in `LoggingSpec` regardless of `NEXUML_LOGS_ROOT`:

```python
from nexuml.core.types import LoggingSpec, TensorBoardSpec, MLflowSpec, DiagramSpec

LoggingSpec(
    tensorboard=TensorBoardSpec(log_dir="/custom/tensorboard"),
    mlflow=MLflowSpec(tracking_uri="sqlite:////custom/mlflow.db"),
    diagram=DiagramSpec(output_dir="/custom/diagrams"),
    temp_artifact_dir="/tmp/my_mlruns",
    experiment_name="MyProject",
    run_name="run_001",
    log_system_metrics=False,
)
```

## Typical setup

```bash
# In .env or shell profile
export NEXUML_DATA_ROOT=/data/nexuml
export NEXUML_LOGS_ROOT=/experiments/nexuml
```

Then in any scenario:

```python
DataSpec(
    source_type="my_dataset",
    # dataset reads from $NEXUML_DATA_ROOT/my_dataset/ automatically
)
```

## Implementation map

- `src/nexuml/core/log_paths.py` — root resolution logic

## See also

- [Tracking and logging](../how-to/tracking.md)
- [Pipeline diagrams](../explanation/diagrams.md)
