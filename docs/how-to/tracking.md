# Tracking and logging

Logging belongs to `ScenarioSpec.logging`. Keep experiment semantics in the scenario while each backend owns its external service/runtime.

## TensorBoard

TensorBoard support is part of the core installation:

```python
from nexuml.core.types import LoggingSpec, TensorBoardSpec

logging = LoggingSpec(
    experiment_name="MyProject",
    tensorboard=TensorBoardSpec(log_dir=".experiments/tensorboard"),
)
```

Run TensorBoard against the configured directory.

## MLflow

Install the tracking extra:

```bash
uv pip install "nexuml[tracking]"
```

Then configure the URI explicitly:

```python
from nexuml.core.types import LoggingSpec, MLflowSpec

logging = LoggingSpec(
    experiment_name="MyProject",
    run_name="baseline",
    mlflow=MLflowSpec(
        tracking_uri="sqlite:///.experiments/mlflow.db",
        experiment_name="MyProject",
        log_model=False,
    ),
)
```

Remote HTTP tracking URIs are also passed through to MLflow. Credentials/service deployment are external to NexuML configuration.

## DVCLive

`LoggingSpec` also supports `DVCLiveSpec`. Install `dvclive` in the consuming environment when you select that backend.

```python
from nexuml.core.types import DVCLiveSpec

DVCLiveSpec(dir=".experiments/dvclive")
```

## Pipeline diagrams

`LoggingSpec.diagram` controls Mermaid export independently from scalar loggers. See [Pipeline diagrams](../explanation/diagrams.md).

## No external logger

`ScenarioSpec.logging=None` disables configured NexuML loggers. A plain `LoggingSpec()` has no TensorBoard/MLflow/DVCLive backend by default, though its diagram configuration is enabled by default.

The optional base library's `default_logging()` helper intentionally chooses its own convenience defaults; do not confuse those helper defaults with core `LoggingSpec` defaults.

## Paths

Relative logging paths are resolved through `NEXUML_LOGS_ROOT` where the relevant NexuML path helper is used. See [Environment roots](../reference/environment.md) for the exact rule.

## See also

- [Train](train.md)
- [Optuna tuning](tune.md)
- [Pipeline diagrams](../explanation/diagrams.md)
