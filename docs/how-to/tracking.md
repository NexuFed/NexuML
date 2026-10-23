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

## Requested tracking failures

An explicitly selected backend is required, not best-effort. Missing dependencies,
invalid destinations and initialization errors fail the run with the backend name
and original diagnostic; NexuML no longer warns and silently omits that logger.
This applies to supervised and reinforcement sessions and standalone evaluation.
Requested MLflow system-metrics initialization also fails visibly.

If a later backend fails during initialization, previously created loggers are
finalized as failed. A newly created MLflow child run is marked failed if its
Lightning logger cannot initialize; a supplied existing run is not terminated by
that construction failure. Cleanup errors retain the original failure.
Failure diagnostics retain the created run ID even when cleanup succeeds. If
finalizing a newly created auto-parent fails, NexuML attempts to mark that owned
parent failed too; it does not terminate a pre-existing parent.

Artifact delivery is not transactional: when one destination succeeds and a later
one fails, the error identifies both, and successful files remain available.
NexuML does not delete earlier uploads or retry silently. To intentionally run
without tracking, use `logging=None` or omit all logger backends.

## Paths

Relative logging paths are resolved through `NEXUML_LOGS_ROOT` where the relevant NexuML path helper is used. See [Environment roots](../reference/environment.md) for the exact rule.

## Private monitoring access

The startup banner suggests loopback-only commands for local stores: TensorBoard
on port 6007 and MLflow on port 5000, separate from Viser's default 8080. Paths
and storage URIs are shell-quoted. Run those commands on the worker containing the
store, then use an existing approved private tunnel to reach it. They are
suggestions, not automatically started services or reserved ports; select unused
ports for concurrent jobs. An HTTP(S) tracking URI uses the existing service
instead of suggesting another `mlflow ui` process. The banner includes its actual
run ID when available.

## See also

- [Train](train.md)
- [Optuna tuning](tune.md)
- [Pipeline diagrams](../explanation/diagrams.md)
