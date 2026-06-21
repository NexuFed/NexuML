# Optuna hyperparameter tuning

NexuML integrates with [Optuna](https://optuna.readthedocs.io/) for hyperparameter search via `nexuml tune`.

## Prerequisites

- NexuML installed
- Optuna installed (included with `uv pip install nexuml[tuning]` or `uv pip install optuna`)
- A registered scenario or a trusted Python scenario file

## Supported scenario sources

`nexuml tune` accepts **exactly one** of:

| Source | Example | Notes |
|---|---|---|
| Registered scenario name | `nexuml tune my-scenario` | Discovered from libraries/entry points |
| Trusted scenario file | `nexuml tune --scenario-file my_experiment.py` | Supports `SEARCH_SPACE`, `build(**params)` |

`nexuml tune` does **not** accept resolved YAML configs via `--config` / `-c`. Use `--scenario-file` for Python-driven tuning or register a scenario that contains a `TuningSpec`.

## Basic usage

```bash
# Tune a registered scenario
nexuml tune my-scenario --n-trials 30

# Tune a trusted Python file
nexuml tune --scenario-file my_experiment.py --n-trials 20
```

## Full option reference

| Option | Description |
|---|---|
| `SCENARIO_NAME` | Registered scenario name (alternative to `--scenario-file`) |
| `--scenario-file PATH` | Trusted Python file defining `scenario()` |
| `--artifact-dir PATH` | Directory for provenance snapshots |
| `--n-trials N` | Number of Optuna trials (overrides `TuningSpec.n_trials`) |
| `--metric TEXT` | Metric key to optimize (overrides `TuningSpec.metric_key`) |
| `--direction TEXT` | `minimize` or `maximize` (overrides `TuningSpec.directions`) |
| `--storage TEXT` | Optuna storage path (overrides `TuningSpec.storage`) |
| `--prune` / `--no-prune` | Enable/disable Optuna pruning |
| `--override` / `-O key=value` | Override any scenario field (repeatable) |

## `TuningSpec` reference

Configure tuning defaults inside a `ScenarioSpec`:

```python
from nexuml.core.types import ScenarioSpec, TuningSpec

ScenarioSpec(
    name="my_scenario",
    tuning=TuningSpec(
        n_trials=50,                      # default number of trials
        directions=["minimize"],          # "minimize" or "maximize"; list for multi-objective
        metric_key="val/loss",            # metric to optimize
        storage=".experiments/optuna/optuna.log",  # Optuna storage path
        prune=False,                      # enable Optuna pruning
    ),
    ...
)
```

CLI flags `--metric`, `--direction`, and `--storage` override the corresponding `TuningSpec` fields. `--n-trials` always overrides `TuningSpec.n_trials` when provided.

## Default search space

When no `SEARCH_SPACE` is defined, tuning uses the built-in default:

```python
DEFAULT_SEARCH_SPACE = {
    "training.lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "training.batch_size": {"type": "categorical", "choices": [32, 64, 128]},
}
```

Only `training.lr` and `training.batch_size` are in the default search space.

## Custom search spaces

See the [Tuning file reference](../reference/tuning-file.md) for the full search-space format, including:

- Scalar dotted-path entries (`training.lr`, `training.max_epochs`)
- Conditional `when` branches
- `derived` entries
- Structural `build(**params)` parameters

## Correctness constraints

### Metric key must exist

The metric key must appear in logged metrics or evaluation results during training. If a trial completes without logging the requested metric, Optuna raises a tuning error.

For evaluation metrics such as `omega` or `auc`:

1. Make sure the evaluation algorithm computes and logs the metric.
2. Surface it to test results via `evaluation.test_result_metrics`:

```python
from nexuml.core.types import EvaluationSpec

EvaluationSpec(
    test_result_metrics=["omega"],   # surfaces eval metrics as logged metrics
    algorithms=[...],
)
```

### Pruning

```bash
nexuml tune my-scenario --prune
```

Optuna pruners stop underperforming trials early. Intermediate values must be reported by the Lightning callbacks (this requires the trainer to call `trial.report()` during validation steps). Pruning is disabled by default.

### Storage

By default, tuning results are persisted to `.experiments/optuna/optuna.log`. To use a SQLite database (required for the Optuna dashboard):

```bash
nexuml tune my-scenario \
  --n-trials 50 \
  --storage sqlite:///.experiments/optuna/study.db
```

Visualize with the Optuna dashboard:

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///.experiments/optuna/study.db
```

### MLflow study runs

Each Optuna trial is logged to MLflow as a nested run under the parent study run. The best trial's parameters and metric value are logged to the parent run.

## Example — full tuning run

```bash
nexuml tune synthetic_ae_tutorial \
  --n-trials 20 \
  --metric val/loss \
  --direction minimize \
  --storage sqlite:///.experiments/optuna/ae_study.db \
  --no-prune
```

Expected output:

```
Trial 1 finished with value: 0.3241
Trial 2 finished with value: 0.2987
...
Best trial: 14, value: 0.1823
Best params: {'training.lr': 0.003, 'training.batch_size': 128}
```

## Expected artifacts

After tuning:

| Artifact | Location |
|---|---|
| Optuna study | `--storage` path (`.experiments/optuna/optuna.log` by default) |
| MLflow runs | `.experiments/mlflow.db` |
| Best checkpoint | `.experiments/checkpoints/<scenario>/` |

## Implementation map

- `src/nexuml/tuning/optuna_tuner.py` — `DEFAULT_SEARCH_SPACE`, `build_objective`, tuning loop
- `src/nexuml/core/types.py` — `TuningSpec`
- `src/nexuml/core/scenario_loader.py` — `SEARCH_SPACE`, `TUNING_SPEC`, `build` loading
- `src/nexuml/cli/main.py` — `tune` command

## See also

- [Tuning file reference](../reference/tuning-file.md)
- [Trusted scenario files](scenario-file.md)
- [CLI lifecycle](cli-lifecycle.md)
- [Tracking and logging](tracking.md)
