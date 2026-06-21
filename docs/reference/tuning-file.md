# Tuning file reference

A trusted Python scenario file used with `nexuml tune` can define `SEARCH_SPACE`, `TUNING_SPEC`, `HYPOTHESIS`, `PARENT`, `TAGS`, and a `build(**params)` factory. This page documents each export and the search-space format.

## Required contract

The file must still define `scenario() -> ScenarioSpec`. Tuning exports are optional.

```python
from nexuml.core.types import ScenarioSpec, DataSpec, TrainingSpec

HYPOTHESIS = "Smaller learning rate converges better on synthetic data"
PARENT = "baseline_experiment"
TAGS = ["synthetic", "lr-sweep"]

def scenario() -> ScenarioSpec:
    return ScenarioSpec(
        name="lr_sweep",
        data=DataSpec(source_type="synthetic", params={"feature_shape": [32], "num_samples": 500}),
        training=TrainingSpec(lr=1e-3, max_epochs=10, loss_keys={"reconstruction_loss": 1.0}),
    )
```

Run:

```bash
nexuml tune --scenario-file my_experiment.py
```

## Exports

| Export | Type | Purpose |
|---|---|---|
| `HYPOTHESIS` | `str` | Human-readable description of what this experiment tests |
| `PARENT` | `str` | Name or path of the parent experiment |
| `TAGS` | `list[str]` or `str` | Labels for grouping and filtering |
| `SEARCH_SPACE` | `dict` | Optuna search space |
| `TUNING_SPEC` | `TuningSpec` or `dict` | Tuning configuration |
| `build` | `callable(**params) -> ScenarioSpec` | Factory for structural/architectural tuning |

## `SEARCH_SPACE` format

Each key is either:

- A dotted path such as `training.lr` (scalar override on the returned spec)
- A structural parameter name such as `hidden_dim` (passed to `build(**params)`)

Each value is a dict describing how Optuna should sample the parameter.

### Scalar types

```python
SEARCH_SPACE = {
    "training.lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "training.max_epochs": {"type": "int", "low": 5, "high": 50},
    "training.batch_size": {"type": "categorical", "choices": [32, 64, 128]},
}
```

| Type | Parameters | Notes |
|---|---|---|
| `float` | `low`, `high`, optional `log` | Maps to `trial.suggest_float` |
| `int` | `low`, `high`, optional `log` | Maps to `trial.suggest_int` |
| `categorical` | `choices` | Maps to `trial.suggest_categorical` |

If `type` is omitted and `choices` is present, the type defaults to `categorical`.

### Conditional entries (`when`)

```python
SEARCH_SPACE = {
    "training.optimizer.type": {
        "type": "categorical",
        "choices": ["torch.optim.Adam", "torch.optim.SGD"],
        "when": {
            "torch.optim.SGD": {
                "training.optimizer.params.momentum": {
                    "type": "float", "low": 0.8, "high": 0.99
                },
            }
        },
    },
}
```

`when` branches activate additional search-space entries depending on the sampled value. This is **Python-only** — not YAML-exportable. Requires `--scenario-file`.

### Derived entries

```python
SEARCH_SPACE = {
    "training.lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "training.optimizer.params.weight_decay": {
        "derived": "training.lr * 0.01"   # string expression or callable
    },
}
```

`derived` entries are computed from other sampled values. They are **Python-only**.

### Structural / `build(**params)` parameters

For architecture parameters that change model structure, define a `build` callable. Optuna passes sampled values as keyword arguments:

```python
from nexuml.core.types import ScenarioSpec, PipelineSpec, LayerSpec, TrainingSpec, DataSpec

SEARCH_SPACE = {
    "hidden_dim": {"type": "int", "low": 8, "high": 64},
}

def build(hidden_dim: int = 16) -> ScenarioSpec:
    return ScenarioSpec(
        name="arch_search",
        data=DataSpec(source_type="synthetic", params={"feature_shape": [64], "num_samples": 500}),
        training=TrainingSpec(lr=1e-3, max_epochs=5, loss_keys={"reconstruction_loss": 1.0}),
        pipeline=PipelineSpec(stages={
            "encode": [
                LayerSpec(
                    type_key="linear_encoder",
                    keys_in=["features"],
                    keys_out=["z"],
                    params={"input_dim": 64, "output_dim": hidden_dim},
                )
            ],
        }),
    )

def scenario() -> ScenarioSpec:
    return build()
```

Scalar dotted-path search-space keys like `training.lr` are applied as attribute overrides on the returned spec. Structural keys like `hidden_dim` are passed directly to `build`.

## `TUNING_SPEC`

Set defaults for the tuning run:

```python
from nexuml.core.types import TuningSpec

TUNING_SPEC = TuningSpec(
    n_trials=30,
    metric_key="val/loss",
    directions=["minimize"],
    storage=".experiments/optuna/lr_sweep.log",
    prune=False,
)
```

CLI flags override these values:

```bash
nexuml tune --scenario-file my_experiment.py \
  --n-trials 20 \
  --metric val/loss \
  --direction minimize \
  --storage sqlite:///.experiments/optuna/lr_sweep.db
```

## Python-only constraints

`when`, `derived`, and `build`-factory parameters cannot be serialized to YAML and require `--scenario-file`. A resolved YAML config cannot drive structural tuning.

## Implementation map

- `src/nexuml/core/scenario_loader.py` — `load_scenario_file`, `SEARCH_SPACE`, `TUNING_SPEC`, `build` loading
- `src/nexuml/tuning/optuna_tuner.py` — `DEFAULT_SEARCH_SPACE`, `build_objective`
- `src/nexuml/cli/main.py` — `tune` command

## See also

- [Trusted scenario files](../how-to/scenario-file.md)
- [Optuna tuning](../how-to/tune.md)
- [Run scenarios](../how-to/run-scenarios.md)
