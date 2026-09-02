# Trusted Python scenario files

NexuML can load a scenario from a plain Python file instead of the built-in registry. This enables locally-authored experiments and agent-driven experiment loops.

!!! warning "Trusted execution"
    Scenario files are executed as Python code with `exec()`. Only load files from trusted sources.

## Prerequisites

- NexuML installed
- The file exposes a callable `scenario()` that returns `ScenarioSpec`

## Minimal example

```python
# my_experiment.py
from nexuml.core.types import (
    ScenarioSpec, TrainingSpec, DataSpec, PipelineSpec, LayerSpec
)
from nexuml_library.data.synthetic import SyntheticDataset
from nexuml_library.layers.loss.reconstruction_loss import ReconstructionLoss
from nexuml_library.layers.model.linear_encoder import LinearEncoder

def scenario() -> ScenarioSpec:
    return ScenarioSpec(
        name="my_experiment",
        data=DataSpec(
            source=SyntheticDataset(feature_shape=(64,), num_samples=1000),
            input_shapes={"features": [64]},
        ),
        training=TrainingSpec(
            lr=1e-3,
            max_epochs=5,
            loss_keys={"reconstruction_loss": 1.0},
        ),
        pipeline=PipelineSpec(stages={
            "encode": [
                LayerSpec(
                    component=LinearEncoder(output_dim=8),
                    keys_in=["features"],
                    keys_out=["z"],
                )
            ],
            "decode": [
                LayerSpec(
                    component=LinearEncoder(output_dim=64),
                    keys_in=["z"],
                    keys_out=["reconstructed"],
                )
            ],
            "loss": [
                LayerSpec(
                    component=ReconstructionLoss(),
                    keys_in=["features", "reconstructed"],
                    keys_out=["reconstruction_loss"],
                )
            ],
        }),
    )
```

Run it:

```bash
nexuml train --scenario-file my_experiment.py
```

## Provenance snapshots

Use `--artifact-dir` to save a copy of the file alongside the run outputs:

```bash
nexuml train --scenario-file my_experiment.py --artifact-dir ./artifacts/exp-001/
```

This writes:

- `artifacts/exp-001/scenario.py` — a copy of the file
- `artifacts/exp-001/scenario_hash.txt` — SHA-256 of the file source

## Optional agent metadata exports

Scenario files can export additional constants used by agent-driven workflows:

| Export | Type | Purpose |
|---|---|---|
| `HYPOTHESIS` | `str` | Human-readable description of what this experiment tests |
| `PARENT` | `str` | Name or path of the parent experiment |
| `TAGS` | `list[str]` or `str` | Labels for grouping and filtering |
| `SEARCH_SPACE` | `dict` | Optuna search space for `nexuml tune` |
| `TUNING_SPEC` | `TuningSpec` or `dict` | Tuning configuration |
| `build` | `callable(**params) -> ScenarioSpec` | Factory for structural/architectural tuning |

## Using with `nexuml tune`

```bash
nexuml tune --scenario-file my_experiment.py \
  --n-trials 20 \
  --metric val/loss \
  --direction minimize
```

For full details on `SEARCH_SPACE`, `TUNING_SPEC`, `build(**params)`, and advanced search-space types, see the [Tuning file reference](../reference/tuning-file.md).

## Implementation map

- `src/nexuml/core/scenario_loader.py` — `load_scenario_file`, `LoadedScenarioFile`
- `src/nexuml/cli/main.py` — `--scenario-file` and `--artifact-dir` options on `train` and `tune`
- `src/nexuml/core/provenance.py` — snapshot writing

## See also

- [Run scenarios](run-scenarios.md)
- [Define a scenario](define-scenario.md)
- [Tuning file reference](../reference/tuning-file.md)
- [Optuna tuning](tune.md)
