# Define a scenario

A *scenario* is a pure-Python object that declares the full spec for data, pipeline, training, and evaluation. NexuML compiles it into a runnable pipeline at resolve time.

## 1. Create a scenario function

Use the `@scenario` decorator from `nexuml.core.discovery` and return a `ScenarioSpec`:

```python
# my_library/scenarios/my_scenario.py
from nexuml.core.discovery import scenario
from nexuml.core.types import (
    ScenarioSpec,
    PipelineSpec,
    LayerSpec,
    DataSpec,
    TrainingSpec,
)

@scenario("my-scenario")
def my_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        name="my-scenario",
        data=DataSpec(
            source_type="synthetic",
            params={"feature_shape": [64], "num_samples": 1000},
        ),
        training=TrainingSpec(
            lr=1e-3,
            max_epochs=10,
            batch_size=64,
            loss_keys={"reconstruction_loss": 1.0},
        ),
        pipeline=PipelineSpec(stages={
            "encode": [
                LayerSpec(
                    type_key="LinearEncoder",
                    keys_in=["features"],
                    keys_out=["z"],
                    params={"input_dim": 64, "output_dim": 8, "hidden_dims": [32]},
                ),
            ],
            "decode": [
                LayerSpec(
                    type_key="LinearEncoder",
                    keys_in=["z"],
                    keys_out=["reconstructed"],
                    params={"input_dim": 8, "output_dim": 64, "hidden_dims": [32]},
                ),
            ],
            "loss": [
                LayerSpec(
                    type_key="ReconstructionLoss",
                    keys_in=["features", "reconstructed"],
                    keys_out=["reconstruction_loss"],
                    params={},
                ),
            ],
        }),
    )
```

Decorating the function registers it under the given name. No separate registration function is required.

## 2. Add to a library

Place the module inside an installable package or a local library root:

```
my_library/
├── pyproject.toml
└── src/
    └── my_library/
        ├── __init__.py
        ├── layers/
        │   └── __init__.py
        └── scenarios/
            ├── __init__.py
            └── my_scenario.py
```

For installable packages, declare the entry point in `pyproject.toml`:

```toml
[project.entry-points."nexuml.libraries"]
my-library = "my_library"
```

The entry-point value must be the importable package name. NexuML scans the package tree and discovers all decorated elements automatically.

## 3. Verify and run

```bash
# Local-root workflow
nexuml library add /path/to/my_library

# Verify registration
nexuml registry list scenarios

# Run by name
nexuml resolve my-scenario
nexuml train my-scenario
```

## ScenarioSpec fields

A complete `ScenarioSpec` contains:

| Field | Purpose |
|---|---|
| `name` | Scenario identifier |
| `pipeline` | `PipelineSpec` with named stages of `LayerSpec` |
| `data` | `DataSpec` for source, splits, loader, targets |
| `training` | `TrainingSpec` for optimizer, scheduler, epochs, batch size |
| `evaluation` | `EvaluationSpec` for metrics and eval algorithms |
| `logging` | `LoggingSpec` for TensorBoard / MLflow / DVCLive / diagrams |
| `callbacks` | List of `CallbackSpec` for Lightning callbacks |
| `tuning` | `TuningSpec` defaults for `nexuml tune` |
| `checkpoint` | `CheckpointLoadSpec` for resume/fine-tune |
| `exports` | List of `ExportSpec` artifacts to produce after training |

See the full annotated reference: [ScenarioSpec](../reference/scenario-spec.md).

## See also

- [Run scenarios](run-scenarios.md)
- [Trusted scenario files](scenario-file.md)
- [ScenarioSpec reference](../reference/scenario-spec.md)
- [Architecture explanation](../explanation/architecture.md)
- [Discovery decorators](../reference/decorators.md)
