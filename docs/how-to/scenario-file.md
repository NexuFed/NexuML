# Trusted Python scenario files

A trusted Python file is useful for local/agent-driven experiments that should not yet be promoted into a discoverable library recipe.

!!! warning "Trusted execution"
    NexuML executes scenario files as Python code. Only run files you trust.

## Contract

The file exposes `scenario() -> ScenarioSpec`:

```python
from nexuml.core.types import DataSpec, LoaderSpec, ScenarioSpec, TrainingSpec
from nexuml.data.loaders.definitions import TorchLoader
from nexuml_library.data.synthetic import SyntheticDataset


def scenario() -> ScenarioSpec:
    return ScenarioSpec(
        name="experiment",
        data=DataSpec(
            source=SyntheticDataset(feature_shape=(64,), num_samples=1000),
            input_shapes={"features": [64]},
            loader=LoaderSpec(backend=TorchLoader()),
        ),
        pipeline=...,
        training=TrainingSpec(max_epochs=5),
    )
```

Run it directly:

```bash
nexuml train --scenario-file experiment.py
```

## Save provenance

```bash
nexuml train \
  --scenario-file experiment.py \
  --artifact-dir artifacts/exp-001
```

The artifact directory records a source snapshot/hash and run provenance so an exploratory file can be tied to the resulting experiment.

## Tuning metadata

Trusted scenario files can also expose the tuning metadata documented in [Tuning file reference](../reference/tuning-file.md), including `SEARCH_SPACE`, `TUNING_SPEC`, and an optional structural `build(**params)` factory.

```bash
nexuml tune --scenario-file experiment.py --n-trials 20
```

## When to promote it

Once a recipe is stable and should be reused by colleagues, move it into an importable library, decorate it with `@scenario`, and expose the package through `nexuml.libraries`. See [Register a library](register-library.md).
