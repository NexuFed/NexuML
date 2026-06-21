# Write a custom composed scenario

This tutorial walks through writing a complete NexuML scenario from scratch using the `@scenario` decorator, running it end to end.

## Prerequisites

- NexuML installed in your own environment (see [Install NexuML](../start/install.md))
- `nexuml_library` installed
- Basic familiarity with the [core mental model](../explanation/architecture.md)

## What we'll build

A synthetic autoencoder scenario that:

1. Generates synthetic feature vectors
2. Encodes them to a lower-dimensional embedding
3. Computes reconstruction loss
4. Trains for 5 epochs

## Step 1 — create the scenario file

```python
# my_scenario.py
from nexuml.core.discovery import scenario
from nexuml.core.types import (
    ScenarioSpec,
    PipelineSpec,
    LayerSpec,
    DataSpec,
    TrainingSpec,
    LoggingSpec,
)


@scenario("synthetic_ae_tutorial")
def synthetic_ae_tutorial() -> ScenarioSpec:
    return ScenarioSpec(
        name="synthetic_ae_tutorial",
        data=DataSpec(
            source_type="synthetic",
            params={
                "feature_shape": [64],
                "num_samples": 1000,
            },
        ),
        training=TrainingSpec(
            lr=1e-3,
            max_epochs=5,
            batch_size=64,
            loss_keys={"reconstruction_loss": 1.0},
            metric_keys=["train/loss", "val/loss"],
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
            "loss": [
                LayerSpec(
                    type_key="reconstruction_loss",
                    keys_in=["z", "features"],
                    keys_out=["reconstruction_loss"],
                    params={"input_dim": 8},
                ),
            ],
        }),
        logging=LoggingSpec(
            experiment_name="tutorial",
            run_name="synthetic_ae_v1",
        ),
    )
```

## Step 2 — train from the file

```bash
nexuml train --scenario-file my_scenario.py --max-epochs 5
```

Expected output:

```
Epoch 1/5: train/loss=0.4231, val/loss=0.3987
Epoch 2/5: train/loss=0.3102, val/loss=0.2964
...
Training complete. Checkpoints saved to .experiments/checkpoints/synthetic_ae_tutorial/
```

## Step 3 — resolve to YAML (optional)

If you want a reproducible YAML snapshot, first register the library containing this file, then resolve:

```bash
nexuml library add .
nexuml resolve synthetic_ae_tutorial -o configs/synthetic_ae_tutorial.yaml
```

The YAML can be re-used without the Python file:

```bash
nexuml build configs/synthetic_ae_tutorial.yaml
nexuml train -c configs/synthetic_ae_tutorial.yaml
```

## Step 4 — smoke test

```bash
nexuml smoke --scenario-file my_scenario.py --max-epochs 2
```

Runs the full resolve → build → train → export → reload → infer cycle in one command.

## Step 5 — export the trained model

```bash
nexuml export synthetic_ae_tutorial -o packages/synthetic_ae_tutorial/
```

Load and run inference in Python:

```python
from nexuml.core.export import load_inference_package
import torch
from tensordict import TensorDict

pkg = load_inference_package("packages/synthetic_ae_tutorial/")
x = TensorDict({"features": torch.randn(1, 64)}, batch_size=[1])
output = pkg.infer(x)
print(output)
```

## TensorDict key contract

Each layer declares its input and output TensorDict keys:

```
DataSpec → TensorDict{"features": Tensor[B, 64]}
                    ↓
linear_encoder: keys_in=["features"] → keys_out=["z"]
                    ↓
TensorDict{"features": ..., "z": Tensor[B, 8]}
                    ↓
reconstruction_loss: keys_in=["z", "features"] → keys_out=["reconstruction_loss"]
                    ↓
TensorDict{"features": ..., "z": ..., "reconstruction_loss": Tensor[B, 1]}
```

The compiler validates that every `keys_in` key is produced by a prior layer before training starts.

## Adding to a library

To make the scenario available by name to `nexuml resolve` without `--scenario-file`, move it to a library package:

```
my_library/
└── src/
    └── my_library/
        ├── __init__.py
        └── scenarios/
            ├── __init__.py
            └── synthetic_ae.py   ← place my_scenario.py here
```

Then register:

```bash
nexuml library add my_library
nexuml registry list scenarios   # should show synthetic_ae_tutorial
nexuml train synthetic_ae_tutorial
```

## Implementation map

- `src/nexuml/core/types.py` — `ScenarioSpec`, `PipelineSpec`, `LayerSpec`, `DataSpec`, `TrainingSpec`
- `src/nexuml/core/discovery.py` — `@scenario` decorator
- `src/nexuml/core/compiler.py` — `ScenarioSpec` → `CompiledPipeline`
- `src/nexuml/core/scenario_loader.py` — `--scenario-file` loading
- `src/nexuml/core/export.py` — `load_inference_package`, `infer`

## See also

- [Discovery decorators](../reference/decorators.md)
- [Add a custom layer](../how-to/custom-layer.md)
- [Add a custom data source](../how-to/custom-data-source.md)
- [Trusted scenario files](../how-to/scenario-file.md)
- [Run scenarios](../how-to/run-scenarios.md)
- [Architecture](../explanation/architecture.md)
