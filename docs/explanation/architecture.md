# Architecture: spec / compile / run

NexuML separates the description of a training pipeline from its execution. The lifecycle has three phases.

## 1. Spec

A *scenario* is a pure-Python dataclass tree. It declares what should happen, not how:

```python
from nexuml.core.discovery import scenario
from nexuml.core.types import ScenarioSpec, PipelineSpec, LayerSpec, DataSpec, TrainingSpec

@scenario("my-architecture-example")
def my_architecture_example() -> ScenarioSpec:
    return ScenarioSpec(
        name="my-architecture-example",
        data=DataSpec(source_type="synthetic", params={"feature_shape": [64], "num_samples": 1000}),
        pipeline=PipelineSpec(stages={
            "encode": [
                LayerSpec(
                    type_key="LinearEncoder",
                    keys_in=["features"],
                    keys_out=["z"],
                    params={"input_dim": 64, "output_dim": 8},
                ),
            ],
            "loss": [
                LayerSpec(
                    type_key="ReconstructionLoss",
                    keys_in=["z", "features"],
                    keys_out=["reconstruction_loss"],
                    params={},
                ),
            ],
        }),
        training=TrainingSpec(max_epochs=10, loss_keys={"reconstruction_loss": 1.0}),
    )
```

No tensors, no PyTorch — just configuration. The resolved spec is serializable to YAML and reloadable without code changes.

## 2. Compile

`nexuml resolve <scenario>` runs the compiler:

1. Resolves layer keys to registered `PipelineLayer` classes.
2. Validates that tensor key contracts are satisfied (output keys of layer N → input keys of layer N+1).
3. Writes `configs/<scenario>.yaml` — the canonical, reproducible form of the pipeline.

`nexuml build <config.yaml>` instantiates the compiled pipeline and reports:

- Layer order and tensor shapes
- Parameter counts per layer
- A Mermaid flowchart (see [Pipeline diagrams](diagrams.md))

## 3. Run

`nexuml train <scenario>` drives the compiled pipeline through PyTorch Lightning:

- Each forward pass routes a `TensorDict` through the layer sequence.
- Gradient flow is handled by Lightning's training loop.
- Callbacks (checkpointing, LR scheduling, early stopping) are declared in the spec.

After training, post-train pipeline layers are fitted on the full training set before evaluation.

## TensorDict data flow

Each layer in the pipeline operates on a `TensorDict` — a dictionary of named tensors. Layers declare their inputs and outputs as key lists:

```python
LayerSpec(
    type_key="linear_encoder",
    keys_in=["features"],       # reads "features" from the TensorDict
    keys_out=["z"],             # writes "z" into the TensorDict
    params={"input_dim": 64, "output_dim": 8},
)
```

The compiler validates that every `keys_in` key is produced by a prior layer (or the data source) before the pipeline can run. The TensorDict accumulates keys as it flows through stages:

```
data source → TensorDict{"features": Tensor[B, 64]}
                    ↓ linear_encoder
              TensorDict{"features": ..., "z": Tensor[B, 8]}
                    ↓ reconstruction_loss
              TensorDict{"features": ..., "z": ..., "reconstruction_loss": Tensor[B, 1]}
```

## Package export

After training, the compiled pipeline can be exported to a portable package:

```bash
nexuml export my-scenario
```

The package contains `state_dict.pt` (weights), `config.yaml` (full reproducible spec), and `metadata.json` (provenance). Load it anywhere without re-running compilation:

```python
from nexuml.core.export import load_inference_package

pkg = load_inference_package("packages/my-scenario/")
output = pkg.infer(tensordict_input)
```

See [Model export and reload](../how-to/export.md).

## Why this separation?

| Concern | Where it lives |
|---|---|
| What to run | Scenario spec (Python / YAML) |
| How to run | PipelineLayer implementations |
| When to run | Lightning Trainer configuration |
| Results | MLflow / log directory |

This means scenarios can be versioned, diff'd, and reproduced independently of the code that executes them.

## Project structure

```
src/nexuml/
├── cli/                  # Typer CLI (nexuml entrypoint)
├── core/
│   ├── types.py          # ScenarioSpec, LayerSpec, PipelineSpec, …
│   ├── base_layer.py     # PipelineLayer base class (TensorDict forward)
│   ├── registry.py       # Global layer registry
│   ├── compiler.py       # ScenarioSpec → CompiledPipeline
│   ├── pipeline.py       # CompiledPipeline forward / loss
│   ├── config.py         # ResolvedConfig (save/load YAML)
│   ├── export.py         # export_package / load_package / infer
│   ├── discovery.py      # decorators (@layer, @data_source, @scenario, @eval_algorithm)
│   └── torch_adapter.py  # Lightning module wrapping CompiledPipeline
├── data/
│   ├── dataset.py        # NexuDataset (TensorDict-based)
│   ├── sources/          # Data source implementations
│   ├── loaders.py        # DataLoader factory
│   └── module.py         # LightningDataModule
├── evaluation/
├── tracking/             # Experiment tracking (extensible)
└── tuning/               # Hyperparameter tuning (extensible)

library/src/nexuml_library/
├── layers/               # Registered layers (feature, model, head, …)
├── data/                 # Data source implementations
├── scenarios/            # Pre-built scenario functions
└── evaluation/           # Evaluation algorithm implementations
```

## Writing a custom layer

```python
from nexuml.core.discovery import layer
from nexuml.core.base_layer import PipelineLayer
import torch

@layer("my_relu")
class MyReLU(PipelineLayer):
    def forward_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)
```

The layer is available in any `LayerSpec` by `type_key = "my_relu"`.

## Writing a custom scenario

```python
from nexuml.core.discovery import scenario
from nexuml.core.types import ScenarioSpec, PipelineSpec, LayerSpec, DataSpec, TrainingSpec

@scenario("my_scenario")
def my_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        name="my_scenario",
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
        data=DataSpec(source_type="synthetic", params={"feature_shape": [64], "num_samples": 500}),
        training=TrainingSpec(
            lr=1e-3, batch_size=32, max_epochs=5, loss_keys={"reconstruction_loss": 1.0}
        ),
    )
```

See [Discovery decorators](../reference/decorators.md), [Define a scenario](../how-to/define-scenario.md), and [Run scenarios](../how-to/run-scenarios.md).
