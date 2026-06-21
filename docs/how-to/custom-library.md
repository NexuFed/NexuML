# Custom library end-to-end

This tutorial builds a small user-owned library package outside the NexuML repository. It adds a custom data source, layer, evaluation algorithm, and a scenario that composes them together with existing base-library components.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- NexuML and `nexuml_library` installed (see [Install NexuML](../start/install.md))

## 1. Create the library package

Create a new directory outside the NexuML repo:

```bash
mkdir -p ~/my_nexu_library/src/my_nexu_library
```

```
~/my_nexu_library/
├── pyproject.toml
└── src/
    └── my_nexu_library/
        ├── __init__.py
        ├── data/
        │   ├── __init__.py
        │   └── noisy_synthetic.py
        ├── layers/
        │   ├── __init__.py
        │   └── scaler.py
        ├── evaluation/
        │   ├── __init__.py
        │   └── mae_eval.py
        └── scenarios/
            ├── __init__.py
            └── noisy_ae.py
```

## 2. Add `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-nexu-library"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "nexuml>=0.1.0",
    "nexuml-library>=0.1.0",
]

[project.entry-points."nexuml.libraries"]
my-nexu-library = "my_nexu_library"

[tool.hatch.build.targets.wheel]
packages = ["src/my_nexu_library"]
```

The entry-point value is the importable package name. No `register()` function is required.

## 3. Add a custom data source

`src/my_nexu_library/data/noisy_synthetic.py`:

```python
import torch
from tensordict import TensorDict

from nexuml.core.discovery import data_source
from nexuml.data.dataset import NexuDataset


@data_source("noisy_synthetic")
class NoisySyntheticDataset(NexuDataset):
    def __init__(self, feature_dim: int = 64, num_samples: int = 1000, noise: float = 0.1, seed: int = 42):
        self.feature_key = "features"
        super().__init__(label_names=["reconstruction_target"])

        generator = torch.Generator().manual_seed(seed)
        self._features = torch.randn(num_samples, feature_dim, generator=generator)
        self._features += torch.randn_like(self._features) * noise

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx: int) -> tuple[TensorDict, TensorDict]:
        x = TensorDict({self.feature_key: self._features[idx]}, batch_size=[])
        y = TensorDict({"reconstruction_target": self._features[idx]}, batch_size=[])
        return x, y
```

## 4. Add a custom layer

`src/my_nexu_library/layers/scaler.py`:

```python
import torch

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.discovery import layer


@layer("standard_scaler")
class StandardScaler(PipelineLayer):
    """Scale features to zero mean and unit variance along the last dim."""

    def forward_tensor(self, features: torch.Tensor) -> torch.Tensor:
        mean = features.mean(dim=-1, keepdim=True)
        std = features.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (features - mean) / std
```

## 5. Add a custom eval algorithm

`src/my_nexu_library/evaluation/mae_eval.py`:

```python
import torch
from tensordict import TensorDict

from nexuml.core.discovery import eval_algorithm
from nexuml.evaluation.algorithm import EvalAlgorithm


@eval_algorithm("mae")
class MaeEval(EvalAlgorithm):
    def __init__(self, feature_key: str = "features", prediction_key: str = "reconstruction"):
        self.feature_key = feature_key
        self.prediction_key = prediction_key
        self._sum = 0.0
        self._count = 0

    def eval_batch(self, x: TensorDict, y: TensorDict | None) -> None:
        error = (x[self.prediction_key] - x[self.feature_key]).abs().flatten(start_dim=1).mean(dim=1)
        self._sum += error.sum().item()
        self._count += error.shape[0]

    def results(self) -> dict[str, float]:
        return {"mae": self._sum / max(1, self._count)}
```

## 6. Compose a scenario

`src/my_nexu_library/scenarios/noisy_ae.py`:

```python
from nexuml.core.discovery import scenario
from nexuml.core.types import (
    ScenarioSpec,
    DataSpec,
    TrainingSpec,
    PipelineSpec,
    LayerSpec,
    EvaluationSpec,
    EvalAlgorithmSpec,
)


@scenario("noisy_ae")
def noisy_ae() -> ScenarioSpec:
    return ScenarioSpec(
        name="noisy_ae",
        data=DataSpec(
            source_type="noisy_synthetic",
            params={"feature_dim": 64, "num_samples": 1000, "noise": 0.2},
        ),
        training=TrainingSpec(
            lr=1e-3,
            max_epochs=5,
            batch_size=64,
            loss_keys={"reconstruction_loss": 1.0},
        ),
        pipeline=PipelineSpec(stages={
            "preprocess": [
                LayerSpec(
                    type_key="standard_scaler",
                    keys_in=["features"],
                    keys_out=["features_scaled"],
                ),
            ],
            "encode": [
                LayerSpec(
                    type_key="linear_encoder",
                    keys_in=["features_scaled"],
                    keys_out=["z"],
                    params={"input_dim": 64, "output_dim": 8, "hidden_dims": [32]},
                ),
            ],
            "decode": [
                LayerSpec(
                    type_key="linear_decoder",
                    keys_in=["z"],
                    keys_out=["reconstruction"],
                    params={"input_dim": 8, "output_dim": 64},
                ),
            ],
            "loss": [
                LayerSpec(
                    type_key="reconstruction_loss",
                    keys_in=["reconstruction", "features_scaled"],
                    keys_out=["reconstruction_loss"],
                    params={"input_dim": 64},
                ),
            ],
        }),
        evaluation=EvaluationSpec(
            algorithms=[
                EvalAlgorithmSpec(
                    type="mae",
                    params={"feature_key": "features_scaled", "prediction_key": "reconstruction"},
                ),
            ],
            test_result_metrics=["mae"],
        ),
    )
```

## 7. Local-root discovery

Without installing the package:

```bash
nexuml library add ~/my_nexu_library
nexuml library list
```

Verify that all elements are discovered:

```bash
nexuml registry list data
nexuml registry list layers
nexuml registry list eval
nexuml registry list scenarios
```

You should see `noisy_synthetic`, `standard_scaler`, `mae`, and `noisy_ae`.

## 8. Run by registry name

```bash
nexuml resolve noisy_ae -o configs/noisy_ae.yaml
nexuml build configs/noisy_ae.yaml
nexuml train noisy_ae --max-epochs 5
```

## 9. Installable package discovery

To make the library discoverable in any environment, install it:

```bash
uv pip install -e ~/my_nexu_library
```

The entry point in `pyproject.toml` causes NexuML to scan `my_nexu_library` automatically. Verify with:

```bash
nexuml library list
nexuml registry list scenarios
```

## 10. Run from a trusted scenario file

You can also run the scenario without registering it as a library by writing a trusted Python file:

```python
# noisy_ae_file.py
from my_nexu_library.scenarios.noisy_ae import noisy_ae

def scenario():
    return noisy_ae()
```

```bash
nexuml train --scenario-file noisy_ae_file.py --max-epochs 5
```

Or put the full `ScenarioSpec` factory directly in the file.

## What just happened?

- Custom elements (`noisy_synthetic`, `standard_scaler`, `mae`) were discovered automatically via decorators.
- Existing base-library elements (`linear_encoder`, `linear_decoder`, `reconstruction_loss`) were reused without copying code.
- The scenario was executed by registry name, by resolved YAML, and by trusted file.
- No changes were made to the NexuML repository.

## See also

- [Register a library](../how-to/register-library.md)
- [Add a custom layer](../how-to/custom-layer.md)
- [Add a custom data source](../how-to/custom-data-source.md)
- [Add a custom eval algorithm](../how-to/custom-eval-algorithm.md)
- [Discovery decorators](../reference/decorators.md)
- [Run scenarios](../how-to/run-scenarios.md)
