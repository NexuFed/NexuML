# Discovery decorators

All decorators live in `nexuml.core.discovery`. They attach metadata to classes and functions so NexuML can find them during package scanning.

## Compact matrix

| Decorator | Registers | Use in spec | Registry kind | Inspect command |
|---|---|---|---|---|
| `@layer(key)` | `PipelineLayer` subclass | `LayerSpec(type_key=key, ...)` | `layers` | `nexuml registry list layers` |
| `@data_source(key)` | Dataset class | `DataSpec(source_type=key, ...)` or `DatasetSpec(type_key=key, ...)` | `data` | `nexuml registry list data` |
| `@scenario(key)` | Callable `() -> ScenarioSpec` | `nexuml train key`, `nexuml resolve key` | `scenarios` | `nexuml registry list scenarios` |
| `@eval_algorithm(key)` | Eval-algorithm class | `EvalAlgorithmSpec(type=key, ...)` | `eval` | `nexuml registry list eval` |

## `@layer`

```python
from nexuml.core.discovery import layer
from nexuml.core.base_layer import PipelineLayer
import torch

@layer("my_relu")
class MyReLU(PipelineLayer):
    def forward_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)
```

Used as:

```python
LayerSpec(type_key="my_relu", keys_in=["x"], keys_out=["x"])
```

## `@data_source`

```python
from nexuml.core.discovery import data_source
from torch.utils.data import Dataset

@data_source("my_dataset")
class MyDataset(Dataset):
    def __init__(self, root: str, split: str = "train", **kwargs):
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int):
        ...
```

Used as:

```python
DataSpec(source_type="my_dataset", params={"root": "/data/my_dataset"})
```

## `@scenario`

```python
from nexuml.core.discovery import scenario
from nexuml.core.types import ScenarioSpec, DataSpec, TrainingSpec

@scenario("my_experiment")
def my_experiment() -> ScenarioSpec:
    return ScenarioSpec(
        name="my_experiment",
        data=DataSpec(source_type="synthetic", params={"feature_shape": [32], "num_samples": 200}),
        training=TrainingSpec(lr=1e-3, max_epochs=5, loss_keys={"reconstruction_loss": 1.0}),
    )
```

Used from the CLI:

```bash
nexuml train my_experiment
nexuml resolve my_experiment
```

## `@eval_algorithm`

```python
from nexuml.core.discovery import eval_algorithm
from nexuml.evaluation.base import EvalAlgorithm

@eval_algorithm("my_eval")
class MyEvalAlgorithm(EvalAlgorithm):
    def fit(self, features, labels):
        ...

    def predict(self, features):
        ...
```

Used as:

```python
from nexuml.core.types import EvaluationSpec, EvalAlgorithmSpec

EvaluationSpec(algorithms=[
    EvalAlgorithmSpec(type="my_eval", feature_key="z")
])
```

## How discovery finds decorated elements

1. NexuML calls `scan_all()` at startup.
2. `scan_all` discovers packages from three sources:
   - Built-in `nexuml_library` (if installed)
   - Packages declared via the `nexuml.libraries` entry-point group
   - Local roots registered with `nexuml library add`
3. Each package is walked with `pkgutil.walk_packages` and every module is imported.
4. Modules are inspected for objects with the `__nexuml_discovered__` attribute (set by the decorators).
5. Decorated objects are registered into the appropriate registry.

Discovery is resilient: a module that fails to import records a `DiscoveryError` and scanning continues. Use `nexuml registry list layers --verbose` to see import failures.

## Keys must be unique per kind

Duplicate keys within the same kind raise a `ValueError` at registration time. Keys are scoped per registry — you can use the same string for a layer and a scenario.

## See also

- [Add a custom layer](../how-to/custom-layer.md)
- [Add a custom data source](../how-to/custom-data-source.md)
- [Add a custom eval algorithm](../how-to/custom-eval-algorithm.md)
- [Define a scenario](../how-to/define-scenario.md)
- [Library discovery](../explanation/library-discovery.md)
- [Registry inspection](registry.md)
