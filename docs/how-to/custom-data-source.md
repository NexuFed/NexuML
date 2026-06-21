# Add a custom data source

This guide walks through adding a custom dataset to a library and using it from a `ScenarioSpec` via `DataSpec(source_type=...)`.

## Prerequisites

- NexuML installed
- A library root registered with `nexuml library add` or an installed entry-point package
- For a new library: see [Register a library](register-library.md)

## 1. Create the dataset file

Place the dataset under `data/` inside your library package:

```
my_library/
└── src/
    └── my_library/
        ├── __init__.py
        └── data/
            ├── __init__.py
            └── my_dataset.py
```

```python
# my_library/src/my_library/data/my_dataset.py
from nexuml.core.discovery import data_source
from nexuml.data.dataset import NexuDataset
from tensordict import TensorDict
import torch


@data_source("my_dataset")
class MyDataset(NexuDataset):
    """Small labeled tensor dataset."""

    def __init__(
        self,
        num_samples: int = 1000,
        feature_dim: int = 64,
        seed: int = 42,
        feature_key: str = "features",
    ):
        self.feature_key = feature_key
        super().__init__(label_names=["label"])

        generator = torch.Generator().manual_seed(seed)
        self._features = torch.randn(num_samples, feature_dim, generator=generator)
        self._labels = torch.randint(0, 10, (num_samples,), generator=generator)

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx: int) -> tuple[TensorDict, TensorDict]:
        x = TensorDict({self.feature_key: self._features[idx]}, batch_size=[])
        y = TensorDict({"label": self._labels[idx]}, batch_size=[])
        return x, y
```

## 2. Register the local root or install the package

For local development:

```bash
nexuml library add my_library
```

For installable packages, declare the entry point:

```toml
[project.entry-points."nexuml.libraries"]
my-library = "my_library"
```

## 3. Verify registration

```bash
nexuml registry list data
```

You should see `my_dataset` in the output. If it doesn't appear, run with `--verbose` to see import errors.

## 4. Use in a scenario

Reference the dataset by `source_type` in a `DataSpec`:

```python
from nexuml.core.discovery import scenario
from nexuml.core.types import ScenarioSpec, DataSpec, TrainingSpec

@scenario("my_dataset_baseline")
def my_dataset_baseline() -> ScenarioSpec:
    return ScenarioSpec(
        name="my_dataset_baseline",
        data=DataSpec(
            source_type="my_dataset",
            params={"num_samples": 2000, "feature_dim": 64},
        ),
        training=TrainingSpec(
            lr=1e-3,
            max_epochs=5,
            batch_size=64,
            loss_keys={"loss": 1.0},
        ),
    )
```

Or use it from a trusted scenario file:

```python
from nexuml.core.types import ScenarioSpec, DataSpec, TrainingSpec

def scenario() -> ScenarioSpec:
    return ScenarioSpec(
        name="my_dataset_file",
        data=DataSpec(source_type="my_dataset", params={"num_samples": 500}),
        training=TrainingSpec(max_epochs=2, loss_keys={"loss": 1.0}),
    )
```

```bash
nexuml train --scenario-file my_experiment.py
```

## Dataset contract

- Inherit from `NexuDataset` (or implement the same `__getitem__` contract).
- Return `(x: TensorDict, y: TensorDict | None)` from `__getitem__`.
- `x` contains input tensors (commonly under the `features` key).
- `y` contains label tensors. Use `super().__init__(label_names=[...])` to declare them.

## See also

- [Discovery decorators](../reference/decorators.md)
- [Register a library](register-library.md)
- [Custom library end-to-end tutorial](custom-library.md)
