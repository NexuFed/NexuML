# Add A Custom Data Source

The public data source is an immutable `DataSourceDefinition`. It builds a private mutable `NexuDataset` runtime.

```python
import torch
from tensordict import TensorDict

from nexuml.core.components import DataSourceDefinition
from nexuml.core.discovery import data_source
from nexuml.data.dataset import NexuDataset


@data_source("my_dataset")
class MyDataset(DataSourceDefinition):
    num_samples: int = 1000
    feature_dim: int = 64
    seed: int = 42

    def build(self) -> NexuDataset:
        return _MyDatasetRuntime(**self.model_dump())


class _MyDatasetRuntime(NexuDataset):
    def __init__(self, num_samples: int, feature_dim: int, seed: int):
        super().__init__(label_names=["label"])
        generator = torch.Generator().manual_seed(seed)
        self.features = torch.randn(num_samples, feature_dim, generator=generator)
        self.labels = torch.randint(0, 10, (num_samples,), generator=generator)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[TensorDict, TensorDict]:
        x = TensorDict({"features": self.features[index]}, batch_size=[])
        y = TensorDict({"label": self.labels[index]}, batch_size=[])
        return x, y
```

Use the definition directly:

```python
from nexuml.core.types import DataSpec, DatasetSpec
from my_library.data.my_dataset import MyDataset

data = DataSpec(
    source=MyDataset(num_samples=2000, feature_dim=64),
    input_shapes={"features": [64]},
)

data_list = DataSpec(
    datasets=[DatasetSpec(source=MyDataset(num_samples=2000))],
    input_shapes={"features": [64]},
)
```

Split, modality, preprocessing, label merging, loader policy, and graph input shapes stay on `DataSpec` or `DatasetSpec`. Dataset-specific values stay on `MyDataset`.

Verify discovery with:

```bash
nexuml library add /path/to/my_library
nexuml registry list data
```

YAML stores `type: my_dataset`, `version: '1'`, and validated parameters. It does not store the Python import path.

## Dataset Contract

- `build()` returns a `NexuDataset` runtime.
- Runtime `__getitem__` returns `(x: TensorDict, y: TensorDict | None)`.
- Loaded tensors and mutable dataset state stay off the public definition.

## See Also

- [Discovery decorators](../reference/decorators.md)
- [Custom layer](custom-layer.md)
- [Custom library](custom-library.md)
