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
    fixed_label_keys = ("label",)

    num_samples: int = 1000
    feature_dim: int = 64
    seed: int = 42

    def build(self) -> NexuDataset:
        return _MyDatasetRuntime(**self.model_dump())


class _MyDatasetRuntime(NexuDataset):
    def __init__(self, num_samples: int, feature_dim: int, seed: int):
        assert MyDataset.fixed_label_keys is not None
        super().__init__(label_names=list(MyDataset.fixed_label_keys))
        generator = torch.Generator().manual_seed(seed)
        self.features = torch.randn(num_samples, feature_dim, generator=generator)
        self.labels = torch.randint(0, 10, (num_samples,), generator=generator)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[TensorDict, TensorDict]:
        x = TensorDict({"features": self.features[index]}, batch_size=[])
        y = TensorDict({self.label_names[0]: self.labels[index]}, batch_size=[])
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

Make the package discoverable through [library registration](register-library.md)
or a [local root](library-cli.md), then verify with `nexuml registry list data`.

YAML stores `type: my_dataset`, `version: '1'`, and validated parameters. It does not store the Python import path.

## Dataset Contract

- `build()` returns a `NexuDataset` runtime.
- Runtime `__getitem__` returns `(x: TensorDict, y: TensorDict | None)`.
- Declare fixed labels once with `fixed_label_keys` and reuse them in the runtime. `None` means unknown; `()` means no labels. This class variable is not a parameter or YAML field.
- Loaded tensors and mutable dataset state stay off the public definition.
- Sources with fixed episode/group partitions can override runtime
  `validate_split_type(split_type)` to reject unsafe repartitioning. The creator
  calls it before changing metadata or composing sources; ordinary sources keep
  the default no-op. Use `split_type="keep"` for the
  [local LeRobot source](lerobot-data.md).
- Dataset composition preserves source-owned `(x, y)` pairs. Metadata-backed
  labels still use the existing aggregate label-merging rules.

## See Also

- [Discovery decorators](../reference/decorators.md)
- [Custom layer](custom-layer.md)
- [Custom library](custom-library.md)
