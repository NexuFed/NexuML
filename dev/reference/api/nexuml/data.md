# nexuml.data

*module*

Data loading primitives for NexuML.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/__init__.py#L1-L1)

## nexuml.data.DaliLoader

*class* · *pydantic model*

```python
class DaliLoader(LoaderBackendDefinition)
```

Bases: `nexuml.core.components.LoaderBackendDefinition`

Re-exported from: `nexuml.data.loaders.definitions`

Use the NVIDIA DALI data loader backend.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/definitions.py#L19-L26)

### nexuml.data.DaliLoader.kind

*attribute* · *pydantic field*

```python
kind = 'loader_backend'
```

Inherited from: `nexuml.core.components.LoaderBackendDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/components.py#L159-L159)

### nexuml.data.DaliLoader.component_name

*attribute* · *class attribute*

```python
component_name: str
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/components.py#L34-L34)

### nexuml.data.DaliLoader.component_version

*attribute* · *class attribute*

```python
component_version: str = '1'
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/components.py#L35-L35)

### nexuml.data.DaliLoader.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid', frozen=True, validate_default=True, allow_inf_nan=False, arbitrary_types_allowed=False)
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/components.py#L25-L31)

### nexuml.data.DaliLoader.build

*method*

```python
def build()
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/definitions.py#L23-L26)

## nexuml.data.NexuDataCreator

*class*

```python
class NexuDataCreator
```

Re-exported from: `nexuml.data.creator`

Build datasets and datamodules from a ``DataSpec``.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/creator.py#L15-L162)

### nexuml.data.NexuDataCreator.build

*method*

```python
def build(spec: DataSpec, default_batch_size: int | AutoBatchSizeSpec = 64) -> NexuDataModule
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/creator.py#L104-L119)

### nexuml.data.NexuDataCreator.build_dataset

*method*

```python
def build_dataset(spec: DataSpec) -> NexuDataset
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/creator.py#L18-L102)

## nexuml.data.NexuDataModule

*class*

```python
class NexuDataModule(L.LightningDataModule)
```

Bases: `L.LightningDataModule`

Re-exported from: `nexuml.data.module`

Lightning DataModule wrapping a ``NexuDataset``.

Supports:
- random splits or metadata-driven splits
- weighted sampling for the native torch backend
- backend-driven loader construction via ``LoaderSpec.backend``

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/module.py#L18-L154)

### nexuml.data.NexuDataModule.dataset

*attribute* · *instance attribute*

```python
dataset = dataset
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/module.py#L38-L38)

### nexuml.data.NexuDataModule.loader_spec

*attribute* · *instance attribute*

```python
loader_spec = loader_spec or LoaderSpec()
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/module.py#L39-L39)

### nexuml.data.NexuDataModule.seed

*attribute* · *instance attribute*

```python
seed = seed
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/module.py#L43-L43)

### nexuml.data.NexuDataModule.split_by_column

*attribute* · *instance attribute*

```python
split_by_column = split_by_column
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/module.py#L44-L44)

### nexuml.data.NexuDataModule.test_split

*attribute* · *instance attribute*

```python
test_split = test_split
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/module.py#L42-L42)

### nexuml.data.NexuDataModule.train_split

*attribute* · *instance attribute*

```python
train_split = train_split
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/module.py#L40-L40)

### nexuml.data.NexuDataModule.val_split

*attribute* · *instance attribute*

```python
val_split = val_split
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/module.py#L41-L41)

### nexuml.data.NexuDataModule.predict_dataloader

*method*

```python
def predict_dataloader()
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/module.py#L153-L154)

### nexuml.data.NexuDataModule.setup

*method*

```python
def setup(stage: str | None = None) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/module.py#L49-L56)

### nexuml.data.NexuDataModule.test_dataloader

*method*

```python
def test_dataloader()
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/module.py#L150-L151)

### nexuml.data.NexuDataModule.train_dataloader

*method*

```python
def train_dataloader()
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/module.py#L138-L145)

### nexuml.data.NexuDataModule.val_dataloader

*method*

```python
def val_dataloader()
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/module.py#L147-L148)

## nexuml.data.NexuDataset

*class*

```python
class NexuDataset(torch.utils.data.Dataset)
```

Bases: `torch.utils.data.Dataset`

Re-exported from: `nexuml.data.dataset`

Base dataset returning ``(x, y)`` TensorDict pairs.

The base class supports two common patterns used throughout NexuML:

- metadata-backed datasets via ``self.meta``
- in-memory datasets via ``self.data``

Subclasses can still override ``__getitem__`` entirely for bespoke behavior.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/dataset.py#L34-L278)

### nexuml.data.NexuDataset.data

*attribute* · *instance attribute*

```python
data = data
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/dataset.py#L58-L58)

### nexuml.data.NexuDataset.label_names

*attribute* · *instance attribute*

```python
label_names: list[str] = list(label_names or [])
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/dataset.py#L60-L60)

### nexuml.data.NexuDataset.meta

*attribute* · *instance attribute*

```python
meta: pd.DataFrame | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/dataset.py#L56-L56)

### nexuml.data.NexuDataset.modality

*attribute* · *instance attribute*

```python
modality = modality
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/dataset.py#L57-L57)

### nexuml.data.NexuDataset.num_classes

*attribute* · *instance attribute*

```python
num_classes: dict[str, int] = {}
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/dataset.py#L61-L61)

### nexuml.data.NexuDataset.split_ratio

*attribute* · *instance attribute*

```python
split_ratio = split_ratio or [0.85, 0.1, 0.05]
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/dataset.py#L59-L59)

### nexuml.data.NexuDataset.clone_with_meta

*method*

```python
def clone_with_meta(meta: pd.DataFrame, data=_KEEP_DATA) -> Self
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/dataset.py#L119-L128)

### nexuml.data.NexuDataset.download

*method*

```python
def download() -> None
```

Override to implement dataset downloading.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/dataset.py#L276-L278)

### nexuml.data.NexuDataset.get_split

*method*

```python
def get_split(split: str) -> Self
```

Return a view of this dataset filtered to a single split.

**Returns**

- (`Self`) — A new :class:`NexuDataset` containing only the requested split.

**Raises**

- `ValueError` — If the dataset has no metadata.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/dataset.py#L188-L201)

### nexuml.data.NexuDataset.load_item

*method*

```python
def load_item(idx: int, row: pd.Series) -> TensorDict
```

Load features for a single sample.

Subclasses can override this to lazily load file-backed content.

**Returns**

- (`TensorDict`) — Feature tensor dictionary for the sample.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/dataset.py#L203-L227)

### nexuml.data.NexuDataset.load_labels

*method*

```python
def load_labels(idx: int, row: pd.Series) -> TensorDict | None
```

Load labels for a single sample.

By default labels are read from ``self.meta`` using ``self.label_names``.
Subclasses can override this for file-backed or computed targets.

**Returns**

- (`TensorDict | None`) — Label tensor dictionary, or ``None`` if no labels are configured.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/dataset.py#L237-L255)

### nexuml.data.NexuDataset.split_meta

*method*

```python
def split_meta(split_ratio: list[float]) -> None
```

Convert 'fit' rows to train/val and 'all' rows to train/val/test.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/dataset.py#L163-L186)

### nexuml.data.NexuDataset.take

*method*

```python
def take(indices: Sequence[int]) -> Self
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/dataset.py#L154-L161)

## nexuml.data.TensorShardsLoader

*class* · *pydantic model*

```python
class TensorShardsLoader(LoaderBackendDefinition)
```

Bases: `nexuml.core.components.LoaderBackendDefinition`

Re-exported from: `nexuml.data.loaders.definitions`

Configure the windowed tensor-shard loader backend.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/definitions.py#L29-L45)

### nexuml.data.TensorShardsLoader.drop_last

*attribute* · *pydantic field*

```python
drop_last: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/definitions.py#L39-L39)

### nexuml.data.TensorShardsLoader.pin_memory

*attribute* · *pydantic field*

```python
pin_memory: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/definitions.py#L38-L38)

### nexuml.data.TensorShardsLoader.prefetch_windows

*attribute* · *pydantic field*

```python
prefetch_windows: int = 2
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/definitions.py#L34-L34)

### nexuml.data.TensorShardsLoader.prefetch_workers

*attribute* · *pydantic field*

```python
prefetch_workers: int = 2
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/definitions.py#L35-L35)

### nexuml.data.TensorShardsLoader.seed

*attribute* · *pydantic field*

```python
seed: int | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/definitions.py#L40-L40)

### nexuml.data.TensorShardsLoader.shards_per_window

*attribute* · *pydantic field*

```python
shards_per_window: int = 6
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/definitions.py#L33-L33)

### nexuml.data.TensorShardsLoader.shuffle_samples

*attribute* · *pydantic field*

```python
shuffle_samples: bool | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/definitions.py#L37-L37)

### nexuml.data.TensorShardsLoader.shuffle_shards

*attribute* · *pydantic field*

```python
shuffle_shards: bool | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/definitions.py#L36-L36)

### nexuml.data.TensorShardsLoader.kind

*attribute* · *pydantic field*

```python
kind = 'loader_backend'
```

Inherited from: `nexuml.core.components.LoaderBackendDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/components.py#L159-L159)

### nexuml.data.TensorShardsLoader.component_name

*attribute* · *class attribute*

```python
component_name: str
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/components.py#L34-L34)

### nexuml.data.TensorShardsLoader.component_version

*attribute* · *class attribute*

```python
component_version: str = '1'
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/components.py#L35-L35)

### nexuml.data.TensorShardsLoader.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid', frozen=True, validate_default=True, allow_inf_nan=False, arbitrary_types_allowed=False)
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/components.py#L25-L31)

### nexuml.data.TensorShardsLoader.build

*method*

```python
def build()
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/definitions.py#L42-L45)

## nexuml.data.TorchLoader

*class* · *pydantic model*

```python
class TorchLoader(LoaderBackendDefinition)
```

Bases: `nexuml.core.components.LoaderBackendDefinition`

Re-exported from: `nexuml.data.loaders.definitions`

Use the standard PyTorch data loader backend.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/definitions.py#L9-L16)

### nexuml.data.TorchLoader.kind

*attribute* · *pydantic field*

```python
kind = 'loader_backend'
```

Inherited from: `nexuml.core.components.LoaderBackendDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/components.py#L159-L159)

### nexuml.data.TorchLoader.component_name

*attribute* · *class attribute*

```python
component_name: str
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/components.py#L34-L34)

### nexuml.data.TorchLoader.component_version

*attribute* · *class attribute*

```python
component_version: str = '1'
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/components.py#L35-L35)

### nexuml.data.TorchLoader.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid', frozen=True, validate_default=True, allow_inf_nan=False, arbitrary_types_allowed=False)
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/core/components.py#L25-L31)

### nexuml.data.TorchLoader.build

*method*

```python
def build()
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/definitions.py#L13-L16)

**Modules**

- `nexuml.data.auto_batch`
- `nexuml.data.creator`
- `nexuml.data.dataset`
- `nexuml.data.export`
- `nexuml.data.exported`
- `nexuml.data.file_io`
- `nexuml.data.loaders`
- `nexuml.data.module`
- `nexuml.data.super_dataset`
