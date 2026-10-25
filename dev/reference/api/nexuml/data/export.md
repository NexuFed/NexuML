# nexuml.data.export

*module*

Dataset export backends for NexuML.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/__init__.py#L1-L1)

## nexuml.data.export.BatchTransform

*attribute* · *module attribute*

```python
BatchTransform: TypeAlias = Callable[[TensorDict, TensorDict | None], tuple[TensorDict, TensorDict | None]]
```

Re-exported from: `nexuml.data.export.runner`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/runner.py#L28-L31)

## nexuml.data.export.ExportBackend

*class*

```python
class ExportBackend(ABC)
```

Bases: `ABC`

Re-exported from: `nexuml.data.export.backend`

Abstract base class for dataset export backends.

Each backend handles serialization of feature tensors to a specific
storage format. The metadata (labels, splits, source info) is always
stored as parquet alongside the backend-specific data directory.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L40-L89)

### nexuml.data.export.ExportBackend.finalize

*method* · *abstract*

```python
def finalize() -> dict[str, Any]
```

Flush/close files. Return backend-specific metadata for config.yaml.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L82-L84)

### nexuml.data.export.ExportBackend.initialize

*method* · *abstract*

```python
def initialize(export_dir: Path, num_samples: int, feature_shapes: dict[str, tuple[int, ...]], dtype: Any | None = None) -> None
```

Prepare storage (create files, allocate mmap, etc.).

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L51-L59)

### nexuml.data.export.ExportBackend.load_sample

*method* · *abstract* · *staticmethod*

```python
def load_sample(export_dir: Path, index: int) -> dict[str, torch.Tensor]
```

Load one sample's feature tensors from an exported directory.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L86-L89)

### nexuml.data.export.ExportBackend.save_batch

*method*

```python
def save_batch(start_index: int, features: dict[str, torch.Tensor]) -> None
```

Save a batch of feature tensors.

Backends can override this for more efficient contiguous writes.

**Raises**

- `ValueError` — If batch sizes across feature tensors are inconsistent.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L65-L80)

### nexuml.data.export.ExportBackend.save_sample

*method* · *abstract*

```python
def save_sample(index: int, features: dict[str, torch.Tensor]) -> None
```

Save one sample's feature tensors.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L61-L63)

## nexuml.data.export.ExportConfig

*class* · *dataclass*

```python
class ExportConfig
```

Re-exported from: `nexuml.data.export.backend`

Schema for config.yaml written alongside exported data.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L19-L37)

### nexuml.data.export.ExportConfig.backend

*attribute* · *instance attribute*

```python
backend: str
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L24-L24)

### nexuml.data.export.ExportConfig.extra

*attribute* · *class attribute* · *instance attribute*

```python
extra: dict[str, Any] = field(default_factory=dict)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L37-L37)

### nexuml.data.export.ExportConfig.feature_shapes

*attribute* · *instance attribute*

```python
feature_shapes: dict[str, list[int]]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L33-L33)

### nexuml.data.export.ExportConfig.format_version

*attribute* · *instance attribute*

```python
format_version: int
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L23-L23)

### nexuml.data.export.ExportConfig.key_specs

*attribute* · *class attribute* · *instance attribute*

```python
key_specs: dict[str, dict[str, Any]] = field(default_factory=dict)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L34-L34)

### nexuml.data.export.ExportConfig.label_names

*attribute* · *instance attribute*

```python
label_names: list[str]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L27-L27)

### nexuml.data.export.ExportConfig.label_prefix

*attribute* · *instance attribute*

```python
label_prefix: str
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L32-L32)

### nexuml.data.export.ExportConfig.merge_labels_config

*attribute* · *class attribute* · *instance attribute*

```python
merge_labels_config: dict[str, Any] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L36-L36)

### nexuml.data.export.ExportConfig.modality

*attribute* · *instance attribute*

```python
modality: str
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L29-L29)

### nexuml.data.export.ExportConfig.num_classes

*attribute* · *instance attribute*

```python
num_classes: dict[str, int]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L28-L28)

### nexuml.data.export.ExportConfig.num_samples

*attribute* · *instance attribute*

```python
num_samples: int
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L26-L26)

### nexuml.data.export.ExportConfig.source_datasets

*attribute* · *class attribute* · *instance attribute*

```python
source_datasets: list[str] = field(default_factory=list)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L35-L35)

### nexuml.data.export.ExportConfig.writer

*attribute* · *instance attribute*

```python
writer: str
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L25-L25)

### nexuml.data.export.ExportConfig.x_keys

*attribute* · *instance attribute*

```python
x_keys: list[str]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L30-L30)

### nexuml.data.export.ExportConfig.y_keys

*attribute* · *instance attribute*

```python
y_keys: list[str]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L31-L31)

## nexuml.data.export.NumpyBackend

*class*

```python
class NumpyBackend(ExportBackend)
```

Bases: `nexuml.data.export.backend.ExportBackend`

Re-exported from: `nexuml.data.export.numpy_files`

Store each key as one `.npy` file per sample.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/numpy_files.py#L21-L98)

### nexuml.data.export.NumpyBackend.finalize

*method*

```python
def finalize() -> dict[str, Any]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/numpy_files.py#L71-L87)

### nexuml.data.export.NumpyBackend.initialize

*method*

```python
def initialize(export_dir: Path, num_samples: int, feature_shapes: dict[str, tuple[int, ...]], dtype: np.dtype[Any] | str | None = None) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/numpy_files.py#L32-L45)

### nexuml.data.export.NumpyBackend.load_sample

*method* · *staticmethod*

```python
def load_sample(export_dir: Path, index: int) -> dict[str, torch.Tensor]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/numpy_files.py#L89-L98)

### nexuml.data.export.NumpyBackend.save_sample

*method*

```python
def save_sample(index: int, features: dict[str, torch.Tensor]) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/numpy_files.py#L47-L69)

### nexuml.data.export.NumpyBackend.save_batch

*method*

```python
def save_batch(start_index: int, features: dict[str, torch.Tensor]) -> None
```

Inherited from: `nexuml.data.export.backend.ExportBackend`

Save a batch of feature tensors.

Backends can override this for more efficient contiguous writes.

**Raises**

- `ValueError` — If batch sizes across feature tensors are inconsistent.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L65-L80)

## nexuml.data.export.NumpyMmapBackend

*class*

```python
class NumpyMmapBackend(ExportBackend)
```

Bases: `nexuml.data.export.backend.ExportBackend`

Re-exported from: `nexuml.data.export.numpy_mmap`

Memory-mapped NumPy backend for large-scale datasets.

Pre-allocates a single `.npy` file per feature key with shape
`(num_samples, *feature_shape)` for efficient sequential writes and
memory-mapped reads. Requires all samples to have uniform feature shapes.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/numpy_mmap.py#L17-L170)

### nexuml.data.export.NumpyMmapBackend.finalize

*method*

```python
def finalize() -> dict[str, Any]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/numpy_mmap.py#L128-L152)

### nexuml.data.export.NumpyMmapBackend.initialize

*method*

```python
def initialize(export_dir: Path, num_samples: int, feature_shapes: dict[str, tuple[int, ...]], dtype: np.dtype[Any] | str | None = None) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/numpy_mmap.py#L34-L65)

### nexuml.data.export.NumpyMmapBackend.load_sample

*method* · *staticmethod*

```python
def load_sample(export_dir: Path, index: int) -> dict[str, torch.Tensor]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/numpy_mmap.py#L154-L170)

### nexuml.data.export.NumpyMmapBackend.save_batch

*method*

```python
def save_batch(start_index: int, features: dict[str, torch.Tensor]) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/numpy_mmap.py#L92-L126)

### nexuml.data.export.NumpyMmapBackend.save_sample

*method*

```python
def save_sample(index: int, features: dict[str, torch.Tensor]) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/numpy_mmap.py#L67-L90)

## nexuml.data.export.TensorDictMemmapBackend

*class*

```python
class TensorDictMemmapBackend(ExportBackend)
```

Bases: `nexuml.data.export.backend.ExportBackend`

Re-exported from: `nexuml.data.export.tensordict_memmap`

Persist the exported dataset as one memmapped TensorDict.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/tensordict_memmap.py#L15-L113)

### nexuml.data.export.TensorDictMemmapBackend.finalize

*method*

```python
def finalize() -> dict[str, Any]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/tensordict_memmap.py#L92-L107)

### nexuml.data.export.TensorDictMemmapBackend.initialize

*method*

```python
def initialize(export_dir: Path, num_samples: int, feature_shapes: dict[str, tuple[int, ...]], dtype: np.dtype[Any] | str | None = None) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/tensordict_memmap.py#L27-L53)

### nexuml.data.export.TensorDictMemmapBackend.load_sample

*method* · *staticmethod*

```python
def load_sample(export_dir: Path, index: int) -> dict[str, torch.Tensor]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/tensordict_memmap.py#L109-L113)

### nexuml.data.export.TensorDictMemmapBackend.save_batch

*method*

```python
def save_batch(start_index: int, features: dict[str, torch.Tensor]) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/tensordict_memmap.py#L72-L90)

### nexuml.data.export.TensorDictMemmapBackend.save_sample

*method*

```python
def save_sample(index: int, features: dict[str, torch.Tensor]) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/tensordict_memmap.py#L55-L70)

## nexuml.data.export.TensorShardsBackend

*class*

```python
class TensorShardsBackend(ExportBackend)
```

Bases: `nexuml.data.export.backend.ExportBackend`

Re-exported from: `nexuml.data.export.tensor_shards`

Write arbitrary fixed-shape tensor keys into sample shards.

A shard is a storage unit with ``N`` samples along dimension zero. It is
NOT a training batch; the runtime loader may create arbitrary batch sizes
from the shard contents.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/tensor_shards.py#L52-L520)

### nexuml.data.export.TensorShardsBackend.samples_per_shard

*attribute* · *instance attribute*

```python
samples_per_shard = int(samples_per_shard)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/tensor_shards.py#L70-L70)

### nexuml.data.export.TensorShardsBackend.end_split

*method*

```python
def end_split(split: str) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/tensor_shards.py#L140-L146)

### nexuml.data.export.TensorShardsBackend.finalize

*method*

```python
def finalize() -> dict[str, Any]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/tensor_shards.py#L208-L260)

### nexuml.data.export.TensorShardsBackend.initialize

*method*

```python
def initialize(export_dir: Path, num_samples: int, feature_shapes: dict[str, tuple[int, ...]], dtype: object | None = None) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/tensor_shards.py#L91-L127)

### nexuml.data.export.TensorShardsBackend.load_manifest

*method* · *staticmethod*

```python
def load_manifest(export_dir: Path) -> dict[str, Any]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/tensor_shards.py#L262-L268)

### nexuml.data.export.TensorShardsBackend.load_sample

*method* · *staticmethod*

```python
def load_sample(export_dir: Path, index: int) -> dict[str, torch.Tensor]
```

Map-style compatibility path; the window loader should use load_shard.

**Returns**

- (`dict[str, torch.Tensor]`) — dict[str, torch.Tensor]: Feature tensors for the requested sample index.

**Raises**

- `IndexError` — If the sample index is outside available shard bounds.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/tensor_shards.py#L330-L360)

### nexuml.data.export.TensorShardsBackend.load_shard

*method* · *staticmethod*

```python
def load_shard(export_dir: Path, shard: int | str | dict[str, Any]) -> dict[str, Any]
```

Load a complete shard for the future windowed loader.

**Returns**

- (`dict[str, Any]`) — dict[str, Any]: Shard payload with metadata and CPU feature tensors.

**Raises**

- `IndexError` — If an integer shard index is out of range.
- `ValueError` — If the shard content or metadata is invalid.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/tensor_shards.py#L270-L328)

### nexuml.data.export.TensorShardsBackend.save_batch

*method*

```python
def save_batch(start_index: int, features: dict[str, torch.Tensor]) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/tensor_shards.py#L158-L206)

### nexuml.data.export.TensorShardsBackend.save_sample

*method*

```python
def save_sample(index: int, features: dict[str, torch.Tensor]) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/tensor_shards.py#L148-L156)

### nexuml.data.export.TensorShardsBackend.start_split

*method*

```python
def start_split(split: str) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/tensor_shards.py#L130-L138)

## nexuml.data.export.TorchBackend

*class*

```python
class TorchBackend(ExportBackend)
```

Bases: `nexuml.data.export.backend.ExportBackend`

Re-exported from: `nexuml.data.export.torch_files`

Store one `.pt` file per sample containing all exported keys.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/torch_files.py#L17-L91)

### nexuml.data.export.TorchBackend.finalize

*method*

```python
def finalize() -> dict[str, Any]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/torch_files.py#L67-L84)

### nexuml.data.export.TorchBackend.initialize

*method*

```python
def initialize(export_dir: Path, num_samples: int, feature_shapes: dict[str, tuple[int, ...]], dtype: torch.dtype | str | None = None) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/torch_files.py#L28-L43)

### nexuml.data.export.TorchBackend.load_sample

*method* · *staticmethod*

```python
def load_sample(export_dir: Path, index: int) -> dict[str, torch.Tensor]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/torch_files.py#L86-L91)

### nexuml.data.export.TorchBackend.save_sample

*method*

```python
def save_sample(index: int, features: dict[str, torch.Tensor]) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/torch_files.py#L45-L65)

### nexuml.data.export.TorchBackend.save_batch

*method*

```python
def save_batch(start_index: int, features: dict[str, torch.Tensor]) -> None
```

Inherited from: `nexuml.data.export.backend.ExportBackend`

Save a batch of feature tensors.

Backends can override this for more efficient contiguous writes.

**Raises**

- `ValueError` — If batch sizes across feature tensors are inconsistent.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L65-L80)

## nexuml.data.export.WebDatasetBackend

*class*

```python
class WebDatasetBackend(ExportBackend)
```

Bases: `nexuml.data.export.backend.ExportBackend`

Re-exported from: `nexuml.data.export.webdataset`

Write lossless NumPy components into indexed WebDataset tar shards.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/webdataset.py#L33-L319)

### nexuml.data.export.WebDatasetBackend.create_index

*attribute* · *instance attribute*

```python
create_index = bool(create_index)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/webdataset.py#L52-L52)

### nexuml.data.export.WebDatasetBackend.samples_per_shard

*attribute* · *instance attribute*

```python
samples_per_shard = int(samples_per_shard)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/webdataset.py#L51-L51)

### nexuml.data.export.WebDatasetBackend.remote_uri

*property*

```python
remote_uri: str | None
```

S3 dataset root for remote exports.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/webdataset.py#L82-L84)

### nexuml.data.export.WebDatasetBackend.end_split

*method*

```python
def end_split(split: str) -> None
```

Finish one dataset split and close its current shard.

**Raises**

- `RuntimeError` — If the requested split is not the active split.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/webdataset.py#L113-L123)

### nexuml.data.export.WebDatasetBackend.finalize

*method*

```python
def finalize() -> dict[str, Any]
```

Close the final shard and return metadata for ``config.yaml``.

**Returns**

- (`dict[str, Any]`) — Backend-specific export metadata.

**Raises**

- `RuntimeError` — If the backend was never initialized.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/webdataset.py#L256-L283)

### nexuml.data.export.WebDatasetBackend.initialize

*method*

```python
def initialize(export_dir: Path, num_samples: int, feature_shapes: dict[str, tuple[int, ...]], dtype: np.dtype[Any] | str | None = None) -> None
```

Prepare local shard staging.

**Parameters**

- `export_dir` (`Path`) — Local export/staging directory.
- `num_samples` (`int`) — Declared number of exported samples.
- `feature_shapes` (`dict[str, tuple[int, ...]]`) — Tensor shapes supplied by the export runner.
- `dtype` (`np.dtype[Any] | str | None`) (default: `None`) — Optional floating-point storage dtype.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/webdataset.py#L86-L104)

### nexuml.data.export.WebDatasetBackend.load_sample

*method* · *staticmethod*

```python
def load_sample(export_dir: Path, index: int) -> dict[str, torch.Tensor]
```

Load one sample from a local WebDataset export.

**Returns**

- (`dict[str, torch.Tensor]`) — Mapping of stored keys to tensors.

**Raises**

- `IndexError` — If the sample is absent from the export index.
- `FileNotFoundError` — If a referenced tar member cannot be read.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/webdataset.py#L292-L319)

### nexuml.data.export.WebDatasetBackend.publish_export_metadata

*method*

```python
def publish_export_metadata(config_path: Path, metadata_path: Path) -> None
```

Upload the final lightweight export metadata for an S3 export.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/webdataset.py#L285-L290)

### nexuml.data.export.WebDatasetBackend.save_sample

*method*

```python
def save_sample(index: int, features: dict[str, torch.Tensor]) -> None
```

Append one tensor sample to the active tar shard.

**Raises**

- `ValueError` — If a component key contains a path separator.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/webdataset.py#L210-L254)

### nexuml.data.export.WebDatasetBackend.start_split

*method*

```python
def start_split(split: str) -> None
```

Start one physically isolated dataset split.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/webdataset.py#L106-L111)

### nexuml.data.export.WebDatasetBackend.save_batch

*method*

```python
def save_batch(start_index: int, features: dict[str, torch.Tensor]) -> None
```

Inherited from: `nexuml.data.export.backend.ExportBackend`

Save a batch of feature tensors.

Backends can override this for more efficient contiguous writes.

**Raises**

- `ValueError` — If batch sizes across feature tensors are inconsistent.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L65-L80)

## nexuml.data.export.export_data_module

*function*

```python
def export_data_module(data_module: NexuDataModule, path: str | Path, *, backend: str | WriterSpec = 'numpy', splits: Sequence[str] | None = None, transform: BatchTransform | None = None, x_keys: Sequence[str] | None = None, y_keys: Sequence[str] | None = None, include_labels: bool = True, label_prefix: str = 'label__', dtype: object | None = None, device: torch.device | str | None = None, **backend_kwargs: object) -> Path | str
```

Re-exported from: `nexuml.data.export.runner`

Export the data exactly as seen through a configured data module.

**Returns**

- (`Path | str`) — Local export path, or the original S3 URI for a remote WebDataset export.

**Raises**

- `ValueError` — If there are no samples or an unsupported remote backend is requested.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/runner.py#L70-L150)

## nexuml.data.export.get_export_backend

*function*

```python
def get_export_backend(name: str) -> type[ExportBackend]
```

Re-exported from: `nexuml.data.export.backend`

Retrieve a registered export backend by name.

**Returns**

- (`type[ExportBackend]`) — The backend class for the given name.

**Raises**

- `KeyError` — If no backend is registered under that name.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L117-L129)

## nexuml.data.export.get_export_backend_name

*function*

```python
def get_export_backend_name(backend_type: type[ExportBackend]) -> str
```

Re-exported from: `nexuml.data.export.backend`

Return the registered name for an export backend class.

**Returns**

- (`str`) — Registered backend name.

**Raises**

- `KeyError` — If the backend class is not registered.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L132-L146)

## nexuml.data.export.list_export_backends

*function*

```python
def list_export_backends() -> list[str]
```

Re-exported from: `nexuml.data.export.backend`

Return names of all registered export backends.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L149-L151)

## nexuml.data.export.register_export_backend

*function*

```python
def register_export_backend(name: str) -> Callable[[type[_ExportBackendT]], type[_ExportBackendT]]
```

Re-exported from: `nexuml.data.export.backend`

Decorator to register an export backend.

**Returns**

- (`Callable[[type[_ExportBackendT]], type[_ExportBackendT]]`) — A decorator that registers the backend class and returns it.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/backend.py#L99-L114)

**Modules**

- `nexuml.data.export.backend`
- `nexuml.data.export.numpy_files`
- `nexuml.data.export.numpy_mmap`
- `nexuml.data.export.runner`
- `nexuml.data.export.tensor_shards`
- `nexuml.data.export.tensordict_memmap`
- `nexuml.data.export.torch_files`
- `nexuml.data.export.webdataset`
