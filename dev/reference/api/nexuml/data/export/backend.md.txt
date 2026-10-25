# nexuml.data.export.backend

*module*

Export backend abstraction for SuperDataset data serialization.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/backend.py#L1-L1)

## nexuml.data.export.backend.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/backend.py#L14-L14)

## nexuml.data.export.backend.ExportBackend

*class*

```python
class ExportBackend(ABC)
```

Bases: `ABC`

Abstract base class for dataset export backends.

Each backend handles serialization of feature tensors to a specific
storage format. The metadata (labels, splits, source info) is always
stored as parquet alongside the backend-specific data directory.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/backend.py#L40-L89)

### nexuml.data.export.backend.ExportBackend.finalize

*method* · *abstract*

```python
def finalize() -> dict[str, Any]
```

Flush/close files. Return backend-specific metadata for config.yaml.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/backend.py#L82-L84)

### nexuml.data.export.backend.ExportBackend.initialize

*method* · *abstract*

```python
def initialize(export_dir: Path, num_samples: int, feature_shapes: dict[str, tuple[int, ...]], dtype: Any | None = None) -> None
```

Prepare storage (create files, allocate mmap, etc.).

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/backend.py#L51-L59)

### nexuml.data.export.backend.ExportBackend.load_sample

*method* · *abstract* · *staticmethod*

```python
def load_sample(export_dir: Path, index: int) -> dict[str, torch.Tensor]
```

Load one sample's feature tensors from an exported directory.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/backend.py#L86-L89)

### nexuml.data.export.backend.ExportBackend.save_batch

*method*

```python
def save_batch(start_index: int, features: dict[str, torch.Tensor]) -> None
```

Save a batch of feature tensors.

Backends can override this for more efficient contiguous writes.

**Raises**

- `ValueError` — If batch sizes across feature tensors are inconsistent.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/backend.py#L65-L80)

### nexuml.data.export.backend.ExportBackend.save_sample

*method* · *abstract*

```python
def save_sample(index: int, features: dict[str, torch.Tensor]) -> None
```

Save one sample's feature tensors.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/backend.py#L61-L63)

## nexuml.data.export.backend.ExportConfig

*class* · *dataclass*

```python
class ExportConfig
```

Schema for config.yaml written alongside exported data.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/backend.py#L19-L37)

### nexuml.data.export.backend.ExportConfig.backend

*attribute* · *instance attribute*

```python
backend: str
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/backend.py#L24-L24)

### nexuml.data.export.backend.ExportConfig.extra

*attribute* · *class attribute* · *instance attribute*

```python
extra: dict[str, Any] = field(default_factory=dict)
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/backend.py#L37-L37)

### nexuml.data.export.backend.ExportConfig.feature_shapes

*attribute* · *instance attribute*

```python
feature_shapes: dict[str, list[int]]
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/backend.py#L33-L33)

### nexuml.data.export.backend.ExportConfig.format_version

*attribute* · *instance attribute*

```python
format_version: int
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/backend.py#L23-L23)

### nexuml.data.export.backend.ExportConfig.key_specs

*attribute* · *class attribute* · *instance attribute*

```python
key_specs: dict[str, dict[str, Any]] = field(default_factory=dict)
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/backend.py#L34-L34)

### nexuml.data.export.backend.ExportConfig.label_names

*attribute* · *instance attribute*

```python
label_names: list[str]
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/backend.py#L27-L27)

### nexuml.data.export.backend.ExportConfig.label_prefix

*attribute* · *instance attribute*

```python
label_prefix: str
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/backend.py#L32-L32)

### nexuml.data.export.backend.ExportConfig.merge_labels_config

*attribute* · *class attribute* · *instance attribute*

```python
merge_labels_config: dict[str, Any] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/backend.py#L36-L36)

### nexuml.data.export.backend.ExportConfig.modality

*attribute* · *instance attribute*

```python
modality: str
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/backend.py#L29-L29)

### nexuml.data.export.backend.ExportConfig.num_classes

*attribute* · *instance attribute*

```python
num_classes: dict[str, int]
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/backend.py#L28-L28)

### nexuml.data.export.backend.ExportConfig.num_samples

*attribute* · *instance attribute*

```python
num_samples: int
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/backend.py#L26-L26)

### nexuml.data.export.backend.ExportConfig.source_datasets

*attribute* · *class attribute* · *instance attribute*

```python
source_datasets: list[str] = field(default_factory=list)
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/backend.py#L35-L35)

### nexuml.data.export.backend.ExportConfig.writer

*attribute* · *instance attribute*

```python
writer: str
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/backend.py#L25-L25)

### nexuml.data.export.backend.ExportConfig.x_keys

*attribute* · *instance attribute*

```python
x_keys: list[str]
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/backend.py#L30-L30)

### nexuml.data.export.backend.ExportConfig.y_keys

*attribute* · *instance attribute*

```python
y_keys: list[str]
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/backend.py#L31-L31)

## nexuml.data.export.backend.get_export_backend

*function*

```python
def get_export_backend(name: str) -> type[ExportBackend]
```

Retrieve a registered export backend by name.

**Returns**

- (`type[ExportBackend]`) — The backend class for the given name.

**Raises**

- `KeyError` — If no backend is registered under that name.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/backend.py#L117-L129)

## nexuml.data.export.backend.get_export_backend_name

*function*

```python
def get_export_backend_name(backend_type: type[ExportBackend]) -> str
```

Return the registered name for an export backend class.

**Returns**

- (`str`) — Registered backend name.

**Raises**

- `KeyError` — If the backend class is not registered.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/backend.py#L132-L146)

## nexuml.data.export.backend.list_export_backends

*function*

```python
def list_export_backends() -> list[str]
```

Return names of all registered export backends.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/backend.py#L149-L151)

## nexuml.data.export.backend.register_export_backend

*function*

```python
def register_export_backend(name: str) -> Callable[[type[_ExportBackendT]], type[_ExportBackendT]]
```

Decorator to register an export backend.

**Returns**

- (`Callable[[type[_ExportBackendT]], type[_ExportBackendT]]`) — A decorator that registers the backend class and returns it.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/backend.py#L99-L114)
