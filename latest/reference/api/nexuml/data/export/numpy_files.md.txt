# nexuml.data.export.numpy_files

*module*

File-per-sample NumPy export backend.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/numpy_files.py#L1-L1)

## nexuml.data.export.numpy_files.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/numpy_files.py#L14-L14)

## nexuml.data.export.numpy_files.NumpyBackend

*class*

```python
class NumpyBackend(ExportBackend)
```

Bases: `nexuml.data.export.backend.ExportBackend`

Store each key as one `.npy` file per sample.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/numpy_files.py#L21-L98)

### nexuml.data.export.numpy_files.NumpyBackend.finalize

*method*

```python
def finalize() -> dict[str, Any]
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/numpy_files.py#L71-L87)

### nexuml.data.export.numpy_files.NumpyBackend.initialize

*method*

```python
def initialize(export_dir: Path, num_samples: int, feature_shapes: dict[str, tuple[int, ...]], dtype: np.dtype[Any] | str | None = None) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/numpy_files.py#L32-L45)

### nexuml.data.export.numpy_files.NumpyBackend.load_sample

*method* · *staticmethod*

```python
def load_sample(export_dir: Path, index: int) -> dict[str, torch.Tensor]
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/numpy_files.py#L89-L98)

### nexuml.data.export.numpy_files.NumpyBackend.save_sample

*method*

```python
def save_sample(index: int, features: dict[str, torch.Tensor]) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/numpy_files.py#L47-L69)

### nexuml.data.export.numpy_files.NumpyBackend.save_batch

*method*

```python
def save_batch(start_index: int, features: dict[str, torch.Tensor]) -> None
```

Inherited from: `nexuml.data.export.backend.ExportBackend`

Save a batch of feature tensors.

Backends can override this for more efficient contiguous writes.

**Raises**

- `ValueError` — If batch sizes across feature tensors are inconsistent.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/backend.py#L65-L80)
