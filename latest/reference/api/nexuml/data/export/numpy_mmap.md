# nexuml.data.export.numpy_mmap

*module*

NumPy memory-mapped export backend.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/numpy_mmap.py#L1-L1)

## nexuml.data.export.numpy_mmap.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/numpy_mmap.py#L14-L14)

## nexuml.data.export.numpy_mmap.NumpyMmapBackend

*class*

```python
class NumpyMmapBackend(ExportBackend)
```

Bases: `nexuml.data.export.backend.ExportBackend`

Memory-mapped NumPy backend for large-scale datasets.

Pre-allocates a single `.npy` file per feature key with shape
`(num_samples, *feature_shape)` for efficient sequential writes and
memory-mapped reads. Requires all samples to have uniform feature shapes.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/numpy_mmap.py#L17-L170)

### nexuml.data.export.numpy_mmap.NumpyMmapBackend.finalize

*method*

```python
def finalize() -> dict[str, Any]
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/numpy_mmap.py#L128-L152)

### nexuml.data.export.numpy_mmap.NumpyMmapBackend.initialize

*method*

```python
def initialize(export_dir: Path, num_samples: int, feature_shapes: dict[str, tuple[int, ...]], dtype: np.dtype[Any] | str | None = None) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/numpy_mmap.py#L34-L65)

### nexuml.data.export.numpy_mmap.NumpyMmapBackend.load_sample

*method* · *staticmethod*

```python
def load_sample(export_dir: Path, index: int) -> dict[str, torch.Tensor]
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/numpy_mmap.py#L154-L170)

### nexuml.data.export.numpy_mmap.NumpyMmapBackend.save_batch

*method*

```python
def save_batch(start_index: int, features: dict[str, torch.Tensor]) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/numpy_mmap.py#L92-L126)

### nexuml.data.export.numpy_mmap.NumpyMmapBackend.save_sample

*method*

```python
def save_sample(index: int, features: dict[str, torch.Tensor]) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/numpy_mmap.py#L67-L90)
