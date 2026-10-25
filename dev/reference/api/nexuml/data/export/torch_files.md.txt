# nexuml.data.export.torch_files

*module*

Torch-file export backend.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/torch_files.py#L1-L1)

## nexuml.data.export.torch_files.TorchBackend

*class*

```python
class TorchBackend(ExportBackend)
```

Bases: `nexuml.data.export.backend.ExportBackend`

Store one `.pt` file per sample containing all exported keys.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/torch_files.py#L17-L91)

### nexuml.data.export.torch_files.TorchBackend.finalize

*method*

```python
def finalize() -> dict[str, Any]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/torch_files.py#L67-L84)

### nexuml.data.export.torch_files.TorchBackend.initialize

*method*

```python
def initialize(export_dir: Path, num_samples: int, feature_shapes: dict[str, tuple[int, ...]], dtype: torch.dtype | str | None = None) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/torch_files.py#L28-L43)

### nexuml.data.export.torch_files.TorchBackend.load_sample

*method* · *staticmethod*

```python
def load_sample(export_dir: Path, index: int) -> dict[str, torch.Tensor]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/torch_files.py#L86-L91)

### nexuml.data.export.torch_files.TorchBackend.save_sample

*method*

```python
def save_sample(index: int, features: dict[str, torch.Tensor]) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/export/torch_files.py#L45-L65)

### nexuml.data.export.torch_files.TorchBackend.save_batch

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
