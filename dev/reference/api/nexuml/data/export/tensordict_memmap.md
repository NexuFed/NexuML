# nexuml.data.export.tensordict_memmap

*module*

TensorDict memmap export backend.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/export/tensordict_memmap.py#L1-L1)

## nexuml.data.export.tensordict_memmap.TensorDictMemmapBackend

*class*

```python
class TensorDictMemmapBackend(ExportBackend)
```

Bases: `nexuml.data.export.backend.ExportBackend`

Persist the exported dataset as one memmapped TensorDict.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/export/tensordict_memmap.py#L15-L113)

### nexuml.data.export.tensordict_memmap.TensorDictMemmapBackend.finalize

*method*

```python
def finalize() -> dict[str, Any]
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/export/tensordict_memmap.py#L92-L107)

### nexuml.data.export.tensordict_memmap.TensorDictMemmapBackend.initialize

*method*

```python
def initialize(export_dir: Path, num_samples: int, feature_shapes: dict[str, tuple[int, ...]], dtype: np.dtype[Any] | str | None = None) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/export/tensordict_memmap.py#L27-L53)

### nexuml.data.export.tensordict_memmap.TensorDictMemmapBackend.load_sample

*method* · *staticmethod*

```python
def load_sample(export_dir: Path, index: int) -> dict[str, torch.Tensor]
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/export/tensordict_memmap.py#L109-L113)

### nexuml.data.export.tensordict_memmap.TensorDictMemmapBackend.save_batch

*method*

```python
def save_batch(start_index: int, features: dict[str, torch.Tensor]) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/export/tensordict_memmap.py#L72-L90)

### nexuml.data.export.tensordict_memmap.TensorDictMemmapBackend.save_sample

*method*

```python
def save_sample(index: int, features: dict[str, torch.Tensor]) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/export/tensordict_memmap.py#L55-L70)
