# nexuml.data.export.tensor_shards

*module*

Non-prebatched tensor-shard export backend.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/tensor_shards.py#L1-L1)

## nexuml.data.export.tensor_shards.TensorShardsBackend

*class*

```python
class TensorShardsBackend(ExportBackend)
```

Bases: `nexuml.data.export.backend.ExportBackend`

Write arbitrary fixed-shape tensor keys into sample shards.

A shard is a storage unit with ``N`` samples along dimension zero. It is
NOT a training batch; the runtime loader may create arbitrary batch sizes
from the shard contents.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/tensor_shards.py#L52-L520)

### nexuml.data.export.tensor_shards.TensorShardsBackend.samples_per_shard

*attribute* · *instance attribute*

```python
samples_per_shard = int(samples_per_shard)
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/tensor_shards.py#L70-L70)

### nexuml.data.export.tensor_shards.TensorShardsBackend.end_split

*method*

```python
def end_split(split: str) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/tensor_shards.py#L140-L146)

### nexuml.data.export.tensor_shards.TensorShardsBackend.finalize

*method*

```python
def finalize() -> dict[str, Any]
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/tensor_shards.py#L208-L260)

### nexuml.data.export.tensor_shards.TensorShardsBackend.initialize

*method*

```python
def initialize(export_dir: Path, num_samples: int, feature_shapes: dict[str, tuple[int, ...]], dtype: object | None = None) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/tensor_shards.py#L91-L127)

### nexuml.data.export.tensor_shards.TensorShardsBackend.load_manifest

*method* · *staticmethod*

```python
def load_manifest(export_dir: Path) -> dict[str, Any]
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/tensor_shards.py#L262-L268)

### nexuml.data.export.tensor_shards.TensorShardsBackend.load_sample

*method* · *staticmethod*

```python
def load_sample(export_dir: Path, index: int) -> dict[str, torch.Tensor]
```

Map-style compatibility path; the window loader should use load_shard.

**Returns**

- (`dict[str, torch.Tensor]`) — dict[str, torch.Tensor]: Feature tensors for the requested sample index.

**Raises**

- `IndexError` — If the sample index is outside available shard bounds.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/tensor_shards.py#L330-L360)

### nexuml.data.export.tensor_shards.TensorShardsBackend.load_shard

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

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/tensor_shards.py#L270-L328)

### nexuml.data.export.tensor_shards.TensorShardsBackend.save_batch

*method*

```python
def save_batch(start_index: int, features: dict[str, torch.Tensor]) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/tensor_shards.py#L158-L206)

### nexuml.data.export.tensor_shards.TensorShardsBackend.save_sample

*method*

```python
def save_sample(index: int, features: dict[str, torch.Tensor]) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/tensor_shards.py#L148-L156)

### nexuml.data.export.tensor_shards.TensorShardsBackend.start_split

*method*

```python
def start_split(split: str) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/export/tensor_shards.py#L130-L138)
