# nexuml.data.loaders.tensor_shards_backend

*module*

Windowed loader backend for tensor-shard exports.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/tensor_shards_backend.py#L1-L1)

## nexuml.data.loaders.tensor_shards_backend.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/tensor_shards_backend.py#L21-L21)

## nexuml.data.loaders.tensor_shards_backend.TensorShardWindowLoader

*class*

```python
class TensorShardWindowLoader(DataLoader)
```

Bases: `DataLoader`

PyTorch DataLoader that receives already-batched shard-window items.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/tensor_shards_backend.py#L287-L324)

## nexuml.data.loaders.tensor_shards_backend.TensorShardsLoaderBackend

*class*

```python
class TensorShardsLoaderBackend
```

Create split-specific loaders for tensor-shard exports.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/tensor_shards_backend.py#L327-L367)

### nexuml.data.loaders.tensor_shards_backend.TensorShardsLoaderBackend.definition

*attribute* · *instance attribute*

```python
definition = definition
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/tensor_shards_backend.py#L331-L331)

### nexuml.data.loaders.tensor_shards_backend.TensorShardsLoaderBackend.create_loader

*method*

```python
def create_loader(module: Any, dataset: Any, *, split: str, shuffle: bool = False, sampler: WeightedRandomSampler | None = None) -> TensorShardWindowLoader
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/tensor_shards_backend.py#L333-L367)
