# nexuml.data.loaders.dali_numpy_external

*module*

DALI external source pipeline for NumPy datasets.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/data/loaders/dali_numpy_external.py#L1-L1)

## nexuml.data.loaders.dali_numpy_external.ExternalInputIterator

*class*

```python
class ExternalInputIterator
```

Iterator that loads NumPy files on demand for DALI external_source.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/data/loaders/dali_numpy_external.py#L17-L48)

### nexuml.data.loaders.dali_numpy_external.ExternalInputIterator.data_set_len

*attribute* · *instance attribute*

```python
data_set_len = len(self.files)
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/data/loaders/dali_numpy_external.py#L23-L23)

### nexuml.data.loaders.dali_numpy_external.ExternalInputIterator.files

*attribute* · *instance attribute*

```python
files = self.files[self.data_set_len * shard_id // num_shards:self.data_set_len * shard_id + 1 // num_shards]
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/data/loaders/dali_numpy_external.py#L25-L29)

### nexuml.data.loaders.dali_numpy_external.ExternalInputIterator.n

*attribute* · *instance attribute*

```python
n = len(self.files)
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/data/loaders/dali_numpy_external.py#L31-L31)

### nexuml.data.loaders.dali_numpy_external.ExternalInputIterator.shuffle

*attribute* · *instance attribute*

```python
shuffle = shuffle
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/data/loaders/dali_numpy_external.py#L22-L22)

## nexuml.data.loaders.dali_numpy_external.PyTorchIterator

*class*

```python
class PyTorchIterator(DALIGenericIterator)
```

Bases: `DALIGenericIterator`

DALIGenericIterator with multi-label support for external source pipelines.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/data/loaders/dali_numpy_external.py#L51-L70)

### nexuml.data.loaders.dali_numpy_external.PyTorchIterator.labels

*attribute* · *instance attribute*

```python
labels = labels
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/data/loaders/dali_numpy_external.py#L57-L57)

## nexuml.data.loaders.dali_numpy_external.DaliNumpyExternalPipeline

*function*

```python
def DaliNumpyExternalPipeline(files: list[str], labels: list[list[int]] | TensorDict, batch_size: int, num_threads: int = -1, prefetch_factor: int = 2, shuffle: bool = False, local_rank: int = 0, global_rank: int = 0, world_size: int = 1, **kwargs) -> PyTorchIterator
```

Build and return a DALI external source pipeline as a PyTorchIterator.

**Returns**

- (`PyTorchIterator`) — A ``PyTorchIterator`` over the built DALI pipeline.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/data/loaders/dali_numpy_external.py#L95-L148)

## nexuml.data.loaders.dali_numpy_external.external_source_pipeline

*function*

```python
def external_source_pipeline(external_source: Iterable, device: str = 'cpu') -> Any
```

DALI pipeline using external source for flexible data loading.

**Returns**

- (`Any`) — Tuple of data tensor and label tensor from the external source.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/data/loaders/dali_numpy_external.py#L73-L92)
