# nexuml.data.loaders.dali_numpy

*module*

DALI NumPy pipeline for file-backed numpy datasets.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/dali_numpy.py#L1-L1)

## nexuml.data.loaders.dali_numpy.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/dali_numpy.py#L13-L13)

## nexuml.data.loaders.dali_numpy.PyTorchIterator

*class*

```python
class PyTorchIterator(DALIGenericIterator)
```

Bases: `DALIGenericIterator`

DALIGenericIterator with filename-based label lookup for NumPy datasets.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/dali_numpy.py#L16-L38)

### nexuml.data.loaders.dali_numpy.PyTorchIterator.labels

*attribute* · *instance attribute*

```python
labels = labels
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/dali_numpy.py#L21-L21)

## nexuml.data.loaders.dali_numpy.DaliNumpyPipeline

*function*

```python
def DaliNumpyPipeline(files: list[str], labels: list[list[int]], batch_size: int, target_sr: int = 16000, target_length: int = 10, num_threads: int = -1, prefetch_factor: int = 2, shuffle: bool = False, local_rank: int = 0, global_rank: int = 0, world_size: int = 1, mono: bool = True, random_crop_size: float | None = None, start_sec: float | None = None, direct_store: bool = False, **kwargs) -> PyTorchIterator
```

Build and return a DALI numpy pipeline as a PyTorchIterator.

**Returns**

- (`PyTorchIterator`) — A ``PyTorchIterator`` over the built DALI pipeline.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/dali_numpy.py#L101-L165)

## nexuml.data.loaders.dali_numpy.numpy_data_pipeline

*function*

```python
def numpy_data_pipeline(files: list[str], filename_len: int, target_sr: int = 16000, target_length: int = 10, shuffle: bool = False, shard_id: int = 0, num_shards: int = 1, device: str = 'cpu', direct_store: bool = False, mono: bool = True, rnd_crop_size: float | None = None, start_sec: float | None = None) -> Any
```

Load pre-encoded NumPy arrays with DALI native reader.

**Returns**

- (`Any`) — Tuple of audio tensor and integer file-index label.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/dali_numpy.py#L41-L98)
