# nexuml.data.loaders.dali_wav

*module*

DALI WAV audio pipeline for file-backed audio datasets.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/data/loaders/dali_wav.py#L1-L1)

## nexuml.data.loaders.dali_wav.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/data/loaders/dali_wav.py#L15-L15)

## nexuml.data.loaders.dali_wav.PyTorchIterator

*class*

```python
class PyTorchIterator(DALIGenericIterator)
```

Bases: `DALIGenericIterator`

DALIGenericIterator with multi-label support via TensorDict lookup.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/data/loaders/dali_wav.py#L23-L47)

### nexuml.data.loaders.dali_wav.PyTorchIterator.labels

*attribute* · *instance attribute*

```python
labels = labels
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/data/loaders/dali_wav.py#L30-L30)

## nexuml.data.loaders.dali_wav.DaliAudioPipeline

*function*

```python
def DaliAudioPipeline(files: list[str], labels: list[list[int]] | TensorDict, batch_size: int, target_sr: int = 16000, target_length: float = 10, mono: bool = True, num_threads: int = -1, prefetch_factor: int = 2, shuffle: bool = False, local_rank: int = 0, global_rank: int = 0, world_size: int = 1, random_crop_size: float | None = None, start_sec: float | None = None, **kwargs) -> PyTorchIterator
```

Build and return a DALI audio pipeline as a PyTorchIterator.

**Returns**

- (`PyTorchIterator`) — A ``PyTorchIterator`` over the built DALI pipeline.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/data/loaders/dali_wav.py#L134-L194)

## nexuml.data.loaders.dali_wav.wav_data_pipeline

*function*

```python
def wav_data_pipeline(files: list[str], target_sr: int = 16000, target_length: float | None = 10, mono: bool = True, shuffle: bool = False, shard_id: int = 0, num_shards: int = 1, device: str = 'cpu', dont_use_mmap: bool = True, rnd_crop_size: float | None = None, start_sec: float | None = None) -> Any
```

Load WAV files with DALI native readers and return audio + file index label.

**Returns**

- (`Any`) — Tuple of audio tensor and integer file-index label.

**Raises**

- `ValueError` — If a crop/start offset is requested without a target length.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/data/loaders/dali_wav.py#L50-L131)
