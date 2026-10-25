# nexuml.data.loaders.dali_multimodal

*module*

General DALI pipeline helpers for keyed TensorDict outputs.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L1-L1)

## nexuml.data.loaders.dali_multimodal.INDEX_OUTPUT

*attribute* · *module attribute*

```python
INDEX_OUTPUT = '__index'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L32-L32)

## nexuml.data.loaders.dali_multimodal.ExternalSampleIterator

*class*

```python
class ExternalSampleIterator
```

External source iterator backed by a dataset's `__getitem__`.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L134-L178)

### nexuml.data.loaders.dali_multimodal.ExternalSampleIterator.dataset

*attribute* · *instance attribute*

```python
dataset = dataset
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L147-L147)

### nexuml.data.loaders.dali_multimodal.ExternalSampleIterator.indices

*attribute* · *instance attribute*

```python
indices = indices[len(indices) * shard_id // num_shards:len(indices) * shard_id + 1 // num_shards]
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L152-L154)

### nexuml.data.loaders.dali_multimodal.ExternalSampleIterator.n

*attribute* · *instance attribute*

```python
n = len(self.indices)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L155-L155)

### nexuml.data.loaders.dali_multimodal.ExternalSampleIterator.shuffle

*attribute* · *instance attribute*

```python
shuffle = shuffle
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L150-L150)

### nexuml.data.loaders.dali_multimodal.ExternalSampleIterator.x_keys

*attribute* · *instance attribute*

```python
x_keys = list(x_keys)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L148-L148)

### nexuml.data.loaders.dali_multimodal.ExternalSampleIterator.y_keys

*attribute* · *instance attribute*

```python
y_keys = list(y_keys)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L149-L149)

## nexuml.data.loaders.dali_multimodal.KeyedDaliIterator

*class*

```python
class KeyedDaliIterator(DALIGenericIterator)
```

Bases: `DALIGenericIterator`

DALI iterator that returns `(x, y)` TensorDict pairs keyed semantically.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L93-L131)

### nexuml.data.loaders.dali_multimodal.KeyedDaliIterator.metadata

*attribute* · *class attribute* · *instance attribute*

```python
metadata: pd.DataFrame | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L96-L96)

### nexuml.data.loaders.dali_multimodal.KeyedDaliIterator.metadata_labels

*attribute* · *instance attribute*

```python
metadata_labels = metadata_labels
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L111-L111)

### nexuml.data.loaders.dali_multimodal.KeyedDaliIterator.x_keys

*attribute* · *instance attribute*

```python
x_keys = list(x_keys)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L109-L109)

### nexuml.data.loaders.dali_multimodal.KeyedDaliIterator.y_keys

*attribute* · *instance attribute*

```python
y_keys = list(y_keys)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L110-L110)

## nexuml.data.loaders.dali_multimodal.WebDatasetComponentSpec

*class* · *dataclass*

```python
class WebDatasetComponentSpec
```

Component specification for a WebDataset tar shard.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L369-L379)

### nexuml.data.loaders.dali_multimodal.WebDatasetComponentSpec.dtype

*attribute* · *class attribute* · *instance attribute*

```python
dtype: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L378-L378)

### nexuml.data.loaders.dali_multimodal.WebDatasetComponentSpec.encoding

*attribute* · *instance attribute*

```python
encoding: str
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L375-L375)

### nexuml.data.loaders.dali_multimodal.WebDatasetComponentSpec.key

*attribute* · *instance attribute*

```python
key: str
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L373-L373)

### nexuml.data.loaders.dali_multimodal.WebDatasetComponentSpec.layout

*attribute* · *class attribute* · *instance attribute*

```python
layout: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L376-L376)

### nexuml.data.loaders.dali_multimodal.WebDatasetComponentSpec.member_ext

*attribute* · *instance attribute*

```python
member_ext: str
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L374-L374)

### nexuml.data.loaders.dali_multimodal.WebDatasetComponentSpec.modality

*attribute* · *class attribute* · *instance attribute*

```python
modality: str = 'generic'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L379-L379)

### nexuml.data.loaders.dali_multimodal.WebDatasetComponentSpec.shape

*attribute* · *class attribute* · *instance attribute*

```python
shape: list[int] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L377-L377)

## nexuml.data.loaders.dali_multimodal.audio_file_pipeline

*function*

```python
def audio_file_pipeline(files: list[str], target_sr: int, mono: bool, num_samples: int | None, shuffle: bool, shard_id: int, num_shards: int, layout: str | None) -> Any
```

Build a DALI pipeline that decodes audio files from paths.

**Returns**

- (`Any`) — Tuple of decoded audio tensor and integer file-index label.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L202-L247)

## nexuml.data.loaders.dali_multimodal.build_external_source_loader

*function*

```python
def build_external_source_loader(dataset: Any, x_keys: Sequence[str], y_keys: Sequence[str], batch_size: int, num_threads: int, prefetch_factor: int | None, shuffle: bool, local_rank: int, global_rank: int, world_size: int) -> KeyedDaliIterator
```

Build a DALI external-source loader from a Python iterable.

**Returns**

- (`KeyedDaliIterator`) — A ``KeyedDaliIterator`` yielding ``(x, y)`` TensorDict pairs.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L457-L504)

## nexuml.data.loaders.dali_multimodal.build_native_file_loader

*function*

```python
def build_native_file_loader(kind: str, files: list[str], x_key: str, metadata_labels: TensorDict | None, batch_size: int, num_threads: int, prefetch_factor: int | None, shuffle: bool, local_rank: int, global_rank: int, world_size: int, sample_layout: str | None = None, sample_rate: int = 16000, sequence_length: int | None = None) -> KeyedDaliIterator
```

Build a DALI native file loader for the given modality.

**Returns**

- (`KeyedDaliIterator`) — A ``KeyedDaliIterator`` yielding ``(x, y)`` TensorDict pairs.

**Raises**

- `ValueError` — If ``kind`` is unsupported or video lacks a sequence length.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L507-L580)

## nexuml.data.loaders.dali_multimodal.build_webdataset_loader

*function*

```python
def build_webdataset_loader(shard_paths: list[str], index_paths: list[str] | None, x_keys: Sequence[str], y_keys: Sequence[str], metadata_labels: TensorDict | None, batch_size: int, num_threads: int, prefetch_factor: int | None, shuffle: bool, local_rank: int, global_rank: int, world_size: int, components: list[WebDatasetComponentSpec]) -> KeyedDaliIterator
```

Build a DALI WebDataset loader from tar archive paths.

**Returns**

- (`KeyedDaliIterator`) — A ``KeyedDaliIterator`` yielding ``(x, y)`` TensorDict pairs.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L583-L628)

## nexuml.data.loaders.dali_multimodal.external_source_pipeline

*function*

```python
def external_source_pipeline(external_source: Iterable, *, num_outputs: int) -> Any
```

Build a DALI pipeline backed by an external Python source.

**Returns**

- (`Any`) — A single DALI output node when ``num_outputs == 1``,
- (`Any`) — otherwise a tuple of output nodes.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L181-L199)

## nexuml.data.loaders.dali_multimodal.image_file_pipeline

*function*

```python
def image_file_pipeline(files: list[str], shuffle: bool, shard_id: int, num_shards: int, layout: str | None) -> Any
```

Build a DALI pipeline that decodes image files from paths.

**Returns**

- (`Any`) — Tuple of decoded image tensor and integer file-index label.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L250-L275)

## nexuml.data.loaders.dali_multimodal.numpy_file_pipeline

*function*

```python
def numpy_file_pipeline(files: list[str], shuffle: bool, shard_id: int, num_shards: int) -> Any
```

Build a DALI pipeline that loads numpy array files from paths.

**Returns**

- (`Any`) — Tuple of loaded numpy data tensor and integer file-index label.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L339-L361)

## nexuml.data.loaders.dali_multimodal.text_file_pipeline

*function*

```python
def text_file_pipeline(files: list[str], shuffle: bool, shard_id: int, num_shards: int) -> Any
```

Build a DALI pipeline that reads text files as byte strings.

**Returns**

- (`Any`) — Tuple of text byte tensor and integer file-index label.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L278-L301)

## nexuml.data.loaders.dali_multimodal.video_file_pipeline

*function*

```python
def video_file_pipeline(files: list[str], shuffle: bool, shard_id: int, num_shards: int, sequence_length: int, layout: str | None, reader_device: str) -> Any
```

Build a DALI pipeline that decodes video files from paths.

**Returns**

- (`Any`) — Tuple of decoded video tensor and integer file-index label.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L304-L336)

## nexuml.data.loaders.dali_multimodal.webdataset_pipeline

*function*

```python
def webdataset_pipeline(paths: list[str], index_paths: list[str] | None, shuffle: bool, shard_id: int, num_shards: int, components: list[WebDatasetComponentSpec]) -> Any
```

Build a DALI pipeline that reads a WebDataset tar archive.

**Returns**

- (`Any`) — Tuple of decoded tensors, one per component.

**Raises**

- `ValueError` — If a component encoding is unsupported.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/loaders/dali_multimodal.py#L382-L454)
