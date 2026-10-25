# nexuml_library.data.audioset.audioset

*module*

AudioSet dataset source for NexuML.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/audioset/audioset.py#L1-L1)

## nexuml_library.data.audioset.audioset.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/audioset/audioset.py#L21-L21)

## nexuml_library.data.audioset.audioset.AudiosetDataset

*class*

```python
class AudiosetDataset(DataSourceDefinition)
```

Bases: `DataSourceDefinition`

AudioSet dataset source.

Scans <data_dir> for .wav or .npy files. File names must encode labels
via the pattern ``<ytid>__<label_str>.<ext>`` where label_str is
underscore-joined AudioSet display names.

A cached metadata CSV is read/written to ``<meta_dir>/<data_dir.name>_meta.csv``
so subsequent loads are fast.

Label columns: class (int index), class_logits (multi-hot list).

**Attributes**

- `data_dir` (`str | Path`) — Directory containing audio/feature files.
- `meta_dir` (`str | Path`) — Directory for class_labels_indices.csv and the cached meta CSV.
- `perform_checks` (`bool`) — Whether to verify each file is loadable (slow on first run).
- `len_seconds` (`float`) — Minimum audio duration in seconds to include a file.
- `sample_rate` (`int`) — Target sample rate for audio loading.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/audioset/audioset.py#L89-L117)

### nexuml_library.data.audioset.audioset.AudiosetDataset.data_dir

*attribute* · *class attribute* · *instance attribute*

```python
data_dir: str | Path = 'data/audioset/eval'
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/audioset/audioset.py#L110-L110)

### nexuml_library.data.audioset.audioset.AudiosetDataset.len_seconds

*attribute* · *class attribute* · *instance attribute*

```python
len_seconds: float = 5.0
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/audioset/audioset.py#L113-L113)

### nexuml_library.data.audioset.audioset.AudiosetDataset.meta_dir

*attribute* · *class attribute* · *instance attribute*

```python
meta_dir: str | Path = 'data/AudioSet/meta'
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/audioset/audioset.py#L111-L111)

### nexuml_library.data.audioset.audioset.AudiosetDataset.perform_checks

*attribute* · *class attribute* · *instance attribute*

```python
perform_checks: bool = True
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/audioset/audioset.py#L112-L112)

### nexuml_library.data.audioset.audioset.AudiosetDataset.sample_rate

*attribute* · *class attribute* · *instance attribute*

```python
sample_rate: int = 16000
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/audioset/audioset.py#L114-L114)

### nexuml_library.data.audioset.audioset.AudiosetDataset.build

*method*

```python
def build() -> NexuDataset
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/audioset/audioset.py#L116-L117)
