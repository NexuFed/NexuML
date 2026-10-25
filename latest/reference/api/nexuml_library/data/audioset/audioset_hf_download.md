# nexuml_library.data.audioset.audioset_hf_download

*module*

Hugging Face-only AudioSet downloader.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/audioset/audioset_hf_download.py#L1-L1)

## nexuml_library.data.audioset.audioset_hf_download.CONFIG_NAME

*attribute* · *module attribute*

```python
CONFIG_NAME = 'full'
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/audioset/audioset_hf_download.py#L12-L12)

## nexuml_library.data.audioset.audioset_hf_download.DATASET_ID

*attribute* · *module attribute*

```python
DATASET_ID = 'agkphysics/AudioSet'
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/audioset/audioset_hf_download.py#L11-L11)

## nexuml_library.data.audioset.audioset_hf_download.SPLITS

*attribute* · *module attribute*

```python
SPLITS = ('bal_train', 'eval')
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/audioset/audioset_hf_download.py#L13-L13)

## nexuml_library.data.audioset.audioset_hf_download.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/audioset/audioset_hf_download.py#L9-L9)

## nexuml_library.data.audioset.audioset_hf_download.download_hf_audioset

*function*

```python
def download_hf_audioset(root: str | Path, decode: bool = True) -> None
```

Download AudioSet from Hugging Face into ``root``.

Uses only ``agkphysics/AudioSet`` config ``full`` and splits
``bal_train``, ``eval``, and ``unbal_train``. No YouTube or yt-dlp code is
imported or invoked.

**Raises**

- `RuntimeError` — If required packages (datasets, torchaudio, tqdm)
are not installed.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/audioset/audioset_hf_download.py#L33-L106)

## nexuml_library.data.audioset.audioset_hf_download.expected_hf_audioset_layout

*function*

```python
def expected_hf_audioset_layout(root: str | Path) -> str
```

Return the expected HuggingFace AudioSet directory layout path.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/audioset/audioset_hf_download.py#L24-L30)

## nexuml_library.data.audioset.audioset_hf_download.is_hf_audioset_complete

*function*

```python
def is_hf_audioset_complete(root: str | Path) -> bool
```

Return whether the Hugging Face AudioSet layout exists.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/audioset/audioset_hf_download.py#L16-L21)
