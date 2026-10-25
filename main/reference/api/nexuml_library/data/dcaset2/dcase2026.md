# nexuml_library.data.dcaset2.dcase2026

*module*

DCASE 2026 dataset sources for Tasks 1, 2, and 7.

All datasets are metadata-backed and fake-fixture friendly: they accept
pre-built DataFrames and tiny local audio files for testing without
requiring network downloads.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L1-L1)

## nexuml_library.data.dcaset2.dcase2026.DCASE2026_T1_BSD10K_RECORD

*attribute* · *module attribute*

```python
DCASE2026_T1_BSD10K_RECORD = 17233905
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L597-L597)

## nexuml_library.data.dcaset2.dcase2026.DCASE2026_T1_OFFICIAL_URL

*attribute* · *module attribute*

```python
DCASE2026_T1_OFFICIAL_URL = 'https://zenodo.org/records/17233905'
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L598-L598)

## nexuml_library.data.dcaset2.dcase2026.DCASE2026_T2_OFFICIAL_URL

*attribute* · *module attribute*

```python
DCASE2026_T2_OFFICIAL_URL = 'https://dcase.community/challenge2026/'
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L907-L907)

## nexuml_library.data.dcaset2.dcase2026.DCASE2026_T7_DEV_RECORD

*attribute* · *module attribute*

```python
DCASE2026_T7_DEV_RECORD = 19335184
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L1057-L1057)

## nexuml_library.data.dcaset2.dcase2026.DCASE2026_T7_OFFICIAL_URL

*attribute* · *module attribute*

```python
DCASE2026_T7_OFFICIAL_URL = 'https://zenodo.org/records/19335184'
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L1058-L1058)

## nexuml_library.data.dcaset2.dcase2026.DownloadMode

*attribute* · *module attribute*

```python
DownloadMode = Literal['disabled', 'dry_run', 'manual', 'already_downloaded', 'download']
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L36-L36)

## nexuml_library.data.dcaset2.dcase2026.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L34-L34)

## nexuml_library.data.dcaset2.dcase2026.DCASE2026T1Dataset

*class*

```python
class DCASE2026T1Dataset(DataSourceDefinition)
```

Bases: `DataSourceDefinition`

DCASE 2026 Task 1 dataset for BSD10k-v1.2 / BSD35k-CS.

Expects a metadata CSV with columns including at least:
``file``, ``sound_id``, ``class_top``, ``class_second``.
Optional columns: ``confidence``, ``uploader``, ``license``, ``title``,
``tags``, ``description``, ``clap_embedding``.

**Attributes**

- `root` (`str | Path`) — Dataset root directory.
- `metadata_csv` (`str`) — Path to metadata CSV relative to root.
- `download_mode` (`DownloadMode`) — One of the DownloadMode literals.
- `checksum` (`str | None`) — Optional expected checksum of the metadata CSV.
- `clap_embeddings_dir` (`str | None`) — Optional directory with pre-computed CLAP .npy files.
- `fold_seed` (`int`) — Seed for reproducible baseline 5-fold splits.
- `fold_id` (`int | None`) — If provided, filter to that fold (0-4).
- `split` (`str`) — ``"fit"``, ``"val"``, ``"test"``, or ``"eval"``.
- `sample_rate` (`int`) — Target audio sample rate.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L725-L758)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T1Dataset.checksum

*attribute* · *class attribute* · *instance attribute*

```python
checksum: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L749-L749)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T1Dataset.clap_embeddings_dir

*attribute* · *class attribute* · *instance attribute*

```python
clap_embeddings_dir: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L750-L750)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T1Dataset.clip_num_samples

*attribute* · *class attribute* · *instance attribute*

```python
clip_num_samples: int | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L755-L755)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T1Dataset.download_mode

*attribute* · *class attribute* · *instance attribute*

```python
download_mode: DownloadMode = 'disabled'
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L748-L748)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T1Dataset.fold_id

*attribute* · *class attribute* · *instance attribute*

```python
fold_id: int | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L752-L752)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T1Dataset.fold_seed

*attribute* · *class attribute* · *instance attribute*

```python
fold_seed: int = 42
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L751-L751)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T1Dataset.metadata_csv

*attribute* · *class attribute* · *instance attribute*

```python
metadata_csv: str = 'metadata.csv'
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L747-L747)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T1Dataset.root

*attribute* · *instance attribute*

```python
root: str | Path
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L746-L746)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T1Dataset.sample_rate

*attribute* · *class attribute* · *instance attribute*

```python
sample_rate: int = 16000
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L754-L754)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T1Dataset.split

*attribute* · *class attribute* · *instance attribute*

```python
split: str = 'fit'
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L753-L753)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T1Dataset.build

*method*

```python
def build() -> NexuDataset
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L757-L758)

## nexuml_library.data.dcaset2.dcase2026.DCASE2026T2Dataset

*class*

```python
class DCASE2026T2Dataset(DCASET2Dataset)
```

Bases: `nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset`

DCASE 2026 Task 2 dataset — metadata index with domain/attribute enrichment.

Extends ``DCASET2Dataset`` with 2026-specific metadata:
- ``domain`` inferred from filename source/target markers
- ``attributes`` merged from the official attributes CSV when available

Audio is 2-channel (stereo). Channel selection happens in pipeline layers,
not here. DALI loads raw stereo audio as [C, T] via ``dali_layout="CT"``.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L944-L960)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T2Dataset.attributes_csv

*attribute* · *class attribute* · *instance attribute*

```python
attributes_csv: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L957-L957)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T2Dataset.dataset_name

*attribute* · *class attribute* · *instance attribute*

```python
dataset_name: str = 'DCASE2026T2'
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L956-L956)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T2Dataset.clip_num_samples

*attribute* · *class attribute* · *instance attribute*

```python
clip_num_samples: int | None = None
```

Inherited from: `nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset`

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L164-L164)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T2Dataset.data_root

*attribute* · *class attribute* · *instance attribute*

```python
data_root: str | Path | None = None
```

Inherited from: `nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset`

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L156-L156)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T2Dataset.data_type

*attribute* · *class attribute* · *instance attribute*

```python
data_type: str = 'dev'
```

Inherited from: `nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset`

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L161-L161)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T2Dataset.download

*attribute* · *class attribute* · *instance attribute*

```python
download: bool = False
```

Inherited from: `nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset`

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L168-L168)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T2Dataset.download_manifest

*attribute* · *class attribute* · *instance attribute*

```python
download_manifest: Path | None = None
```

Inherited from: `nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset`

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L169-L169)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T2Dataset.machine_id

*attribute* · *class attribute* · *instance attribute*

```python
machine_id: int | None = None
```

Inherited from: `nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset`

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L166-L166)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T2Dataset.machine_label

*attribute* · *class attribute* · *instance attribute*

```python
machine_label: str | None = None
```

Inherited from: `nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset`

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L167-L167)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T2Dataset.machine_type

*attribute* · *class attribute* · *instance attribute*

```python
machine_type: str = 'ToyCar'
```

Inherited from: `nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset`

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L158-L158)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T2Dataset.machine_types

*attribute* · *class attribute* · *instance attribute*

```python
machine_types: list[str] | None = None
```

Inherited from: `nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset`

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L165-L165)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T2Dataset.root

*attribute* · *class attribute* · *instance attribute*

```python
root: str | Path | None = None
```

Inherited from: `nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset`

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L155-L155)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T2Dataset.sample_rate

*attribute* · *class attribute* · *instance attribute*

```python
sample_rate: int = 16000
```

Inherited from: `nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset`

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L163-L163)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T2Dataset.section_ids

*attribute* · *class attribute* · *instance attribute*

```python
section_ids: list[str] | None = None
```

Inherited from: `nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset`

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L160-L160)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T2Dataset.section_keyword

*attribute* · *class attribute* · *instance attribute*

```python
section_keyword: str = 'section'
```

Inherited from: `nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset`

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L159-L159)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T2Dataset.train

*attribute* · *class attribute* · *instance attribute*

```python
train: bool = True
```

Inherited from: `nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset`

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L162-L162)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T2Dataset.build

*method*

```python
def build() -> NexuDataset
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L959-L960)

## nexuml_library.data.dcaset2.dcase2026.DCASE2026T7Dataset

*class*

```python
class DCASE2026T7Dataset(DataSourceDefinition)
```

Bases: `DataSourceDefinition`

DCASE 2026 Task 7 DIL-DCASE26 dataset.

Expects a root directory with subfolders ``D2/`` and ``D3/`` (and
optionally ``D1/`` for baseline reference).  Each domain folder may
contain:

- ``metadata.csv`` with columns ``file``, ``class``, ``domain``
- Raw audio files (optionally with ``evaluation_setup/`` split files)

The dataset exposes a sequence-of-increments interface via
``get_increment()``.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L1179-L1203)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T7Dataset.checksum

*attribute* · *class attribute* · *instance attribute*

```python
checksum: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L1197-L1197)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T7Dataset.clip_num_samples

*attribute* · *class attribute* · *instance attribute*

```python
clip_num_samples: int | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L1200-L1200)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T7Dataset.d1_baseline_dir

*attribute* · *class attribute* · *instance attribute*

```python
d1_baseline_dir: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L1198-L1198)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T7Dataset.domains

*attribute* · *class attribute* · *instance attribute*

```python
domains: list[str] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L1195-L1195)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T7Dataset.download_mode

*attribute* · *class attribute* · *instance attribute*

```python
download_mode: DownloadMode = 'disabled'
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L1196-L1196)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T7Dataset.root

*attribute* · *instance attribute*

```python
root: str | Path
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L1194-L1194)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T7Dataset.sample_rate

*attribute* · *class attribute* · *instance attribute*

```python
sample_rate: int = 16000
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L1199-L1199)

### nexuml_library.data.dcaset2.dcase2026.DCASE2026T7Dataset.build

*method*

```python
def build() -> NexuDataset
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L1202-L1203)

## nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset

*class*

```python
class DCASET2Dataset(DataSourceDefinition)
```

Bases: `DataSourceDefinition`

DCASE Task 2 anomalous sound detection dataset.

Builds a metadata DataFrame from the DCASE directory structure and
lazily loads audio/features in load_item().

> **Directory layout expected**
>
> <root>/<dataset_name>/<data_type>_data/raw/<machine_type>/train/
> <root>/<dataset_name>/<data_type>_data/raw/<machine_type>/test/

> **machine (int index), y_true (0=normal,1=anomalous),**
>
> condition (int), target (0/1).

**Attributes**

- `root` (`str | Path | None`) — Path to the dataset root directory.
- `dataset_name` (`str`) — e.g. "DCASE2024T2"
- `machine_type` (`str`) — e.g. "ToyCar", "fan", etc.
- `section_keyword` (`str`) — prefix for section names, e.g. "section"
- `section_ids` (`list[str] | None`) — list of section id strings, e.g. ["00", "01"]
- `data_type` (`str`) — "dev" or "eval"
- `train` (`bool`) — if True load train split, else test split
- `sample_rate` (`int`) — target sample rate for audio loading
- `machine_types` (`list[str] | None`) — list of all machine types (for machine index mapping)

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L129-L172)

### nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset.clip_num_samples

*attribute* · *class attribute* · *instance attribute*

```python
clip_num_samples: int | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L164-L164)

### nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset.data_root

*attribute* · *class attribute* · *instance attribute*

```python
data_root: str | Path | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L156-L156)

### nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset.data_type

*attribute* · *class attribute* · *instance attribute*

```python
data_type: str = 'dev'
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L161-L161)

### nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset.dataset_name

*attribute* · *class attribute* · *instance attribute*

```python
dataset_name: str = 'DCASE2024T2'
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L157-L157)

### nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset.download

*attribute* · *class attribute* · *instance attribute*

```python
download: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L168-L168)

### nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset.download_manifest

*attribute* · *class attribute* · *instance attribute*

```python
download_manifest: Path | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L169-L169)

### nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset.machine_id

*attribute* · *class attribute* · *instance attribute*

```python
machine_id: int | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L166-L166)

### nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset.machine_label

*attribute* · *class attribute* · *instance attribute*

```python
machine_label: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L167-L167)

### nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset.machine_type

*attribute* · *class attribute* · *instance attribute*

```python
machine_type: str = 'ToyCar'
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L158-L158)

### nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset.machine_types

*attribute* · *class attribute* · *instance attribute*

```python
machine_types: list[str] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L165-L165)

### nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset.root

*attribute* · *class attribute* · *instance attribute*

```python
root: str | Path | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L155-L155)

### nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset.sample_rate

*attribute* · *class attribute* · *instance attribute*

```python
sample_rate: int = 16000
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L163-L163)

### nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset.section_ids

*attribute* · *class attribute* · *instance attribute*

```python
section_ids: list[str] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L160-L160)

### nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset.section_keyword

*attribute* · *class attribute* · *instance attribute*

```python
section_keyword: str = 'section'
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L159-L159)

### nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset.train

*attribute* · *class attribute* · *instance attribute*

```python
train: bool = True
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L162-L162)

### nexuml_library.data.dcaset2.dcase2026.DCASET2Dataset.build

*method*

```python
def build() -> NexuDataset
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/data/dcaset2/dcase2026.py#L171-L172)
