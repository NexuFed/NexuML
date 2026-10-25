# nexuml.data.export.webdataset

*module*

WebDataset tar-shard export backend.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/webdataset.py#L1-L1)

## nexuml.data.export.webdataset.WebDatasetBackend

*class*

```python
class WebDatasetBackend(ExportBackend)
```

Bases: `nexuml.data.export.backend.ExportBackend`

Write lossless NumPy components into indexed WebDataset tar shards.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/webdataset.py#L33-L319)

### nexuml.data.export.webdataset.WebDatasetBackend.create_index

*attribute* · *instance attribute*

```python
create_index = bool(create_index)
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/webdataset.py#L52-L52)

### nexuml.data.export.webdataset.WebDatasetBackend.samples_per_shard

*attribute* · *instance attribute*

```python
samples_per_shard = int(samples_per_shard)
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/webdataset.py#L51-L51)

### nexuml.data.export.webdataset.WebDatasetBackend.remote_uri

*property*

```python
remote_uri: str | None
```

S3 dataset root for remote exports.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/webdataset.py#L82-L84)

### nexuml.data.export.webdataset.WebDatasetBackend.end_split

*method*

```python
def end_split(split: str) -> None
```

Finish one dataset split and close its current shard.

**Raises**

- `RuntimeError` — If the requested split is not the active split.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/webdataset.py#L113-L123)

### nexuml.data.export.webdataset.WebDatasetBackend.finalize

*method*

```python
def finalize() -> dict[str, Any]
```

Close the final shard and return metadata for ``config.yaml``.

**Returns**

- (`dict[str, Any]`) — Backend-specific export metadata.

**Raises**

- `RuntimeError` — If the backend was never initialized.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/webdataset.py#L256-L283)

### nexuml.data.export.webdataset.WebDatasetBackend.initialize

*method*

```python
def initialize(export_dir: Path, num_samples: int, feature_shapes: dict[str, tuple[int, ...]], dtype: np.dtype[Any] | str | None = None) -> None
```

Prepare local shard staging.

**Parameters**

- `export_dir` (`Path`) — Local export/staging directory.
- `num_samples` (`int`) — Declared number of exported samples.
- `feature_shapes` (`dict[str, tuple[int, ...]]`) — Tensor shapes supplied by the export runner.
- `dtype` (`np.dtype[Any] | str | None`) (default: `None`) — Optional floating-point storage dtype.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/webdataset.py#L86-L104)

### nexuml.data.export.webdataset.WebDatasetBackend.load_sample

*method* · *staticmethod*

```python
def load_sample(export_dir: Path, index: int) -> dict[str, torch.Tensor]
```

Load one sample from a local WebDataset export.

**Returns**

- (`dict[str, torch.Tensor]`) — Mapping of stored keys to tensors.

**Raises**

- `IndexError` — If the sample is absent from the export index.
- `FileNotFoundError` — If a referenced tar member cannot be read.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/webdataset.py#L292-L319)

### nexuml.data.export.webdataset.WebDatasetBackend.publish_export_metadata

*method*

```python
def publish_export_metadata(config_path: Path, metadata_path: Path) -> None
```

Upload the final lightweight export metadata for an S3 export.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/webdataset.py#L285-L290)

### nexuml.data.export.webdataset.WebDatasetBackend.save_sample

*method*

```python
def save_sample(index: int, features: dict[str, torch.Tensor]) -> None
```

Append one tensor sample to the active tar shard.

**Raises**

- `ValueError` — If a component key contains a path separator.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/webdataset.py#L210-L254)

### nexuml.data.export.webdataset.WebDatasetBackend.start_split

*method*

```python
def start_split(split: str) -> None
```

Start one physically isolated dataset split.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/webdataset.py#L106-L111)

### nexuml.data.export.webdataset.WebDatasetBackend.save_batch

*method*

```python
def save_batch(start_index: int, features: dict[str, torch.Tensor]) -> None
```

Inherited from: `nexuml.data.export.backend.ExportBackend`

Save a batch of feature tensors.

Backends can override this for more efficient contiguous writes.

**Raises**

- `ValueError` — If batch sizes across feature tensors are inconsistent.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/export/backend.py#L65-L80)
