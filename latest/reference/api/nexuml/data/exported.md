# nexuml.data.exported

*module*

Dataset source for loading NexuML exported datasets.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/exported.py#L1-L1)

## nexuml.data.exported.ExportedDataset

*class*

```python
class ExportedDataset(NexuDataset)
```

Bases: `nexuml.data.dataset.NexuDataset`

Load a dataset previously written by ``export_data_module``.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/exported.py#L25-L393)

### nexuml.data.exported.ExportedDataset.backend

*attribute* · *instance attribute*

```python
backend = str(config.get('writer') or config.get('backend', 'numpy'))
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/exported.py#L55-L55)

### nexuml.data.exported.ExportedDataset.config

*attribute* · *instance attribute*

```python
config = cast(dict[str, Any], config)
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/exported.py#L54-L54)

### nexuml.data.exported.ExportedDataset.extra

*attribute* · *instance attribute*

```python
extra = cast(dict[str, Any], config.get('extra', {}) or {})
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/exported.py#L71-L71)

### nexuml.data.exported.ExportedDataset.feature_key

*attribute* · *instance attribute*

```python
feature_key = self.x_keys[0]
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/exported.py#L135-L135)

### nexuml.data.exported.ExportedDataset.feature_shapes

*attribute* · *instance attribute*

```python
feature_shapes = {key: tuple(shape) for key, shape in config.get('feature_shapes', {}) or {}.items()}
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/exported.py#L67-L69)

### nexuml.data.exported.ExportedDataset.file_label_keys

*attribute* · *instance attribute*

```python
file_label_keys = [key for key in requested_y if key in stored_y]
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/exported.py#L106-L106)

### nexuml.data.exported.ExportedDataset.key_specs

*attribute* · *instance attribute*

```python
key_specs = cast(dict[str, dict[str, Any]], config.get('key_specs', {}) or {})
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/exported.py#L70-L70)

### nexuml.data.exported.ExportedDataset.label_names

*attribute* · *instance attribute*

```python
label_names = list(self.y_keys)
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/exported.py#L133-L133)

### nexuml.data.exported.ExportedDataset.label_prefix

*attribute* · *instance attribute*

```python
label_prefix = str(config.get('label_prefix', label_prefix))
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/exported.py#L59-L59)

### nexuml.data.exported.ExportedDataset.meta_label_keys

*attribute* · *instance attribute*

```python
meta_label_keys = [key for key in requested_y if key not in self.file_label_keys and key in metadata.columns]
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/exported.py#L107-L111)

### nexuml.data.exported.ExportedDataset.modality

*attribute* · *instance attribute*

```python
modality = str(config.get('modality', 'generic'))
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/exported.py#L58-L58)

### nexuml.data.exported.ExportedDataset.num_classes

*attribute* · *instance attribute*

```python
num_classes = {key: int(config_num_classes[key]) for key in self.label_names if key in config_num_classes}
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/exported.py#L137-L141)

### nexuml.data.exported.ExportedDataset.root

*attribute* · *instance attribute*

```python
root: Path | S3Path = S3Path.parse(root_text) if self._s3 is not None else Path(root)
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/exported.py#L51-L51)

### nexuml.data.exported.ExportedDataset.supports_dali_loader

*attribute* · *instance attribute*

```python
supports_dali_loader = self.backend in {'numpy', 'numpy_mmap', 'webdataset', 'tensordict_memmap', 'torch'}
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/exported.py#L60-L66)

### nexuml.data.exported.ExportedDataset.x_keys

*attribute* · *instance attribute*

```python
x_keys = requested_x
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/exported.py#L105-L105)

### nexuml.data.exported.ExportedDataset.y_keys

*attribute* · *instance attribute*

```python
y_keys = self.file_label_keys + self.meta_label_keys
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/exported.py#L112-L112)

### nexuml.data.exported.ExportedDataset.data

*attribute* · *instance attribute*

```python
data = data
```

Inherited from: `nexuml.data.dataset.NexuDataset`

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/dataset.py#L58-L58)

### nexuml.data.exported.ExportedDataset.meta

*attribute* · *instance attribute*

```python
meta: pd.DataFrame | None = None
```

Inherited from: `nexuml.data.dataset.NexuDataset`

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/dataset.py#L56-L56)

### nexuml.data.exported.ExportedDataset.split_ratio

*attribute* · *instance attribute*

```python
split_ratio = split_ratio or [0.85, 0.1, 0.05]
```

Inherited from: `nexuml.data.dataset.NexuDataset`

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/dataset.py#L59-L59)

### nexuml.data.exported.ExportedDataset.is_remote

*property*

```python
is_remote: bool
```

Whether tensors are stored in S3.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/exported.py#L145-L147)

### nexuml.data.exported.ExportedDataset.clone_with_meta

*method*

```python
def clone_with_meta(meta: pd.DataFrame, data=_KEEP_DATA) -> 'ExportedDataset'
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/exported.py#L352-L365)

### nexuml.data.exported.ExportedDataset.load_item

*method*

```python
def load_item(idx: int, row: pd.Series) -> TensorDict
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/exported.py#L367-L376)

### nexuml.data.exported.ExportedDataset.load_labels

*method*

```python
def load_labels(idx: int, row: pd.Series) -> TensorDict | None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/exported.py#L378-L393)

### nexuml.data.exported.ExportedDataset.download

*method*

```python
def download() -> None
```

Inherited from: `nexuml.data.dataset.NexuDataset`

Override to implement dataset downloading.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/dataset.py#L276-L278)

### nexuml.data.exported.ExportedDataset.get_split

*method*

```python
def get_split(split: str) -> Self
```

Inherited from: `nexuml.data.dataset.NexuDataset`

Return a view of this dataset filtered to a single split.

**Returns**

- (`Self`) — A new :class:`NexuDataset` containing only the requested split.

**Raises**

- `ValueError` — If the dataset has no metadata.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/dataset.py#L188-L201)

### nexuml.data.exported.ExportedDataset.split_meta

*method*

```python
def split_meta(split_ratio: list[float]) -> None
```

Inherited from: `nexuml.data.dataset.NexuDataset`

Convert 'fit' rows to train/val and 'all' rows to train/val/test.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/dataset.py#L163-L186)

### nexuml.data.exported.ExportedDataset.take

*method*

```python
def take(indices: Sequence[int]) -> Self
```

Inherited from: `nexuml.data.dataset.NexuDataset`

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/data/dataset.py#L154-L161)
