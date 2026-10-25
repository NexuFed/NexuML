# nexuml_library.data.exported

*module*

Typed definition for loading NexuML exported datasets.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/data/exported.py#L1-L1)

## nexuml_library.data.exported.ExportedDataset

*class*

```python
class ExportedDataset(DataSourceDefinition)
```

Bases: `DataSourceDefinition`

Load a local or S3 dataset previously written by ``export_data_module``.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/data/exported.py#L11-L25)

### nexuml_library.data.exported.ExportedDataset.feature_keys

*attribute* · *class attribute* · *instance attribute*

```python
feature_keys: list[str] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/data/exported.py#L17-L17)

### nexuml_library.data.exported.ExportedDataset.label_keys

*attribute* · *class attribute* · *instance attribute*

```python
label_keys: list[str] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/data/exported.py#L18-L18)

### nexuml_library.data.exported.ExportedDataset.label_prefix

*attribute* · *class attribute* · *instance attribute*

```python
label_prefix: str = 'label__'
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/data/exported.py#L19-L19)

### nexuml_library.data.exported.ExportedDataset.root

*attribute* · *instance attribute*

```python
root: str | Path
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/data/exported.py#L15-L15)

### nexuml_library.data.exported.ExportedDataset.s3_endpoint_url

*attribute* · *class attribute* · *instance attribute*

```python
s3_endpoint_url: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/data/exported.py#L20-L20)

### nexuml_library.data.exported.ExportedDataset.s3_profile

*attribute* · *class attribute* · *instance attribute*

```python
s3_profile: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/data/exported.py#L22-L22)

### nexuml_library.data.exported.ExportedDataset.s3_region

*attribute* · *class attribute* · *instance attribute*

```python
s3_region: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/data/exported.py#L21-L21)

### nexuml_library.data.exported.ExportedDataset.split

*attribute* · *class attribute* · *instance attribute*

```python
split: str | list[str] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/data/exported.py#L16-L16)

### nexuml_library.data.exported.ExportedDataset.build

*method*

```python
def build() -> NexuDataset
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/data/exported.py#L24-L25)
