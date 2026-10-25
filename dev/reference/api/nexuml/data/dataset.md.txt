# nexuml.data.dataset

*module*

Base dataset contracts and metadata-backed helpers for NexuML.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/dataset.py#L1-L1)

## nexuml.data.dataset.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/dataset.py#L16-L16)

## nexuml.data.dataset.NexuDataset

*class*

```python
class NexuDataset(torch.utils.data.Dataset)
```

Bases: `torch.utils.data.Dataset`

Base dataset returning ``(x, y)`` TensorDict pairs.

The base class supports two common patterns used throughout NexuML:

- metadata-backed datasets via ``self.meta``
- in-memory datasets via ``self.data``

Subclasses can still override ``__getitem__`` entirely for bespoke behavior.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/dataset.py#L34-L278)

### nexuml.data.dataset.NexuDataset.data

*attribute* · *instance attribute*

```python
data = data
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/dataset.py#L58-L58)

### nexuml.data.dataset.NexuDataset.label_names

*attribute* · *instance attribute*

```python
label_names: list[str] = list(label_names or [])
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/dataset.py#L60-L60)

### nexuml.data.dataset.NexuDataset.meta

*attribute* · *instance attribute*

```python
meta: pd.DataFrame | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/dataset.py#L56-L56)

### nexuml.data.dataset.NexuDataset.modality

*attribute* · *instance attribute*

```python
modality = modality
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/dataset.py#L57-L57)

### nexuml.data.dataset.NexuDataset.num_classes

*attribute* · *instance attribute*

```python
num_classes: dict[str, int] = {}
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/dataset.py#L61-L61)

### nexuml.data.dataset.NexuDataset.split_ratio

*attribute* · *instance attribute*

```python
split_ratio = split_ratio or [0.85, 0.1, 0.05]
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/dataset.py#L59-L59)

### nexuml.data.dataset.NexuDataset.clone_with_meta

*method*

```python
def clone_with_meta(meta: pd.DataFrame, data=_KEEP_DATA) -> Self
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/dataset.py#L119-L128)

### nexuml.data.dataset.NexuDataset.download

*method*

```python
def download() -> None
```

Override to implement dataset downloading.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/dataset.py#L276-L278)

### nexuml.data.dataset.NexuDataset.get_split

*method*

```python
def get_split(split: str) -> Self
```

Return a view of this dataset filtered to a single split.

**Returns**

- (`Self`) — A new :class:`NexuDataset` containing only the requested split.

**Raises**

- `ValueError` — If the dataset has no metadata.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/dataset.py#L188-L201)

### nexuml.data.dataset.NexuDataset.load_item

*method*

```python
def load_item(idx: int, row: pd.Series) -> TensorDict
```

Load features for a single sample.

Subclasses can override this to lazily load file-backed content.

**Returns**

- (`TensorDict`) — Feature tensor dictionary for the sample.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/dataset.py#L203-L227)

### nexuml.data.dataset.NexuDataset.load_labels

*method*

```python
def load_labels(idx: int, row: pd.Series) -> TensorDict | None
```

Load labels for a single sample.

By default labels are read from ``self.meta`` using ``self.label_names``.
Subclasses can override this for file-backed or computed targets.

**Returns**

- (`TensorDict | None`) — Label tensor dictionary, or ``None`` if no labels are configured.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/dataset.py#L237-L255)

### nexuml.data.dataset.NexuDataset.split_meta

*method*

```python
def split_meta(split_ratio: list[float]) -> None
```

Convert 'fit' rows to train/val and 'all' rows to train/val/test.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/dataset.py#L163-L186)

### nexuml.data.dataset.NexuDataset.take

*method*

```python
def take(indices: Sequence[int]) -> Self
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/data/dataset.py#L154-L161)
