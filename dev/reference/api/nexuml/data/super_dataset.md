# nexuml.data.super_dataset

*module*

SuperDataset: merges multiple datasets into one unified dataset.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/super_dataset.py#L1-L1)

## nexuml.data.super_dataset.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/super_dataset.py#L13-L13)

## nexuml.data.super_dataset.SuperDataset

*class*

```python
class SuperDataset(NexuDataset)
```

Bases: `nexuml.data.dataset.NexuDataset`

Merges multiple datasets (keyed by name) into one dataset.

Adds a "dataset" column to the merged DataFrame to track origin.
Supports label merging and dataset subdivision for federated/multi-client scenarios.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/super_dataset.py#L16-L256)

### nexuml.data.super_dataset.SuperDataset.meta_data_list

*attribute* · *instance attribute*

```python
meta_data_list = dict(meta_data_list)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/super_dataset.py#L30-L30)

### nexuml.data.super_dataset.SuperDataset.data

*attribute* · *instance attribute*

```python
data = data
```

Inherited from: `nexuml.data.dataset.NexuDataset`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/dataset.py#L58-L58)

### nexuml.data.super_dataset.SuperDataset.label_names

*attribute* · *instance attribute*

```python
label_names: list[str] = list(label_names or [])
```

Inherited from: `nexuml.data.dataset.NexuDataset`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/dataset.py#L60-L60)

### nexuml.data.super_dataset.SuperDataset.meta

*attribute* · *instance attribute*

```python
meta: pd.DataFrame | None = None
```

Inherited from: `nexuml.data.dataset.NexuDataset`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/dataset.py#L56-L56)

### nexuml.data.super_dataset.SuperDataset.modality

*attribute* · *instance attribute*

```python
modality = modality
```

Inherited from: `nexuml.data.dataset.NexuDataset`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/dataset.py#L57-L57)

### nexuml.data.super_dataset.SuperDataset.num_classes

*attribute* · *instance attribute*

```python
num_classes: dict[str, int] = {}
```

Inherited from: `nexuml.data.dataset.NexuDataset`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/dataset.py#L61-L61)

### nexuml.data.super_dataset.SuperDataset.split_ratio

*attribute* · *instance attribute*

```python
split_ratio = split_ratio or [0.85, 0.1, 0.05]
```

Inherited from: `nexuml.data.dataset.NexuDataset`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/dataset.py#L59-L59)

### nexuml.data.super_dataset.SuperDataset.create_subset_dataset

*method*

```python
def create_subset_dataset(idx: int, do_split: bool = True) -> NexuDataset
```

Return a dataset for a single assignment subset.

**Returns**

- `A` (`NexuDataset`) — class:`NexuDataset` containing only samples assigned to ``idx``.

**Raises**

- `ValueError` — If metadata is missing.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/super_dataset.py#L190-L207)

### nexuml.data.super_dataset.SuperDataset.divide_dataset

*method*

```python
def divide_dataset(num_splits: int, key: str | None = None, alpha_in: float = 5.0, alpha_out: float = 0.0, labels_per_cluster: int = 3, num_clusters: int | None = None, seed: int | None = None) -> None
```

Assign samples to splits via Dirichlet cluster skew distribution.

**Raises**

- `ValueError` — If ``key`` is ``None`` or metadata is missing.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/super_dataset.py#L121-L156)

### nexuml.data.super_dataset.SuperDataset.divide_dataset_by_key

*method*

```python
def divide_dataset_by_key(key: str) -> int
```

Assign each unique key value to its own split index.

**Returns**

- (`int`) — Number of unique splits created.

**Raises**

- `ValueError` — If metadata is missing.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/super_dataset.py#L158-L174)

### nexuml.data.super_dataset.SuperDataset.load_item

*method*

```python
def load_item(idx: int, row: pd.Series) -> TensorDict
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/super_dataset.py#L209-L221)

### nexuml.data.super_dataset.SuperDataset.clone_with_meta

*method*

```python
def clone_with_meta(meta: pd.DataFrame, data=_KEEP_DATA) -> Self
```

Inherited from: `nexuml.data.dataset.NexuDataset`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/dataset.py#L119-L128)

### nexuml.data.super_dataset.SuperDataset.download

*method*

```python
def download() -> None
```

Inherited from: `nexuml.data.dataset.NexuDataset`

Override to implement dataset downloading.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/dataset.py#L276-L278)

### nexuml.data.super_dataset.SuperDataset.get_split

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

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/dataset.py#L188-L201)

### nexuml.data.super_dataset.SuperDataset.load_labels

*method*

```python
def load_labels(idx: int, row: pd.Series) -> TensorDict | None
```

Inherited from: `nexuml.data.dataset.NexuDataset`

Load labels for a single sample.

By default labels are read from ``self.meta`` using ``self.label_names``.
Subclasses can override this for file-backed or computed targets.

**Returns**

- (`TensorDict | None`) — Label tensor dictionary, or ``None`` if no labels are configured.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/dataset.py#L237-L255)

### nexuml.data.super_dataset.SuperDataset.split_meta

*method*

```python
def split_meta(split_ratio: list[float]) -> None
```

Inherited from: `nexuml.data.dataset.NexuDataset`

Convert 'fit' rows to train/val and 'all' rows to train/val/test.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/dataset.py#L163-L186)

### nexuml.data.super_dataset.SuperDataset.take

*method*

```python
def take(indices: Sequence[int]) -> Self
```

Inherited from: `nexuml.data.dataset.NexuDataset`

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/data/dataset.py#L154-L161)
