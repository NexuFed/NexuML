# nexuml.data.module

*module*

Lightning DataModule for NexuML datasets.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/module.py#L1-L1)

## nexuml.data.module.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/module.py#L15-L15)

## nexuml.data.module.NexuDataModule

*class*

```python
class NexuDataModule(L.LightningDataModule)
```

Bases: `L.LightningDataModule`

Lightning DataModule wrapping a ``NexuDataset``.

Supports:
- random splits or metadata-driven splits
- weighted sampling for the native torch backend
- backend-driven loader construction via ``LoaderSpec.backend``

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/module.py#L18-L154)

### nexuml.data.module.NexuDataModule.dataset

*attribute* · *instance attribute*

```python
dataset = dataset
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/module.py#L38-L38)

### nexuml.data.module.NexuDataModule.loader_spec

*attribute* · *instance attribute*

```python
loader_spec = loader_spec or LoaderSpec()
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/module.py#L39-L39)

### nexuml.data.module.NexuDataModule.seed

*attribute* · *instance attribute*

```python
seed = seed
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/module.py#L43-L43)

### nexuml.data.module.NexuDataModule.split_by_column

*attribute* · *instance attribute*

```python
split_by_column = split_by_column
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/module.py#L44-L44)

### nexuml.data.module.NexuDataModule.test_split

*attribute* · *instance attribute*

```python
test_split = test_split
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/module.py#L42-L42)

### nexuml.data.module.NexuDataModule.train_split

*attribute* · *instance attribute*

```python
train_split = train_split
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/module.py#L40-L40)

### nexuml.data.module.NexuDataModule.val_split

*attribute* · *instance attribute*

```python
val_split = val_split
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/module.py#L41-L41)

### nexuml.data.module.NexuDataModule.predict_dataloader

*method*

```python
def predict_dataloader()
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/module.py#L153-L154)

### nexuml.data.module.NexuDataModule.setup

*method*

```python
def setup(stage: str | None = None) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/module.py#L49-L56)

### nexuml.data.module.NexuDataModule.test_dataloader

*method*

```python
def test_dataloader()
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/module.py#L150-L151)

### nexuml.data.module.NexuDataModule.train_dataloader

*method*

```python
def train_dataloader()
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/module.py#L138-L145)

### nexuml.data.module.NexuDataModule.val_dataloader

*method*

```python
def val_dataloader()
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/module.py#L147-L148)
