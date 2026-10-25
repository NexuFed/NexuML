# nexuml.data.creator

*module*

Unified dataset and datamodule creation flow.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/creator.py#L1-L1)

## nexuml.data.creator.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('lightning.pytorch')
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/creator.py#L12-L12)

## nexuml.data.creator.NexuDataCreator

*class*

```python
class NexuDataCreator
```

Build datasets and datamodules from a ``DataSpec``.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/creator.py#L15-L162)

### nexuml.data.creator.NexuDataCreator.build

*method*

```python
def build(spec: DataSpec, default_batch_size: int | AutoBatchSizeSpec = 64) -> NexuDataModule
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/creator.py#L104-L119)

### nexuml.data.creator.NexuDataCreator.build_dataset

*method*

```python
def build_dataset(spec: DataSpec) -> NexuDataset
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/creator.py#L18-L102)
