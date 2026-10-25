# nexuml.data.loaders.dali_backend

*module*

DALI loader backend using native readers when possible.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/dali_backend.py#L1-L1)

## nexuml.data.loaders.dali_backend.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/dali_backend.py#L17-L17)

## nexuml.data.loaders.dali_backend.DaliLoaderBackend

*class*

```python
class DaliLoaderBackend
```

DALI backend that prefers native readers and falls back deliberately.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/dali_backend.py#L175-L386)

### nexuml.data.loaders.dali_backend.DaliLoaderBackend.create_loader

*method*

```python
def create_loader(module: Any, dataset: Any, *, split: str, shuffle: bool = False, sampler: WeightedRandomSampler | None = None) -> Any
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/dali_backend.py#L178-L386)
