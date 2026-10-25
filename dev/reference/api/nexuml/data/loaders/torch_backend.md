# nexuml.data.loaders.torch_backend

*module*

Native PyTorch DataLoader backend.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/torch_backend.py#L1-L1)

## nexuml.data.loaders.torch_backend.TorchLoaderBackend

*class*

```python
class TorchLoaderBackend
```

Native PyTorch dataloader backend.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/torch_backend.py#L29-L54)

### nexuml.data.loaders.torch_backend.TorchLoaderBackend.create_loader

*method*

```python
def create_loader(module, dataset, *, split: str, shuffle: bool = False, sampler: WeightedRandomSampler | None = None) -> DataLoader
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/data/loaders/torch_backend.py#L32-L54)
