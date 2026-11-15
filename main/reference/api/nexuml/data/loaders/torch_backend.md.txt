# nexuml.data.loaders.torch_backend

*module*

Native PyTorch DataLoader backend.

[View source](https://github.com/NexuFed/NexuML/blob/484c461e3d84309527692c64ae567c7de3524802/src/nexuml/data/loaders/torch_backend.py#L1-L1)

## nexuml.data.loaders.torch_backend.TorchLoaderBackend

*class*

```python
class TorchLoaderBackend
```

Native PyTorch dataloader backend.

[View source](https://github.com/NexuFed/NexuML/blob/484c461e3d84309527692c64ae567c7de3524802/src/nexuml/data/loaders/torch_backend.py#L29-L54)

### nexuml.data.loaders.torch_backend.TorchLoaderBackend.create_loader

*method*

```python
def create_loader(module, dataset, *, split: str, shuffle: bool = False, sampler: WeightedRandomSampler | None = None) -> DataLoader
```

[View source](https://github.com/NexuFed/NexuML/blob/484c461e3d84309527692c64ae567c7de3524802/src/nexuml/data/loaders/torch_backend.py#L32-L54)
