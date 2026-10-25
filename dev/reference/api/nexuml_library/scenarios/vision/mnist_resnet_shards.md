# nexuml_library.scenarios.vision.mnist_resnet_shards

*module*

MNIST ResNet classification using tensor-shard data loading.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/vision/mnist_resnet_shards.py#L1-L1)

## nexuml_library.scenarios.vision.mnist_resnet_shards.mnist_resnet_shards

*function*

```python
def mnist_resnet_shards(download: bool = True, resnet_type: ResNetType = 'resnet18', pretrained: bool = False, cifar_stem: bool = True, lr: float = 0.001, batch_size: int = 64, max_epochs: int = 10, samples_per_shard: int = 4096, shards_per_window: int = 6, prefetch_windows: int = 2, prefetch_workers: int = 2, shuffle_shards: bool = True, shuffle_samples: bool = True, pin_memory: bool = False, overwrite_shards: bool = False) -> ScenarioSpec
```

MNIST ResNet using materialized tensor shards.

**Returns**

- `ScenarioSpec` (`ScenarioSpec`) — Assembled scenario configured for tensor-shard loading.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/scenarios/vision/mnist_resnet_shards.py#L15-L86)
