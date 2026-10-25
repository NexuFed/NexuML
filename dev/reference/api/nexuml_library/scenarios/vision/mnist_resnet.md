# nexuml_library.scenarios.vision.mnist_resnet

*module*

MNIST and FashionMNIST ResNet classification scenarios.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/scenarios/vision/mnist_resnet.py#L1-L1)

## nexuml_library.scenarios.vision.mnist_resnet.fashionmnist_data

*function*

```python
def fashionmnist_data(download: bool = True, root: str = 'fashionmnist') -> DataSpec
```

Create a DataSpec for FashionMNIST image classification.

**Returns**

- `DataSpec` (`DataSpec`) — FashionMNIST dataset specification with fit and test splits.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/scenarios/vision/mnist_resnet.py#L56-L79)

## nexuml_library.scenarios.vision.mnist_resnet.fashionmnist_resnet

*function*

```python
def fashionmnist_resnet(download: bool = True, resnet_type: ResNetType = 'resnet18', pretrained: bool = False, cifar_stem: bool = True, lr: float = 0.001, batch_size: int = 64, max_epochs: int = 10) -> ScenarioSpec
```

FashionMNIST image classification with ResNet backbone.

**Returns**

- `ScenarioSpec` (`ScenarioSpec`) — Assembled scenario with pipeline, training, data and evaluation.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/scenarios/vision/mnist_resnet.py#L110-L135)

## nexuml_library.scenarios.vision.mnist_resnet.mnist_data

*function*

```python
def mnist_data(download: bool = True, root: str = 'mnist', normalize: bool = False) -> DataSpec
```

Create a DataSpec for MNIST image classification.

**Returns**

- `DataSpec` (`DataSpec`) — MNIST dataset specification with fit and test splits.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/scenarios/vision/mnist_resnet.py#L16-L53)

## nexuml_library.scenarios.vision.mnist_resnet.mnist_resnet

*function*

```python
def mnist_resnet(download: bool = True, resnet_type: ResNetType = 'resnet18', pretrained: bool = False, cifar_stem: bool = True, lr: float = 0.001, batch_size: int = 64, max_epochs: int = 10) -> ScenarioSpec
```

MNIST image classification with ResNet backbone.

**Returns**

- `ScenarioSpec` (`ScenarioSpec`) — Assembled scenario with pipeline, training, data and evaluation.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/scenarios/vision/mnist_resnet.py#L82-L107)
