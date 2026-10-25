# nexuml_library.layers.model.resnet

*module*

ResNet pipeline layer for image classification backbones.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/model/resnet.py#L1-L1)

## nexuml_library.layers.model.resnet.ResNetType

*attribute* · *module attribute*

```python
ResNetType = Literal['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/model/resnet.py#L17-L17)

## nexuml_library.layers.model.resnet.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/model/resnet.py#L15-L15)

## nexuml_library.layers.model.resnet.ResNet

*class*

```python
class ResNet(LayerDefinition)
```

Bases: `LayerDefinition`

ResNet backbone that emits feature embeddings.

Wraps ``torchvision.models.resnet*`` and replaces the final
fully-connected layer with an identity so downstream heads
can operate on embeddings.

**Attributes**

- `resnet_type` (`ResNetType`) — Architecture variant (e.g. ``"resnet18"``).
- `pretrained` (`bool`) — Whether to load pretrained ImageNet weights.
- `cifar_stem` (`bool | None`) — Use a CIFAR-friendly stem (3x3 stride-1, no maxpool). Defaults to ``False`` when pretrained, ``True`` otherwise.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/model/resnet.py#L20-L40)

### nexuml_library.layers.model.resnet.ResNet.cifar_stem

*attribute* · *class attribute* · *instance attribute*

```python
cifar_stem: bool | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/model/resnet.py#L37-L37)

### nexuml_library.layers.model.resnet.ResNet.pretrained

*attribute* · *class attribute* · *instance attribute*

```python
pretrained: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/model/resnet.py#L36-L36)

### nexuml_library.layers.model.resnet.ResNet.resnet_type

*attribute* · *class attribute* · *instance attribute*

```python
resnet_type: ResNetType = 'resnet18'
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/model/resnet.py#L35-L35)

### nexuml_library.layers.model.resnet.ResNet.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/model/resnet.py#L39-L40)
