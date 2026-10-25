# nexuml_library.layers.model.unet

*module*

Time-conditioned U-Net for image-valued vector fields.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/model/unet.py#L1-L1)

## nexuml_library.layers.model.unet.TimeConditionedUNet

*class*

```python
class TimeConditionedUNet(LayerDefinition)
```

Bases: `LayerDefinition`

Small time-conditioned U-Net for image-valued continuous flows.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/model/unet.py#L67-L78)

### nexuml_library.layers.model.unet.TimeConditionedUNet.base_channels

*attribute* · *class attribute* · *instance attribute*

```python
base_channels: int = Field(default=32, gt=0)
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/model/unet.py#L71-L71)

### nexuml_library.layers.model.unet.TimeConditionedUNet.time_embedding_dim

*attribute* · *class attribute* · *instance attribute*

```python
time_embedding_dim: int = Field(default=128, gt=0, multiple_of=2)
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/model/unet.py#L72-L72)

### nexuml_library.layers.model.unet.TimeConditionedUNet.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/model/unet.py#L74-L78)
