# nexuml_library.layers.model.flow

*module*

Continuous flow model layer.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/model/flow.py#L1-L1)

## nexuml_library.layers.model.flow.Flow

*class*

```python
class Flow(LayerDefinition)
```

Bases: `LayerDefinition`

Trainable vector field with Euler sampling.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/model/flow.py#L20-L45)

### nexuml_library.layers.model.flow.Flow.hidden_dims

*attribute* · *class attribute* · *instance attribute*

```python
hidden_dims: list[int] = Field(default_factory=lambda : [128, 128])
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/model/flow.py#L24-L24)

### nexuml_library.layers.model.flow.Flow.num_steps

*attribute* · *class attribute* · *instance attribute*

```python
num_steps: int = Field(default=100, gt=0)
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/model/flow.py#L25-L25)

### nexuml_library.layers.model.flow.Flow.record_trajectory

*attribute* · *class attribute* · *instance attribute*

```python
record_trajectory: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/model/flow.py#L26-L26)

### nexuml_library.layers.model.flow.Flow.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/model/flow.py#L28-L45)

## nexuml_library.layers.model.flow.UNetFlow

*class*

```python
class UNetFlow(LayerDefinition)
```

Bases: `LayerDefinition`

Image flow using a time-conditioned U-Net vector field.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/model/flow.py#L48-L75)

### nexuml_library.layers.model.flow.UNetFlow.base_channels

*attribute* · *class attribute* · *instance attribute*

```python
base_channels: int = Field(default=32, gt=0)
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/model/flow.py#L52-L52)

### nexuml_library.layers.model.flow.UNetFlow.num_steps

*attribute* · *class attribute* · *instance attribute*

```python
num_steps: int = Field(default=100, gt=0)
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/model/flow.py#L54-L54)

### nexuml_library.layers.model.flow.UNetFlow.record_trajectory

*attribute* · *class attribute* · *instance attribute*

```python
record_trajectory: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/model/flow.py#L55-L55)

### nexuml_library.layers.model.flow.UNetFlow.time_embedding_dim

*attribute* · *class attribute* · *instance attribute*

```python
time_embedding_dim: int = Field(default=128, gt=0)
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/model/flow.py#L53-L53)

### nexuml_library.layers.model.flow.UNetFlow.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/model/flow.py#L57-L75)
