# nexuml_library.layers.generative.flow.vector_field

*module*

Time-conditioned vector field for Flow Matching.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/generative/flow/vector_field.py#L1-L1)

## nexuml_library.layers.generative.flow.vector_field.TimeConditionedVectorField

*class*

```python
class TimeConditionedVectorField(LayerDefinition)
```

Bases: `LayerDefinition`

Vector field using the reusable linear projector as its backbone.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/generative/flow/vector_field.py#L18-L41)

### nexuml_library.layers.generative.flow.vector_field.TimeConditionedVectorField.hidden_dims

*attribute* · *class attribute* · *instance attribute*

```python
hidden_dims: list[int] = Field(default_factory=lambda : [128, 128])
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/generative/flow/vector_field.py#L22-L22)

### nexuml_library.layers.generative.flow.vector_field.TimeConditionedVectorField.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/generative/flow/vector_field.py#L24-L41)
