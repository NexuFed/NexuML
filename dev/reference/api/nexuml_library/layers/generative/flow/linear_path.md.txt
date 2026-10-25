# nexuml_library.layers.generative.flow.linear_path

*module*

Linear probability path for Flow Matching.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/layers/generative/flow/linear_path.py#L1-L1)

## nexuml_library.layers.generative.flow.linear_path.LinearFlowPath

*class*

```python
class LinearFlowPath(LayerDefinition)
```

Bases: `LayerDefinition`

Sample the linear path from standard Gaussian noise to data.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/layers/generative/flow/linear_path.py#L24-L29)

### nexuml_library.layers.generative.flow.linear_path.LinearFlowPath.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/layers/generative/flow/linear_path.py#L28-L29)
