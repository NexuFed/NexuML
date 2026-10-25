# nexuml_library.layers.head.regression_head

*module*

Regression head for latent space.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/head/regression_head.py#L1-L1)

## nexuml_library.layers.head.regression_head.LatentRegressionHead

*class*

```python
class LatentRegressionHead(LayerDefinition)
```

Bases: `LayerDefinition`

Regression head operating on latent vectors.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/head/regression_head.py#L15-L23)

### nexuml_library.layers.head.regression_head.LatentRegressionHead.hidden_dims

*attribute* · *class attribute* · *instance attribute*

```python
hidden_dims: list[int] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/head/regression_head.py#L20-L20)

### nexuml_library.layers.head.regression_head.LatentRegressionHead.num_outputs

*attribute* · *class attribute* · *instance attribute*

```python
num_outputs: int = 1
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/head/regression_head.py#L19-L19)

### nexuml_library.layers.head.regression_head.LatentRegressionHead.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/head/regression_head.py#L22-L23)
