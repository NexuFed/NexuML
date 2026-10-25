# nexuml_library.layers.head.classification_head

*module*

Classification head for latent space.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/head/classification_head.py#L1-L1)

## nexuml_library.layers.head.classification_head.LatentClassificationHead

*class*

```python
class LatentClassificationHead(LayerDefinition)
```

Bases: `LayerDefinition`

Classification head operating on latent vectors.

Produces logits for multiclass or multilabel classification.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/head/classification_head.py#L15-L28)

### nexuml_library.layers.head.classification_head.LatentClassificationHead.hidden_dims

*attribute* · *class attribute* · *instance attribute*

```python
hidden_dims: list[int] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/head/classification_head.py#L22-L22)

### nexuml_library.layers.head.classification_head.LatentClassificationHead.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/head/classification_head.py#L24-L28)
