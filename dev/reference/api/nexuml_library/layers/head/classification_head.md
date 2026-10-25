# nexuml_library.layers.head.classification_head

*module*

Classification head for latent space.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/head/classification_head.py#L1-L1)

## nexuml_library.layers.head.classification_head.LatentClassificationHead

*class*

```python
class LatentClassificationHead(LayerDefinition)
```

Bases: `LayerDefinition`

Classification head operating on latent vectors.

Produces logits for multiclass or multilabel classification.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/head/classification_head.py#L15-L28)

### nexuml_library.layers.head.classification_head.LatentClassificationHead.hidden_dims

*attribute* · *class attribute* · *instance attribute*

```python
hidden_dims: list[int] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/head/classification_head.py#L22-L22)

### nexuml_library.layers.head.classification_head.LatentClassificationHead.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/head/classification_head.py#L24-L28)
