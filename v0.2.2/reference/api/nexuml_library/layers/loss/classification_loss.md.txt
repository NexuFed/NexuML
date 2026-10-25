# nexuml_library.layers.loss.classification_loss

*module*

Classification loss layer.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/layers/loss/classification_loss.py#L1-L1)

## nexuml_library.layers.loss.classification_loss.ClassificationLoss

*class*

```python
class ClassificationLoss(LayerDefinition)
```

Bases: `LayerDefinition`

Computes classification loss from logits and labels.

keys_in: [logits_key]
keys_out: [loss_key]
Reads labels from y TensorDict using label_key.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/layers/loss/classification_loss.py#L15-L32)

### nexuml_library.layers.loss.classification_loss.ClassificationLoss.label_smoothing

*attribute* · *class attribute* · *instance attribute*

```python
label_smoothing: float = 0.0
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/layers/loss/classification_loss.py#L25-L25)

### nexuml_library.layers.loss.classification_loss.ClassificationLoss.loss_type

*attribute* · *class attribute* · *instance attribute*

```python
loss_type: str = 'cross_entropy'
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/layers/loss/classification_loss.py#L24-L24)

### nexuml_library.layers.loss.classification_loss.ClassificationLoss.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/layers/loss/classification_loss.py#L27-L32)

## nexuml_library.layers.loss.classification_loss.LogitsToClass

*class*

```python
class LogitsToClass(LayerDefinition)
```

Bases: `LayerDefinition`

Converts class logits to predicted class integers via argmax.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/layers/loss/classification_loss.py#L87-L92)

### nexuml_library.layers.loss.classification_loss.LogitsToClass.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/layers/loss/classification_loss.py#L91-L92)
