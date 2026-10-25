# nexuml_library.layers.model.linear_encoder

*module*

Linear encoder/decoder layer for autoencoder architectures.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/model/linear_encoder.py#L1-L1)

## nexuml_library.layers.model.linear_encoder.LinearEncoder

*class*

```python
class LinearEncoder(LayerDefinition)
```

Bases: `LayerDefinition`

Stacked fully-connected layers with optional batch norm and activation.

Used for both encoder and decoder in linear autoencoders.
Input is flattened to 2D before processing.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/model/linear_encoder.py#L15-L30)

### nexuml_library.layers.model.linear_encoder.LinearEncoder.activation

*attribute* · *class attribute* · *instance attribute*

```python
activation: str = 'torch.nn.ReLU'
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/model/linear_encoder.py#L25-L25)

### nexuml_library.layers.model.linear_encoder.LinearEncoder.bias

*attribute* · *class attribute* · *instance attribute*

```python
bias: bool = True
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/model/linear_encoder.py#L27-L27)

### nexuml_library.layers.model.linear_encoder.LinearEncoder.hidden_dims

*attribute* · *class attribute* · *instance attribute*

```python
hidden_dims: list[int] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/model/linear_encoder.py#L23-L23)

### nexuml_library.layers.model.linear_encoder.LinearEncoder.last_activation

*attribute* · *class attribute* · *instance attribute*

```python
last_activation: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/model/linear_encoder.py#L26-L26)

### nexuml_library.layers.model.linear_encoder.LinearEncoder.output_dim

*attribute* · *class attribute* · *instance attribute*

```python
output_dim: int = 8
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/model/linear_encoder.py#L24-L24)

### nexuml_library.layers.model.linear_encoder.LinearEncoder.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/model/linear_encoder.py#L29-L30)
