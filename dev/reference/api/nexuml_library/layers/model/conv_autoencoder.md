# nexuml_library.layers.model.conv_autoencoder

*module*

Convolutional autoencoder building blocks.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/model/conv_autoencoder.py#L1-L1)

## nexuml_library.layers.model.conv_autoencoder.ConvolutionalDecoder

*class*

```python
class ConvolutionalDecoder(LayerDefinition)
```

Bases: `LayerDefinition`

Decode a latent vector back into a 2D tensor.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/model/conv_autoencoder.py#L210-L226)

### nexuml_library.layers.model.conv_autoencoder.ConvolutionalDecoder.activation

*attribute* · *class attribute* · *instance attribute*

```python
activation: str = 'relu'
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/model/conv_autoencoder.py#L218-L218)

### nexuml_library.layers.model.conv_autoencoder.ConvolutionalDecoder.channel_schedule

*attribute* · *class attribute* · *instance attribute*

```python
channel_schedule: list[int] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/model/conv_autoencoder.py#L215-L215)

### nexuml_library.layers.model.conv_autoencoder.ConvolutionalDecoder.kernel_sizes

*attribute* · *class attribute* · *instance attribute*

```python
kernel_sizes: list[int | tuple[int, int]] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/model/conv_autoencoder.py#L216-L216)

### nexuml_library.layers.model.conv_autoencoder.ConvolutionalDecoder.output_shape

*attribute* · *instance attribute*

```python
output_shape: tuple[int, ...]
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/model/conv_autoencoder.py#L214-L214)

### nexuml_library.layers.model.conv_autoencoder.ConvolutionalDecoder.strides

*attribute* · *class attribute* · *instance attribute*

```python
strides: list[int | tuple[int, int]] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/model/conv_autoencoder.py#L217-L217)

### nexuml_library.layers.model.conv_autoencoder.ConvolutionalDecoder.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/model/conv_autoencoder.py#L220-L226)

## nexuml_library.layers.model.conv_autoencoder.ConvolutionalEncoder

*class*

```python
class ConvolutionalEncoder(LayerDefinition)
```

Bases: `LayerDefinition`

Encode a 2D tensor into a latent vector.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/model/conv_autoencoder.py#L81-L92)

### nexuml_library.layers.model.conv_autoencoder.ConvolutionalEncoder.activation

*attribute* · *class attribute* · *instance attribute*

```python
activation: str = 'relu'
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/model/conv_autoencoder.py#L89-L89)

### nexuml_library.layers.model.conv_autoencoder.ConvolutionalEncoder.channel_schedule

*attribute* · *class attribute* · *instance attribute*

```python
channel_schedule: list[int] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/model/conv_autoencoder.py#L86-L86)

### nexuml_library.layers.model.conv_autoencoder.ConvolutionalEncoder.kernel_sizes

*attribute* · *class attribute* · *instance attribute*

```python
kernel_sizes: list[int | tuple[int, int]] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/model/conv_autoencoder.py#L87-L87)

### nexuml_library.layers.model.conv_autoencoder.ConvolutionalEncoder.output_dim

*attribute* · *class attribute* · *instance attribute*

```python
output_dim: int = 128
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/model/conv_autoencoder.py#L85-L85)

### nexuml_library.layers.model.conv_autoencoder.ConvolutionalEncoder.strides

*attribute* · *class attribute* · *instance attribute*

```python
strides: list[int | tuple[int, int]] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/model/conv_autoencoder.py#L88-L88)

### nexuml_library.layers.model.conv_autoencoder.ConvolutionalEncoder.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/model/conv_autoencoder.py#L91-L92)

## nexuml_library.layers.model.conv_autoencoder.VariationalLatent

*class*

```python
class VariationalLatent(LayerDefinition)
```

Bases: `LayerDefinition`

Variational bottleneck layer emitting latent sample, moments, and KL loss.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/model/conv_autoencoder.py#L151-L159)

### nexuml_library.layers.model.conv_autoencoder.VariationalLatent.beta

*attribute* · *class attribute* · *instance attribute*

```python
beta: float = 1.0
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/model/conv_autoencoder.py#L156-L156)

### nexuml_library.layers.model.conv_autoencoder.VariationalLatent.latent_dim

*attribute* · *class attribute* · *instance attribute*

```python
latent_dim: int = 32
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/model/conv_autoencoder.py#L155-L155)

### nexuml_library.layers.model.conv_autoencoder.VariationalLatent.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/layers/model/conv_autoencoder.py#L158-L159)
