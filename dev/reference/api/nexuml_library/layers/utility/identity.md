# nexuml_library.layers.utility.identity

*module*

Identity and noise layers.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/utility/identity.py#L1-L1)

## nexuml_library.layers.utility.identity.AdditiveNoise

*class*

```python
class AdditiveNoise(LayerDefinition)
```

Bases: `LayerDefinition`

Adds Gaussian noise scaled by the input's standard deviation.

Useful as a data augmentation layer (noise only applied during training).

**Attributes**

- `noise_level` (`float`) — Noise scale relative to the input std.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/utility/identity.py#L52-L65)

### nexuml_library.layers.utility.identity.AdditiveNoise.noise_level

*attribute* · *class attribute* · *instance attribute*

```python
noise_level: float = 0.1
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/utility/identity.py#L62-L62)

### nexuml_library.layers.utility.identity.AdditiveNoise.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/utility/identity.py#L64-L65)

## nexuml_library.layers.utility.identity.MergeLayer

*class*

```python
class MergeLayer(LayerDefinition)
```

Bases: `LayerDefinition`

Merge multiple input tensors into one by concatenation along the last axis.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/utility/identity.py#L15-L22)

### nexuml_library.layers.utility.identity.MergeLayer.dim

*attribute* · *class attribute* · *instance attribute*

```python
dim: int = -2
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/utility/identity.py#L19-L19)

### nexuml_library.layers.utility.identity.MergeLayer.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/utility/identity.py#L21-L22)
