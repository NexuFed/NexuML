# nexuml_library.layers.feature.projector

*module*

Linear projection layers.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/projector.py#L1-L1)

## nexuml_library.layers.feature.projector.Conv1dProjector

*class*

```python
class Conv1dProjector(LayerDefinition)
```

Bases: `LayerDefinition`

Pointwise Conv1d projector for sequence inputs (B, T, C).

Equivalent to Linear but uses Conv1d for better memory access patterns.

**Attributes**

- `target_dim` (`int | None`) — Output channel dimension. If None, uses output_sizes.
- `hidden_dim` (`int`) — Hidden size for multi-layer projectors.
- `n_layers` (`int`) — Number of projection layers.
- `bias` (`bool`) — Whether to use bias.
- `activation` (`str | None`) — Dotted class path for activation.
- `normalization` (`str | None`) — Dotted class path for normalization.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/projector.py#L130-L153)

### nexuml_library.layers.feature.projector.Conv1dProjector.activation

*attribute* · *class attribute* · *instance attribute*

```python
activation: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/projector.py#L149-L149)

### nexuml_library.layers.feature.projector.Conv1dProjector.bias

*attribute* · *class attribute* · *instance attribute*

```python
bias: bool = True
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/projector.py#L148-L148)

### nexuml_library.layers.feature.projector.Conv1dProjector.hidden_dim

*attribute* · *class attribute* · *instance attribute*

```python
hidden_dim: int = 256
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/projector.py#L146-L146)

### nexuml_library.layers.feature.projector.Conv1dProjector.n_layers

*attribute* · *class attribute* · *instance attribute*

```python
n_layers: int = 1
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/projector.py#L147-L147)

### nexuml_library.layers.feature.projector.Conv1dProjector.normalization

*attribute* · *class attribute* · *instance attribute*

```python
normalization: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/projector.py#L150-L150)

### nexuml_library.layers.feature.projector.Conv1dProjector.target_dim

*attribute* · *class attribute* · *instance attribute*

```python
target_dim: int | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/projector.py#L145-L145)

### nexuml_library.layers.feature.projector.Conv1dProjector.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/projector.py#L152-L153)

## nexuml_library.layers.feature.projector.Linear

*class*

```python
class Linear(LayerDefinition)
```

Bases: `LayerDefinition`

Multi-layer linear projector with optional activation and normalization.

**Attributes**

- `target_dim` (`int | None`) — Output dimension. If None, uses output_sizes.
- `hidden_dim` (`int`) — Hidden layer size (for n_layers > 1).
- `hidden_dims` (`list[int] | None`) — Explicit hidden-layer sizes. When provided, overrides hidden_dim/n_layers for layer construction.
- `n_layers` (`int`) — Number of linear layers.
- `linear_bias` (`bool`) — Whether to use bias.
- `activation` (`str | None`) — Dotted class path, e.g. "torch.nn.GELU".
- `normalization` (`str | None`) — Dotted class path, e.g. "torch.nn.BatchNorm1d".
- `skip_last_activation` (`bool`) — Skip activation after the last layer.
- `flatten_dims` (`tuple[int, int] | None`) — Optional (start, end) dims to flatten before projection.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/projector.py#L34-L62)

### nexuml_library.layers.feature.projector.Linear.activation

*attribute* · *class attribute* · *instance attribute*

```python
activation: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/projector.py#L56-L56)

### nexuml_library.layers.feature.projector.Linear.flatten_dims

*attribute* · *class attribute* · *instance attribute*

```python
flatten_dims: tuple[int, int] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/projector.py#L59-L59)

### nexuml_library.layers.feature.projector.Linear.hidden_dim

*attribute* · *class attribute* · *instance attribute*

```python
hidden_dim: int = 256
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/projector.py#L52-L52)

### nexuml_library.layers.feature.projector.Linear.hidden_dims

*attribute* · *class attribute* · *instance attribute*

```python
hidden_dims: list[int] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/projector.py#L53-L53)

### nexuml_library.layers.feature.projector.Linear.linear_bias

*attribute* · *class attribute* · *instance attribute*

```python
linear_bias: bool = True
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/projector.py#L55-L55)

### nexuml_library.layers.feature.projector.Linear.n_layers

*attribute* · *class attribute* · *instance attribute*

```python
n_layers: int = 1
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/projector.py#L54-L54)

### nexuml_library.layers.feature.projector.Linear.normalization

*attribute* · *class attribute* · *instance attribute*

```python
normalization: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/projector.py#L57-L57)

### nexuml_library.layers.feature.projector.Linear.skip_last_activation

*attribute* · *class attribute* · *instance attribute*

```python
skip_last_activation: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/projector.py#L58-L58)

### nexuml_library.layers.feature.projector.Linear.target_dim

*attribute* · *class attribute* · *instance attribute*

```python
target_dim: int | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/projector.py#L51-L51)

### nexuml_library.layers.feature.projector.Linear.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/projector.py#L61-L62)

## nexuml_library.layers.feature.projector.LowRankProjector

*class*

```python
class LowRankProjector(LayerDefinition)
```

Bases: `LayerDefinition`

Low-rank factorized projection W ≈ U @ V.

Reduces parameters from O(C*D) to O((C+D)*r) when r << min(C, D).

**Attributes**

- `target_dim` (`int | None`) — Output dimension. If None, uses output_sizes.
- `rank` (`int`) — Bottleneck rank.
- `bias` (`bool`) — Whether to use bias.
- `activation` (`str | None`) — Optional activation between U and V.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/projector.py#L204-L223)

### nexuml_library.layers.feature.projector.LowRankProjector.activation

*attribute* · *class attribute* · *instance attribute*

```python
activation: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/projector.py#L220-L220)

### nexuml_library.layers.feature.projector.LowRankProjector.bias

*attribute* · *class attribute* · *instance attribute*

```python
bias: bool = True
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/projector.py#L219-L219)

### nexuml_library.layers.feature.projector.LowRankProjector.rank

*attribute* · *class attribute* · *instance attribute*

```python
rank: int = 64
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/projector.py#L218-L218)

### nexuml_library.layers.feature.projector.LowRankProjector.target_dim

*attribute* · *class attribute* · *instance attribute*

```python
target_dim: int | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/projector.py#L217-L217)

### nexuml_library.layers.feature.projector.LowRankProjector.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/projector.py#L222-L223)
