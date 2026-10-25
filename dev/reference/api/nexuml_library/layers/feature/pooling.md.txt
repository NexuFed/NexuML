# nexuml_library.layers.feature.pooling

*module*

Pooling layers for token/sequence aggregation.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/pooling.py#L1-L1)

## nexuml_library.layers.feature.pooling.AttentionPool

*class*

```python
class AttentionPool(LayerDefinition)
```

Bases: `LayerDefinition`

Learned attention pooling using a query token.

**Attributes**

- `dim` (`int`) — Token dimension to pool over.
- `remove_dim` (`bool`) — Whether to remove the pooled dimension.
- `n_heads` (`int`) — Number of attention heads.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/pooling.py#L184-L199)

### nexuml_library.layers.feature.pooling.AttentionPool.dim

*attribute* · *class attribute* · *instance attribute*

```python
dim: int = -2
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/pooling.py#L194-L194)

### nexuml_library.layers.feature.pooling.AttentionPool.n_heads

*attribute* · *class attribute* · *instance attribute*

```python
n_heads: int = 8
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/pooling.py#L196-L196)

### nexuml_library.layers.feature.pooling.AttentionPool.remove_dim

*attribute* · *class attribute* · *instance attribute*

```python
remove_dim: bool = True
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/pooling.py#L195-L195)

### nexuml_library.layers.feature.pooling.AttentionPool.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/pooling.py#L198-L199)

## nexuml_library.layers.feature.pooling.AttentiveStatisticsPool

*class*

```python
class AttentiveStatisticsPool(LayerDefinition)
```

Bases: `LayerDefinition`

Attentive Statistics Pooling (Okabe et al., 2018).

Computes attention-weighted mean and std, then projects to input dim.

**Attributes**

- `dim` (`int`) — Dimension to pool over.
- `remove_dim` (`bool`) — Whether to remove the pooled dimension.
- `hidden_dim` (`int`) — Hidden dimension for the attention MLP.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/pooling.py#L230-L247)

### nexuml_library.layers.feature.pooling.AttentiveStatisticsPool.dim

*attribute* · *class attribute* · *instance attribute*

```python
dim: int = -2
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/pooling.py#L242-L242)

### nexuml_library.layers.feature.pooling.AttentiveStatisticsPool.hidden_dim

*attribute* · *class attribute* · *instance attribute*

```python
hidden_dim: int = 128
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/pooling.py#L244-L244)

### nexuml_library.layers.feature.pooling.AttentiveStatisticsPool.remove_dim

*attribute* · *class attribute* · *instance attribute*

```python
remove_dim: bool = True
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/pooling.py#L243-L243)

### nexuml_library.layers.feature.pooling.AttentiveStatisticsPool.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/pooling.py#L246-L247)

## nexuml_library.layers.feature.pooling.TokenPool

*class*

```python
class TokenPool(LayerDefinition)
```

Bases: `LayerDefinition`

Pool tokens along a specified dimension.

**Attributes**

- `dim` (`int`) — Dimension to pool over (negative indexing supported, e.g. -2 for time).
- `pooling_type` (`Literal['mean', 'max', 'std', 'mean_std', 'cls', 'multilayer']`) — - "mean": average over the pool dim - "max": max over the pool dim - "std": standard deviation over the pool dim - "mean_std" (alias: "mean+std", "mean_plus_std"): concatenate   mean and std along the last axis - "cls": select the first index along the pool dim (CLS-style) - "multilayer": when the input is multi-layer hidden states   (e.g. ``[B, L, N, D]`` from an encoder that exposes all   hidden states), concatenate per-layer mean+std along the   last axis, preserving the per-layer ordering.
- `remove_dim` (`bool`) — Whether to squeeze the pooled dimension.
- `skip_first` (`bool`) — Skip the first token (e.g. CLS token) before pooling.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/pooling.py#L21-L48)

### nexuml_library.layers.feature.pooling.TokenPool.dim

*attribute* · *class attribute* · *instance attribute*

```python
dim: int = -2
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/pooling.py#L42-L42)

### nexuml_library.layers.feature.pooling.TokenPool.pooling_type

*attribute* · *class attribute* · *instance attribute*

```python
pooling_type: Literal['mean', 'max', 'std', 'mean_std', 'cls', 'multilayer'] = 'mean'
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/pooling.py#L43-L43)

### nexuml_library.layers.feature.pooling.TokenPool.remove_dim

*attribute* · *class attribute* · *instance attribute*

```python
remove_dim: bool = True
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/pooling.py#L44-L44)

### nexuml_library.layers.feature.pooling.TokenPool.skip_first

*attribute* · *class attribute* · *instance attribute*

```python
skip_first: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/pooling.py#L45-L45)

### nexuml_library.layers.feature.pooling.TokenPool.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/feature/pooling.py#L47-L48)
