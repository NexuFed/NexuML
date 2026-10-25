# nexuml_library.layers.feature

*module*

Feature extraction and embedding layers.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/layers/feature/__init__.py#L1-L1)

## nexuml_library.layers.feature.L2Normalize

*class*

```python
class L2Normalize(LayerDefinition)
```

Bases: `LayerDefinition`

Re-exported from: `nexuml_library.layers.feature.l2_normalize`

L2-normalize the last dimension of the input.

Stateless — no trainable parameters and no fitted state. Equivalent
to ``torch.nn.functional.normalize(x, p=2, dim=-1, eps=eps)``.

**Attributes**

- `eps` (`float`) — Small epsilon added to the denominator to avoid division by zero. Defaults to ``1e-12`` (matches ``F.normalize`` default).

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/layers/feature/l2_normalize.py#L24-L39)

### nexuml_library.layers.feature.L2Normalize.eps

*attribute* · *class attribute* · *instance attribute*

```python
eps: float = 1e-12
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/layers/feature/l2_normalize.py#L36-L36)

### nexuml_library.layers.feature.L2Normalize.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/layers/feature/l2_normalize.py#L38-L39)

**Modules**

- `nexuml_library.layers.feature.l2_normalize`
- `nexuml_library.layers.feature.lmbe`
- `nexuml_library.layers.feature.pooling`
- `nexuml_library.layers.feature.projector`
