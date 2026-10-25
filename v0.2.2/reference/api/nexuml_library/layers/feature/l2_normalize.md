# nexuml_library.layers.feature.l2_normalize

*module*

L2-normalization pipeline layer.

A trivial stateless postproc: applies L2 normalization to the last
dimension of the input. Has no trainable state and no ``fit()``/``is_fitted``
contract — it is an ordinary pipeline composition primitive, not a
train-fitted postprocessor.

This is the canonical "normalize embeddings" step the DCASE 2026 P3
target pipeline uses. It is intentionally a separate layer from
:class:`FeaturePostproc` so the target scenario does not need a
train-fitted postproc / pre-materialization bridge contract.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/layers/feature/l2_normalize.py#L1-L1)

## nexuml_library.layers.feature.l2_normalize.L2Normalize

*class*

```python
class L2Normalize(LayerDefinition)
```

Bases: `LayerDefinition`

L2-normalize the last dimension of the input.

Stateless — no trainable parameters and no fitted state. Equivalent
to ``torch.nn.functional.normalize(x, p=2, dim=-1, eps=eps)``.

**Attributes**

- `eps` (`float`) — Small epsilon added to the denominator to avoid division by zero. Defaults to ``1e-12`` (matches ``F.normalize`` default).

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/layers/feature/l2_normalize.py#L24-L39)

### nexuml_library.layers.feature.l2_normalize.L2Normalize.eps

*attribute* · *class attribute* · *instance attribute*

```python
eps: float = 1e-12
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/layers/feature/l2_normalize.py#L36-L36)

### nexuml_library.layers.feature.l2_normalize.L2Normalize.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/layers/feature/l2_normalize.py#L38-L39)
