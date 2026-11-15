# nexuml.evaluation.registry

*module*

Explicit evaluation-algorithm materialization.

[View source](https://github.com/NexuFed/NexuML/blob/484c461e3d84309527692c64ae567c7de3524802/src/nexuml/evaluation/registry.py#L1-L1)

## nexuml.evaluation.registry.create_algorithm

*function*

```python
def create_algorithm(spec: EvalAlgorithmSpec) -> EvalAlgorithm
```

Materialize and validate an evaluation algorithm definition.

**Returns**

- (`EvalAlgorithm`) — Runtime evaluation algorithm.

**Raises**

- `TypeError` — If the definition builds the wrong runtime type.

[View source](https://github.com/NexuFed/NexuML/blob/484c461e3d84309527692c64ae567c7de3524802/src/nexuml/evaluation/registry.py#L10-L27)
