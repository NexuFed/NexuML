# nexuml.evaluation.registry

*module*

Explicit evaluation-algorithm materialization.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/evaluation/registry.py#L1-L1)

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

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/evaluation/registry.py#L10-L27)
