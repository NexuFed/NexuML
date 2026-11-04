# nexuml.evaluation.registry

*module*

Explicit evaluation-algorithm materialization.

[View source](https://github.com/NexuFed/NexuML/blob/446d626f6712b0ca5963be5d1275ccf1bcd31bdd/src/nexuml/evaluation/registry.py#L1-L1)

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

[View source](https://github.com/NexuFed/NexuML/blob/446d626f6712b0ca5963be5d1275ccf1bcd31bdd/src/nexuml/evaluation/registry.py#L10-L27)
