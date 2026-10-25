# nexuml.execution.semantics

*module*

Shared guard against plausible but incorrect rank-local finalization.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/execution/semantics.py#L1-L1)

## nexuml.execution.semantics.ExecutionError

*class*

```python
class ExecutionError(RuntimeError)
```

Bases: `RuntimeError`

An execution request cannot preserve NexuML semantics or ownership.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/execution/semantics.py#L6-L7)

## nexuml.execution.semantics.ensure_distributed_semantics

*function*

```python
def ensure_distributed_semantics(scenario: Any) -> None
```

Reject stateful phases until global distributed finalization exists.

**Raises**

- `ExecutionError` — If rank-sharded execution would change scenario semantics.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/execution/semantics.py#L10-L31)
