# nexuml.execution.worker

*module*

Canonical remote worker entrypoint; never submit the outer job recursively.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/execution/worker.py#L1-L1)

## nexuml.execution.worker.launch_options

*function*

```python
def launch_options(scenario: Any, environment: dict[str, str]) -> dict[str, Any]
```

Validate torchrun rank/world-size rather than run independent replicas.

**Returns**

- (`dict[str, Any]`) — Native Lightning placement preserving the reviewed training request.

**Raises**

- `ValueError` — For incomplete or inconsistent native distributed launch settings.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/execution/worker.py#L14-L47)

## nexuml.execution.worker.main

*function*

```python
def main() -> None
```

Read one explicit worker config and dispatch the existing lifecycle.

**Raises**

- `ValueError` — If the worker receives an outer job instead of a derived config.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/execution/worker.py#L50-L70)
