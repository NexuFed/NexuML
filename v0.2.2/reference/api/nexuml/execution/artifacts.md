# nexuml.execution.artifacts

*module*

Requested exports reuse existing NexuML export functions and native results.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/execution/artifacts.py#L1-L1)

## nexuml.execution.artifacts.export_requests

*function*

```python
def export_requests(source: Any, requests: list[Any], checkpoint: str | None = None) -> list[dict[str, str]]
```

Export configured formats from the actual pipeline/trainer.

**Returns**

- (`list[dict[str, str]]`) — Existing-format local artifact references, not invented remote downloads.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/execution/artifacts.py#L7-L37)
