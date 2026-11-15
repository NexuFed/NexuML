# nexuml.execution.artifacts

*module*

Requested exports reuse existing NexuML export functions and native results.

[View source](https://github.com/NexuFed/NexuML/blob/484c461e3d84309527692c64ae567c7de3524802/src/nexuml/execution/artifacts.py#L1-L1)

## nexuml.execution.artifacts.export_requests

*function*

```python
def export_requests(source: Any, requests: list[Any], checkpoint: str | None = None) -> list[dict[str, str]]
```

Export configured formats from the actual pipeline/trainer.

**Returns**

- (`list[dict[str, str]]`) — Existing-format local artifact references, not invented remote downloads.

[View source](https://github.com/NexuFed/NexuML/blob/484c461e3d84309527692c64ae567c7de3524802/src/nexuml/execution/artifacts.py#L7-L37)
