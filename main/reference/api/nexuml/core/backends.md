# nexuml.core.backends

*module*

Existing backend descriptions shared by CLI and local API.

[View source](https://github.com/NexuFed/NexuML/blob/484c461e3d84309527692c64ae567c7de3524802/src/nexuml/core/backends.py#L1-L1)

## nexuml.core.backends.backend_rows

*function*

```python
def backend_rows() -> list[tuple[str, str, str]]
```

Describe existing backends without pretending every dependency is installed.

**Returns**

- (`list[tuple[str, str, str]]`) — Sorted category/name/import-target rows (availability is checked on use).

[View source](https://github.com/NexuFed/NexuML/blob/484c461e3d84309527692c64ae567c7de3524802/src/nexuml/core/backends.py#L4-L34)
