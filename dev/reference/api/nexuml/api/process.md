# nexuml.api.process

*module*

Fresh selected-interpreter calls for discovery and configuration validation.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/process.py#L1-L1)

## nexuml.api.process.call_worker

*function*

```python
def call_worker(action: str, payload: dict, directory: Path) -> dict
```

Run one existing NexuML call in a fresh selected-interpreter process.

**Returns**

- (`dict`) — Plain result data from the worker's private response file.

**Raises**

- `ApiError` — If validation, initialization or the bounded call fails.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/process.py#L26-L67)

## nexuml.api.process.worker_environment

*function*

```python
def worker_environment() -> dict[str, str]
```

Inherit the selected environment without forwarding API credentials.

**Returns**

- (`dict[str, str]`) — Worker environment, with private connection settings removed.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/process.py#L13-L23)
