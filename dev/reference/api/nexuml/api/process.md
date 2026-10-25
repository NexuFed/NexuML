# nexuml.api.process

*module*

Fresh selected-interpreter calls for discovery and configuration validation.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/process.py#L1-L1)

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

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/process.py#L26-L67)

## nexuml.api.process.worker_environment

*function*

```python
def worker_environment() -> dict[str, str]
```

Inherit the selected environment without forwarding API credentials.

**Returns**

- (`dict[str, str]`) — Worker environment, with private connection settings removed.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/process.py#L13-L23)
