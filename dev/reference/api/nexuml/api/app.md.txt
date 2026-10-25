# nexuml.api.app

*module*

FastAPI presentation of the selected local NexuML installation.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/api/app.py#L1-L1)

## nexuml.api.app.create_app

*function*

```python
def create_app(settings: Settings) -> FastAPI
```

Create an isolated local API application.

**Returns**

- (`FastAPI`) — Authenticated FastAPI app for this selected runtime and directory.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/api/app.py#L20-L92)
