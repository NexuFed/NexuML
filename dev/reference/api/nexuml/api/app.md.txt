# nexuml.api.app

*module*

FastAPI presentation of the selected local NexuML installation.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/app.py#L1-L1)

## nexuml.api.app.create_app

*function*

```python
def create_app(settings: Settings) -> FastAPI
```

Create an isolated local API application.

**Returns**

- (`FastAPI`) — Authenticated FastAPI app for this selected runtime and directory.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/api/app.py#L20-L92)
