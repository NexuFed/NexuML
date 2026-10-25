# nexuml.api.app

*module*

FastAPI presentation of the selected local NexuML installation.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/app.py#L1-L1)

## nexuml.api.app.create_app

*function*

```python
def create_app(settings: Settings) -> FastAPI
```

Create an isolated local API application.

**Returns**

- (`FastAPI`) — Authenticated FastAPI app for this selected runtime and directory.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/app.py#L20-L92)
