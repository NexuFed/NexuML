# nexuml.api.app

*module*

FastAPI presentation of the selected local NexuML installation.

[View source](https://github.com/NexuFed/NexuML/blob/446d626f6712b0ca5963be5d1275ccf1bcd31bdd/src/nexuml/api/app.py#L1-L1)

## nexuml.api.app.create_app

*function*

```python
def create_app(settings: Settings) -> FastAPI
```

Create an isolated local API application.

**Returns**

- (`FastAPI`) — Authenticated FastAPI app for this selected runtime and directory.

[View source](https://github.com/NexuFed/NexuML/blob/446d626f6712b0ca5963be5d1275ccf1bcd31bdd/src/nexuml/api/app.py#L20-L92)
