# nexuml.training.callbacks

*module*

Build configured Lightning callbacks.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/training/callbacks.py#L1-L1)

## nexuml.training.callbacks.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/training/callbacks.py#L13-L13)

## nexuml.training.callbacks.build_callbacks

*function*

```python
def build_callbacks(callback_specs: list[CallbackSpec]) -> list[Callback]
```

Instantiate Lightning callbacks from portable factory specs.

**Returns**

- (`list[Callback]`) — List of instantiated Lightning callback objects.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/training/callbacks.py#L16-L37)
