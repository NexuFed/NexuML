# nexuml.training.callbacks

*module*

Build configured Lightning callbacks.

[View source](https://github.com/NexuFed/NexuML/blob/484c461e3d84309527692c64ae567c7de3524802/src/nexuml/training/callbacks.py#L1-L1)

## nexuml.training.callbacks.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/484c461e3d84309527692c64ae567c7de3524802/src/nexuml/training/callbacks.py#L13-L13)

## nexuml.training.callbacks.build_callbacks

*function*

```python
def build_callbacks(callback_specs: list[CallbackSpec]) -> list[Callback]
```

Instantiate Lightning callbacks from portable factory specs.

**Returns**

- (`list[Callback]`) — List of instantiated Lightning callback objects.

[View source](https://github.com/NexuFed/NexuML/blob/484c461e3d84309527692c64ae567c7de3524802/src/nexuml/training/callbacks.py#L16-L37)
