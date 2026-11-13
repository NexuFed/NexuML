# nexuml.training.callbacks

*module*

Build configured Lightning callbacks.

[View source](https://github.com/NexuFed/NexuML/blob/1787e5e6ec271ff0045b9659199a85fdf92d1e06/src/nexuml/training/callbacks.py#L1-L1)

## nexuml.training.callbacks.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/1787e5e6ec271ff0045b9659199a85fdf92d1e06/src/nexuml/training/callbacks.py#L13-L13)

## nexuml.training.callbacks.build_callbacks

*function*

```python
def build_callbacks(callback_specs: list[CallbackSpec]) -> list[Callback]
```

Instantiate Lightning callbacks from portable factory specs.

**Returns**

- (`list[Callback]`) — List of instantiated Lightning callback objects.

[View source](https://github.com/NexuFed/NexuML/blob/1787e5e6ec271ff0045b9659199a85fdf92d1e06/src/nexuml/training/callbacks.py#L16-L37)
