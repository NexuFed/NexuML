# nexuml.training.callbacks

*module*

Build configured Lightning callbacks.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/callbacks.py#L1-L1)

## nexuml.training.callbacks.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/callbacks.py#L13-L13)

## nexuml.training.callbacks.build_callbacks

*function*

```python
def build_callbacks(callback_specs: list[CallbackSpec]) -> list[Callback]
```

Instantiate Lightning callbacks from portable factory specs.

**Returns**

- (`list[Callback]`) — List of instantiated Lightning callback objects.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/training/callbacks.py#L16-L37)
