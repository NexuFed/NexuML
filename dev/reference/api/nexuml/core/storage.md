# nexuml.core.storage

*module*

Shared storage for inter-layer communication.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/storage.py#L1-L1)

## nexuml.core.storage.SharedStorage

*class*

```python
class SharedStorage
```

Manages shared memory for inter-layer communication in the pipeline.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/storage.py#L10-L71)

### nexuml.core.storage.SharedStorage.data

*attribute* · *instance attribute*

```python
data = TensorDict({}, batch_size=[])
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/storage.py#L21-L21)

### nexuml.core.storage.SharedStorage.map_location

*attribute* · *instance attribute*

```python
map_location = map_location
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/storage.py#L19-L19)

### nexuml.core.storage.SharedStorage.memory_mapped

*attribute* · *instance attribute*

```python
memory_mapped = memory_mapped
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/storage.py#L18-L18)

### nexuml.core.storage.SharedStorage.append

*method*

```python
def append(key: str, value: torch.Tensor | TensorDict)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/storage.py#L38-L46)

### nexuml.core.storage.SharedStorage.clear

*method*

```python
def clear(key: str)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/storage.py#L69-L71)

### nexuml.core.storage.SharedStorage.get

*method*

```python
def get(key: str)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/storage.py#L66-L67)

### nexuml.core.storage.SharedStorage.set

*method*

```python
def set(key: str, value: torch.Tensor | TensorDict)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/storage.py#L23-L36)

### nexuml.core.storage.SharedStorage.set_ring_buffer

*method*

```python
def set_ring_buffer(key: str, value: torch.Tensor | TensorDict, counter: int)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/storage.py#L48-L64)
