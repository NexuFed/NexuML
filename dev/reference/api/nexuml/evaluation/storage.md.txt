# nexuml.evaluation.storage

*module*

Temporary storage helpers for evaluation algorithms.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/evaluation/storage.py#L1-L1)

## nexuml.evaluation.storage.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/evaluation/storage.py#L15-L15)

## nexuml.evaluation.storage.AppendableTensorDictBuffer

*class*

```python
class AppendableTensorDictBuffer
```

Append TensorDict batches into temporary storage up to a fixed capacity.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/evaluation/storage.py#L136-L170)

### nexuml.evaluation.storage.AppendableTensorDictBuffer.max_samples

*attribute* · *instance attribute*

```python
max_samples = max_samples
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/evaluation/storage.py#L145-L145)

### nexuml.evaluation.storage.AppendableTensorDictBuffer.truncated

*attribute* · *instance attribute*

```python
truncated = False
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/evaluation/storage.py#L151-L151)

### nexuml.evaluation.storage.AppendableTensorDictBuffer.add_batch

*method*

```python
def add_batch(data: TensorDict) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/evaluation/storage.py#L153-L161)

### nexuml.evaluation.storage.AppendableTensorDictBuffer.finalize

*method*

```python
def finalize() -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/evaluation/storage.py#L166-L167)

### nexuml.evaluation.storage.AppendableTensorDictBuffer.get

*method*

```python
def get() -> TensorDict | None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/evaluation/storage.py#L163-L164)

## nexuml.evaluation.storage.ReservoirTensorDictBuffer

*class*

```python
class ReservoirTensorDictBuffer
```

Reservoir-sample TensorDict rows into temporary storage.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/evaluation/storage.py#L173-L211)

### nexuml.evaluation.storage.ReservoirTensorDictBuffer.max_samples

*attribute* · *instance attribute*

```python
max_samples = max_samples
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/evaluation/storage.py#L182-L182)

### nexuml.evaluation.storage.ReservoirTensorDictBuffer.n_seen

*attribute* · *instance attribute*

```python
n_seen = 0
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/evaluation/storage.py#L188-L188)

### nexuml.evaluation.storage.ReservoirTensorDictBuffer.add_batch

*method*

```python
def add_batch(data: TensorDict) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/evaluation/storage.py#L190-L202)

### nexuml.evaluation.storage.ReservoirTensorDictBuffer.finalize

*method*

```python
def finalize() -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/evaluation/storage.py#L207-L208)

### nexuml.evaluation.storage.ReservoirTensorDictBuffer.get

*method*

```python
def get() -> TensorDict | None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/evaluation/storage.py#L204-L205)

## nexuml.evaluation.storage.create_temporary_storage

*function*

```python
def create_temporary_storage(backend: str = 'memory', max_samples: int = 2000, storage_path: str | Path | None = None) -> _BaseTensorDictStorage
```

Create a temporary TensorDict storage backend.

**Returns**

- (`_BaseTensorDictStorage`) — The requested ``_BaseTensorDictStorage`` instance.

**Raises**

- `ValueError` — If *backend* is not ``"memory"`` or ``"memmap"``.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/evaluation/storage.py#L116-L133)
