# nexuml.api.observation

*module*

Record actual Lightning callbacks without replacing the session lifecycle.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/observation.py#L1-L1)

## nexuml.api.observation.Observation

*class*

```python
class Observation(Callback)
```

Bases: `Callback`

Small callback appended to the existing Trainer's callback list.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/observation.py#L28-L166)

### nexuml.api.observation.Observation.completed

*attribute* · *instance attribute*

```python
completed = 0
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/observation.py#L35-L35)

### nexuml.api.observation.Observation.last_phase

*attribute* · *instance attribute*

```python
last_phase = ''
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/observation.py#L34-L34)

### nexuml.api.observation.Observation.last_progress

*attribute* · *instance attribute*

```python
last_progress = 0.0
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/observation.py#L33-L33)

### nexuml.api.observation.Observation.path

*attribute* · *instance attribute*

```python
path = path
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/observation.py#L32-L32)

### nexuml.api.observation.Observation.metrics

*method*

```python
def metrics(trainer, prefixes, **payload)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/observation.py#L57-L67)

### nexuml.api.observation.Observation.on_fit_end

*method*

```python
def on_fit_end(trainer, pl_module)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/observation.py#L156-L157)

### nexuml.api.observation.Observation.on_fit_start

*method*

```python
def on_fit_start(trainer, pl_module)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/observation.py#L80-L81)

### nexuml.api.observation.Observation.on_predict_batch_end

*method*

```python
def on_predict_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/observation.py#L150-L154)

### nexuml.api.observation.Observation.on_predict_epoch_start

*method*

```python
def on_predict_epoch_start(trainer, pl_module)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/observation.py#L144-L148)

### nexuml.api.observation.Observation.on_predict_start

*method*

```python
def on_predict_start(trainer, pl_module)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/observation.py#L162-L163)

### nexuml.api.observation.Observation.on_test_batch_end

*method*

```python
def on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/observation.py#L140-L142)

### nexuml.api.observation.Observation.on_test_end

*method*

```python
def on_test_end(trainer, pl_module)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/observation.py#L77-L78)

### nexuml.api.observation.Observation.on_test_epoch_start

*method*

```python
def on_test_epoch_start(trainer, pl_module)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/observation.py#L136-L138)

### nexuml.api.observation.Observation.on_test_start

*method*

```python
def on_test_start(trainer, pl_module)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/observation.py#L165-L166)

### nexuml.api.observation.Observation.on_train_batch_end

*method*

```python
def on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/observation.py#L110-L113)

### nexuml.api.observation.Observation.on_train_epoch_end

*method*

```python
def on_train_epoch_end(trainer, pl_module)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/observation.py#L69-L70)

### nexuml.api.observation.Observation.on_train_epoch_start

*method*

```python
def on_train_epoch_start(trainer, pl_module)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/observation.py#L107-L108)

### nexuml.api.observation.Observation.on_validation_batch_end

*method*

```python
def on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/observation.py#L125-L134)

### nexuml.api.observation.Observation.on_validation_end

*method*

```python
def on_validation_end(trainer, pl_module)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/observation.py#L72-L75)

### nexuml.api.observation.Observation.on_validation_epoch_start

*method*

```python
def on_validation_epoch_start(trainer, pl_module)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/observation.py#L115-L123)

### nexuml.api.observation.Observation.on_validation_start

*method*

```python
def on_validation_start(trainer, pl_module)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/observation.py#L159-L160)

### nexuml.api.observation.Observation.progress

*method*

```python
def progress(trainer, phase, batch, total, *, force=False) -> bool
```

Publish real counters at most four times per second, plus boundaries.

**Returns**

- (`bool`) — Whether a progress sample was published on this tick.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/observation.py#L83-L105)

### nexuml.api.observation.Observation.record

*method*

```python
def record(kind: str, trainer=None, **payload)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/observation.py#L37-L55)

## nexuml.api.observation.scalar_values

*function*

```python
def scalar_values(values: dict) -> dict
```

Keep only observable finite numeric scalars.

**Returns**

- (`dict`) — Plain scalar metrics, not tensor payloads or invented measurements.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/observation.py#L11-L25)
