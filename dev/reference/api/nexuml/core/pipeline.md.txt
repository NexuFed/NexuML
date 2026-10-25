# nexuml.core.pipeline

*module*

Compiled pipeline: the runtime model assembled from specs.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/pipeline.py#L1-L1)

## nexuml.core.pipeline.CompiledPipeline

*class*

```python
class CompiledPipeline(nn.Module)
```

Bases: `nn.Module`

Executable pipeline assembled from compiled layer specs.

Holds ordered stages of layers, executes sequentially over TensorDicts,
and provides optimizer/scheduler factory methods.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/pipeline.py#L17-L143)

### nexuml.core.pipeline.CompiledPipeline.input_sizes

*attribute* · *instance attribute*

```python
input_sizes = input_sizes or {}
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/pipeline.py#L49-L49)

### nexuml.core.pipeline.CompiledPipeline.loss_keys

*attribute* · *instance attribute*

```python
loss_keys = loss_keys
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/pipeline.py#L36-L36)

### nexuml.core.pipeline.CompiledPipeline.metric_keys

*attribute* · *instance attribute*

```python
metric_keys = metric_keys
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/pipeline.py#L37-L37)

### nexuml.core.pipeline.CompiledPipeline.resolved_config

*attribute* · *instance attribute*

```python
resolved_config = resolved_config
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/pipeline.py#L38-L38)

### nexuml.core.pipeline.CompiledPipeline.stages

*attribute* · *instance attribute*

```python
stages = stages
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/pipeline.py#L35-L35)

### nexuml.core.pipeline.CompiledPipeline.call_layer_hook

*method*

```python
def call_layer_hook(hook_name: str) -> None
```

Propagate a lifecycle hook to all pipeline layers.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/pipeline.py#L129-L143)

### nexuml.core.pipeline.CompiledPipeline.create_optimizer

*method*

```python
def create_optimizer() -> torch.optim.Optimizer
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/pipeline.py#L109-L116)

### nexuml.core.pipeline.CompiledPipeline.create_scheduler

*method*

```python
def create_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/pipeline.py#L118-L127)

### nexuml.core.pipeline.CompiledPipeline.forward

*method*

```python
def forward(x: TensorDict, y: TensorDict | None = None) -> tuple[TensorDict, TensorDict | None]
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/pipeline.py#L63-L68)

### nexuml.core.pipeline.CompiledPipeline.forward_until

*method*

```python
def forward_until(x: TensorDict, y: TensorDict | None = None, *, x_keys: Sequence[str] | None = None, y_keys: Sequence[str] | None = None) -> tuple[TensorDict, TensorDict | None]
```

Run the pipeline until the requested keys are available.

This is useful for feature export when the codebase does not yet model
preprocessing layers separately from the rest of the network.

**Returns**

- (`tuple[TensorDict, TensorDict | None]`) — Tuple of ``(x_out, y_out)`` with the requested keys present.

**Raises**

- `KeyError` — If the pipeline did not produce the requested keys.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/pipeline.py#L70-L107)

### nexuml.core.pipeline.CompiledPipeline.iter_layers

*method*

```python
def iter_layers() -> Iterator[tuple[str, str, nn.Module]]
```

Yield pipeline layers in execution order.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/pipeline.py#L51-L61)
