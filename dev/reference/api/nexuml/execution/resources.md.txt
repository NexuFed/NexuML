# nexuml.execution.resources

*module*

Bounded read-only Local and Ray resource probes; no implicit runtime startup.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/resources.py#L1-L1)

## nexuml.execution.resources.execution

*attribute* · *module attribute*

```python
execution = restore_component(kind='execution_backend', value=request['execution'])
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/resources.py#L317-L317)

## nexuml.execution.resources.request

*attribute* · *module attribute*

```python
request = json.loads(sys.stdin.read())
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/resources.py#L316-L316)

## nexuml.execution.resources.result

*attribute* · *module attribute*

```python
result = _ray_resources(execution, request['timeout'], request['limit'])
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/resources.py#L318-L318)

## nexuml.execution.resources.DiscoveryBudget

*class*

```python
class DiscoveryBudget
```

One finite deadline and result bound shared by all probes in a request.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/resources.py#L15-L28)

### nexuml.execution.resources.DiscoveryBudget.deadline

*attribute* · *instance attribute*

```python
deadline = time.monotonic() + timeout
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/resources.py#L21-L21)

### nexuml.execution.resources.DiscoveryBudget.limit

*attribute* · *instance attribute*

```python
limit = limit
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/resources.py#L22-L22)

### nexuml.execution.resources.DiscoveryBudget.remaining

*method*

```python
def remaining() -> float
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/resources.py#L24-L28)

## nexuml.execution.resources.local_resources

*function*

```python
def local_resources(timeout: float = 10, limit: int = 1000, **kwargs: Any) -> ResourceSnapshot
```

Inspect process-effective limits and visible accelerators without training.

**Returns**

- (`ResourceSnapshot`) — Known limits plus explicit unknown/device-probe diagnostics.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/resources.py#L116-L183)

## nexuml.execution.resources.ray_resources

*function*

```python
def ray_resources(execution: Any, timeout: float = 10, limit: int = 1000, **kwargs: Any) -> ResourceSnapshot
```

Isolate Ray inspection so a wedged client cannot outlive the finite budget.

**Returns**

- (`ResourceSnapshot`) — A sourced snapshot or an explicit unknown diagnostic.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/resources.py#L186-L237)
