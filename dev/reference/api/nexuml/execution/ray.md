# nexuml.execution.ray

*module*

Thin Ray Train execution for the canonical NexuML Lightning session.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/ray.py#L1-L1)

## nexuml.execution.ray.RayExecutionError

*class*

```python
class RayExecutionError(ExecutionError)
```

Bases: `nexuml.execution.semantics.ExecutionError`

Ray placement cannot preserve the requested execution semantics.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/ray.py#L19-L20)

## nexuml.execution.ray.run_ray

*function*

```python
def run_ray(scenario: ScenarioSpec) -> Any
```

Execute one resolved scenario with Ray Train and return Ray's native Result.

**Returns**

- (`Any`) — The ``ray.train.Result`` returned by ``TorchTrainer.fit``.

**Raises**

- `RayExecutionError` — If the scenario cannot be executed safely with Ray or
Ray dependencies are unavailable.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/ray.py#L296-L325)

## nexuml.execution.ray.train_loop_per_worker

*function*

```python
def train_loop_per_worker(config: dict[str, Any]) -> None
```

Run one Ray worker through the normal NexuML session lifecycle.

**Raises**

- `RayExecutionError` — If Ray Train is unavailable in the worker environment.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/ray.py#L138-L158)
