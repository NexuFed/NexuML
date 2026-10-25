# nexuml.execution

*module*

Installation-owned execution, with lazy optional runtime imports.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/__init__.py#L1-L1)

## nexuml.execution.RayExecutionError

*class*

```python
class RayExecutionError(ExecutionError)
```

Bases: `nexuml.execution.semantics.ExecutionError`

Re-exported from: `nexuml.execution.ray`

Ray placement cannot preserve the requested execution semantics.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/ray.py#L19-L20)

## nexuml.execution.cancel

*function*

```python
def cancel(reference: NativeReference) -> dict[str, Any]
```

Re-exported from: `nexuml.execution.service`

Request supported native cancellation through the registered backend.

**Returns**

- (`dict[str, Any]`) — Confirmed cancellation or an explicit unsupported/unknown error.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/service.py#L113-L123)

## nexuml.execution.catalog

*function*

```python
def catalog() -> list[dict[str, Any]]
```

Re-exported from: `nexuml.execution.service`

Describe execution definitions from the selected Python installation.

**Returns**

- (`list[dict[str, Any]]`) — Registered identities, actual schemas, provenance and installation capabilities.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/service.py#L25-L49)

## nexuml.execution.discover

*function*

```python
def discover(execution: ExecutionBackendDefinition, **kwargs: Any) -> ResourceSnapshot
```

Re-exported from: `nexuml.execution.service`

Probe only the caller-selected backend and target.

**Returns**

- (`ResourceSnapshot`) — An advisory snapshot, including unavailable/unknown diagnostics.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/service.py#L52-L66)

## nexuml.execution.inspect

*function*

```python
def inspect(reference: NativeReference) -> dict[str, Any]
```

Re-exported from: `nexuml.execution.service`

Inspect a native reference without resubmitting it.

**Returns**

- (`dict[str, Any]`) — Actual supported native observation from the registered backend.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/service.py#L100-L110)

## nexuml.execution.preflight

*function*

```python
def preflight(scenario: Any, **kwargs: Any) -> dict[str, Any]
```

Re-exported from: `nexuml.execution.service`

Revalidate a frozen selection before any worker allocation or submission.

**Returns**

- (`dict[str, Any]`) — Backend-owned review information.

**Raises**

- `ValueError` — If dependencies, scenario semantics or launch inputs are invalid.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/service.py#L69-L84)

## nexuml.execution.run

*function*

```python
def run(scenario: Any, **kwargs: Any) -> Any
```

Re-exported from: `nexuml.execution.service`

Launch through the same installation-owned dispatch in every interface.

**Returns**

- (`Any`) — The backend's native result, never a second training result hierarchy.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/service.py#L87-L97)

## nexuml.execution.run_ray

*function*

```python
def run_ray(scenario: ScenarioSpec) -> Any
```

Re-exported from: `nexuml.execution.ray`

Execute one resolved scenario with Ray Train and return Ray's native Result.

**Returns**

- (`Any`) — The ``ray.train.Result`` returned by ``TorchTrainer.fit``.

**Raises**

- `RayExecutionError` — If the scenario cannot be executed safely with Ray or
Ray dependencies are unavailable.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/ray.py#L296-L325)

**Modules**

- `nexuml.execution.artifacts`
- `nexuml.execution.capacity`
- `nexuml.execution.definitions`
- `nexuml.execution.kubernetes`
- `nexuml.execution.ray`
- `nexuml.execution.resources`
- `nexuml.execution.schemas`
- `nexuml.execution.semantics`
- `nexuml.execution.service`
- `nexuml.execution.templates`
- `nexuml.execution.worker`
