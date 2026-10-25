# nexuml.execution.service

*module*

Shared Python entrypoints; CLI and API only transport these operations.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/service.py#L1-L1)

## nexuml.execution.service.cancel

*function*

```python
def cancel(reference: NativeReference) -> dict[str, Any]
```

Request supported native cancellation through the registered backend.

**Returns**

- (`dict[str, Any]`) — Confirmed cancellation or an explicit unsupported/unknown error.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/service.py#L113-L123)

## nexuml.execution.service.catalog

*function*

```python
def catalog() -> list[dict[str, Any]]
```

Describe execution definitions from the selected Python installation.

**Returns**

- (`list[dict[str, Any]]`) — Registered identities, actual schemas, provenance and installation capabilities.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/service.py#L25-L49)

## nexuml.execution.service.dependency_diagnostics

*function*

```python
def dependency_diagnostics(definition: type[ExecutionBackendDefinition]) -> list[str]
```

Check installation support without importing optional runtimes.

**Returns**

- (`list[str]`) — Missing dependency reasons, not cluster readiness claims.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/service.py#L12-L22)

## nexuml.execution.service.discover

*function*

```python
def discover(execution: ExecutionBackendDefinition, **kwargs: Any) -> ResourceSnapshot
```

Probe only the caller-selected backend and target.

**Returns**

- (`ResourceSnapshot`) — An advisory snapshot, including unavailable/unknown diagnostics.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/service.py#L52-L66)

## nexuml.execution.service.inspect

*function*

```python
def inspect(reference: NativeReference) -> dict[str, Any]
```

Inspect a native reference without resubmitting it.

**Returns**

- (`dict[str, Any]`) — Actual supported native observation from the registered backend.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/service.py#L100-L110)

## nexuml.execution.service.preflight

*function*

```python
def preflight(scenario: Any, **kwargs: Any) -> dict[str, Any]
```

Revalidate a frozen selection before any worker allocation or submission.

**Returns**

- (`dict[str, Any]`) — Backend-owned review information.

**Raises**

- `ValueError` — If dependencies, scenario semantics or launch inputs are invalid.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/service.py#L69-L84)

## nexuml.execution.service.run

*function*

```python
def run(scenario: Any, **kwargs: Any) -> Any
```

Launch through the same installation-owned dispatch in every interface.

**Returns**

- (`Any`) — The backend's native result, never a second training result hierarchy.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/execution/service.py#L87-L97)
