# nexuml.execution.kubernetes

*module*

One native Kubernetes adapter for template-owned Job, RayJob and PyTorchJob.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/execution/kubernetes.py#L1-L1)

## nexuml.execution.kubernetes.cancel

*function*

```python
def cancel(reference: NativeReference) -> dict
```

Delete only a UID-verified owned workload and confirm foreground termination.

**Returns**

- (`dict`) — A confirmed cancellation or an explicit unknown/error state.

**Raises**

- `ExecutionError` — If cancellation cannot stop the native workload safely.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/execution/kubernetes.py#L572-L595)

## nexuml.execution.kubernetes.discover_resources

*function*

```python
def discover_resources(execution: Any, timeout: float = 10, limit: int = 1000, authorized_root: Path | None = None, **kwargs: Any) -> ResourceSnapshot
```

Bound context authentication and selected-scope read-only capacity queries.

**Returns**

- (`ResourceSnapshot`) — Independent successes with unknown/forbidden/timeout diagnostics.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/execution/kubernetes.py#L464-L505)

## nexuml.execution.kubernetes.inspect

*function*

```python
def inspect(reference: NativeReference) -> dict
```

Inspect only the recorded native identity through its trusted context.

**Returns**

- (`dict`) — Actual workload state and safe reference, without fabricated telemetry.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/execution/kubernetes.py#L555-L569)

## nexuml.execution.kubernetes.main

*function*

```python
def main() -> None
```

Run a private bounded probe, never a workload submission endpoint.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/execution/kubernetes.py#L758-L779)

## nexuml.execution.kubernetes.native_status

*function*

```python
def native_status(resource: dict) -> str
```

Interpret actual native workload conditions, never submitter exit.

**Returns**

- (`str`) — Submitted/pending/running or a verified terminal state.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/execution/kubernetes.py#L275-L304)

## nexuml.execution.kubernetes.prepare_job

*function*

```python
def prepare_job(scenario: Any, authorized_root: Path | None = None, frozen_plan: dict | None = None, **kwargs: Any) -> dict
```

Review exact inputs, native create access and admission without submitting.

**Returns**

- (`dict`) — A private frozen plan with safe review summary and native dry-run evidence.

**Raises**

- `ExecutionError` — On stale template, denied create access or admission rejection.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/execution/kubernetes.py#L508-L552)

## nexuml.execution.kubernetes.run_job

*function*

```python
def run_job(scenario: Any, observer: Any = None, frozen_plan: dict | None = None, authorized_root: Path | None = None, **kwargs: Any) -> dict
```

Submit once, publish the reference promptly, then observe the native workload.

**Returns**

- (`dict`) — Native reference and actual terminal state.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/execution/kubernetes.py#L598-L649)
