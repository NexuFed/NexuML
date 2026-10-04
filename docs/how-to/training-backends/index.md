# Execution modes

NexuML has one model/training lifecycle and different places where that lifecycle can run.

| Execution | Placement | Training lifecycle |
| --- | --- | --- |
| Local | current process | `NexuSession.run()` |
| RayCluster | existing Ray Train workers | the same `NexuSession.run()` with Ray-aware Lightning setup |
| RayJob | KubeRay RayJob template | RayCluster execution inside the remote driver |
| Kubernetes Job | one template-owned training pod | the same `NexuSession.run()` |
| Kubernetes PyTorchJob | Training Operator replicas | the same lifecycle with native distributed launch |

Local execution is the `ScenarioSpec` default:

```python
from nexuml.execution.definitions import LocalExecution

execution = LocalExecution()
```

Ray is selected through the scenario rather than through a second training API:

```python
from nexuml.execution.definitions import RayClusterTarget, RayClusterExecution

execution = RayClusterExecution(
    target=RayClusterTarget(address="ray://ray.example.org:10001"),
    workers=4,
    resources_per_worker={"CPU": 4, "GPU": 1},
)
```

Training behavior such as epochs, precision, optimizer, and Lightning strategy remains in `TrainingSpec`. Execution configuration describes placement/resources.

See [Ray execution](ray.md) for the supported distributed boundary and current restrictions.
See [native jobs and capacity review](jobs.md) for scoped discovery, handoff/templates,
resource units and the separately authorized smoke procedure.

Python, CLI and Studio share the selected installation's registered definitions:

```python
from nexuml.execution import catalog, discover, preflight, run

backends = catalog()                # definitions and dependencies; no cluster probes
snapshot = discover(execution)      # read-only, selected target only
review = preflight(scenario)        # no worker allocation or job creation
result = run(scenario)              # explicit launch; backend's native result
```

Persisted execution uses `type`/`version`/`params`, as with other registered components.
Use `@execution_backend("my-backend")` on an `ExecutionBackendDefinition` subclass in
an installed `nexuml.libraries` package or configured local library root. Implement `run`
and, where supported, `preflight` and `discover`; publish truthful capability metadata.
Do not import cluster runtimes at module scope. Unknown discovery stays unknown, not zero.
`run(scenario, **kwargs)` receives the `review` returned by preflight and optional caller
options/observer. Keep review JSON-safe and browser-facing summaries credential-free.
