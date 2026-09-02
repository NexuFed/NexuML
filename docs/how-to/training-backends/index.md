# Execution modes

NexuML has one model/training lifecycle and different places where that lifecycle can run.

| Execution | Placement | Training lifecycle |
| --- | --- | --- |
| Local | current process | `NexuSession.run()` |
| Ray | Ray Train workers | the same `NexuSession.run()` with Ray-aware Lightning setup |

Local execution is the `ScenarioSpec` default:

```python
from nexuml.core.types import LocalExecutionSpec

execution = LocalExecutionSpec()
```

Ray is selected through the scenario rather than through a second training API:

```python
from nexuml.core.types import RayClusterTarget, RayExecutionSpec

execution = RayExecutionSpec(
    target=RayClusterTarget(address="ray://ray.example.org:10001"),
    workers=4,
    resources_per_worker={"CPU": 4, "GPU": 1},
)
```

Training behavior such as epochs, precision, optimizer, and Lightning strategy remains in `TrainingSpec`. Execution configuration describes placement/resources.

See [Ray execution](ray.md) for the supported distributed boundary and current restrictions.
