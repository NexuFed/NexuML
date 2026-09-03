# Training backends

NexuML has one training lifecycle and two execution choices.

| Execution | What runs the training loop | Where it runs |
| --- | --- | --- |
| Local | PyTorch Lightning | Current process |
| Ray | PyTorch Lightning through Ray Train | Ray workers |

`NexuSession.run()` remains the canonical fit → validate → post-train → test lifecycle in both cases. Ray changes placement and distributed process setup; it does not introduce another NexuML training implementation.

Local execution is the default and needs no additional configuration:

```yaml
execution:
  kind: local
```

Use the [Ray training backend](ray.md) when a scenario should run across an existing Ray cluster or a temporary KubeRay cluster.
