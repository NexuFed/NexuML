# nexuml_library.scenarios.flow.synthetic_flow_matching

*module*

Synthetic Flow Matching reference scenario.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/scenarios/flow/synthetic_flow_matching.py#L1-L1)

## nexuml_library.scenarios.flow.synthetic_flow_matching.synthetic_flow_matching

*function*

```python
def synthetic_flow_matching(feature_shape: tuple[int, ...] = (2), num_samples: int = 2048, num_clusters: int = 4, hidden_dims: list[int] | None = None, num_steps: int = 100, lr: float = 0.001, batch_size: int = 64, max_epochs: int = 20, seed: int = 42) -> ScenarioSpec
```

Build a self-contained synthetic Flow Matching scenario.

**Returns**

- (`ScenarioSpec`) — Synthetic Flow Matching scenario.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/scenarios/flow/synthetic_flow_matching.py#L13-L61)
