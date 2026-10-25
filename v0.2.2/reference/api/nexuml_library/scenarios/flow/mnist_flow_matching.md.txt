# nexuml_library.scenarios.flow.mnist_flow_matching

*module*

MNIST Flow Matching reference scenario.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/scenarios/flow/mnist_flow_matching.py#L1-L1)

## nexuml_library.scenarios.flow.mnist_flow_matching.mnist_flow_matching

*function*

```python
def mnist_flow_matching(base_channels: int = 32, time_embedding_dim: int = 128, num_steps: int = 300, lr: float = 0.0001, batch_size: int = 128, max_epochs: int = 20, download: bool = True) -> ScenarioSpec
```

Build an unconditional MNIST Flow Matching scenario.

**Returns**

- (`ScenarioSpec`) — MNIST Flow Matching scenario.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/scenarios/flow/mnist_flow_matching.py#L13-L57)
