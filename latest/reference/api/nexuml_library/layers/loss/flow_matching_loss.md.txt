# nexuml_library.layers.loss.flow_matching_loss

*module*

Flow Matching objective.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/layers/loss/flow_matching_loss.py#L1-L1)

## nexuml_library.layers.loss.flow_matching_loss.FlowMatchingLoss

*class*

```python
class FlowMatchingLoss(LayerDefinition)
```

Bases: `LayerDefinition`

Per-sample MSE between predicted and target velocity.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/layers/loss/flow_matching_loss.py#L15-L20)

### nexuml_library.layers.loss.flow_matching_loss.FlowMatchingLoss.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/layers/loss/flow_matching_loss.py#L19-L20)
