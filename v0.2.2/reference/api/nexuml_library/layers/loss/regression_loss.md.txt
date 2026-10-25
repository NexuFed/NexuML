# nexuml_library.layers.loss.regression_loss

*module*

Regression loss layer.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/layers/loss/regression_loss.py#L1-L1)

## nexuml_library.layers.loss.regression_loss.RegressionLoss

*class*

```python
class RegressionLoss(LayerDefinition)
```

Bases: `LayerDefinition`

Computes MSE regression loss between predictions and targets.

keys_in: [predictions_key]
keys_out: [loss_key]
Reads targets from y TensorDict using label_key.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/layers/loss/regression_loss.py#L15-L25)

### nexuml_library.layers.loss.regression_loss.RegressionLoss.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/layers/loss/regression_loss.py#L24-L25)
