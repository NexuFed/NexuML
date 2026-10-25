# nexuml_library.layers.loss.regression_loss

*module*

Regression loss layer.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/loss/regression_loss.py#L1-L1)

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

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/loss/regression_loss.py#L15-L25)

### nexuml_library.layers.loss.regression_loss.RegressionLoss.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/loss/regression_loss.py#L24-L25)
