# nexuml_library.layers.loss.multilabel_map_metrics

*module*

Multi-label mAP metric layer.

[View source](https://github.com/NexuFed/NexuML/blob/1787e5e6ec271ff0045b9659199a85fdf92d1e06/library/src/nexuml_library/layers/loss/multilabel_map_metrics.py#L1-L1)

## nexuml_library.layers.loss.multilabel_map_metrics.MultiLabelMAPMetrics

*class*

```python
class MultiLabelMAPMetrics(LayerDefinition)
```

Bases: `LayerDefinition`

Pipeline layer computing multi-label mean Average Precision.

Consumes logits from keys_in[0] and multi-hot labels from y[label_key].
Applies sigmoid to logits before updating the wrapped
torchmetrics MultilabelAveragePrecision metric.

[View source](https://github.com/NexuFed/NexuML/blob/1787e5e6ec271ff0045b9659199a85fdf92d1e06/library/src/nexuml_library/layers/loss/multilabel_map_metrics.py#L15-L28)

### nexuml_library.layers.loss.multilabel_map_metrics.MultiLabelMAPMetrics.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/1787e5e6ec271ff0045b9659199a85fdf92d1e06/library/src/nexuml_library/layers/loss/multilabel_map_metrics.py#L24-L28)
