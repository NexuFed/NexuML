# nexuml_library.layers.loss.classification_metrics

*module*

Classification metrics layer.

Supports both multiclass and multilabel classification metrics via the
``multi_label`` flag.  In multilabel mode, sigmoid is applied to logits
before updating metrics and labels are kept as multi-hot vectors.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/loss/classification_metrics.py#L1-L1)

## nexuml_library.layers.loss.classification_metrics.ClassificationMetrics

*class*

```python
class ClassificationMetrics(LayerDefinition)
```

Bases: `LayerDefinition`

Accumulate classification metrics with torchmetrics.

When ``multi_label`` is ``False`` (default) the layer uses multiclass
metrics (``MulticlassAccuracy``, ``MulticlassF1Score``,
``MulticlassAveragePrecision``).  Labels are expected as integer class
indices with shape ``[batch]``.

When ``multi_label`` is ``True`` the layer uses multilabel metrics
(``MultilabelAccuracy``, ``MultilabelF1Score``,
``MultilabelAveragePrecision``).  Sigmoid is applied to logits before
passing them to the metrics and labels are expected as multi-hot float
tensors with shape ``[batch, num_labels]``.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/loss/classification_metrics.py#L21-L49)

### nexuml_library.layers.loss.classification_metrics.ClassificationMetrics.average

*attribute* · *class attribute* · *instance attribute*

```python
average: str = 'macro'
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/loss/classification_metrics.py#L38-L38)

### nexuml_library.layers.loss.classification_metrics.ClassificationMetrics.metrics

*attribute* · *class attribute* · *instance attribute*

```python
metrics: list[str] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/loss/classification_metrics.py#L37-L37)

### nexuml_library.layers.loss.classification_metrics.ClassificationMetrics.multi_label

*attribute* · *class attribute* · *instance attribute*

```python
multi_label: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/loss/classification_metrics.py#L40-L40)

### nexuml_library.layers.loss.classification_metrics.ClassificationMetrics.top_k

*attribute* · *class attribute* · *instance attribute*

```python
top_k: int = 1
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/loss/classification_metrics.py#L39-L39)

### nexuml_library.layers.loss.classification_metrics.ClassificationMetrics.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/loss/classification_metrics.py#L42-L49)
