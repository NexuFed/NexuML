# nexuml_library.layers.head.anomaly_score

*module*

Anomaly scoring layers.

[View source](https://github.com/NexuFed/NexuML/blob/484c461e3d84309527692c64ae567c7de3524802/library/src/nexuml_library/layers/head/anomaly_score.py#L1-L1)

## nexuml_library.layers.head.anomaly_score.AnomalyScore

*class*

```python
class AnomalyScore(LayerDefinition)
```

Bases: `LayerDefinition`

Compute a per-sample anomaly score from original and reconstructed tensors.

[View source](https://github.com/NexuFed/NexuML/blob/484c461e3d84309527692c64ae567c7de3524802/library/src/nexuml_library/layers/head/anomaly_score.py#L14-L21)

### nexuml_library.layers.head.anomaly_score.AnomalyScore.reduction

*attribute* · *class attribute* · *instance attribute*

```python
reduction: str = 'mean'
```

[View source](https://github.com/NexuFed/NexuML/blob/484c461e3d84309527692c64ae567c7de3524802/library/src/nexuml_library/layers/head/anomaly_score.py#L18-L18)

### nexuml_library.layers.head.anomaly_score.AnomalyScore.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/484c461e3d84309527692c64ae567c7de3524802/library/src/nexuml_library/layers/head/anomaly_score.py#L20-L21)
