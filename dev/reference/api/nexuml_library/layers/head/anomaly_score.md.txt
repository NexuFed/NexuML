# nexuml_library.layers.head.anomaly_score

*module*

Anomaly scoring layers.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/layers/head/anomaly_score.py#L1-L1)

## nexuml_library.layers.head.anomaly_score.AnomalyScore

*class*

```python
class AnomalyScore(LayerDefinition)
```

Bases: `LayerDefinition`

Compute a per-sample anomaly score from original and reconstructed tensors.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/layers/head/anomaly_score.py#L14-L21)

### nexuml_library.layers.head.anomaly_score.AnomalyScore.reduction

*attribute* · *class attribute* · *instance attribute*

```python
reduction: str = 'mean'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/layers/head/anomaly_score.py#L18-L18)

### nexuml_library.layers.head.anomaly_score.AnomalyScore.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/layers/head/anomaly_score.py#L20-L21)
