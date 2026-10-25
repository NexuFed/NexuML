# nexuml_library.layers.head

*module*

Output head layers.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/head/__init__.py#L1-L1)

## nexuml_library.layers.head.AnomalyScore

*class*

```python
class AnomalyScore(LayerDefinition)
```

Bases: `LayerDefinition`

Re-exported from: `nexuml_library.layers.head.anomaly_score`

Compute a per-sample anomaly score from original and reconstructed tensors.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/head/anomaly_score.py#L14-L21)

### nexuml_library.layers.head.AnomalyScore.reduction

*attribute* · *class attribute* · *instance attribute*

```python
reduction: str = 'mean'
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/head/anomaly_score.py#L18-L18)

### nexuml_library.layers.head.AnomalyScore.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/head/anomaly_score.py#L20-L21)

## nexuml_library.layers.head.DecisionRulePipelineLayer

*class*

```python
class DecisionRulePipelineLayer(LayerDefinition)
```

Bases: `LayerDefinition`

Re-exported from: `nexuml_library.layers.head.decision_rule`

Fits a decision threshold on train scores; emits binary decisions at inference.

Replaces DecisionRuleAlgorithm as a proper pipeline layer.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/head/decision_rule.py#L23-L46)

### nexuml_library.layers.head.DecisionRulePipelineLayer.fit_label_key

*attribute* · *class attribute* · *instance attribute*

```python
fit_label_key: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/head/decision_rule.py#L34-L34)

### nexuml_library.layers.head.DecisionRulePipelineLayer.fit_mask_key

*attribute* · *class attribute* · *instance attribute*

```python
fit_mask_key: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/head/decision_rule.py#L33-L33)

### nexuml_library.layers.head.DecisionRulePipelineLayer.normal_label_value

*attribute* · *class attribute* · *instance attribute*

```python
normal_label_value: int | float = 0
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/head/decision_rule.py#L35-L35)

### nexuml_library.layers.head.DecisionRulePipelineLayer.requires_post_train_fit

*attribute* · *class attribute* · *instance attribute*

```python
requires_post_train_fit = True
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/head/decision_rule.py#L30-L30)

### nexuml_library.layers.head.DecisionRulePipelineLayer.rule

*attribute* · *class attribute* · *instance attribute*

```python
rule: DecisionRuleSpec = Field(default_factory=lambda : decision_rule(PercentileThresholdRule))
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/head/decision_rule.py#L32-L32)

### nexuml_library.layers.head.DecisionRulePipelineLayer.build

*method*

```python
def build(context: LayerBuildContext) -> PostTrainFitLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/head/decision_rule.py#L37-L46)

## nexuml_library.layers.head.LatentClassificationHead

*class*

```python
class LatentClassificationHead(LayerDefinition)
```

Bases: `LayerDefinition`

Re-exported from: `nexuml_library.layers.head.classification_head`

Classification head operating on latent vectors.

Produces logits for multiclass or multilabel classification.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/head/classification_head.py#L15-L28)

### nexuml_library.layers.head.LatentClassificationHead.hidden_dims

*attribute* · *class attribute* · *instance attribute*

```python
hidden_dims: list[int] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/head/classification_head.py#L22-L22)

### nexuml_library.layers.head.LatentClassificationHead.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/head/classification_head.py#L24-L28)

## nexuml_library.layers.head.LatentRegressionHead

*class*

```python
class LatentRegressionHead(LayerDefinition)
```

Bases: `LayerDefinition`

Re-exported from: `nexuml_library.layers.head.regression_head`

Regression head operating on latent vectors.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/head/regression_head.py#L15-L23)

### nexuml_library.layers.head.LatentRegressionHead.hidden_dims

*attribute* · *class attribute* · *instance attribute*

```python
hidden_dims: list[int] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/head/regression_head.py#L20-L20)

### nexuml_library.layers.head.LatentRegressionHead.num_outputs

*attribute* · *class attribute* · *instance attribute*

```python
num_outputs: int = 1
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/head/regression_head.py#L19-L19)

### nexuml_library.layers.head.LatentRegressionHead.build

*method*

```python
def build(context: LayerBuildContext) -> PipelineLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/layers/head/regression_head.py#L22-L23)

**Modules**

- `nexuml_library.layers.head.anomaly_score`
- `nexuml_library.layers.head.classification_head`
- `nexuml_library.layers.head.decision_rule`
- `nexuml_library.layers.head.regression_head`
