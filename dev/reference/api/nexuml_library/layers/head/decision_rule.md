# nexuml_library.layers.head.decision_rule

*module*

DecisionRulePipelineLayer — post-training fit layer for binary decision thresholding.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/head/decision_rule.py#L1-L1)

## nexuml_library.layers.head.decision_rule.DecisionRulePipelineLayer

*class*

```python
class DecisionRulePipelineLayer(LayerDefinition)
```

Bases: `LayerDefinition`

Fits a decision threshold on train scores; emits binary decisions at inference.

Replaces DecisionRuleAlgorithm as a proper pipeline layer.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/head/decision_rule.py#L23-L46)

### nexuml_library.layers.head.decision_rule.DecisionRulePipelineLayer.fit_label_key

*attribute* · *class attribute* · *instance attribute*

```python
fit_label_key: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/head/decision_rule.py#L34-L34)

### nexuml_library.layers.head.decision_rule.DecisionRulePipelineLayer.fit_mask_key

*attribute* · *class attribute* · *instance attribute*

```python
fit_mask_key: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/head/decision_rule.py#L33-L33)

### nexuml_library.layers.head.decision_rule.DecisionRulePipelineLayer.normal_label_value

*attribute* · *class attribute* · *instance attribute*

```python
normal_label_value: int | float = 0
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/head/decision_rule.py#L35-L35)

### nexuml_library.layers.head.decision_rule.DecisionRulePipelineLayer.requires_post_train_fit

*attribute* · *class attribute* · *instance attribute*

```python
requires_post_train_fit = True
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/head/decision_rule.py#L30-L30)

### nexuml_library.layers.head.decision_rule.DecisionRulePipelineLayer.rule

*attribute* · *class attribute* · *instance attribute*

```python
rule: DecisionRuleSpec = Field(default_factory=lambda : decision_rule(PercentileThresholdRule))
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/head/decision_rule.py#L32-L32)

### nexuml_library.layers.head.decision_rule.DecisionRulePipelineLayer.build

*method*

```python
def build(context: LayerBuildContext) -> PostTrainFitLayer
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/layers/head/decision_rule.py#L37-L46)
