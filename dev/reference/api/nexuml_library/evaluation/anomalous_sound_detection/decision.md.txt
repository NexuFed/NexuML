# nexuml_library.evaluation.anomalous_sound_detection.decision

*module*

Decision rules: fit thresholds from train scores, emit binary decisions.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/anomalous_sound_detection/decision.py#L1-L1)

## nexuml_library.evaluation.anomalous_sound_detection.decision.P

*attribute* · *module attribute*

```python
P = ParamSpec('P')
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/anomalous_sound_detection/decision.py#L150-L150)

## nexuml_library.evaluation.anomalous_sound_detection.decision.DecisionRule

*class*

```python
class DecisionRule(nn.Module)
```

Bases: `nn.Module`

Abstract decision rule.

> **Lifecycle**
>
> 1. fit(train_scores) — fit threshold from normal train scores
> 2. forward(scores) -> decisions — apply threshold to produce binary decisions

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/anomalous_sound_detection/decision.py#L15-L37)

### nexuml_library.evaluation.anomalous_sound_detection.decision.DecisionRule.threshold

*property*

```python
threshold: float | None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/anomalous_sound_detection/decision.py#L32-L37)

### nexuml_library.evaluation.anomalous_sound_detection.decision.DecisionRule.fit

*method*

```python
def fit(train_scores: torch.Tensor) -> None
```

Fit the decision threshold from train-only scores.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/anomalous_sound_detection/decision.py#L23-L25)

### nexuml_library.evaluation.anomalous_sound_detection.decision.DecisionRule.forward

*method*

```python
def forward(scores: torch.Tensor) -> torch.Tensor
```

Return 1 (anomalous) / 0 (normal) binary decisions.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/anomalous_sound_detection/decision.py#L27-L29)

## nexuml_library.evaluation.anomalous_sound_detection.decision.DecisionRuleSpec

*class*

```python
class DecisionRuleSpec(FactorySpec)
```

Bases: `FactorySpec`

Portable constructor configuration for a decision rule.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/anomalous_sound_detection/decision.py#L40-L41)

## nexuml_library.evaluation.anomalous_sound_detection.decision.GammaPercentileRule

*class*

```python
class GammaPercentileRule(DecisionRule)
```

Bases: `nexuml_library.evaluation.anomalous_sound_detection.decision.DecisionRule`

Gamma-fit threshold at a given percentile of train scores.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/anomalous_sound_detection/decision.py#L127-L147)

### nexuml_library.evaluation.anomalous_sound_detection.decision.GammaPercentileRule.percentile

*attribute* · *instance attribute*

```python
percentile = percentile
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/anomalous_sound_detection/decision.py#L134-L134)

### nexuml_library.evaluation.anomalous_sound_detection.decision.GammaPercentileRule.threshold

*property*

```python
threshold: float | None
```

Inherited from: `nexuml_library.evaluation.anomalous_sound_detection.decision.DecisionRule`

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/anomalous_sound_detection/decision.py#L32-L37)

### nexuml_library.evaluation.anomalous_sound_detection.decision.GammaPercentileRule.fit

*method*

```python
def fit(train_scores: torch.Tensor) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/anomalous_sound_detection/decision.py#L137-L139)

### nexuml_library.evaluation.anomalous_sound_detection.decision.GammaPercentileRule.forward

*method*

```python
def forward(scores: torch.Tensor) -> torch.Tensor
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/anomalous_sound_detection/decision.py#L141-L147)

## nexuml_library.evaluation.anomalous_sound_detection.decision.PercentileThresholdRule

*class*

```python
class PercentileThresholdRule(DecisionRule)
```

Bases: `nexuml_library.evaluation.anomalous_sound_detection.decision.DecisionRule`

Threshold at a given percentile of the train score distribution.

``percentile`` in [0, 100]; equivalent to QuantileThresholdRule with
quantile = percentile / 100.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/anomalous_sound_detection/decision.py#L70-L94)

### nexuml_library.evaluation.anomalous_sound_detection.decision.PercentileThresholdRule.percentile

*attribute* · *instance attribute*

```python
percentile = percentile
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/anomalous_sound_detection/decision.py#L81-L81)

### nexuml_library.evaluation.anomalous_sound_detection.decision.PercentileThresholdRule.threshold

*property*

```python
threshold: float | None
```

Inherited from: `nexuml_library.evaluation.anomalous_sound_detection.decision.DecisionRule`

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/anomalous_sound_detection/decision.py#L32-L37)

### nexuml_library.evaluation.anomalous_sound_detection.decision.PercentileThresholdRule.fit

*method*

```python
def fit(train_scores: torch.Tensor) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/anomalous_sound_detection/decision.py#L84-L86)

### nexuml_library.evaluation.anomalous_sound_detection.decision.PercentileThresholdRule.forward

*method*

```python
def forward(scores: torch.Tensor) -> torch.Tensor
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/anomalous_sound_detection/decision.py#L88-L94)

## nexuml_library.evaluation.anomalous_sound_detection.decision.QuantileThresholdRule

*class*

```python
class QuantileThresholdRule(DecisionRule)
```

Bases: `nexuml_library.evaluation.anomalous_sound_detection.decision.DecisionRule`

Threshold at a given quantile of the train score distribution.

``quantile`` in [0, 1]; e.g. 0.95 means the top-5% of train scores.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/anomalous_sound_detection/decision.py#L44-L67)

### nexuml_library.evaluation.anomalous_sound_detection.decision.QuantileThresholdRule.quantile

*attribute* · *instance attribute*

```python
quantile = quantile
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/anomalous_sound_detection/decision.py#L54-L54)

### nexuml_library.evaluation.anomalous_sound_detection.decision.QuantileThresholdRule.threshold

*property*

```python
threshold: float | None
```

Inherited from: `nexuml_library.evaluation.anomalous_sound_detection.decision.DecisionRule`

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/anomalous_sound_detection/decision.py#L32-L37)

### nexuml_library.evaluation.anomalous_sound_detection.decision.QuantileThresholdRule.fit

*method*

```python
def fit(train_scores: torch.Tensor) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/anomalous_sound_detection/decision.py#L57-L59)

### nexuml_library.evaluation.anomalous_sound_detection.decision.QuantileThresholdRule.forward

*method*

```python
def forward(scores: torch.Tensor) -> torch.Tensor
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/anomalous_sound_detection/decision.py#L61-L67)

## nexuml_library.evaluation.anomalous_sound_detection.decision.decision_rule

*function*

```python
def decision_rule(factory: Callable[P, DecisionRule], *args: P.args, **kwargs: P.kwargs) -> DecisionRuleSpec
```

Configure an importable decision-rule factory.

**Returns**

- (`DecisionRuleSpec`) — Portable decision-rule specification.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/anomalous_sound_detection/decision.py#L153-L161)
