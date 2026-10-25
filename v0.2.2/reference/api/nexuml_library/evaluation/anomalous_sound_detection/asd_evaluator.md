# nexuml_library.evaluation.anomalous_sound_detection.asd_evaluator

*module*

Composable ASD reporting eval algorithms: AnomalyEvaluator and AnomalyVisualizer.

Score-producing components (GroupDistancePipelineLayer, ScoreCalibrationPipelineLayer,
ScoreReductionPipelineLayer, DecisionRulePipelineLayer) are now in
nexuml_library.layers.head as PostTrainFitLayer subclasses.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/anomalous_sound_detection/asd_evaluator.py#L1-L1)

## nexuml_library.evaluation.anomalous_sound_detection.asd_evaluator.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/anomalous_sound_detection/asd_evaluator.py#L32-L32)

## nexuml_library.evaluation.anomalous_sound_detection.asd_evaluator.AnomalyEvaluator

*class*

```python
class AnomalyEvaluator(EvalAlgorithmDefinition)
```

Bases: `EvalAlgorithmDefinition`

Reporting algorithm: computes AUC, pAUC, and DCASE metrics.

Reads anomaly_score from the pipeline output TensorDict. Does not produce
any score keys — score production is handled by pipeline layers.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/anomalous_sound_detection/asd_evaluator.py#L74-L95)

### nexuml_library.evaluation.anomalous_sound_detection.asd_evaluator.AnomalyEvaluator.context_inputs

*attribute* · *class attribute* · *instance attribute*

```python
context_inputs = EvalContextInputs(label_key=ContextInput(default='y_true'))
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/anomalous_sound_detection/asd_evaluator.py#L82-L82)

### nexuml_library.evaluation.anomalous_sound_detection.asd_evaluator.AnomalyEvaluator.dcase_metric_axes

*attribute* · *class attribute* · *instance attribute*

```python
dcase_metric_axes: dict[str, str] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/anomalous_sound_detection/asd_evaluator.py#L86-L86)

### nexuml_library.evaluation.anomalous_sound_detection.asd_evaluator.AnomalyEvaluator.decision_key

*attribute* · *class attribute* · *instance attribute*

```python
decision_key: FeatureKey | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/anomalous_sound_detection/asd_evaluator.py#L84-L84)

### nexuml_library.evaluation.anomalous_sound_detection.asd_evaluator.AnomalyEvaluator.group_keys

*attribute* · *class attribute* · *instance attribute*

```python
group_keys: list[str] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/anomalous_sound_detection/asd_evaluator.py#L85-L85)

### nexuml_library.evaluation.anomalous_sound_detection.asd_evaluator.AnomalyEvaluator.max_fpr

*attribute* · *class attribute* · *instance attribute*

```python
max_fpr: float = 0.1
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/anomalous_sound_detection/asd_evaluator.py#L87-L87)

### nexuml_library.evaluation.anomalous_sound_detection.asd_evaluator.AnomalyEvaluator.score_key

*attribute* · *class attribute* · *instance attribute*

```python
score_key: FeatureKey = 'anomaly_score'
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/anomalous_sound_detection/asd_evaluator.py#L83-L83)

### nexuml_library.evaluation.anomalous_sound_detection.asd_evaluator.AnomalyEvaluator.build

*method*

```python
def build(context: EvalBuildContext) -> EvalAlgorithm
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/anomalous_sound_detection/asd_evaluator.py#L89-L95)

## nexuml_library.evaluation.anomalous_sound_detection.asd_evaluator.AnomalyVisualizer

*class*

```python
class AnomalyVisualizer(EvalAlgorithmDefinition)
```

Bases: `EvalAlgorithmDefinition`

Diagnostic visualization consuming declared score, feature, and grouping axes.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/anomalous_sound_detection/asd_evaluator.py#L190-L209)

### nexuml_library.evaluation.anomalous_sound_detection.asd_evaluator.AnomalyVisualizer.context_inputs

*attribute* · *class attribute* · *instance attribute*

```python
context_inputs = EvalContextInputs(feature_key=ContextInput(default='latent'), label_key=ContextInput(default='y_true'))
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/anomalous_sound_detection/asd_evaluator.py#L194-L197)

### nexuml_library.evaluation.anomalous_sound_detection.asd_evaluator.AnomalyVisualizer.group_keys

*attribute* · *class attribute* · *instance attribute*

```python
group_keys: list[str] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/anomalous_sound_detection/asd_evaluator.py#L199-L199)

### nexuml_library.evaluation.anomalous_sound_detection.asd_evaluator.AnomalyVisualizer.max_plot_samples

*attribute* · *class attribute* · *instance attribute*

```python
max_plot_samples: int = 2000
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/anomalous_sound_detection/asd_evaluator.py#L200-L200)

### nexuml_library.evaluation.anomalous_sound_detection.asd_evaluator.AnomalyVisualizer.score_key

*attribute* · *class attribute* · *instance attribute*

```python
score_key: FeatureKey = 'anomaly_score'
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/anomalous_sound_detection/asd_evaluator.py#L198-L198)

### nexuml_library.evaluation.anomalous_sound_detection.asd_evaluator.AnomalyVisualizer.build

*method*

```python
def build(context: EvalBuildContext) -> EvalAlgorithm
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/anomalous_sound_detection/asd_evaluator.py#L202-L209)
