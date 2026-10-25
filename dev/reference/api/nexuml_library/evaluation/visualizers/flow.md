# nexuml_library.evaluation.visualizers.flow

*module*

Flow sample and trajectory visualizer.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/evaluation/visualizers/flow.py#L1-L1)

## nexuml_library.evaluation.visualizers.flow.FlowVisualizer

*class*

```python
class FlowVisualizer(EvalAlgorithmDefinition)
```

Bases: `EvalAlgorithmDefinition`

Visualize target, initial, generated, and optional trajectory samples.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/evaluation/visualizers/flow.py#L19-L34)

### nexuml_library.evaluation.visualizers.flow.FlowVisualizer.generated_key

*attribute* · *class attribute* · *instance attribute*

```python
generated_key: str = 'flow_sample'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/evaluation/visualizers/flow.py#L24-L24)

### nexuml_library.evaluation.visualizers.flow.FlowVisualizer.image_range

*attribute* · *class attribute* · *instance attribute*

```python
image_range: Literal['zero_one', 'minus_one_one'] = 'zero_one'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/evaluation/visualizers/flow.py#L28-L28)

### nexuml_library.evaluation.visualizers.flow.FlowVisualizer.initial_key

*attribute* · *class attribute* · *instance attribute*

```python
initial_key: str = 'flow_initial'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/evaluation/visualizers/flow.py#L23-L23)

### nexuml_library.evaluation.visualizers.flow.FlowVisualizer.max_samples

*attribute* · *class attribute* · *instance attribute*

```python
max_samples: int = Field(default=512, gt=0)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/evaluation/visualizers/flow.py#L26-L26)

### nexuml_library.evaluation.visualizers.flow.FlowVisualizer.max_trajectories

*attribute* · *class attribute* · *instance attribute*

```python
max_trajectories: int = Field(default=32, gt=0)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/evaluation/visualizers/flow.py#L27-L27)

### nexuml_library.evaluation.visualizers.flow.FlowVisualizer.trajectory_key

*attribute* · *class attribute* · *instance attribute*

```python
trajectory_key: str = 'flow_trajectory'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/evaluation/visualizers/flow.py#L25-L25)

### nexuml_library.evaluation.visualizers.flow.FlowVisualizer.build

*method*

```python
def build(context: EvalBuildContext) -> EvalAlgorithm
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/evaluation/visualizers/flow.py#L30-L34)
