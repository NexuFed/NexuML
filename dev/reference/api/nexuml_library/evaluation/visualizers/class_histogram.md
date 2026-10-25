# nexuml_library.evaluation.visualizers.class_histogram

*module*

Class histogram visualizer.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/visualizers/class_histogram.py#L1-L1)

## nexuml_library.evaluation.visualizers.class_histogram.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/visualizers/class_histogram.py#L26-L26)

## nexuml_library.evaluation.visualizers.class_histogram.ClassHistogramVisualizer

*class*

```python
class ClassHistogramVisualizer(EvalAlgorithmDefinition)
```

Bases: `EvalAlgorithmDefinition`

Bar chart of class label distribution in train and test sets.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/visualizers/class_histogram.py#L29-L43)

### nexuml_library.evaluation.visualizers.class_histogram.ClassHistogramVisualizer.context_inputs

*attribute* · *class attribute* · *instance attribute*

```python
context_inputs = EvalContextInputs(label_key=ContextInput(default='y_true'))
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/visualizers/class_histogram.py#L33-L33)

### nexuml_library.evaluation.visualizers.class_histogram.ClassHistogramVisualizer.title

*attribute* · *class attribute* · *instance attribute*

```python
title: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/visualizers/class_histogram.py#L35-L35)

### nexuml_library.evaluation.visualizers.class_histogram.ClassHistogramVisualizer.build

*method*

```python
def build(context: EvalBuildContext) -> EvalAlgorithm
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/visualizers/class_histogram.py#L37-L43)
