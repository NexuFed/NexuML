# nexuml_library.evaluation.visualizers.class_histogram

*module*

Class histogram visualizer.

[View source](https://github.com/NexuFed/NexuML/blob/1787e5e6ec271ff0045b9659199a85fdf92d1e06/library/src/nexuml_library/evaluation/visualizers/class_histogram.py#L1-L1)

## nexuml_library.evaluation.visualizers.class_histogram.ClassHistogramVisualizer

*class*

```python
class ClassHistogramVisualizer(EvalAlgorithmDefinition)
```

Bases: `EvalAlgorithmDefinition`

Bar chart of class label distribution in train and test sets.

[View source](https://github.com/NexuFed/NexuML/blob/1787e5e6ec271ff0045b9659199a85fdf92d1e06/library/src/nexuml_library/evaluation/visualizers/class_histogram.py#L29-L43)

### nexuml_library.evaluation.visualizers.class_histogram.ClassHistogramVisualizer.context_inputs

*attribute* · *class attribute* · *instance attribute*

```python
context_inputs = EvalContextInputs(label_key=ContextInput(default='y_true'))
```

[View source](https://github.com/NexuFed/NexuML/blob/1787e5e6ec271ff0045b9659199a85fdf92d1e06/library/src/nexuml_library/evaluation/visualizers/class_histogram.py#L33-L33)

### nexuml_library.evaluation.visualizers.class_histogram.ClassHistogramVisualizer.title

*attribute* · *class attribute* · *instance attribute*

```python
title: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1787e5e6ec271ff0045b9659199a85fdf92d1e06/library/src/nexuml_library/evaluation/visualizers/class_histogram.py#L35-L35)

### nexuml_library.evaluation.visualizers.class_histogram.ClassHistogramVisualizer.build

*method*

```python
def build(context: EvalBuildContext) -> EvalAlgorithm
```

[View source](https://github.com/NexuFed/NexuML/blob/1787e5e6ec271ff0045b9659199a85fdf92d1e06/library/src/nexuml_library/evaluation/visualizers/class_histogram.py#L37-L43)
