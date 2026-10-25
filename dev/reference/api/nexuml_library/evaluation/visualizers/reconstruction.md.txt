# nexuml_library.evaluation.visualizers.reconstruction

*module*

Reconstruction visualizer.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/evaluation/visualizers/reconstruction.py#L1-L1)

## nexuml_library.evaluation.visualizers.reconstruction.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/evaluation/visualizers/reconstruction.py#L28-L28)

## nexuml_library.evaluation.visualizers.reconstruction.ReconstructionVisualizer

*class*

```python
class ReconstructionVisualizer(EvalAlgorithmDefinition)
```

Bases: `EvalAlgorithmDefinition`

Shows side-by-side original vs reconstructed samples.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/evaluation/visualizers/reconstruction.py#L31-L51)

### nexuml_library.evaluation.visualizers.reconstruction.ReconstructionVisualizer.context_inputs

*attribute* · *class attribute* · *instance attribute*

```python
context_inputs = EvalContextInputs(feature_key=ContextInput(required=True))
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/evaluation/visualizers/reconstruction.py#L35-L35)

### nexuml_library.evaluation.visualizers.reconstruction.ReconstructionVisualizer.label_keys

*attribute* · *class attribute* · *instance attribute*

```python
label_keys: list[LabelKey] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/evaluation/visualizers/reconstruction.py#L39-L39)

### nexuml_library.evaluation.visualizers.reconstruction.ReconstructionVisualizer.mask_key

*attribute* · *class attribute* · *instance attribute*

```python
mask_key: FeatureKey | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/evaluation/visualizers/reconstruction.py#L38-L38)

### nexuml_library.evaluation.visualizers.reconstruction.ReconstructionVisualizer.n_samples

*attribute* · *class attribute* · *instance attribute*

```python
n_samples: int = 8
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/evaluation/visualizers/reconstruction.py#L40-L40)

### nexuml_library.evaluation.visualizers.reconstruction.ReconstructionVisualizer.patch_size

*attribute* · *class attribute* · *instance attribute*

```python
patch_size: int | tuple[int, int] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/evaluation/visualizers/reconstruction.py#L41-L41)

### nexuml_library.evaluation.visualizers.reconstruction.ReconstructionVisualizer.reconstructed_key

*attribute* · *instance attribute*

```python
reconstructed_key: FeatureKey
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/evaluation/visualizers/reconstruction.py#L37-L37)

### nexuml_library.evaluation.visualizers.reconstruction.ReconstructionVisualizer.storage_backend

*attribute* · *class attribute* · *instance attribute*

```python
storage_backend: str = 'memory'
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/evaluation/visualizers/reconstruction.py#L42-L42)

### nexuml_library.evaluation.visualizers.reconstruction.ReconstructionVisualizer.storage_path

*attribute* · *class attribute* · *instance attribute*

```python
storage_path: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/evaluation/visualizers/reconstruction.py#L43-L43)

### nexuml_library.evaluation.visualizers.reconstruction.ReconstructionVisualizer.build

*method*

```python
def build(context: EvalBuildContext) -> EvalAlgorithm
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/library/src/nexuml_library/evaluation/visualizers/reconstruction.py#L45-L51)
