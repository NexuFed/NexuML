# nexuml_library.evaluation.visualizers

*module*

Generic anomaly detection visualizers.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/visualizers/__init__.py#L1-L1)

## nexuml_library.evaluation.visualizers.ClassHistogramVisualizer

*class*

```python
class ClassHistogramVisualizer(EvalAlgorithmDefinition)
```

Bases: `EvalAlgorithmDefinition`

Re-exported from: `nexuml_library.evaluation.visualizers.class_histogram`

Bar chart of class label distribution in train and test sets.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/visualizers/class_histogram.py#L29-L43)

### nexuml_library.evaluation.visualizers.ClassHistogramVisualizer.context_inputs

*attribute* · *class attribute* · *instance attribute*

```python
context_inputs = EvalContextInputs(label_key=ContextInput(default='y_true'))
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/visualizers/class_histogram.py#L33-L33)

### nexuml_library.evaluation.visualizers.ClassHistogramVisualizer.title

*attribute* · *class attribute* · *instance attribute*

```python
title: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/visualizers/class_histogram.py#L35-L35)

### nexuml_library.evaluation.visualizers.ClassHistogramVisualizer.build

*method*

```python
def build(context: EvalBuildContext) -> EvalAlgorithm
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/visualizers/class_histogram.py#L37-L43)

## nexuml_library.evaluation.visualizers.LatentVisualizer

*class*

```python
class LatentVisualizer(EvalAlgorithmDefinition)
```

Bases: `EvalAlgorithmDefinition`

Re-exported from: `nexuml_library.evaluation.visualizers.latent`

Visualizes latent space using t-SNE or UMAP projection.

Uses reservoir sampling to handle large datasets efficiently.
Train and test samples are projected together and colored by label/split.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/visualizers/latent.py#L31-L56)

### nexuml_library.evaluation.visualizers.LatentVisualizer.context_inputs

*attribute* · *class attribute* · *instance attribute*

```python
context_inputs = EvalContextInputs(feature_key=ContextInput(default='latent'), label_key=ContextInput(default='y_true'))
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/visualizers/latent.py#L39-L42)

### nexuml_library.evaluation.visualizers.LatentVisualizer.max_samples

*attribute* · *class attribute* · *instance attribute*

```python
max_samples: int = 2000
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/visualizers/latent.py#L44-L44)

### nexuml_library.evaluation.visualizers.LatentVisualizer.method

*attribute* · *class attribute* · *instance attribute*

```python
method: Literal['tsne', 'umap'] = 'tsne'
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/visualizers/latent.py#L43-L43)

### nexuml_library.evaluation.visualizers.LatentVisualizer.perplexity

*attribute* · *class attribute* · *instance attribute*

```python
perplexity: int = 30
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/visualizers/latent.py#L45-L45)

### nexuml_library.evaluation.visualizers.LatentVisualizer.storage_backend

*attribute* · *class attribute* · *instance attribute*

```python
storage_backend: str = 'memory'
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/visualizers/latent.py#L46-L46)

### nexuml_library.evaluation.visualizers.LatentVisualizer.storage_path

*attribute* · *class attribute* · *instance attribute*

```python
storage_path: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/visualizers/latent.py#L47-L47)

### nexuml_library.evaluation.visualizers.LatentVisualizer.build

*method*

```python
def build(context: EvalBuildContext) -> EvalAlgorithm
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/visualizers/latent.py#L49-L56)

## nexuml_library.evaluation.visualizers.ReconstructionVisualizer

*class*

```python
class ReconstructionVisualizer(EvalAlgorithmDefinition)
```

Bases: `EvalAlgorithmDefinition`

Re-exported from: `nexuml_library.evaluation.visualizers.reconstruction`

Shows side-by-side original vs reconstructed samples.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/visualizers/reconstruction.py#L31-L51)

### nexuml_library.evaluation.visualizers.ReconstructionVisualizer.context_inputs

*attribute* · *class attribute* · *instance attribute*

```python
context_inputs = EvalContextInputs(feature_key=ContextInput(required=True))
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/visualizers/reconstruction.py#L35-L35)

### nexuml_library.evaluation.visualizers.ReconstructionVisualizer.label_keys

*attribute* · *class attribute* · *instance attribute*

```python
label_keys: list[LabelKey] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/visualizers/reconstruction.py#L39-L39)

### nexuml_library.evaluation.visualizers.ReconstructionVisualizer.mask_key

*attribute* · *class attribute* · *instance attribute*

```python
mask_key: FeatureKey | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/visualizers/reconstruction.py#L38-L38)

### nexuml_library.evaluation.visualizers.ReconstructionVisualizer.n_samples

*attribute* · *class attribute* · *instance attribute*

```python
n_samples: int = 8
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/visualizers/reconstruction.py#L40-L40)

### nexuml_library.evaluation.visualizers.ReconstructionVisualizer.patch_size

*attribute* · *class attribute* · *instance attribute*

```python
patch_size: int | tuple[int, int] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/visualizers/reconstruction.py#L41-L41)

### nexuml_library.evaluation.visualizers.ReconstructionVisualizer.reconstructed_key

*attribute* · *instance attribute*

```python
reconstructed_key: FeatureKey
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/visualizers/reconstruction.py#L37-L37)

### nexuml_library.evaluation.visualizers.ReconstructionVisualizer.storage_backend

*attribute* · *class attribute* · *instance attribute*

```python
storage_backend: str = 'memory'
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/visualizers/reconstruction.py#L42-L42)

### nexuml_library.evaluation.visualizers.ReconstructionVisualizer.storage_path

*attribute* · *class attribute* · *instance attribute*

```python
storage_path: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/visualizers/reconstruction.py#L43-L43)

### nexuml_library.evaluation.visualizers.ReconstructionVisualizer.build

*method*

```python
def build(context: EvalBuildContext) -> EvalAlgorithm
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/evaluation/visualizers/reconstruction.py#L45-L51)

**Modules**

- `nexuml_library.evaluation.visualizers.class_histogram`
- `nexuml_library.evaluation.visualizers.flow`
- `nexuml_library.evaluation.visualizers.latent`
- `nexuml_library.evaluation.visualizers.reconstruction`
