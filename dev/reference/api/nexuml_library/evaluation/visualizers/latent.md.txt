# nexuml_library.evaluation.visualizers.latent

*module*

Latent space visualizer using t-SNE or UMAP.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/visualizers/latent.py#L1-L1)

## nexuml_library.evaluation.visualizers.latent.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/visualizers/latent.py#L28-L28)

## nexuml_library.evaluation.visualizers.latent.LatentVisualizer

*class*

```python
class LatentVisualizer(EvalAlgorithmDefinition)
```

Bases: `EvalAlgorithmDefinition`

Visualizes latent space using t-SNE or UMAP projection.

Uses reservoir sampling to handle large datasets efficiently.
Train and test samples are projected together and colored by label/split.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/visualizers/latent.py#L31-L56)

### nexuml_library.evaluation.visualizers.latent.LatentVisualizer.context_inputs

*attribute* · *class attribute* · *instance attribute*

```python
context_inputs = EvalContextInputs(feature_key=ContextInput(default='latent'), label_key=ContextInput(default='y_true'))
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/visualizers/latent.py#L39-L42)

### nexuml_library.evaluation.visualizers.latent.LatentVisualizer.max_samples

*attribute* · *class attribute* · *instance attribute*

```python
max_samples: int = 2000
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/visualizers/latent.py#L44-L44)

### nexuml_library.evaluation.visualizers.latent.LatentVisualizer.method

*attribute* · *class attribute* · *instance attribute*

```python
method: Literal['tsne', 'umap'] = 'tsne'
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/visualizers/latent.py#L43-L43)

### nexuml_library.evaluation.visualizers.latent.LatentVisualizer.perplexity

*attribute* · *class attribute* · *instance attribute*

```python
perplexity: int = 30
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/visualizers/latent.py#L45-L45)

### nexuml_library.evaluation.visualizers.latent.LatentVisualizer.storage_backend

*attribute* · *class attribute* · *instance attribute*

```python
storage_backend: str = 'memory'
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/visualizers/latent.py#L46-L46)

### nexuml_library.evaluation.visualizers.latent.LatentVisualizer.storage_path

*attribute* · *class attribute* · *instance attribute*

```python
storage_path: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/visualizers/latent.py#L47-L47)

### nexuml_library.evaluation.visualizers.latent.LatentVisualizer.build

*method*

```python
def build(context: EvalBuildContext) -> EvalAlgorithm
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/evaluation/visualizers/latent.py#L49-L56)
