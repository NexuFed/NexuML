# nexuml_library.data.synthetic

*module*

Synthetic dataset for testing architectures without real data.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/synthetic.py#L1-L1)

## nexuml_library.data.synthetic.SyntheticDataset

*class*

```python
class SyntheticDataset(DataSourceDefinition)
```

Bases: `DataSourceDefinition`

Universal synthetic tensor dataset for architecture testing.

Generates arbitrary feature tensors with optional targets:
- reconstruction (features are their own target)
- multiclass labels (from cluster assignments)
- multilabel labels (random binary vectors)
- multi-output regression targets (linear combination + noise)

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/synthetic.py#L30-L50)

### nexuml_library.data.synthetic.SyntheticDataset.feature_key

*attribute* · *class attribute* · *instance attribute*

```python
feature_key: str = 'features'
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/synthetic.py#L47-L47)

### nexuml_library.data.synthetic.SyntheticDataset.feature_shape

*attribute* · *class attribute* · *instance attribute*

```python
feature_shape: tuple[int, ...] = (128)
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/synthetic.py#L41-L41)

### nexuml_library.data.synthetic.SyntheticDataset.noise_type

*attribute* · *class attribute* · *instance attribute*

```python
noise_type: str = 'gaussian'
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/synthetic.py#L43-L43)

### nexuml_library.data.synthetic.SyntheticDataset.num_clusters

*attribute* · *class attribute* · *instance attribute*

```python
num_clusters: int | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/synthetic.py#L44-L44)

### nexuml_library.data.synthetic.SyntheticDataset.num_samples

*attribute* · *class attribute* · *instance attribute*

```python
num_samples: int = 1000
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/synthetic.py#L42-L42)

### nexuml_library.data.synthetic.SyntheticDataset.seed

*attribute* · *class attribute* · *instance attribute*

```python
seed: int = 42
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/synthetic.py#L46-L46)

### nexuml_library.data.synthetic.SyntheticDataset.targets

*attribute* · *class attribute* · *instance attribute*

```python
targets: list[TargetConfig] = Field(default_factory=list)
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/synthetic.py#L45-L45)

### nexuml_library.data.synthetic.SyntheticDataset.build

*method*

```python
def build() -> NexuDataset
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/synthetic.py#L49-L50)

## nexuml_library.data.synthetic.TargetConfig

*class* · *dataclass*

```python
class TargetConfig
```

Configuration for a synthetic target.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/synthetic.py#L18-L27)

### nexuml_library.data.synthetic.TargetConfig.key

*attribute* · *instance attribute*

```python
key: str
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/synthetic.py#L23-L23)

### nexuml_library.data.synthetic.TargetConfig.label_density

*attribute* · *class attribute* · *instance attribute*

```python
label_density: float = 0.3
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/synthetic.py#L26-L26)

### nexuml_library.data.synthetic.TargetConfig.num_classes

*attribute* · *class attribute* · *instance attribute*

```python
num_classes: int | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/synthetic.py#L24-L24)

### nexuml_library.data.synthetic.TargetConfig.num_outputs

*attribute* · *class attribute* · *instance attribute*

```python
num_outputs: int | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/synthetic.py#L25-L25)

### nexuml_library.data.synthetic.TargetConfig.positive_fraction

*attribute* · *class attribute* · *instance attribute*

```python
positive_fraction: float | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/synthetic.py#L27-L27)

### nexuml_library.data.synthetic.TargetConfig.type

*attribute* · *instance attribute*

```python
type: str
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/synthetic.py#L22-L22)
