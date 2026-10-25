# nexuml_library.data.image.mnist

*module*

MNIST dataset source for in-memory image experiments and benchmarks.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/image/mnist.py#L1-L1)

## nexuml_library.data.image.mnist.MNISTDataset

*class*

```python
class MNISTDataset(DataSourceDefinition)
```

Bases: `DataSourceDefinition`

In-memory MNIST dataset backed by ``self.data`` and metadata labels.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/image/mnist.py#L17-L30)

### nexuml_library.data.image.mnist.MNISTDataset.download

*attribute* · *class attribute* · *instance attribute*

```python
download: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/image/mnist.py#L25-L25)

### nexuml_library.data.image.mnist.MNISTDataset.fixed_label_keys

*attribute* · *class attribute* · *instance attribute*

```python
fixed_label_keys = ('digit')
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/image/mnist.py#L21-L21)

### nexuml_library.data.image.mnist.MNISTDataset.normalize

*attribute* · *class attribute* · *instance attribute*

```python
normalize: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/image/mnist.py#L27-L27)

### nexuml_library.data.image.mnist.MNISTDataset.root

*attribute* · *class attribute* · *instance attribute*

```python
root: str | Path = 'data/mnist'
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/image/mnist.py#L23-L23)

### nexuml_library.data.image.mnist.MNISTDataset.split

*attribute* · *class attribute* · *instance attribute*

```python
split: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/image/mnist.py#L26-L26)

### nexuml_library.data.image.mnist.MNISTDataset.train

*attribute* · *class attribute* · *instance attribute*

```python
train: bool = True
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/image/mnist.py#L24-L24)

### nexuml_library.data.image.mnist.MNISTDataset.build

*method*

```python
def build() -> NexuDataset
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/library/src/nexuml_library/data/image/mnist.py#L29-L30)
