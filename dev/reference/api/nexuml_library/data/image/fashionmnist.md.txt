# nexuml_library.data.image.fashionmnist

*module*

FashionMNIST dataset source for in-memory image experiments and benchmarks.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/data/image/fashionmnist.py#L1-L1)

## nexuml_library.data.image.fashionmnist.FashionMNISTDataset

*class*

```python
class FashionMNISTDataset(DataSourceDefinition)
```

Bases: `DataSourceDefinition`

In-memory FashionMNIST dataset backed by ``self.data`` and metadata labels.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/data/image/fashionmnist.py#L17-L29)

### nexuml_library.data.image.fashionmnist.FashionMNISTDataset.download

*attribute* · *class attribute* · *instance attribute*

```python
download: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/data/image/fashionmnist.py#L25-L25)

### nexuml_library.data.image.fashionmnist.FashionMNISTDataset.fixed_label_keys

*attribute* · *class attribute* · *instance attribute*

```python
fixed_label_keys = ('category')
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/data/image/fashionmnist.py#L21-L21)

### nexuml_library.data.image.fashionmnist.FashionMNISTDataset.root

*attribute* · *class attribute* · *instance attribute*

```python
root: str | Path = 'data/fashionmnist'
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/data/image/fashionmnist.py#L23-L23)

### nexuml_library.data.image.fashionmnist.FashionMNISTDataset.split

*attribute* · *class attribute* · *instance attribute*

```python
split: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/data/image/fashionmnist.py#L26-L26)

### nexuml_library.data.image.fashionmnist.FashionMNISTDataset.train

*attribute* · *class attribute* · *instance attribute*

```python
train: bool = True
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/data/image/fashionmnist.py#L24-L24)

### nexuml_library.data.image.fashionmnist.FashionMNISTDataset.build

*method*

```python
def build() -> NexuDataset
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/library/src/nexuml_library/data/image/fashionmnist.py#L28-L29)
