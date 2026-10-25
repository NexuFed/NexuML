# nexuml_library.data.image.cifar

*module*

CIFAR dataset sources for image classification experiments.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/data/image/cifar.py#L1-L1)

## nexuml_library.data.image.cifar.CIFAR100Dataset

*class*

```python
class CIFAR100Dataset(DataSourceDefinition)
```

Bases: `DataSourceDefinition`

CIFAR-100 dataset backed by ``self.data`` and metadata labels.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/data/image/cifar.py#L106-L118)

### nexuml_library.data.image.cifar.CIFAR100Dataset.download

*attribute* · *class attribute* · *instance attribute*

```python
download: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/data/image/cifar.py#L114-L114)

### nexuml_library.data.image.cifar.CIFAR100Dataset.fixed_label_keys

*attribute* · *class attribute* · *instance attribute*

```python
fixed_label_keys = ('class_labels')
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/data/image/cifar.py#L110-L110)

### nexuml_library.data.image.cifar.CIFAR100Dataset.root

*attribute* · *class attribute* · *instance attribute*

```python
root: str | Path = 'data/cifar100'
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/data/image/cifar.py#L112-L112)

### nexuml_library.data.image.cifar.CIFAR100Dataset.split

*attribute* · *class attribute* · *instance attribute*

```python
split: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/data/image/cifar.py#L115-L115)

### nexuml_library.data.image.cifar.CIFAR100Dataset.train

*attribute* · *class attribute* · *instance attribute*

```python
train: bool = True
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/data/image/cifar.py#L113-L113)

### nexuml_library.data.image.cifar.CIFAR100Dataset.build

*method*

```python
def build() -> NexuDataset
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/data/image/cifar.py#L117-L118)

## nexuml_library.data.image.cifar.CIFAR10Dataset

*class*

```python
class CIFAR10Dataset(DataSourceDefinition)
```

Bases: `DataSourceDefinition`

CIFAR-10 dataset backed by ``self.data`` and metadata labels.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/data/image/cifar.py#L17-L29)

### nexuml_library.data.image.cifar.CIFAR10Dataset.download

*attribute* · *class attribute* · *instance attribute*

```python
download: bool = False
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/data/image/cifar.py#L25-L25)

### nexuml_library.data.image.cifar.CIFAR10Dataset.fixed_label_keys

*attribute* · *class attribute* · *instance attribute*

```python
fixed_label_keys = ('class_labels')
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/data/image/cifar.py#L21-L21)

### nexuml_library.data.image.cifar.CIFAR10Dataset.root

*attribute* · *class attribute* · *instance attribute*

```python
root: str | Path = 'data/cifar10'
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/data/image/cifar.py#L23-L23)

### nexuml_library.data.image.cifar.CIFAR10Dataset.split

*attribute* · *class attribute* · *instance attribute*

```python
split: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/data/image/cifar.py#L26-L26)

### nexuml_library.data.image.cifar.CIFAR10Dataset.train

*attribute* · *class attribute* · *instance attribute*

```python
train: bool = True
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/data/image/cifar.py#L24-L24)

### nexuml_library.data.image.cifar.CIFAR10Dataset.build

*method*

```python
def build() -> NexuDataset
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/data/image/cifar.py#L28-L29)
