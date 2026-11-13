# nexuml_library.scenarios.data.cifar

*module*

CIFAR data scenario fragments.

[View source](https://github.com/NexuFed/NexuML/blob/1787e5e6ec271ff0045b9659199a85fdf92d1e06/library/src/nexuml_library/scenarios/data/cifar.py#L1-L1)

## nexuml_library.scenarios.data.cifar.cifar100_data

*function*

```python
def cifar100_data(root: str = 'cifar100', download: bool = True) -> DataSpec
```

Create a DataSpec for CIFAR-100 image classification.

**Returns**

- `DataSpec` (`DataSpec`) — CIFAR-100 dataset specification.

[View source](https://github.com/NexuFed/NexuML/blob/1787e5e6ec271ff0045b9659199a85fdf92d1e06/library/src/nexuml_library/scenarios/data/cifar.py#L34-L55)

## nexuml_library.scenarios.data.cifar.cifar10_data

*function*

```python
def cifar10_data(root: str = 'cifar10', download: bool = True) -> DataSpec
```

Create a DataSpec for CIFAR-10 image classification.

**Returns**

- `DataSpec` (`DataSpec`) — CIFAR-10 dataset specification.

[View source](https://github.com/NexuFed/NexuML/blob/1787e5e6ec271ff0045b9659199a85fdf92d1e06/library/src/nexuml_library/scenarios/data/cifar.py#L10-L31)
