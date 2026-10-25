# nexuml_library.scenarios.vision.cifar_resnet

*module*

Composed CIFAR + ResNet classification scenarios.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/scenarios/vision/cifar_resnet.py#L1-L1)

## nexuml_library.scenarios.vision.cifar_resnet.cifar_resnet

*function*

```python
def cifar_resnet(dataset: str = 'cifar10', download: bool = True, resnet_type: ResNetType = 'resnet18', pretrained: bool = False, cifar_stem: bool | None = None, lr: float = 0.001, batch_size: int = 64, max_epochs: int = 10) -> ScenarioSpec
```

CIFAR image classification with ResNet backbone.

**Parameters**

- `dataset` (`str`) (default: `'cifar10'`) — ``"cifar10"`` or ``"cifar100"``.
- `download` (`bool`) (default: `True`) — Auto-download dataset if not present.
- `resnet_type` (`ResNetType`) (default: `'resnet18'`) — ResNet variant (e.g. ``"resnet18"``).
- `pretrained` (`bool`) (default: `False`) — Load pretrained ImageNet weights.
- `cifar_stem` (`bool | None`) (default: `None`) — Use CIFAR-friendly stem. Defaults to ``True`` when ``pretrained=False``.
- `lr` (`float`) (default: `0.001`) — Learning rate.
- `batch_size` (`int`) (default: `64`) — Training batch size.
- `max_epochs` (`int`) (default: `10`) — Maximum training epochs.

**Returns**

- `ScenarioSpec` (`ScenarioSpec`) — Assembled scenario with pipeline, training, data and evaluation.

**Raises**

- `ValueError` — If ``dataset`` is not ``"cifar10"`` or ``"cifar100"``.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/library/src/nexuml_library/scenarios/vision/cifar_resnet.py#L23-L86)
