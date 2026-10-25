# nexuml_library.training.schedulers.warmup_cosine

*module*

Learning-rate schedulers for NexuML library scenarios.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/training/schedulers/warmup_cosine.py#L1-L1)

## nexuml_library.training.schedulers.warmup_cosine.WarmupCosineLR

*class*

```python
class WarmupCosineLR(torch.optim.lr_scheduler.LRScheduler)
```

Bases: `torch.optim.lr_scheduler.LRScheduler`

Linear warmup followed by cosine decay to an absolute LR floor.

**Parameters**

- `optimizer` (`torch.optim.Optimizer`) — Optimizer whose parameter groups will be scheduled.
- `warmup_epochs` (`int`) — Number of initial epochs spent linearly increasing from ``min_lr`` to ``base_lr``. Use ``0`` to disable warmup.
- `max_epochs` (`int`) — Total scheduler horizon. At ``epoch == max_epochs`` the LR is ``min_lr``.
- `min_lr` (`float`) — Absolute minimum learning rate, not a multiplier.
- `last_epoch` (`int`) (default: `-1`) — Last scheduler epoch, forwarded to PyTorch's scheduler base.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/training/schedulers/warmup_cosine.py#L10-L63)

### nexuml_library.training.schedulers.warmup_cosine.WarmupCosineLR.max_epochs

*attribute* · *instance attribute*

```python
max_epochs = max_epochs
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/training/schedulers/warmup_cosine.py#L46-L46)

### nexuml_library.training.schedulers.warmup_cosine.WarmupCosineLR.min_lr

*attribute* · *instance attribute*

```python
min_lr = min_lr
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/training/schedulers/warmup_cosine.py#L47-L47)

### nexuml_library.training.schedulers.warmup_cosine.WarmupCosineLR.warmup_epochs

*attribute* · *instance attribute*

```python
warmup_epochs = warmup_epochs
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/training/schedulers/warmup_cosine.py#L45-L45)

### nexuml_library.training.schedulers.warmup_cosine.WarmupCosineLR.get_lr

*method*

```python
def get_lr() -> list[float | torch.Tensor]
```

Return scheduled learning rates for the current epoch.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/training/schedulers/warmup_cosine.py#L50-L53)
