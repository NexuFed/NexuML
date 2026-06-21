"""Learning-rate schedulers for NexuML library scenarios."""

from __future__ import annotations

import math

import torch


class WarmupCosineLR(torch.optim.lr_scheduler.LRScheduler):
    """Linear warmup followed by cosine decay to an absolute LR floor.

    Args:
        optimizer: Optimizer whose parameter groups will be scheduled.
        warmup_epochs: Number of initial epochs spent linearly increasing from
            ``min_lr`` to ``base_lr``. Use ``0`` to disable warmup.
        max_epochs: Total scheduler horizon. At ``epoch == max_epochs`` the LR is
            ``min_lr``.
        min_lr: Absolute minimum learning rate, not a multiplier.
        last_epoch: Last scheduler epoch, forwarded to PyTorch's scheduler base.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        min_lr: float,
        last_epoch: int = -1,
    ) -> None:
        if max_epochs <= 0:
            raise ValueError("max_epochs must be greater than 0")
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be greater than or equal to 0")
        if warmup_epochs >= max_epochs:
            raise ValueError("warmup_epochs must be less than max_epochs")
        if min_lr < 0:
            raise ValueError("min_lr must be greater than or equal to 0")

        base_lrs = [group["lr"] for group in optimizer.param_groups]
        too_low = [base_lr for base_lr in base_lrs if min_lr > base_lr]
        if too_low:
            raise ValueError("min_lr must not exceed any optimizer base learning rate")

        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float | torch.Tensor]:
        """Return scheduled learning rates for the current epoch."""
        epoch = max(self.last_epoch, 0)
        return [self._lr_for_epoch(float(base_lr), epoch) for base_lr in self.base_lrs]

    def _lr_for_epoch(self, base_lr: float, epoch: int) -> float:
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            warmup_progress = float(epoch) / float(self.warmup_epochs)
            return self.min_lr + (base_lr - self.min_lr) * warmup_progress

        decay_epochs = self.max_epochs - self.warmup_epochs
        decay_epoch = min(max(epoch - self.warmup_epochs, 0), decay_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_epoch / decay_epochs))
        return self.min_lr + (base_lr - self.min_lr) * cosine
