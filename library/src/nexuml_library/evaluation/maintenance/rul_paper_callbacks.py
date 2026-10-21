"""Paper-matched validation-plateau learning rate schedule for native NexuML."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback


class Goel2026PlateauLR(Callback):
    """Reduce RMSprop learning rate on validation-loss plateaus.

    Implemented as a library callback because the current NexuML scheduler
    slot assumes epoch-stepped schedulers without a monitor argument.
    Native NexuML fit/validation/checkpoint execution remains unchanged.
    """

    def __init__(self, factor: float = 0.5, patience: int = 5) -> None:
        super().__init__()
        self.factor = factor
        self.patience = patience
        self._scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
        self._pending_state: dict[str, Any] | None = None

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Attach plateau scheduler to NexuML's existing optimizer."""
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            trainer.optimizers[0], mode="min", factor=self.factor, patience=self.patience
        )
        if self._pending_state is not None:
            self._scheduler.load_state_dict(self._pending_state)
            self._pending_state = None

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Update scheduler after complete validation metrics are available."""
        if trainer.sanity_checking or self._scheduler is None:
            return
        metric = trainer.callback_metrics.get("val/loss")
        if metric is None:
            raise RuntimeError("Goel2026 schedule requires val/loss from native NexuML")
        self._scheduler.step(float(metric))

    def state_dict(self) -> dict[str, Any]:
        """Persist scheduler state through regular Lightning checkpoints.

        Returns:
            Learning-rate scheduler state.
        """
        if self._scheduler is None:
            return dict(self._pending_state or {})
        return self._scheduler.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore plateau state on native Lightning checkpoint resume."""
        self._pending_state = dict(state_dict)
