"""Paper-matched validation-plateau learning rate schedule for native NexuML."""

from __future__ import annotations

import json
from pathlib import Path
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


class Goel2026RestoreBestWeights(Callback):
    """Restore the monitored best validation weights after native NexuML fit.

    NexuML's default session validates and tests the supplied live module.
    This callback makes the paper's best-epoch inference rule explicit
    without changing the core NexuML training or testing framework.

    Checkpoints are created locally by the current run. The callback loads
    only the model state_dict and never restores optimizer/scheduler state,
    because no additional fitting follows this operation.
    """

    def __init__(self, output_file: str | None = None) -> None:
        super().__init__()
        self.output_file = output_file

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Load weights from the lowest-validation-loss Lightning checkpoint.

        Raises:
            RuntimeError: If there is no valid saved best model.
        """
        from lightning.pytorch.callbacks import ModelCheckpoint

        monitors = [
            item
            for item in trainer.callbacks
            if isinstance(item, ModelCheckpoint)
            and item.monitor == "val/loss"
            and item.mode == "min"
        ]
        if len(monitors) != 1 or not monitors[0].best_model_path:
            raise RuntimeError(
                "Goel2026 reproduction requires exactly one monitored best checkpoint"
            )
        path = monitors[0].best_model_path
        # Trust boundary: load a checkpoint written by this local training run.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise RuntimeError("Goel2026 best checkpoint has no model state_dict")
        pl_module.load_state_dict(checkpoint["state_dict"], strict=True)
        if self.output_file is not None:
            destination = Path(self.output_file).expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            score = monitors[0].best_model_score
            details = {
                "selection": "best minimum val/loss",
                "best_model_path": path,
                "best_model_score": float(score) if score is not None else None,
                "checkpoint_epoch": checkpoint.get("epoch"),
                "checkpoint_global_step": checkpoint.get("global_step"),
                "loaded_strict_model_state": True,
            }
            destination.write_text(
                json.dumps(details, indent=2) + "\n", encoding="utf-8"
            )
