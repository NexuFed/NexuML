"""Persist the actual last fitted state independently of validation-best scoring."""

from __future__ import annotations

from pathlib import Path

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


class SaveLatestEpoch(Callback):
    """Overwrite the latest checkpoint at every completed training epoch.

    In Lightning 2.6.5, ModelCheckpoint(save_top_k=1, save_last=True)
    can leave last.ckpt stale when the monitored validation score does not
    improve. Keep this independent of the validation-best checkpoint.
    """

    def __init__(self, dirpath: str | Path) -> None:
        """Configure the dedicated last-state checkpoint path."""
        super().__init__()
        self.path = Path(dirpath) / "last.ckpt"

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save model, optimizer, scheduler and callback state after each epoch."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Call on every rank; the strategy handles distributed checkpoint I/O.
        trainer.save_checkpoint(str(self.path))
