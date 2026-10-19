"""Persist the actual last fitted state independently of validation-best scoring."""

from __future__ import annotations

from pathlib import Path

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Checkpoint


class SaveLatestEpoch(Checkpoint):
    """Overwrite the latest checkpoint at every completed training epoch.

    In Lightning 2.6.5, ModelCheckpoint(save_top_k=1, save_last=True)
    can leave last.ckpt stale when the monitored validation score does not
    improve. Keep this independent of the validation-best checkpoint.
    """

    def __init__(self, dirpath: str | Path) -> None:
        """Configure the dedicated last-state checkpoint path."""
        super().__init__()
        self.path = Path(dirpath) / "last.ckpt"

    def _save(self, trainer: Trainer) -> None:
        """Write the latest full-state checkpoint across all distributed ranks."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # All ranks participate; the Lightning strategy manages the actual write.
        trainer.save_checkpoint(str(self.path))

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Persist latest weights every epoch, even without validation improvement."""
        self._save(trainer)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Capture the final loop and best-model callback state after training."""
        self._save(trainer)
