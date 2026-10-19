"""Regression test for latest-state preservation when validation does not improve."""

from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from nexuml_library.evaluation.maintenance.ablation_checkpoint import SaveLatestEpoch


class _WorseningValidationModel(L.LightningModule):
    """Small model whose parameter changes while monitored validation gets worse."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        return self.weight.square()

    def validation_step(self, batch: object, batch_idx: int) -> None:
        self.log("val/loss", float(self.current_epoch + 1), on_epoch=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return a real optimizer that updates model parameters."""
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def train_dataloader(self) -> DataLoader:
        """Return two training batches per epoch."""
        return DataLoader(TensorDataset(torch.zeros(4)), batch_size=2)

    def val_dataloader(self) -> DataLoader:
        """Return one validation batch per epoch."""
        return DataLoader(TensorDataset(torch.zeros(2)), batch_size=2)


def test_last_checkpoint_updates_when_validation_best_stays_at_epoch_zero(
    tmp_path: Path,
) -> None:
    """Best remains at epoch zero while last retains the final parameter state."""
    best_callback = ModelCheckpoint(
        dirpath=tmp_path,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    latest_callback = SaveLatestEpoch(dirpath=tmp_path)
    trainer = L.Trainer(
        max_epochs=3,
        accelerator="cpu",
        devices=1,
        logger=False,
        callbacks=[best_callback, latest_callback],
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(_WorseningValidationModel())

    best = torch.load(best_callback.best_model_path, map_location="cpu", weights_only=False)
    final = torch.load(latest_callback.path, map_location="cpu", weights_only=False)
    assert trainer.global_step == 6
    assert best["global_step"] == 2
    assert final["global_step"] == 6
    assert best["epoch"] == 0
    assert final["epoch"] == 2
    best_paths = {
        state["best_model_path"]
        for state in final["callbacks"].values()
        if isinstance(state, dict) and "best_model_path" in state
    }
    assert best_paths == {best_callback.best_model_path}
    assert not torch.equal(
        best["state_dict"]["weight"],
        final["state_dict"]["weight"],
    )
