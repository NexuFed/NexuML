"""Deterministic training-time RNG initialization for NexuML RUL scenarios."""

from __future__ import annotations

import lightning as L
import torch


class RULSeedOnFitStart(L.Callback):
    """Seed data sampling and training RNG after deterministic model construction."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        self.seed = seed

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Seed PyTorch/NumPy/Python RNGs before training data are iterated."""
        L.seed_everything(self.seed, workers=True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
