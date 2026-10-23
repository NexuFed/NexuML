"""Comparable full-split evaluation-mode RUL learning curves and engine diagnostics.

Runs within native Lightning validation, without changing NexuML core logging.
The training loader is re-created without shuffle or sampling, and only
train/validation data are inspected. Test data remain sealed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import lightning as L
import torch
from lightning.pytorch.trainer.states import TrainerFn
from tensordict import TensorDict
from torch import Tensor

from nexuml.data.module import NexuDataModule
from nexuml.training.lightning import NexuLightningModule

_FIELDS = (
    "epoch", "train_eval_mse", "val_eval_mse", "train_eval_rmse",
    "val_eval_rmse", "mse_gap", "train_windows", "val_windows",
    "val_engine_count", "val_group_max_mse", "val_group_std_mse",
)


def _evaluate_split(
    model: NexuLightningModule,
    loader: Any,
    *,
    metadata: Any | None = None,
) -> tuple[float, int, dict[str, tuple[float, int]]]:
    """Measure sample-weighted prediction MSE with dropout disabled.

    Args:
        model: The native compiled RUL model in evaluation mode.
        loader: Deterministic, complete and unshuffled split data loader.
        metadata: Validation split metadata in local sample-index order.

    Returns:
        Total squared error, number of windows and per-engine sum/count.

    Raises:
        ValueError: If RUL shapes or validation sample identity are invalid.
    """
    squared_sum = 0.0
    count = 0
    engines: dict[str, tuple[float, int]] = {}
    group_error: dict[str, float] = defaultdict(float)
    group_count: dict[str, int] = defaultdict(int)
    for x, y in loader:
        if not isinstance(x, TensorDict) or not isinstance(y, TensorDict):
            raise ValueError("RUL diagnostics need feature and label TensorDicts")
        indices = cast(Tensor, x["sample_index"]).view(-1).tolist()
        x = x.to(model.device)
        y = y.to(model.device)
        predicted_x, _ = model.pipeline(x, y)
        estimate = cast(Tensor, predicted_x["predicted_rul"])
        label = cast(Tensor, y["rul"])
        if estimate.shape != label.shape or estimate.ndim != 2:
            raise ValueError("RUL diagnostic requires aligned [batch,1] tensors")
        residual = (estimate - label).detach().cpu().flatten()
        if not bool(torch.isfinite(residual).all()):
            raise ValueError("RUL diagnostics found nonfinite residuals")
        errors = residual.square().tolist()
        squared_sum += sum(errors)
        count += len(errors)
        if metadata is not None:
            for local_index, squared in zip(indices, errors):
                if not 0 <= local_index < len(metadata):
                    raise ValueError("Validation sample index outside engine metadata")
                row = metadata.iloc[local_index]
                if row["split"] != "val":
                    raise ValueError(
                        "Validation diagnostics encountered a non-val asset"
                    )
                engine = str(row["asset_id"])
                group_error[engine] += float(squared)
                group_count[engine] += 1
    if count == 0:
        raise ValueError("Cannot diagnose an empty RUL split")
    for engine, total in group_error.items():
        engines[engine] = (total, group_count[engine])
    return squared_sum, count, engines


def _validation_engine_groups(
    engines: dict[str, tuple[float, int]], groups: int = 4
) -> list[dict[str, float | int]]:
    """Create reproducible, disjoint validation-engine stress groups.

    These groups partition *existing* upstream-held-out validation engines.
    They are not independent training folds or new encoder/flow validation
    splits, and are never used as training samples.

    Returns:
        Per-group engine count and window-weighted/engine-mean RUL MSE.

    Raises:
        ValueError: If too few distinct validation assets are present.
    """
    if groups < 2 or len(engines) < groups:
        raise ValueError("Need at least two nonempty validation engine groups")
    assets = sorted(
        engines, key=lambda name: (hashlib.sha256(name.encode()).digest(), name)
    )
    results: list[dict[str, float | int]] = []
    for index in range(groups):
        members = [name for i, name in enumerate(assets) if i % groups == index]
        total_sse = sum(engines[name][0] for name in members)
        windows = sum(engines[name][1] for name in members)
        if windows <= 0:
            raise ValueError("Validation group cannot be empty")
        results.append({
            "group": index,
            "engines": len(members),
            "windows": windows,
            "window_mse": total_sse / windows,
            "engine_mean_mse": statistics.mean(
                engines[name][0] / engines[name][1] for name in members
            ),
        })
    return results


class SequenceRULGeneralizationDiagnostics(L.Callback):
    """Export full epoch curves and validation per-engine losses for RUL."""

    def __init__(self, output_file: str) -> None:
        super().__init__()
        self.output_file = Path(output_file)
        self._best_validation = float("inf")

    @torch.inference_mode()
    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Re-evaluate full train/val splits after each fitting epoch only.

        Raises:
            TypeError: If NexuML's expected Lightning/data contracts differ.
            ValueError: If train/val engine metadata or predictions are invalid.
        """
        # NexuML calls validate() again after fit, with best weights restored.
        # That standalone pass is not a training epoch and must not enter curves.
        if trainer.sanity_checking or trainer.state.fn != TrainerFn.FITTING:
            return
        if not isinstance(pl_module, NexuLightningModule):
            raise TypeError("Sequence RUL diagnostics expect NexuLightningModule")
        data = getattr(trainer, "datamodule", None)
        if not isinstance(data, NexuDataModule):
            raise TypeError("Sequence RUL diagnostics expect NexuDataModule")
        if trainer.world_size != 1:
            raise ValueError("RUL diagnostics currently require one trainer process")
        train_dataset = data._train_ds
        val_dataset = data._val_ds
        if train_dataset is None or val_dataset is None:
            raise ValueError("RUL diagnostic splits have not been prepared")
        train_meta = getattr(train_dataset, "meta", None)
        val_meta = getattr(val_dataset, "meta", None)
        if train_meta is None or val_meta is None:
            raise ValueError("RUL diagnostics require engine metadata per split")
        if set(train_meta["asset_id"]) & set(val_meta["asset_id"]):
            raise ValueError("Engine identity leakage across train and validation")
        was_training = pl_module.training
        pl_module.eval()
        try:
            # Loader initialization must not perturb training RNG streams.
            with torch.random.fork_rng(devices=[]):
                train_loader = data._loader(
                    train_dataset, split="val", shuffle=False
                )
                train_sse, train_count, _ = _evaluate_split(
                    pl_module, train_loader
                )
                val_sse, val_count, engines = _evaluate_split(
                    pl_module, data.val_dataloader(), metadata=val_meta
                )
        finally:
            pl_module.train(was_training)
        if not engines:
            raise ValueError("Validation engine diagnostics found no groups")
        train_mse = train_sse / train_count
        val_mse = val_sse / val_count
        groups = _validation_engine_groups(engines)
        epoch = trainer.current_epoch
        result = {
            "epoch": epoch,
            "train_eval_mse": train_mse,
            "val_eval_mse": val_mse,
            "train_eval_rmse": train_mse ** 0.5,
            "val_eval_rmse": val_mse ** 0.5,
            "mse_gap": val_mse - train_mse,
            "train_windows": train_count,
            "val_windows": val_count,
            "val_engine_count": len(engines),
            "val_group_max_mse": max(group["window_mse"] for group in groups),
            "val_group_std_mse": statistics.stdev(
                float(group["window_mse"]) for group in groups
            ),
        }
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with self.output_file.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(_FIELDS))
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(result)
        pl_module.log(
            "val/generalization_train_eval_mse", train_mse,
            on_step=False, on_epoch=True, sync_dist=False
        )
        pl_module.log(
            "val/generalization_val_eval_mse", val_mse,
            on_step=False, on_epoch=True, sync_dist=False
        )
        pl_module.log(
            "val/generalization_gap", val_mse - train_mse,
            on_step=False, on_epoch=True, sync_dist=False
        )
        if val_mse < self._best_validation:
            self._best_validation = val_mse
            per_engine = [
                {
                    "asset_id": asset,
                    "windows": observations,
                    "mse": squared_error / observations,
                    "rmse": (squared_error / observations) ** 0.5,
                }
                for asset, (squared_error, observations) in sorted(engines.items())
            ]
            report = {
                "selection": "minimum evaluation-mode validation-window MSE",
                "best_epoch": epoch,
                "train_eval_mse": train_mse,
                "val_eval_mse": val_mse,
                "validation_engines": per_engine,
                "validation_engine_groups": groups,
                "test_labels_used": False,
            }
            destination = self.output_file.with_suffix(".validation_engines.json")
            destination.write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )

    def on_fit_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Reset per-run diagnostics to avoid appending stale epoch histories."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file.unlink(missing_ok=True)
        self.output_file.with_suffix(".validation_engines.json").unlink(
            missing_ok=True
        )
        self._best_validation = float("inf")
