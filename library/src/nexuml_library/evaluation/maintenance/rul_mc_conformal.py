"""Post-selection MC-dropout uncertainty and diagnostic conformal calibration.

The upstream encoder and RUL checkpoint have already been selected using the
validation split. Consequently the intervals are *exploratory*, with no claim
of exchangeability-based marginal coverage. No official test labels are read.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, cast

import lightning as L
import torch
from torch import Tensor, nn

from nexuml.data.module import NexuDataModule
from nexuml.training.lightning import NexuLightningModule


def _quantile(values: list[float], alpha: float) -> float:
    """Return the finite-sample conformal rank with an infinite fallback.

    Returns:
        Absolute-residual threshold for the requested miscoverage fraction.

    Raises:
        ValueError: If no finite residuals are available.
    """
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Conformal calibration requires finite residuals")
    rank = math.ceil((len(values) + 1) * (1.0 - alpha))
    return sorted(values)[rank - 1] if rank <= len(values) else math.inf


def _selected_calibration_windows(metadata: Any) -> list[int]:
    """Pick one hash-fixed validation window per held-out validation engine.

    Returns:
        Sorted, distinct local dataset indices for engine-level calibration.

    Raises:
        ValueError: If validation engine metadata is missing or duplicated.
    """
    if metadata is None:
        raise ValueError("Missing calibration validation metadata")
    groups: dict[str, list[int]] = {}
    for index in range(len(metadata)):
        row = metadata.iloc[index]
        if row["split"] != "val":
            raise ValueError("Calibration must use held-out validation engines")
        groups.setdefault(str(row["asset_id"]), []).append(index)
    if len(groups) < 2:
        raise ValueError("Conformal diagnostics need at least two validation engines")
    chosen = []
    for engine, positions in sorted(groups.items()):
        digest = hashlib.sha256(engine.encode("utf-8")).digest()
        chosen.append(positions[int.from_bytes(digest[:8], "big") % len(positions)])
    return sorted(chosen)


class RULMCConformalDiagnostics(L.Callback):
    """Compute dropout ensemble uncertainty and calibration residuals post-fit."""

    def __init__(
        self,
        output_file: str,
        *,
        stochastic_samples: int = 20,
        alpha: float = 0.1,
    ) -> None:
        super().__init__()
        if stochastic_samples < 2 or not 0.0 < alpha < 1.0:
            raise ValueError("Invalid MC dropout or calibration parameters")
        self.output_file = Path(output_file)
        self.stochastic_samples = stochastic_samples
        self.alpha = alpha

    @torch.inference_mode()
    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Record exploratory calibration after restoring the best RUL checkpoint.

        Raises:
            TypeError: If the native NexuML model and dataset are unavailable.
            ValueError: If calibration windows or predictions are inconsistent.
        """
        if not isinstance(pl_module, NexuLightningModule):
            raise TypeError("MC RUL diagnostic expects a native NexuML model")
        data = trainer.datamodule
        if not isinstance(data, NexuDataModule) or data._val_ds is None:
            raise TypeError("MC RUL diagnostic requires a prepared validation dataset")
        meta = data._val_ds.meta
        indices = _selected_calibration_windows(meta)
        subset = data._val_ds.take(indices)
        loader = data._loader(subset, split="val", shuffle=False)
        was_training = pl_module.training
        pl_module.eval()
        stochastic = [
            module for module in pl_module.modules() if isinstance(module, nn.Dropout)
        ]
        means: list[float] = []
        scales: list[float] = []
        targets: list[float] = []
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(1987)
                for x, y in loader:
                    if y is None:
                        raise ValueError("MC conformal diagnostics require RUL labels")
                    x = x.to(pl_module.device)
                    y = y.to(pl_module.device)
                    batch_means = []
                    for module in stochastic:
                        module.train()
                    for _ in range(self.stochastic_samples):
                        result, _ = pl_module.pipeline(x.clone(), y)
                        batch_means.append(
                            cast(Tensor, result["predicted_rul"]).detach()
                        )
                    predictions = torch.stack(batch_means, dim=0)
                    means.extend(predictions.mean(dim=0).flatten().cpu().tolist())
                    scales.extend(
                        predictions.std(dim=0, unbiased=False).flatten().cpu().tolist()
                    )
                    targets.extend(cast(Tensor, y["rul"]).flatten().cpu().tolist())
        finally:
            pl_module.train(was_training)
        if len(means) != len(indices) or len(scales) != len(targets):
            raise ValueError("MC RUL calibration engine count does not match metadata")
        residuals = [abs(pred - target) for pred, target in zip(means, targets)]
        threshold = _quantile(residuals, self.alpha)
        coverage = statistics.mean(
            float(residual <= threshold) for residual in residuals
        )
        report = {
            "study": "Exploratory MC dropout and calibration diagnostic",
            "calibration_scope": "one hash-selected window per validation engine",
            "selection_disjoint_from_calibration": False,
            "official_test_labels_used": False,
            "nominal_coverage_guarantee": False,
            "alpha": self.alpha,
            "stochastic_samples": self.stochastic_samples,
            "calibration_engines": len(indices),
            "dropout_predictive_std_mean": statistics.mean(scales),
            "mean_absolute_calibration_residual": statistics.mean(residuals),
            "symmetric_absolute_residual_quantile": (
                threshold if math.isfinite(threshold) else None
            ),
            "empirical_in_sample_coverage": coverage,
            "interval_full_width": (
                2.0 * threshold if math.isfinite(threshold) else None
            ),
        }
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
