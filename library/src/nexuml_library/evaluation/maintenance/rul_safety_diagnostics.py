"""Train/validation-only sensor-stage evidence for RUL safety experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightning as L
import torch
from torch import Tensor

from nexuml_library.data.maintenance.rul import read_rul_artifact

_STAGE_BOUNDS = (
    (0.0, 30.0),
    (30.0, 60.0),
    (60.0, 90.0),
    (90.0, 120.0),
    (120.0, float("inf")),
)


def summarize_rul_sensor_stages(payload: dict[str, Any]) -> dict[str, Any]:
    """Compare last-cycle sensor features by RUL stage without using test rows.

    Input features have already been normalized using the fit-engine split.
    All statistics are estimated separately from train and validation windows.

    Returns:
        Counts, standardized sensor profiles, and training-only stage contrasts.

    Raises:
        ValueError: If train/validation features, metadata, or split IDs disagree.
    """
    features = payload["features"]
    targets = payload["targets"]
    rows = payload["rows"]
    settings = payload["settings"]
    if (
        not isinstance(features, Tensor)
        or not isinstance(targets, Tensor)
        or features.ndim != 3
        or targets.shape != (len(features), 1)
        or len(rows) != len(features)
    ):
        raise ValueError("RUL stage audit requires aligned windows and scalar labels")
    split_assets = settings["split_assets"]
    if any(
        set(split_assets[a]) & set(split_assets[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise ValueError("RUL stage audit detected overlapping engine splits")
    normalized_rul = targets[:, 0].detach().cpu() * float(settings["target_scale"])
    final_features = features[:, -1, :].detach().cpu()
    result: dict[str, Any] = {
        "source": "C-MAPSS prepared windows",
        "purpose": "Train/validation-only sensor-stage comparison; not test-tuned",
        "feature_names": settings["feature_names"],
        "stage_bounds_cycles": [
            [low, high if high != float("inf") else None]
            for low, high in _STAGE_BOUNDS
        ],
        "splits": {},
    }
    for split in ("train", "val"):
        members = torch.tensor(
            [row["split"] == split for row in rows], dtype=torch.bool
        )
        details: dict[str, Any] = {}
        for low, high in _STAGE_BOUNDS:
            chosen = members & (normalized_rul >= low) & (normalized_rul < high)
            observed = final_features[chosen]
            label = f"{int(low)}-{int(high)}" if high != float("inf") else "120+"
            if observed.shape[0] == 0:
                details[label] = {"count": 0, "mean": None, "std": None}
            else:
                details[label] = {
                    "count": len(observed),
                    "mean": observed.mean(dim=0).tolist(),
                    "std": observed.std(dim=0, unbiased=False).tolist(),
                }
        result["splits"][split] = details

    # Rank sensors by training-set midlife (60-90) vs late-stage (0-30)
    # separation; validation and test observations do not determine ranks.
    fit = result["splits"]["train"]
    late = fit["0-30"]
    mid = fit["60-90"]
    contrasts: list[dict[str, float | str]] = []
    if late["count"] and mid["count"]:
        for name, mean_mid, mean_late, std_late in zip(
            settings["feature_names"],
            mid["mean"],
            late["mean"],
            late["std"],
        ):
            contrast = (mean_mid - mean_late) / max(std_late, 1e-3)
            contrasts.append({"sensor": name, "separation": float(contrast)})
        contrasts.sort(key=lambda entry: abs(float(entry["separation"])), reverse=True)
    result["train_midlife_late_sensor_contrast"] = contrasts
    return result


class RULSensorStageAudit(L.Callback):
    """Save split-safe RUL stage diagnostics before ordinary NexuML fitting."""

    def __init__(self, artifact_path: str, output_file: str) -> None:
        super().__init__()
        self.artifact_path = artifact_path
        self.output_file = output_file

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Export sensor-stage statistics using only fit and validation engines."""
        payload = read_rul_artifact(self.artifact_path)
        summary = summarize_rul_sensor_stages(payload)
        destination = Path(self.output_file).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
