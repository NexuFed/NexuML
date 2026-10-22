"""Labeled bimodal temperature/load histories for industrial DMM-JA.

The source never invents time-to-failure labels for an ETDataset time series.
It requires explicit unit-level RUL supervision and train/val/test asset splits.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Self, cast

import numpy as np
import pandas as pd
import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor

from nexuml.core.components import DataSourceDefinition
from nexuml.core.discovery import data_source
from nexuml.data.dataset import NexuDataset


@data_source("LabeledBimodalRULWindows")
class LabeledBimodalRULWindows(DataSourceDefinition):
    """Window explicit temperature/load measurements with observed RUL labels."""

    path: str = "data/maintenance/bimodal_rul.csv"
    window: int = Field(default=24, gt=1)
    normalizer_file: str | None = None

    def build(self) -> NexuDataset:
        """Validate provenance, fit train-only scales and construct causal windows.

        Returns:
            RUL-labeled NexuML dataset with exactly two modality channels.

        Raises:
            FileNotFoundError: If the user has not supplied a labeled CSV.
            ValueError: If labels, split provenance or modalities are invalid.
        """
        path = Path(self.path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Expected labeled temperature/load RUL CSV: {path}")
        table = pd.read_csv(path)
        required = {"asset_id", "cycle", "split", "temperature", "load", "rul"}
        if not required.issubset(table.columns):
            missing = sorted(required - set(table.columns))
            raise ValueError(f"Missing bimodal RUL columns: {missing}")
        numeric = ["cycle", "temperature", "load", "rul"]
        table[numeric] = table[numeric].apply(pd.to_numeric, errors="raise")
        if (
            not np.isfinite(table[numeric].to_numpy(dtype=np.float64)).all()
            or bool((table["rul"] < 0).any())
            or not set(table["split"]).issubset({"train", "val", "test"})
        ):
            raise ValueError("Bimodal RUL requires finite nonnegative labels and splits")
        split_assets: dict[str, set[str]] = {
            split: set(table.loc[table["split"] == split, "asset_id"])
            for split in ("train", "val", "test")
        }
        if any(not values for values in split_assets.values()) or any(
            split_assets[a] & split_assets[b]
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        ):
            raise ValueError("Bimodal RUL assets must have disjoint nonempty splits")
        training = table.loc[table["split"] == "train", ["temperature", "load"]]
        means = training.mean().to_numpy(dtype=np.float64)
        scales = np.maximum(training.std(ddof=0).to_numpy(dtype=np.float64), 1e-6)
        if self.normalizer_file is not None:
            destination = Path(self.normalizer_file).expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            metadata = {
                "fit_scope": "training_assets_only",
                "columns": ["temperature", "load"],
                "mean": means.tolist(),
                "std": scales.tolist(),
                "split_assets": {
                    name: sorted(ids) for name, ids in split_assets.items()
                },
            }
            destination.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        windows: list[Tensor] = []
        targets: list[float] = []
        rows: list[dict[str, int | str]] = []
        for asset, group in table.groupby("asset_id", sort=True):
            ordered = group.sort_values("cycle")
            cycles = ordered["cycle"].to_numpy(dtype=np.float64)
            if bool(np.any(np.diff(cycles) <= 0)):
                raise ValueError(f"Repeated or nonmonotonic cycles: {asset}")
            values = (
                ordered[["temperature", "load"]].to_numpy(dtype=np.float64) - means
            ) / scales
            labels = ordered["rul"].to_numpy(dtype=np.float64)
            split = str(ordered["split"].iloc[0])
            for stop in range(len(values)):
                segment = values[max(0, stop - self.window + 1):stop + 1]
                if len(segment) < self.window:
                    prefix = np.repeat(
                        segment[:1], self.window - len(segment), axis=0
                    )
                    segment = np.concatenate((prefix, segment))
                windows.append(torch.tensor(segment.copy(), dtype=torch.float32))
                targets.append(float(labels[stop]))
                rows.append(
                    {"asset_id": str(asset), "step": stop, "split": split}
                )
        return _BimodalRULRuntime(
            features=torch.stack(windows),
            targets=torch.tensor(targets, dtype=torch.float32).unsqueeze(-1),
            meta=pd.DataFrame.from_records(rows),
        )


class _BimodalRULRuntime(NexuDataset):
    """Native two-channel supervised RUL dataset with immutable split views."""

    def __init__(self, *, features: Tensor, targets: Tensor, meta: pd.DataFrame) -> None:
        self._targets = targets
        super().__init__(
            meta=meta, data=features, label_names=[], modality="sensor", do_split=False
        )

    def take(self, indices: Sequence[int]) -> Self:
        """Subset labels together with native split view metadata.

        Returns:
            NexuML subset aligned by training/validation/test indices.
        """
        selection = list(indices)
        subset = super().take(selection)
        subset._targets = self._targets[selection]
        return subset

    def load_item(self, idx: int, row: pd.Series) -> TensorDict:
        """Load an observed, standardized temperature/load history.

        Returns:
            Two-feature RUL history with shape [window, 2].
        """
        return TensorDict(
            {"rul_history": cast(Tensor, self.data)[idx]}, batch_size=[]
        )

    def load_labels(self, idx: int, row: pd.Series) -> TensorDict:
        """Return observed RUL directly, never derived from test trajectories.

        Returns:
            One-dimensional physical-unit RUL label.
        """
        return TensorDict({"rul": self._targets[idx]}, batch_size=[])
