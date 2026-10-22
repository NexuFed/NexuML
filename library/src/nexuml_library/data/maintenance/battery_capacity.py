"""Chronological battery capacity forecasting windows for PatchFormer/RUL-Mamba.

Expect one CSV per cell, with 'cycle' and 'capacity' columns. This adapter
predicts observed NEXT-cycle capacity in Ah, not an invented RUL label.
Battery life requires separate extrapolation to the documented EOL threshold.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
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


@data_source("BatteryCapacityWindows")
class BatteryCapacityWindows(DataSourceDefinition):
    """Real capacity observations with ordered train/val and held-out cells."""

    root: str = "data/battery/nasa"
    lookback: int = Field(default=30, gt=1)
    val_fraction: float = Field(default=0.2, gt=0, lt=0.5)
    train_cells: list[str] = Field(
        default_factory=lambda: ["B0006", "B0007", "B0018"]
    )
    test_cell: str = "B0005"
    end_of_life_ah: float = Field(default=1.4, gt=0)
    normalizer_file: str | None = None

    def build(self) -> NexuDataset:
        """Read real CSV curves and fit capacity scaler to training cycles only.

        Returns:
            NexuML sensor dataset of causal windows with next capacity labels.

        Raises:
            ValueError: If battery identities or chronological splits are invalid.
            FileNotFoundError: If required capacity CSV files have not been supplied.
        """
        if (
            not self.train_cells
            or len(set(self.train_cells)) != len(self.train_cells)
            or self.test_cell in self.train_cells
        ):
            raise ValueError("Battery train/test cell identities must be disjoint")
        root = Path(self.root).expanduser()
        series: dict[str, np.ndarray] = {}
        for cell in [*self.train_cells, self.test_cell]:
            path = root / f"{cell}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing battery capacity trace: {path}")
            frame = pd.read_csv(path)
            if not {"cycle", "capacity"}.issubset(frame.columns):
                raise ValueError(f"Battery file needs cycle and capacity columns: {path}")
            cycles = frame["cycle"].to_numpy(dtype=np.float64)
            capacities = frame["capacity"].to_numpy(dtype=np.float64)
            if (
                len(cycles) <= self.lookback + 1
                or not np.isfinite(cycles).all()
                or not np.isfinite(capacities).all()
                or bool(np.any(np.diff(cycles) <= 0))
                or bool(np.any(capacities <= 0))
            ):
                raise ValueError(f"Malformed battery degradation curve: {cell}")
            series[cell] = capacities
        boundaries = {
            cell: max(
                self.lookback + 1,
                min(
                    len(series[cell]) - 1,
                    round(len(series[cell]) * (1.0 - self.val_fraction)),
                ),
            )
            for cell in self.train_cells
        }
        train_values = np.concatenate(
            [series[cell][:boundaries[cell]] for cell in self.train_cells]
        )
        minimum = float(train_values.min())
        spread = max(float(train_values.max()) - minimum, 1e-6)
        if self.normalizer_file is not None:
            path = Path(self.normalizer_file).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            metadata = {
                "fit_scope": "training_cycles_only",
                "train_cells": self.train_cells,
                "held_out_cell": self.test_cell,
                "capacity_offset_ah": minimum,
                "capacity_scale_ah": spread,
                "end_of_life_ah": self.end_of_life_ah,
            }
            path.write_text(json.dumps(metadata, indent=2) + "\\n", encoding="utf-8")
        features: list[Tensor] = []
        labels: list[float] = []
        meta: list[dict[str, int | str]] = []
        for cell, values in series.items():
            for stop in range(self.lookback, len(values)):
                if cell == self.test_cell:
                    split = "test"
                else:
                    split = "train" if stop < boundaries[cell] else "val"
                history = (values[stop - self.lookback:stop] - minimum) / spread
                features.append(torch.tensor(history, dtype=torch.float32)[:, None])
                labels.append(float(values[stop]))
                meta.append({"asset_id": cell, "split": split, "step": stop})
        if any(not any(row["split"] == split for row in meta)
               for split in ("train", "val", "test")):
            raise ValueError("Battery dataset needs nonempty chronological splits")
        return _BatteryCapacityRuntime(
            features=torch.stack(features),
            targets=torch.tensor(labels, dtype=torch.float32).unsqueeze(-1),
            meta=pd.DataFrame.from_records(meta),
        )


class _BatteryCapacityRuntime(NexuDataset):
    """Per-observation native features and next-cycle capacity targets."""

    def __init__(self, *, features: Tensor, targets: Tensor, meta: pd.DataFrame) -> None:
        self._targets = targets
        super().__init__(
            meta=meta,
            data=features,
            label_names=[],
            modality="sensor",
            do_split=False,
        )

    def take(self, indices: Sequence[int]) -> Self:
        """Preserve capacity targets alongside NexuML split views.

        Returns:
            Split-safe dataset copy with aligned next-cycle capacity labels.
        """
        selected = list(indices)
        subset = super().take(selected)
        subset._targets = self._targets[selected]
        return subset

    def load_item(self, idx: int, row: pd.Series) -> TensorDict:
        """Return normalized historical capacities before the predicted cycle.

        Returns:
            Feature TensorDict with one univariate capacity time series.
        """
        return TensorDict(
            {"capacity_history": cast(Tensor, self.data)[idx]},
            batch_size=[],
        )

    def load_labels(self, idx: int, row: pd.Series) -> TensorDict:
        """Return observed following-cycle capacity in Ah, never derived test RUL.

        Returns:
            Supervised scalar future_capacity label in ampere-hours.
        """
        return TensorDict({"future_capacity": self._targets[idx]}, batch_size=[])
