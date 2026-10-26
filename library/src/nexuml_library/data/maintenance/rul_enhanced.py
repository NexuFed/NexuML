"""Aligned raw C-MAPSS windows and real adjacent-engine pairs for RUL studies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Self, cast

import pandas as pd
import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor

from nexuml.core.components import DataSourceDefinition
from nexuml.core.discovery import data_source
from nexuml.data.dataset import NexuDataset
from nexuml_library.data.maintenance.rul import (
    AutoCMapSSRULWindows,
    read_rul_artifact,
)

_FD001 = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]


@data_source("AutoPairedCMapSSRULWindows")
class AutoPairedCMapSSRULWindows(DataSourceDefinition):
    """Prepare raw windows and same-engine one-cycle predecessors with no leakage."""

    subset: Literal["FD001", "FD003"] = "FD003"
    seed: int = Field(default=42, ge=0)
    root: str = "data/cmapss"
    artifact_path: str = ".experiments/rul/prepared/cmapss-FD003-seed42.pt"

    def build(self) -> NexuDataset:
        """Materialize the controlled RUL protocol and paired predecessor views.

        Returns:
            NexuML dataset with exact train/val/test asset boundaries.

        Raises:
            ValueError: If the artifact does not match the frozen-upstream protocol.
        """
        sensors = sorted([*_FD001, 6]) if self.subset == "FD003" else _FD001
        AutoCMapSSRULWindows(
            subset=self.subset,
            seed=self.seed,
            root=self.root,
            artifact_path=self.artifact_path,
            window=30,
            stride=1,
            rul_cap=130.0,
            sensor_ids=sensors,
            normalization="zscore",
            padding_mode="zero",
            train_full_windows_only=True,
            normalize_targets=False,
            download=True,
        ).build()
        payload = read_rul_artifact(self.artifact_path)
        settings = cast(dict[str, Any], payload["settings"])
        if (
            settings.get("subset") != self.subset
            or settings.get("seed") != self.seed
            or settings.get("window") != 30
            or settings.get("rul_cap") != 130.0
            or settings.get("normalization") != "zscore"
            or settings.get("sensor_ids") != sensors
        ):
            raise ValueError("Paired RUL source has incompatible training protocol")
        return _PairedRULRuntime(
            features=cast(Tensor, payload["features"]),
            targets=cast(Tensor, payload["targets"]),
            raw=cast(Tensor, payload["raw_targets"]),
            rows=cast(list[dict[str, Any]], payload["rows"]),
        )


class _PairedRULRuntime(NexuDataset):
    """Return actual predecessor windows, never crossing engine or split."""

    def __init__(
        self,
        *,
        features: Tensor,
        targets: Tensor,
        raw: Tensor,
        rows: list[dict[str, Any]],
    ) -> None:
        index = {
            (str(row["split"]), str(row["asset_id"]), int(row["step"])): pos
            for pos, row in enumerate(rows)
        }
        predecessors = []
        for pos, row in enumerate(rows):
            key = (str(row["split"]), str(row["asset_id"]), int(row["step"]) - 1)
            predecessors.append(index.get(key, -1))
        self._target = targets
        self._raw = raw
        self._predecessors = torch.tensor(predecessors, dtype=torch.long)
        self._previous = features[self._predecessors.clamp_min(0)].clone()
        self._mask = (self._predecessors >= 0).float().unsqueeze(-1)
        super().__init__(
            meta=pd.DataFrame.from_records(rows),
            data=features,
            label_names=[],
            modality="sensor",
            do_split=False,
        )

    def take(self, indices: Sequence[int]) -> Self:
        """Preserve aligned predecessor tensors through native split slicing.

        Returns:
            Engine-disjoint dataset subset.
        """
        selected = list(indices)
        subset = super().take(selected)
        # Materialized predecessor features remain aligned even when taking
        # further subsets, e.g. one validation window per calibration engine.
        subset._previous = self._previous[selected]
        subset._mask = self._mask[selected]
        subset._target = self._target[selected]
        subset._raw = self._raw[selected]
        subset._predecessors = self._predecessors[selected]
        return subset

    def load_item(self, idx: int, row: pd.Series) -> TensorDict:
        """Return current and immediately preceding same-engine sensor windows.

        Returns:
            Sensor window, preceding window and availability indicator.
        """
        data = cast(Tensor, self.data)
        previous_sample = self._previous[idx]
        is_paired = float(self._mask[idx, 0])
        return TensorDict(
            {
                "rul_history": data[idx],
                "previous_rul_history": previous_sample,
                "pair_mask": torch.tensor([is_paired], dtype=data.dtype),
            },
            batch_size=[],
        )

    def load_labels(self, idx: int, row: pd.Series) -> TensorDict:
        """Return exact capped and uncapped per-engine RUL labels.

        Returns:
            TensorDict with scalar RUL labels in cycles.
        """
        return TensorDict(
            {"rul": self._target[idx], "rul_raw": self._raw[idx]},
            batch_size=[],
        )
