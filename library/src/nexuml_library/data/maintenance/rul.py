"""Typed NexuML datasource for immutable labeled run-to-failure RUL windows."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self, cast

import pandas as pd
import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor

from nexuml.core.components import DataSourceDefinition
from nexuml.core.discovery import data_source
from nexuml.data.dataset import NexuDataset


def validate_rul_payload(payload: dict[str, Any]) -> None:
    """Reject leakage, invalid targets and inconsistent normalized trajectories.

    Raises:
        ValueError: If the prepared dataset violates the RUL split contract.
    """
    if not isinstance(payload, dict) or payload.get("format") != "nexuml-rul-windows-v1":
        raise ValueError("Unknown RUL artifact schema")
    features = payload.get("features")
    targets = payload.get("targets")
    rows = payload.get("rows")
    settings = payload.get("settings")
    if (
        not isinstance(features, Tensor)
        or features.ndim != 3
        or features.shape[0] == 0
        or features.shape[1] == 0
        or features.shape[2] == 0
        or not features.is_floating_point()
        or not bool(torch.isfinite(features).all())
        or not isinstance(targets, Tensor)
        or targets.shape != (len(features), 1)
        or not targets.is_floating_point()
        or not bool(torch.isfinite(targets).all())
        or bool((targets < 0).any())
        or not isinstance(rows, list)
        or len(rows) != len(features)
        or not isinstance(settings, dict)
        or settings.get("target_unit") not in {"cycles", "recordings"}
        or float(settings.get("target_scale", 0)) <= 0
    ):
        raise ValueError("Malformed or nonfinite RUL windows")
    asset_split: dict[str, str] = {}
    identifiers: set[tuple[str, int]] = set()
    actual = {"train": set(), "val": set(), "test": set()}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Malformed RUL sample provenance")
        split = row.get("split")
        asset = row.get("asset_id")
        step = row.get("step")
        if (
            split not in actual
            or not isinstance(asset, str)
            or not asset
            or not isinstance(step, int)
            or step < 0
            or (asset, step) in identifiers
        ):
            raise ValueError("Invalid or repeated asset-step RUL observation")
        if asset in asset_split and asset_split[asset] != split:
            raise ValueError("RUL windows from one asset span multiple splits")
        asset_split[asset] = split
        actual[split].add(asset)
        identifiers.add((asset, step))
    if any(not groups for groups in actual.values()):
        raise ValueError("Require nonempty train, val and test asset groups")
    saved = settings.get("split_assets")
    if not isinstance(saved, dict) or any(
        sorted(actual[split]) != saved.get(split) for split in actual
    ):
        raise ValueError("RUL split provenance does not match the tensor rows")
    scaler = settings.get("scaler")
    if not isinstance(scaler, dict) or any(
        not isinstance(scaler.get(key), list)
        or len(scaler[key]) != features.shape[-1]
        for key in ("mean", "std")
    ):
        raise ValueError("RUL feature normalization metadata is inconsistent")


def read_rul_artifact(path: str | Path) -> dict[str, Any]:
    """Load a safe, versioned tensor artifact.

    Returns:
        Validated tensor features, normalized labels and asset metadata.
    """
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    validate_rul_payload(payload)
    return cast(dict[str, Any], payload)


@data_source("PreparedRULWindows")
class PreparedRULWindows(DataSourceDefinition):
    """Load immutable RUL windows and preserve prepared asset-disjoint splits."""

    path: str = Field(min_length=1)

    def build(self) -> NexuDataset:
        """Materialize the prepared features through the normal NexuML dataset API.

        Returns:
            Dataset of sensor history windows and normalized RUL labels.
        """
        return _PreparedRULRuntime(self.path)


class _PreparedRULRuntime(NexuDataset):
    """Tensor-backed causal observations and supervised normalized RUL labels."""

    def __init__(self, path: str) -> None:
        payload = read_rul_artifact(path)
        self._rul_targets = cast(Tensor, payload["targets"])
        super().__init__(
            meta=pd.DataFrame.from_records(payload["rows"]),
            data=cast(Tensor, payload["features"]),
            label_names=[],
            modality="sensor",
            do_split=False,
        )

    def take(self, indices: Sequence[int]) -> Self:
        """Return a split-safe copy with corresponding RUL labels.

        Returns:
            Dataset view with aligned window tensors, provenance and targets.
        """
        selected = list(indices)
        subset = super().take(selected)
        subset._rul_targets = self._rul_targets[selected]
        return subset

    def load_item(self, idx: int, row: pd.Series) -> TensorDict:
        """Expose causal sensor sequence to the model.

        Returns:
            Feature TensorDict with shape [L,D] per observation.
        """
        return TensorDict(
            {"rul_history": cast(Tensor, self.data)[idx]},
            batch_size=[],
        )

    def load_labels(self, idx: int, row: pd.Series) -> TensorDict:
        """Provide normalized RUL targets with scalar output dimension.

        Returns:
            Label TensorDict with rul [1].
        """
        return TensorDict({"rul": self._rul_targets[idx]}, batch_size=[])
