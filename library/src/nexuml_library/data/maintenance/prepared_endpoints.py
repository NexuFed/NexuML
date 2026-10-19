"""Prepared endpoint embeddings for leakage-safe predictive-maintenance ablations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch
from pydantic import Field
from tensordict import TensorDict

from nexuml.core.components import DataSourceDefinition
from nexuml.core.discovery import data_source
from nexuml.data.dataset import NexuDataset


def read_prepared_artifact(path: str | Path) -> dict[str, Any]:
    """Validate an explicitly prepared, local feature artifact.

    Args:
        path: Trusted local feature artifact written by ablation preparation.

    Returns:
        Tensor features, recording metadata, training statistics and settings.

    Raises:
        FileNotFoundError: If preparation has not been performed.
        ValueError: If schema, splits or feature tensors are invalid.
    """
    file = Path(path).expanduser()
    if not file.is_file():
        raise FileNotFoundError(f"Prepare endpoint embeddings before training: {file}")
    payload = torch.load(file, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "nex233-features-v1":
        raise ValueError("Unsupported NEX-233 feature artifact format")
    features = payload.get("features")
    rows = payload.get("rows")
    if (
        not isinstance(features, torch.Tensor)
        or features.ndim != 3
        or features.shape[1] != 2
        or not features.is_floating_point()
        or not torch.isfinite(features).all()
        or not isinstance(rows, list)
        or len(rows) != features.shape[0]
    ):
        raise ValueError("Prepared endpoints must be finite [N,2,D] tensors with aligned rows")
    if not rows or not all(
        isinstance(row, dict)
        and isinstance(row.get("split"), str)
        and isinstance(row.get("group_id"), str)
        for row in rows
    ):
        raise ValueError("Prepared rows must contain split and group metadata")
    splits = [row["split"] for row in rows]
    if any(part not in {"train", "val", "test"} for part in splits):
        raise ValueError("Prepared feature splits must be train, val or test")
    if "train" not in splits or "val" not in splits:
        raise ValueError("Prepared features require nonempty train and val partitions")
    settings = payload.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("Prepared features require experiment provenance settings")
    if settings.get("split_protocol") == "within_machine_recording_holdout":
        groups = {row["group_id"] for row in rows}
        if len(groups) != 1 or "test" in splits:
            raise ValueError("Within-machine holdout must contain one machine and no test split")
        if settings.get("machine_group") not in groups:
            raise ValueError("Within-machine artifact must identify its sole machine")
        for key in ("healthy_file", "faulty_file"):
            names = [row.get(key) for row in rows]
            if not all(isinstance(name, str) and name for name in names):
                raise ValueError(f"Within-machine holdout requires {key} provenance")
            if len(set(names)) != len(names):
                raise ValueError(f"Within-machine holdout has duplicate {key} recordings")
    else:
        group_splits: dict[str, str] = {}
        for row in rows:
            group = row["group_id"]
            split = row["split"]
            previous = group_splits.setdefault(group, split)
            if previous != split:
                raise ValueError("Machine identity overlaps train, val or test")
    return cast(dict[str, Any], payload)


@data_source("PreparedMaintenanceEndpointPairs")
class PreparedMaintenanceEndpointPairs(DataSourceDefinition):
    """Load immutable, already normalized/coupled endpoint feature pairs.

    Attributes:
        path: File produced by the NEX-233 offline feature preparation script.
    """

    path: str = Field(min_length=1)

    def build(self) -> NexuDataset:
        """Materialize feature tensors and keep original machine-disjoint splits.

        Returns:
            Dataset of two flattened embedding endpoints and paired labels.
        """
        return _PreparedMaintenanceEndpointRuntime(path=self.path)


class _PreparedMaintenanceEndpointRuntime(NexuDataset):
    """Tensor-backed dataset preserving take/get_split semantics."""

    def __init__(self, path: str) -> None:
        payload = read_prepared_artifact(path)
        self.settings = payload["settings"]
        features = cast(torch.Tensor, payload["features"])
        super().__init__(
            meta=pd.DataFrame.from_records(payload["rows"]),
            label_names=["paired"],
            modality="audio",
            do_split=False,
            data=features,
        )

    def load_item(self, idx: int, row: pd.Series) -> TensorDict:
        """Read matching source and target embeddings from the tensor backing.

        Returns:
            TensorDict with the endpoint names required by the existing flow.
        """
        data = cast(torch.Tensor, self.data)
        features = data[idx]
        return TensorDict(
            {"healthy_spectrum": features[0], "faulty_spectrum": features[1]},
            batch_size=[],
        )
