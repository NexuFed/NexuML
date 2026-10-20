"""Immutable healthy-reference FISHER datasets for cross-machine Flow Matching.

Each query's conditioning set contains exclusively separate healthy support
recordings from the same known machine. A leave-one-machine-out assessment
artifact is separate from the training datasource.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor

from nexuml.core.components import DataSourceDefinition
from nexuml.core.discovery import data_source
from nexuml.data.dataset import NexuDataset


def read_healthy_reference_training(path: str | Path) -> dict[str, Any]:
    """Validate a reference-conditioned train/validation prepared artifact.

    Returns:
        Train-only support sets and query endpoint tensors, with provenance.

    Raises:
        ValueError: If tensor shapes, split roles, or support/query identities
            violate the conditioning and validation isolation contract.
    """
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "nex233-healthy-context-train-v1":
        raise ValueError("Invalid healthy-reference training artifact")
    features = payload.get("features")
    references = payload.get("references")
    rows = payload.get("rows")
    names = payload.get("machine_groups")
    support_files = payload.get("support_healthy_files")
    settings = payload.get("settings")
    if (
        not isinstance(features, Tensor)
        or features.ndim != 3
        or features.shape[1] != 2
        or not features.is_floating_point()
        or not bool(torch.isfinite(features).all())
        or not isinstance(references, Tensor)
        or references.ndim != 3
        or not references.is_floating_point()
        or not bool(torch.isfinite(references).all())
        or features.shape[0] == 0
        or not isinstance(rows, list)
        or len(rows) != len(features)
        or not isinstance(names, list)
        or len(names) < 2
        or len(set(names)) != len(names)
        or not isinstance(support_files, list)
        or len(support_files) != len(names)
        or not isinstance(settings, dict)
    ):
        raise ValueError("Malformed or nonfinite healthy-reference training data")
    if (
        references.shape[0] != len(names)
        or references.shape[1] == 0
        or references.shape[2] != features.shape[2]
    ):
        raise ValueError("Reference set must be [machines,K,feature_dim]")
    query_files: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("split") not in {"train", "val"}:
            raise ValueError("Training artifact must contain only train and val queries")
        if row.get("group_id") not in names:
            raise ValueError("Query machine is absent from reference sets")
        index = row.get("reference_index")
        if (
            not isinstance(index, int)
            or not 0 <= index < len(names)
            or names[index] != row["group_id"]
        ):
            raise ValueError("Query references do not match their machine")
        for key in ("healthy_file", "faulty_file"):
            filename = row.get(key)
            if not isinstance(filename, str) or not filename or filename in query_files:
                raise ValueError("Query recordings must have distinct file identities")
            query_files.add(filename)
    if {row["split"] for row in rows} != {"train", "val"}:
        raise ValueError("Require both nonempty training and validation")
    all_support: list[str] = []
    for item in support_files:
        if (
            not isinstance(item, list)
            or len(item) != references.shape[1]
            or not all(isinstance(name, str) and name for name in item)
        ):
            raise ValueError("Invalid healthy reference support provenance")
        all_support.extend(item)
    if len(all_support) != len(set(all_support)) or not query_files.isdisjoint(all_support):
        raise ValueError("Healthy reference recordings overlap query recordings")
    if settings.get("scaling_fit") != "training_machine_healthy_only":
        raise ValueError("Conditional flow requires healthy-training-only scaling")
    return cast(dict[str, Any], payload)


def read_healthy_reference_assessment(path: str | Path) -> dict[str, Any]:
    """Validate the distinct unseen-machine post-training evaluation artifact.

    Returns:
        Healthy-only reference set and disjoint assessment endpoint pairs.

    Raises:
        ValueError: If reference/assessment samples overlap or are invalid.
    """
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "nex233-healthy-context-eval-v1":
        raise ValueError("Invalid held-out machine assessment artifact")
    references = payload.get("references")
    features = payload.get("features")
    rows = payload.get("rows")
    support_files = payload.get("support_healthy_files")
    if (
        not isinstance(references, Tensor)
        or references.ndim != 2
        or not isinstance(features, Tensor)
        or features.ndim != 3
        or features.shape[1] != 2
        or features.shape[2] != references.shape[-1]
        or not features.is_floating_point()
        or not references.is_floating_point()
        or not bool(torch.isfinite(references).all())
        or not bool(torch.isfinite(features).all())
        or len(features) == 0
        or len(references) == 0
        or not isinstance(rows, list)
        or len(rows) != len(features)
        or not isinstance(support_files, list)
    ):
        raise ValueError("Malformed held-out assessment artifact")
    identities = [row.get(key) for row in rows for key in ("healthy_file", "faulty_file")]
    if not all(isinstance(name, str) and name for name in identities):
        raise ValueError("Assessment requires recording identities")
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate held-out assessment file")
    group = payload.get("heldout_machine")
    if (
        not isinstance(group, str)
        or not group
        or any(
            row.get("group_id") != group or row.get("split") != "assessment"
            for row in rows
        )
    ):
        raise ValueError("Assessment rows must all belong to the held-out machine")
    if (
        len(support_files) != len(references)
        or len(support_files) != len(set(support_files))
        or not set(support_files).isdisjoint(identities)
    ):
        raise ValueError("Held-out healthy support and assessment queries overlap")
    return cast(dict[str, Any], payload)


@data_source("HealthyReferenceMaintenancePairs")
class HealthyReferenceMaintenancePairs(DataSourceDefinition):
    """Expose prepared healthy reference sets through ordinary NexuML routing."""

    path: str = Field(min_length=1)

    def build(self) -> NexuDataset:
        """Return a split-preserving TensorDict-backed training dataset."""
        return _HealthyReferenceMaintenanceRuntime(path=self.path)


class _HealthyReferenceMaintenanceRuntime(NexuDataset):
    """Map each query to its machine's immutable healthy support recordings."""

    def __init__(self, path: str) -> None:
        artifact = read_healthy_reference_training(path)
        self.references = cast(Tensor, artifact["references"])
        features = cast(Tensor, artifact["features"])
        super().__init__(
            meta=pd.DataFrame.from_records(artifact["rows"]),
            label_names=["paired"],
            modality="audio",
            do_split=False,
            data=features,
        )

    def load_item(self, idx: int, row: pd.Series) -> TensorDict:
        """Load query endpoints and the support set of the same machine.

        Returns:
            TensorDict of healthy, faulty, and reference feature tensors.
        """
        data = cast(Tensor, self.data)[idx]
        index = int(row["reference_index"])
        return TensorDict(
            {
                "healthy_spectrum": data[0],
                "faulty_spectrum": data[1],
                "healthy_reference": self.references[index],
            },
            batch_size=[],
        )
