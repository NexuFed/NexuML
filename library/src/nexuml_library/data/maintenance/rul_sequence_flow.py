"""Split-safe full-token PatchFormer sequence RUL features with optional flow.

The original PatchFormer RUL/FAN head is never called. The same full DPAN
embeddings and train-endpoint normalization feed a matched temporal readout,
with either frozen sequence-flow displacement or identically shaped zeros.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Self, cast

import pandas as pd
import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor

from nexuml.core.components import DataSourceDefinition
from nexuml.core.discovery import data_source
from nexuml.data.dataset import NexuDataset
from nexuml_library.data.maintenance.patchformer_flow import (
    _trusted_patchformer_weights,
    resolve_patchformer_checkpoint,
)
from nexuml_library.data.maintenance.prepared_endpoints import read_prepared_artifact
from nexuml_library.data.maintenance.rul import AutoCMapSSRULWindows, read_rul_artifact
from nexuml_library.evaluation.maintenance.rul_sequence_flow_checkpoint import (
    SequenceArchitecture,
    _file_sha256,
    load_frozen_sequence_flow,
    resolve_sequence_flow_checkpoint,
    sequence_flow_run,
)
from nexuml_library.layers.model.rul_advanced import _PatchFormerModel

_SENSOR_IDS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]


@data_source("AutoPatchFormerSequenceFlowRULFeatures")
class AutoPatchFormerSequenceFlowRULFeatures(DataSourceDefinition):
    """Frozen full-sequence embeddings and optional transported displacement."""

    subset: Literal["FD001", "FD003"] = "FD001"
    seed: int = Field(default=42, ge=0)
    architecture: SequenceArchitecture = "transformer"
    use_flow: bool = True
    root: str = "data/cmapss"
    prepared_rul_path: str = ".experiments/rul/prepared/cmapss-FD001-seed42.pt"
    endpoint_artifact: str = ".experiments/rul/prepared/sequence-endpoints.pt"
    diagnostics_file: str = ".experiments/rul/audits/sequence-rul.geometry.json"
    encoding_batch_size: int = Field(default=32, gt=0)
    encoding_device: Literal["auto", "cpu"] = "auto"

    def build(self) -> NexuDataset:
        """Prepare immutable features for either paired RUL comparison arm.

        Returns:
            Engine-disjoint original train/val/test RUL labels and [L,2D] inputs.

        Raises:
            FileNotFoundError: If upstream frozen embeddings or Flow are missing.
            ValueError: If upstream provenance or dimensionality differs.
        """
        sensors = (
            sorted([*_SENSOR_IDS, 6])
            if self.subset == "FD003" else _SENSOR_IDS
        )
        AutoCMapSSRULWindows(
            root=self.root,
            artifact_path=self.prepared_rul_path,
            subset=self.subset,
            seed=self.seed,
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
        original = read_rul_artifact(self.prepared_rul_path)
        endpoint_file = Path(self.endpoint_artifact).expanduser().resolve()
        endpoints = read_prepared_artifact(endpoint_file)
        info = cast(dict[str, Any], endpoints["settings"])
        if (
            info.get("subset") != self.subset
            or info.get("seed") != self.seed
            or info.get("feature_layout") != "sequence"
            or info.get("sequence_length") != 30
            or info.get("embedding_dim") != 128
            or info.get("feature_normalization") != "train_endpoints_only"
            or info.get("held_out_test_engine_labels_used") is not False
            or Path(str(info.get("source_rul_artifact"))).resolve()
            != Path(self.prepared_rul_path).resolve()
        ):
            raise ValueError(
                "Sequence endpoints do not match the official RUL protocol"
            )
        checkpoint = resolve_patchformer_checkpoint(
            self.subset,
            self.seed,
            checkpoint_path=str(info["frozen_patchformer_checkpoint_path"]),
        )
        encoder = _PatchFormerModel(30, len(sensors), 64, 5, 2, 0.1)
        encoder_hash = _trusted_patchformer_weights(encoder, checkpoint)
        if encoder_hash != info["frozen_patchformer_checkpoint_sha256"]:
            raise ValueError(
                "PatchFormer weights differ from the sequence flow encoder"
            )
        mean = torch.as_tensor(info["feature_mean"], dtype=torch.float32)
        std = torch.as_tensor(info["feature_std"], dtype=torch.float32)
        if (
            mean.shape != (30, 128) or std.shape != (30, 128)
            or not bool(torch.isfinite(mean).all())
            or not bool(torch.isfinite(std).all())
            or not bool((std > 0).all())
        ):
            raise ValueError("Invalid train-endpoint-only sequence normalization")
        flow_hash: str | None = None
        flow = None
        if self.use_flow:
            flow_checkpoint = resolve_sequence_flow_checkpoint(
                self.subset, self.seed, self.architecture
            )
            flow, flow_hash = load_frozen_sequence_flow(
                flow_checkpoint, self.architecture, self.seed
            )
        device = torch.device(
            "cuda"
            if self.encoding_device == "auto" and torch.cuda.is_available()
            else "cpu"
        )
        encoder.eval().requires_grad_(False).to(device)
        if flow is not None:
            flow.eval().requires_grad_(False).to(device)
        mean, std = mean.to(device), std.to(device)
        windows = cast(Tensor, original["features"])
        rows = cast(list[dict[str, Any]], original["rows"])
        batches: list[Tensor] = []
        with torch.no_grad():
            for batch in windows.split(self.encoding_batch_size):
                embedding = encoder.forward_features(batch.to(device))
                normalized = (embedding - mean) / std
                if flow is None:
                    displacement = torch.zeros_like(normalized)
                else:
                    displacement = flow.sample(normalized) - normalized
                inputs = torch.cat((normalized, displacement), dim=-1)
                if not bool(torch.isfinite(inputs).all()):
                    raise ValueError("Nonfinite full-sequence Flow RUL embeddings")
                batches.append(inputs.detach().cpu())
        features = torch.cat(batches)
        if features.shape != (len(rows), 30, 256):
            raise ValueError("Sequence flow RUL inputs must be [N,30,256]")
        report_file = Path(self.diagnostics_file).expanduser().resolve()
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(
            json.dumps(
                {
                    "subset": self.subset,
                    "seed": self.seed,
                    "architecture": self.architecture,
                    "use_flow": self.use_flow,
                    "feature_shape": [30, 256],
                    "frozen_flow_run": (
                        sequence_flow_run(self.subset, self.seed, self.architecture)
                        if self.use_flow else None
                    ),
                    "flow_checkpoint_sha256": flow_hash,
                    "patchformer_upstream_run": checkpoint.parent.name,
                    "patchformer_checkpoint_sha256": encoder_hash,
                    "endpoint_artifact_sha256": _file_sha256(endpoint_file),
                    "embedding_normalization": "matching_training_flow_endpoints",
                    "uses_patchformer_regression_head": False,
                    "test_labels_used_in_feature_fitting": False,
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        return _SequenceFlowRULRuntime(
            features=features,
            targets=cast(Tensor, original["targets"]),
            raw_targets=cast(Tensor, original["raw_targets"]),
            meta=pd.DataFrame.from_records(rows),
        )


class _SequenceFlowRULRuntime(NexuDataset):
    """Native aligned [L,2D] frozen sequence feature and RUL label dataset."""

    def __init__(
        self, *, features: Tensor, targets: Tensor,
        raw_targets: Tensor, meta: pd.DataFrame,
    ) -> None:
        self._rul_targets = targets
        self._rul_raw_targets = raw_targets
        super().__init__(
            meta=meta,
            data=features,
            label_names=[],
            modality="sensor",
            do_split=False,
        )

    def take(self, indices: Sequence[int]) -> Self:
        """Return engine-disjoint positions with exactly aligned RUL targets.

        Returns:
            Split-safe native NexuML dataset.
        """
        selected = list(indices)
        subset = super().take(selected)
        subset._rul_targets = self._rul_targets[selected]
        subset._rul_raw_targets = self._rul_raw_targets[selected]
        return subset

    def load_item(self, idx: int, row: pd.Series) -> TensorDict:
        """Expose frozen full-length sequence without temporal pooling.

        Returns:
            TensorDict with one [30,256] feature sequence.
        """
        return TensorDict(
            {"sequence_rul_features": cast(Tensor, self.data)[idx]},
            batch_size=[],
        )

    def load_labels(self, idx: int, row: pd.Series) -> TensorDict:
        """Expose untouched capped and official raw RUL labels.

        Returns:
            TensorDict containing capped and raw RUL in cycles.
        """
        return TensorDict(
            {"rul": self._rul_targets[idx], "rul_raw": self._rul_raw_targets[idx]},
            batch_size=[],
        )
