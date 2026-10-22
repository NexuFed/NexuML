"""NexuML datasource for frozen PatchFormer + Flow Matching RUL calibration.

All expensive pretrained features are computed once at data preparation.
The official test targets remain labels only, never inputs to calibrators,
feature transforms, reference selection or checkpoint selection.
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
from nexuml_library.data.maintenance.rul import AutoCMapSSRULWindows, read_rul_artifact
from nexuml_library.evaluation.maintenance.rul_flow_geometry import (
    healthy_reference_trajectories,
    load_frozen_latent_flow,
    pearson_flow_progress,
    project_flow_time,
    resolve_latent_flow_checkpoint,
)
from nexuml_library.layers.model.rul_advanced import _PatchFormerModel


@data_source("AutoPatchFormerFlowRULFeatures")
class AutoPatchFormerFlowRULFeatures(DataSourceDefinition):
    """Prepare one common, label-safe flow-time and baseline prediction dataset."""

    subset: Literal["FD001", "FD003"] = "FD001"
    seed: int = Field(default=42, ge=0)
    root: str = "data/cmapss"
    prepared_rul_path: str = ".experiments/rul/prepared/cmapss-FD001-seed42.pt"
    endpoint_artifact: str = ".experiments/rul/prepared/patchformer-flow.pt"
    diagnostics_file: str = ".experiments/rul/audits/patchformer-flowtime.json"
    encoding_batch_size: int = Field(default=64, gt=0)
    reference_count: int = Field(default=8, ge=1, le=32)
    feature_mode: Literal["baseline_only", "with_flow"] = "with_flow"

    def build(self) -> NexuDataset:
        """Extract frozen model outputs and project on training health trajectories.

        Returns:
            Split-preserving, original-index-aligned RUL calibration dataset.

        Raises:
            FileNotFoundError: If trained PatchFormer or Flow checkpoints are absent.
            ValueError: If feature provenance is inconsistent or nonfinite.
        """
        sensor_ids = [
            2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21
        ]
        if self.subset == "FD003":
            sensor_ids = sorted([*sensor_ids, 6])
        source = AutoCMapSSRULWindows(
            root=self.root,
            artifact_path=self.prepared_rul_path,
            subset=self.subset,
            seed=self.seed,
            window=30,
            stride=1,
            rul_cap=130.0,
            sensor_ids=sensor_ids,
            normalization="zscore",
            padding_mode="zero",
            train_full_windows_only=True,
            normalize_targets=False,
            download=True,
        )
        source.build()
        source_payload = read_rul_artifact(self.prepared_rul_path)
        if self.feature_mode == "baseline_only":
            return self._build_baseline_only(source_payload, sensor_ids)
        endpoint_file = Path(self.endpoint_artifact).expanduser().resolve()
        if not endpoint_file.is_file():
            raise FileNotFoundError(
                f"Train maintenance-rul-patchformer-flow-{self.subset.lower()}-"
                f"s{self.seed} before RUL calibration: {endpoint_file}"
            )
        from nexuml_library.data.maintenance.prepared_endpoints import (
            read_prepared_artifact,
        )

        endpoint = read_prepared_artifact(endpoint_file)
        provenance = cast(dict[str, Any], endpoint["settings"])
        if (
            provenance.get("subset") != self.subset
            or provenance.get("seed") != self.seed
            or provenance.get("feature_normalization") != "train_endpoints_only"
            or provenance.get("held_out_test_engine_labels_used") is not False
            or Path(str(provenance.get("source_rul_artifact"))).resolve()
            != Path(self.prepared_rul_path).resolve()
        ):
            raise ValueError("Pretrained flow endpoints do not match the RUL split")
        patch_path = str(provenance["frozen_patchformer_checkpoint_path"])
        checkpoint = resolve_patchformer_checkpoint(
            self.subset, self.seed, checkpoint_path=patch_path
        )
        patch = _PatchFormerModel(30, len(sensor_ids), 64, 5, 2, 0.1)
        patch_digest = _trusted_patchformer_weights(patch, checkpoint)
        if patch_digest != provenance["frozen_patchformer_checkpoint_sha256"]:
            raise ValueError("PatchFormer backbone changed since the flow was trained")
        patch.eval()
        flow_checkpoint = resolve_latent_flow_checkpoint(self.subset, self.seed)
        flow, flow_digest = load_frozen_latent_flow(
            flow_checkpoint, seed=self.seed
        )
        reference_rows = cast(list[dict[str, Any]], endpoint["rows"])
        train_indices = [
            i for i, row in enumerate(reference_rows) if row["split"] == "train"
        ]
        endpoints = cast(Tensor, endpoint["features"])
        if not train_indices:
            raise ValueError("Frozen flow has no training healthy references")
        healthy = endpoints[train_indices, 0]
        trajectories = healthy_reference_trajectories(
            flow, healthy, references=self.reference_count
        )
        mean = torch.as_tensor(provenance["feature_mean"], dtype=torch.float32)
        std = torch.as_tensor(provenance["feature_std"], dtype=torch.float32)
        if (
            mean.shape != (128,)
            or std.shape != (128,)
            or not bool(torch.isfinite(mean).all())
            or not bool(torch.isfinite(std).all())
            or not bool((std > 0).all())
        ):
            raise ValueError("Invalid train-only PatchFormer flow feature normalization")
        original = cast(Tensor, source_payload["features"])
        rows = cast(list[dict[str, Any]], source_payload["rows"])
        features: list[Tensor] = []
        with torch.no_grad():
            for sequence in original.split(self.encoding_batch_size):
                raw = patch.forward_features(sequence).mean(dim=1)
                baseline = patch(sequence)
                latent = (raw - mean) / std
                coordinates = project_flow_time(latent, trajectories)
                batch = torch.cat((coordinates, baseline), dim=1)
                if not bool(torch.isfinite(batch).all()):
                    raise ValueError("Nonfinite frozen PatchFormer/Flow RUL features")
                features.append(batch)
        prepared_features = torch.cat(features, dim=0)
        if prepared_features.shape != (len(rows), 3):
            raise ValueError("Prepared flow RUL features disagree with source rows")
        val_indices = [
            i for i, row in enumerate(rows) if row["split"] == "val"
        ]
        training_indices = [
            i for i, row in enumerate(rows) if row["split"] == "train"
        ]
        # Geometry diagnostics are descriptive and use no official test targets.
        labels = cast(Tensor, source_payload["targets"])
        val_asset_indices: dict[str, list[int]] = {}
        for index in val_indices:
            asset = str(rows[index]["asset_id"])
            val_asset_indices.setdefault(asset, []).append(index)
        progression = [
            pearson_flow_progress(
                prepared_features[indices, 0],
                torch.tensor(
                    [int(rows[index]["step"]) for index in indices],
                    dtype=torch.float32,
                ),
            )
            for indices in val_asset_indices.values()
            if len(indices) > 1
        ]
        valid_progression = [value for value in progression if value is not None]
        geometry = {
            "subset": self.subset,
            "seed": self.seed,
            "flow_checkpoint_sha256": flow_digest,
            "patchformer_checkpoint_sha256": patch_digest,
            "patchformer_upstream_run": checkpoint.parent.name,
            "reference_count": int(trajectories.shape[0]),
            "selection": "training_healthy_endpoints_only",
            "val/flow_time_rul_pearson": pearson_flow_progress(
                prepared_features[val_indices, 0], labels[val_indices]
            ),
            "train/flow_time_rul_pearson": pearson_flow_progress(
                prepared_features[training_indices, 0],
                labels[training_indices],
            ),
            "val/engine_progression_pearson_mean": (
                sum(valid_progression) / len(valid_progression)
                if valid_progression else None
            ),
            "val/engine_progression_positive_fraction": (
                sum(value > 0 for value in valid_progression)
                / len(valid_progression)
                if valid_progression else None
            ),
            "val/engine_progression_count": len(valid_progression),
            "test_labels_used_in_feature_fitting": False,
        }
        destination = Path(self.diagnostics_file).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(geometry, indent=2) + "\n", encoding="utf-8"
        )
        return _FlowRULFeatureRuntime(
            features=prepared_features,
            targets=cast(Tensor, source_payload["targets"]),
            raw_targets=cast(Tensor, source_payload["raw_targets"]),
            meta=pd.DataFrame.from_records(rows),
        )


    def _build_baseline_only(
        self, payload: dict[str, Any], sensor_ids: list[int]
    ) -> NexuDataset:
        """Cache frozen PatchFormer predictions without loading a Flow model.

        The endpoint manifest is consulted only when one already exists, to
        reuse the *exact* upstream PatchFormer of the flow/hybrid arms. It is
        not required for baseline-only training or validation.

        Args:
            payload: Prepared immutable original C-MAPSS window observations.
            sensor_ids: Paper-matched FD001/FD003 sensor identifiers.

        Returns:
            Split-safe baseline-only calibration dataset with zeroed flow inputs.

        Raises:
            ValueError: If a present flow endpoint references a different model.
        """
        checkpoint_hint: str | None = None
        expected_digest: str | None = None
        endpoint_file = Path(self.endpoint_artifact).expanduser()
        if endpoint_file.is_file():
            from nexuml_library.data.maintenance.prepared_endpoints import (
                read_prepared_artifact,
            )

            endpoint = read_prepared_artifact(endpoint_file)
            settings = cast(dict[str, Any], endpoint["settings"])
            if (
                settings.get("subset") != self.subset
                or settings.get("seed") != self.seed
                or settings.get("held_out_test_engine_labels_used") is not False
                or Path(str(settings.get("source_rul_artifact"))).resolve()
                != Path(self.prepared_rul_path).resolve()
            ):
                raise ValueError("Baseline-only endpoint provenance mismatch")
            checkpoint_hint = str(settings["frozen_patchformer_checkpoint_path"])
            expected_digest = str(settings["frozen_patchformer_checkpoint_sha256"])
        checkpoint = resolve_patchformer_checkpoint(
            self.subset, self.seed, checkpoint_path=checkpoint_hint
        )
        model = _PatchFormerModel(30, len(sensor_ids), 64, 5, 2, 0.1)
        digest = _trusted_patchformer_weights(model, checkpoint)
        if expected_digest is not None and digest != expected_digest:
            raise ValueError("Baseline-only checkpoint differs from flow's PatchFormer")
        model.eval()
        windows = cast(Tensor, payload["features"])
        rows = cast(list[dict[str, Any]], payload["rows"])
        predictions: list[Tensor] = []
        with torch.no_grad():
            for batch in windows.split(self.encoding_batch_size):
                predictions.append(model(batch).detach().float())
        baseline = torch.cat(predictions, dim=0)
        features = torch.cat(
            (torch.zeros((len(windows), 2), dtype=baseline.dtype), baseline),
            dim=1,
        )
        if not bool(torch.isfinite(features).all()):
            raise ValueError("Nonfinite baseline PatchFormer predictions")
        path = Path(self.diagnostics_file).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "subset": self.subset,
                    "seed": self.seed,
                    "feature_mode": "baseline_only",
                    "selection": "frozen_patchformer_without_flow",
                    "patchformer_upstream_run": checkpoint.parent.name,
                    "patchformer_checkpoint_sha256": digest,
                    "flow_checkpoint_sha256": None,
                    "reference_count": 0,
                    "test_labels_used_in_feature_fitting": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return _FlowRULFeatureRuntime(
            features=features,
            targets=cast(Tensor, payload["targets"]),
            raw_targets=cast(Tensor, payload["raw_targets"]),
            meta=pd.DataFrame.from_records(rows),
        )


class _FlowRULFeatureRuntime(NexuDataset):
    """Same row order and official split indices as the original RUL artifact."""

    def __init__(
        self, *, features: Tensor, targets: Tensor,
        raw_targets: Tensor, meta: pd.DataFrame,
    ) -> None:
        self._rul_targets = targets
        self._rul_raw_targets = raw_targets
        super().__init__(
            meta=meta, data=features,
            label_names=[], modality="sensor", do_split=False,
        )

    def take(self, indices: Sequence[int]) -> Self:
        """Create a native split without losing label alignment.

        Returns:
            A split subset with the corresponding capped/raw RUL labels.
        """
        positions = list(indices)
        subset = super().take(positions)
        subset._rul_targets = self._rul_targets[positions]
        subset._rul_raw_targets = self._rul_raw_targets[positions]
        return subset

    def load_item(self, idx: int, row: pd.Series) -> TensorDict:
        """Read frozen flow-time, distance and baseline prediction features.

        Returns:
            Batch-free [3] features for the calibrated RUL model.
        """
        return TensorDict(
            {"flow_rul_features": cast(Tensor, self.data)[idx]},
            batch_size=[],
        )

    def load_labels(self, idx: int, row: pd.Series) -> TensorDict:
        """Read original RUL labels exclusively via the supervised label path.

        Returns:
            Capped RUL and raw official RUL targets in cycles.
        """
        return TensorDict(
            {"rul": self._rul_targets[idx], "rul_raw": self._rul_raw_targets[idx]},
            batch_size=[],
        )
