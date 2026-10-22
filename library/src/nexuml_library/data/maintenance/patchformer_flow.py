"""Frozen PatchFormer DPAN embeddings for native healthy-to-degraded Flow Matching.

Only asset-disjoint C-MAPSS train/validation windows participate in the flow
objective and validation metrics. Official FD001/FD003 test RUL is never used
to prepare endpoints, estimate feature normalization or select checkpoints.
"""

from __future__ import annotations

import hashlib
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
from nexuml_library.data.maintenance.prepared_endpoints import read_prepared_artifact
from nexuml_library.data.maintenance.rul import (
    AutoCMapSSRULWindows,
    read_rul_artifact,
)
from nexuml_library.layers.model.rul_advanced import _PatchFormerModel

_FD001_SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]


def _trusted_patchformer_weights(
    model: _PatchFormerModel,
    checkpoint_path: Path,
) -> str:
    """Load only matching backbone tensors from a trusted NexuML checkpoint.

    Args:
        model: Matching, previously trained PatchFormer backbone.
        checkpoint_path: Local trusted Lightning checkpoint file.

    Returns:
        SHA-256 checksum to bind feature artifacts to the upstream model.

    Raises:
        FileNotFoundError: If the user has not supplied a trained checkpoint.
        ValueError: If the checkpoint has incomplete or incorrect weights.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Train PatchFormer first and set its best checkpoint: {checkpoint_path}"
        )
    # The user explicitly selects this trusted local Lightning checkpoint.
    # Lightning callback/optimizer metadata is not necessarily weights-only safe.
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if not isinstance(payload, dict):
        raise ValueError("Expected a NexuML checkpoint dictionary")
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise ValueError("Expected state_dict in PatchFormer checkpoint")
    matched: dict[str, Tensor] = {}
    for name, expected in model.state_dict().items():
        candidates = [
            value for key, value in state.items()
            if isinstance(key, str)
            and (key == name or key.endswith(f".backbone.{name}"))
        ]
        if (
            len(candidates) != 1
            or not isinstance(candidates[0], Tensor)
            or candidates[0].shape != expected.shape
        ):
            raise ValueError(
                f"PatchFormer checkpoint has missing or ambiguous backbone key {name}"
            )
        matched[name] = candidates[0]
    model.load_state_dict(matched, strict=True)
    sha = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


@data_source("AutoPatchFormerFlowEndpoints")
class AutoPatchFormerFlowEndpoints(DataSourceDefinition):
    """Derive split-safe flow endpoint embeddings from a frozen RUL checkpoint."""

    subset: Literal["FD001", "FD003"] = "FD001"
    seed: int = Field(default=42, ge=0)
    root: str = "data/cmapss"
    checkpoint_path: str = ""
    prepared_rul_path: str = ".experiments/rul/prepared/cmapss-FD001-seed42.pt"
    output_file: str = ".experiments/rul/prepared/patchformer-flow.pt"
    healthy_min_rul: float = Field(default=100.0, ge=0)
    faulty_max_rul: float = Field(default=30.0, ge=0)
    train_pairs: int = Field(default=512, gt=0)
    val_pairs: int = Field(default=128, gt=0)
    embedding_batch_size: int = Field(default=64, gt=0)

    def build(self) -> NexuDataset:
        """Prepare immutable paired endpoints without official test-label leakage.

        Returns:
            Native feature-pair dataset with train and validation splits only.

        Raises:
            ValueError: If endpoint thresholds, checkpoint or dataset are invalid.
            FileNotFoundError: If the frozen PatchFormer checkpoint is absent.
        """
        if self.healthy_min_rul <= self.faulty_max_rul:
            raise ValueError("Healthy RUL threshold must exceed faulty RUL threshold")
        if not self.checkpoint_path:
            raise FileNotFoundError(
                "Set NEXUML_PATCHFORMER_CHECKPOINT to a trained PatchFormer .ckpt"
            )
        sensor_ids = (
            sorted([*_FD001_SENSORS, 6])
            if self.subset == "FD003"
            else _FD001_SENSORS
        )
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
        prepared = read_rul_artifact(self.prepared_rul_path)
        model = _PatchFormerModel(
            length=30,
            channels=len(sensor_ids),
            width=64,
            patch_size=5,
            heads=2,
            dropout=0.1,
        )
        digest = _trusted_patchformer_weights(
            model, Path(self.checkpoint_path).expanduser().resolve()
        )
        model.eval()
        windows = cast(Tensor, prepared["features"])
        labels = cast(Tensor, prepared["targets"]).flatten()
        source_rows = cast(list[dict[str, Any]], prepared["rows"])
        paired_raw: dict[str, tuple[Tensor, Tensor, list[dict[str, str]]]] = {}
        with torch.no_grad():
            for split, limit, offset in (
                ("train", self.train_pairs, 0),
                ("val", self.val_pairs, 1),
            ):
                candidates = [
                    i for i, row in enumerate(source_rows)
                    if row["split"] == split
                ]
                healthy = [
                    i for i in candidates if labels[i] >= self.healthy_min_rul
                ]
                faulty = [
                    i for i in candidates if labels[i] <= self.faulty_max_rul
                ]
                count = min(limit, len(healthy), len(faulty))
                if count < 2:
                    raise ValueError(
                        f"Insufficient {split} PatchFormer flow endpoints "
                        f"(healthy={len(healthy)}, faulty={len(faulty)})"
                    )
                generator = torch.Generator().manual_seed(self.seed + offset)
                h_ids = [
                    healthy[i] for i in
                    torch.randperm(len(healthy), generator=generator)[:count].tolist()
                ]
                f_ids = [
                    faulty[i] for i in
                    torch.randperm(len(faulty), generator=generator)[:count].tolist()
                ]
                def encode(indices: list[int]) -> Tensor:
                    """Encode one fixed endpoint collection without gradient."""
                    parts = [
                        model.forward_features(windows[batch]).mean(dim=1)
                        for batch in (
                            indices[start : start + self.embedding_batch_size]
                            for start in range(0, len(indices), self.embedding_batch_size)
                        )
                    ]
                    return torch.cat(parts)

                meta = [
                    {
                        "split": split,
                        "group_id": f"{split}-pair-{i}",
                        "healthy_asset": str(source_rows[hi]["asset_id"]),
                        "faulty_asset": str(source_rows[fi]["asset_id"]),
                    }
                    for i, (hi, fi) in enumerate(zip(h_ids, f_ids))
                ]
                paired_raw[split] = (encode(h_ids), encode(f_ids), meta)
        train_h, train_f, _ = paired_raw["train"]
        train_embeddings = torch.cat((train_h, train_f), dim=0)
        mean = train_embeddings.mean(dim=0)
        std = train_embeddings.std(dim=0, unbiased=False).clamp_min(1e-6)
        rows: list[dict[str, str]] = []
        chunks: list[Tensor] = []
        for split in ("train", "val"):
            source_h, source_f, items = paired_raw[split]
            standardized = torch.stack(
                ((source_h - mean) / std, (source_f - mean) / std),
                dim=1,
            ).float()
            if not bool(torch.isfinite(standardized).all()):
                raise ValueError("PatchFormer flow endpoints contain nonfinite features")
            chunks.append(standardized)
            rows.extend(items)
        artifact = Path(self.output_file).expanduser().resolve()
        artifact.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "nex233-features-v1",
                "features": torch.cat(chunks),
                "rows": rows,
                "settings": {
                    "split_protocol": "cmapss_asset_disjoint",
                    "dataset": "cmapss",
                    "subset": self.subset,
                    "seed": self.seed,
                    "healthy_min_rul": self.healthy_min_rul,
                    "faulty_max_rul": self.faulty_max_rul,
                    "embedding_dim": int(mean.numel()),
                    "feature_normalization": "train_endpoints_only",
                    "feature_mean": mean.tolist(),
                    "feature_std": std.tolist(),
                    "source_rul_artifact": str(
                        Path(self.prepared_rul_path).resolve()
                    ),
                    "frozen_patchformer_checkpoint_sha256": digest,
                    "held_out_test_engine_labels_used": False,
                },
            },
            artifact,
        )
        read_prepared_artifact(artifact)
        return _PatchFormerFlowRuntime(str(artifact))


class _PatchFormerFlowRuntime(NexuDataset):
    """Load a frozen [healthy,faulty] feature-pair artifact with split metadata."""

    def __init__(self, file: str) -> None:
        data = read_prepared_artifact(file)
        super().__init__(
            meta=pd.DataFrame.from_records(data["rows"]),
            data=cast(Tensor, data["features"]),
            label_names=[],
            modality="sensor",
            do_split=False,
        )

    def take(self, indices: Sequence[int]) -> Self:
        """Preserve the feature pairs when taking an asset-disjoint view.

        Returns:
            NexuML split-safe endpoint subset.
        """
        return super().take(indices)

    def load_item(self, idx: int, row: pd.Series) -> TensorDict:
        """Expose the already standardized frozen endpoint embeddings.

        Returns:
            Healthy and degraded [embedding_dim] TensorDict entries.
        """
        features = cast(Tensor, self.data)[idx]
        return TensorDict(
            {"healthy_spectrum": features[0], "faulty_spectrum": features[1]},
            batch_size=[],
        )
