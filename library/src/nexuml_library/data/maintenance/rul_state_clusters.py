"""Train-engine-only operating/failure-state clusters for conditioned RUL.

Cluster targets may use an engine's complete TRAIN trajectory. They never
become model inputs: inference uses predicted soft memberships only.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal, cast

import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor

from nexuml.core.discovery import data_source
from nexuml_library.data.maintenance.patchformer_flow import (
    _trusted_patchformer_weights,
    resolve_patchformer_checkpoint,
)
from nexuml_library.data.maintenance.rul import read_rul_artifact
from nexuml_library.data.maintenance.rul_enhanced import (
    AutoPairedCMapSSRULWindows,
    _PairedRULRuntime,
)
from nexuml_library.layers.model.rul_advanced import _PatchFormerModel

CLUSTER_FORMAT = "nex233-state-clusters-v1"
DEFAULT_CLUSTERS = ".experiments/rul/prepared/fd003-state-clusters-seed42.pt"


def _sha256(path: Path) -> str:
    """Hash exact source files for immutable cluster provenance.

    Returns:
        SHA-256 hexadecimal string.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kmeans(features: Tensor, clusters: int, seed: int, iterations: int = 40) -> tuple[Tensor, Tensor]:
    """Fit deterministic nonempty-centroid K-means on training-only points.

    Returns:
        Cluster centroids and per-point hard assignments.

    Raises:
        ValueError: If there are too few finite observations.
    """
    if features.ndim != 2 or not bool(torch.isfinite(features).all()):
        raise ValueError("K-means needs finite [engines, features] inputs")
    if clusters < 2 or len(features) < clusters:
        raise ValueError("K-means needs at least two clusters and enough training engines")
    x = features.detach().cpu().float()
    # Deterministic farthest-first initialization, with seeded first member.
    start = int(torch.randint(len(x), (1,), generator=torch.Generator().manual_seed(seed)))
    selected = [start]
    for _ in range(1, clusters):
        distances = torch.cdist(x, x[selected]).square().min(dim=1).values
        distances[selected] = -1
        selected.append(int(distances.argmax()))
    centroids = x[selected].clone()
    for _ in range(iterations):
        distances = torch.cdist(x, centroids).square()
        assignment = distances.argmin(dim=1)
        updated = centroids.clone()
        for index in range(clusters):
            assigned = x[assignment == index]
            if len(assigned):
                updated[index] = assigned.mean(dim=0)
        if torch.allclose(updated, centroids, atol=1e-6, rtol=0):
            break
        centroids = updated
    assignment = torch.cdist(x, centroids).square().argmin(dim=1)
    if len(assignment.unique()) != clusters:
        raise ValueError("Degenerate K-means: empty cluster; reduce cluster count")
    return centroids, assignment


@torch.inference_mode()
def prepare_state_clusters(
    prepared_path: str = ".experiments/rul/prepared/cmapss-FD003-seed42-controlled-w30-s42.pt",
    output_file: str = DEFAULT_CLUSTERS,
    *,
    clusters: int = 2,
    seed: int = 42,
    endpoint_windows: int = 3,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Fit working/failure prototypes from train-engine prefixes/endpoints.

    Each training engine supplies an early and a last-run-to-failure embedding.
    The frozen PatchFormer is shared with the original CfC reference. Official
    test engines and validation engines are never inspected for fitting.

    Returns:
        Metadata for the persisted immutable cluster artifact.

    Raises:
        ValueError: For unexpected splits, dimensions or incompatible settings.
        FileExistsError: To avoid silently changing an experiment's pseudo-labels.
    """
    source = Path(prepared_path).expanduser().resolve()
    target = Path(output_file).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite cluster artifact: {target}")
    if clusters < 2 or endpoint_windows < 1 or batch_size < 1:
        raise ValueError("Invalid cluster preparation configuration")
    payload = read_rul_artifact(source)
    settings = cast(dict[str, Any], payload["settings"])
    if (
        settings.get("subset") != "FD003"
        or settings.get("seed") != 42
        or settings.get("window") != 30
        or settings.get("sensor_ids") != sorted([2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21])
        or settings.get("train_full_windows_only") is not True
    ):
        raise ValueError("State clustering requires the unchanged FD003 controlled protocol")
    rows = cast(list[dict[str, Any]], payload["rows"])
    windows = cast(Tensor, payload["features"])
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        if row["split"] == "train":
            grouped.setdefault(str(row["asset_id"]), []).append(index)
    if len(grouped) < clusters:
        raise ValueError("Too few training engines for cluster preparation")
    indices: list[int] = []
    chosen: list[tuple[str, list[int], list[int]]] = []
    for asset, positions in sorted(grouped.items()):
        positions.sort(key=lambda i: int(rows[i]["step"]))
        early, late = positions[:endpoint_windows], positions[-endpoint_windows:]
        if len(early) != endpoint_windows or len(late) != endpoint_windows:
            raise ValueError(f"Insufficient windows for training asset {asset}")
        chosen.append((asset, early, late))
        indices.extend([*early, *late])
    backbone = _PatchFormerModel(30, 15, 64, 5, 2, 0.1)
    checkpoint = resolve_patchformer_checkpoint("FD003", 42)
    encoder_hash = _trusted_patchformer_weights(backbone, checkpoint)
    backbone.eval().requires_grad_(False)
    embeddings: list[Tensor] = []
    for batch in windows[indices].split(batch_size):
        embeddings.append(backbone.forward_features(batch).mean(dim=1).cpu())
    vectors = torch.cat(embeddings)
    per_engine = vectors.reshape(len(chosen), 2, endpoint_windows, 128).mean(dim=2)
    working, working_ids = kmeans(per_engine[:, 0], clusters, seed)
    failure, failure_ids = kmeans(per_engine[:, 1], clusters, seed + 1)
    assignments = {
        asset: [int(working_ids[i]), int(failure_ids[i])]
        for i, (asset, _, _) in enumerate(chosen)
    }
    output = {
        "format": CLUSTER_FORMAT,
        "subset": "FD003",
        "seed": 42,
        "cluster_count": clusters,
        "endpoint_windows": endpoint_windows,
        "working_centroids": working,
        "failure_centroids": failure,
        "training_assignments": assignments,
        "prepared_rul_sha256": _sha256(source),
        "patchformer_sha256": encoder_hash,
        "training_engines": sorted(grouped),
        "uses_val_or_test_embeddings": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, target)
    return {"path": str(target), "engines": len(grouped), "clusters": clusters}


def load_state_clusters(path: str, prepared_path: str) -> dict[str, Any]:
    """Validate the saved train-only cluster artifact and its source digest.

    Returns:
        Verified tensors and engine assignments.

    Raises:
        ValueError: If clustering provenance does not match the prepared RUL data.
    """
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or artifact.get("format") != CLUSTER_FORMAT:
        raise ValueError("Invalid state-cluster artifact format")
    count = artifact.get("cluster_count")
    assignments = artifact.get("training_assignments")
    if (
        artifact.get("subset") != "FD003"
        or artifact.get("seed") != 42
        or artifact.get("uses_val_or_test_embeddings") is not False
        or artifact.get("prepared_rul_sha256") != _sha256(Path(prepared_path))
        or not isinstance(count, int)
        or count < 2
        or not isinstance(assignments, dict)
        or not assignments
    ):
        raise ValueError("State clusters have incompatible training provenance")
    for key in ("working_centroids", "failure_centroids"):
        centroids = artifact.get(key)
        if not isinstance(centroids, Tensor) or centroids.shape != (count, 128):
            raise ValueError("Invalid state-cluster centroid dimensions")
        if not bool(torch.isfinite(centroids).all()):
            raise ValueError("Nonfinite cluster centroid")
    if any(
        not isinstance(ids, list)
        or len(ids) != 2
        or any(not isinstance(i, int) or not 0 <= i < count for i in ids)
        for ids in assignments.values()
    ):
        raise ValueError("Invalid engine cluster identities")
    return cast(dict[str, Any], artifact)


@data_source("AutoClusteredCMapSSRULWindows")
class AutoClusteredCMapSSRULWindows(AutoPairedCMapSSRULWindows):
    """Attach teacher cluster IDs to TRAIN labels only, never observable inputs."""

    cluster_artifact: str = Field(default=DEFAULT_CLUSTERS, min_length=1)

    def build(self) -> _ClusteredRULRuntime:
        """Materialize the same FD003 windows with privileged train-only labels.

        Returns:
            Tensor dataset with masked cluster teacher labels in nontrain splits.

        Raises:
            ValueError: If teacher labels could leak validation/test information.
        """
        if self.subset != "FD003" or self.seed != 42:
            raise ValueError("State clusters are defined on FD003 seed-42 training")
        base = super().build()
        artifact = load_state_clusters(self.cluster_artifact, self.artifact_path)
        ids = cast(dict[str, list[int]], artifact["training_assignments"])
        meta = base.meta
        labels = torch.full((len(meta), 2), -1, dtype=torch.long)
        for i, row in meta.reset_index(drop=True).iterrows():
            asset = str(row["asset_id"])
            if row["split"] == "train":
                if asset not in ids:
                    raise ValueError(f"Missing cluster identity for training engine {asset}")
                labels[i] = torch.tensor(ids[asset], dtype=torch.long)
            elif asset in ids:
                raise ValueError("Clustering leaked train-engine identity into held-out split")
        return _ClusteredRULRuntime(
            features=cast(Tensor, base.data),
            targets=base._target,
            raw=base._raw,
            rows=meta.to_dict(orient="records"),
            memberships=labels,
        )


class _ClusteredRULRuntime(_PairedRULRuntime):
    """Keep oracle assignments solely in training labels with split-safe slicing."""

    def __init__(
        self,
        *,
        features: Tensor,
        targets: Tensor,
        raw: Tensor,
        rows: list[dict[str, Any]],
        memberships: Tensor,
    ) -> None:
        super().__init__(features=features, targets=targets, raw=raw, rows=rows)
        self._memberships = memberships

    def take(self, indices: list[int]) -> _ClusteredRULRuntime:
        """Keep cluster teachers aligned under engine-disjoint split selection.

        Returns:
            Split subset with corresponding membership labels.
        """
        subset = super().take(indices)
        subset._memberships = self._memberships[indices]
        return subset

    def load_labels(self, idx: int, row: Any) -> TensorDict:
        """Expose teacher identities in y, not x.

        Returns:
            RUL and cluster targets; -1 denotes unavailable oracle labels.
        """
        labels = super().load_labels(idx, row)
        labels["working_cluster"] = self._memberships[idx, 0].reshape(1)
        labels["failure_cluster"] = self._memberships[idx, 1].reshape(1)
        return labels
