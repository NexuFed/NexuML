"""Train-engine-only healthy/failure embedding clusters for conditional RUL.

Every cluster target is created from an upstream TRAIN engine. Validation and
test engines have -1 auxiliary targets, never a future-derived condition.
The CfC head consumes its own predicted SOFT cluster probabilities at both
training and inference, not these oracle labels.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Self, cast

import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor

from nexuml.core.discovery import data_source
from nexuml.data.dataset import NexuDataset
from nexuml_library.data.maintenance.patchformer_flow import (
    _trusted_patchformer_weights,
    resolve_patchformer_checkpoint,
)
from nexuml_library.data.maintenance.prepared_endpoints import read_prepared_artifact
from nexuml_library.data.maintenance.rul import read_rul_artifact
from nexuml_library.data.maintenance.rul_enhanced import (
    AutoPairedCMapSSRULWindows,
    _PairedRULRuntime,
)
from nexuml_library.evaluation.maintenance.rul_sequence_flow_checkpoint import (
    sequence_flow_run,
)
from nexuml_library.layers.model.rul_advanced import _PatchFormerModel

_CLUSTER_FORMAT = "nex233-cfc-conditional-clusters-v1"
_SENSORS_FD003 = 15


def _sha256(path: Path) -> str:
    """Hash one immutable local artifact.

    Returns:
        Lowercase SHA-256 hex string.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _kmeans(embeddings: Tensor, clusters: int, seed: int) -> tuple[Tensor, Tensor]:
    """Deterministically fit engine-balanced squared-Euclidean K-means.

    Each row represents ONE training engine, not a correlated sensor window.

    Returns:
        Cluster centroids and per-engine integer assignments.

    Raises:
        ValueError: If clusters are unsupported or a cluster is empty.
    """
    if embeddings.ndim != 2 or not 1 < clusters <= embeddings.shape[0]:
        raise ValueError("K-means requires 2..N training-engine embeddings")
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError("Nonfinite training-engine embeddings")
    data = embeddings.to(dtype=torch.float32, device="cpu")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    first = int(torch.randint(len(data), (1,), generator=generator))
    chosen = [first]
    while len(chosen) < clusters:
        distances = torch.cdist(data, data[chosen]).square().min(dim=1).values
        distances[chosen] = -1.0
        nxt = int(distances.argmax())
        if nxt in chosen:
            raise ValueError("Degenerate K-means initialization")
        chosen.append(nxt)
    centroids = data[chosen].clone()
    for _ in range(80):
        labels = torch.cdist(data, centroids).square().argmin(dim=1)
        updated = []
        for cluster in range(clusters):
            members = data[labels == cluster]
            if len(members) == 0:
                raise ValueError("Empty K-means cluster; lower the cluster count")
            updated.append(members.mean(dim=0))
        next_centroids = torch.stack(updated)
        if bool(torch.allclose(centroids, next_centroids, atol=1e-6, rtol=0)):
            break
        centroids = next_centroids
    labels = torch.cdist(data, centroids).square().argmin(dim=1)
    if len(torch.unique(labels)) != clusters:
        raise ValueError("K-means collapsed to fewer clusters than requested")
    return centroids, labels


def read_cluster_artifact(
    file: str | Path,
    *,
    subset: str = "FD003",
    seed: int = 42,
    working_clusters: int = 3,
    failure_clusters: int = 2,
    prepared_sha256: str | None = None,
    encoder_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate train-only cluster prototype provenance and tensor shapes.

    Returns:
        Trusted artifact with standardized cluster centers and TRAIN-only IDs.

    Raises:
        ValueError: If shapes, identities or source fingerprints disagree.
        FileNotFoundError: If the artifact has not been prepared.
    """
    path = Path(file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Prepare CfC clusters before fitting: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != _CLUSTER_FORMAT:
        raise ValueError("Unsupported conditional CfC cluster artifact")
    settings = payload.get("settings")
    if not isinstance(settings, dict) or (
        settings.get("subset") != subset
        or settings.get("seed") != seed
        or settings.get("working_clusters") != working_clusters
        or settings.get("failure_clusters") != failure_clusters
        or settings.get("training_engines_only") is not True
        or settings.get("uses_test_or_val_endpoints") is not False
        or (prepared_sha256 is not None
            and settings.get("prepared_sha256") != prepared_sha256)
        or (encoder_sha256 is not None
            and settings.get("encoder_sha256") != encoder_sha256)
    ):
        raise ValueError("Cluster artifact provenance does not match this RUL study")
    for prefix, count in (("working", working_clusters), ("failure", failure_clusters)):
        centers = payload.get(f"{prefix}_centroids")
        offset = payload.get(f"{prefix}_mean")
        scale = payload.get(f"{prefix}_std")
        if (
            not isinstance(centers, Tensor) or centers.shape != (count, 128)
            or not isinstance(offset, Tensor) or offset.shape != (128,)
            or not isinstance(scale, Tensor) or scale.shape != (128,)
            or not all(bool(torch.isfinite(t).all()) for t in (centers, offset, scale))
            or not bool((scale > 0).all())
        ):
            raise ValueError(f"Invalid {prefix} cluster statistics")
    assigned = payload.get("train_engine_assignments")
    if not isinstance(assigned, dict) or len(assigned) < max(working_clusters, failure_clusters):
        raise ValueError("Missing engine-level train cluster assignments")
    for name, pair in assigned.items():
        if (
            not isinstance(name, str)
            or not name.startswith("train:FD003:")
            or not isinstance(pair, list) or len(pair) != 2
            or not all(isinstance(i, int) and not isinstance(i, bool) for i in pair)
            or not 0 <= pair[0] < working_clusters
            or not 0 <= pair[1] < failure_clusters
        ):
            raise ValueError("Invalid or non-training cluster assignment")
    return cast(dict[str, Any], payload)


def prepare_cluster_artifact(
    *,
    prepared_path: str,
    output_path: str,
    seed: int = 42,
    working_clusters: int = 3,
    failure_clusters: int = 2,
    endpoint_windows: int = 4,
    encoding_batch_size: int = 32,
) -> dict[str, Any]:
    """Fit engine-balanced cluster prototypes from train-engine endpoints only.

    Uses first/last full 30-cycle windows of each TRAIN engine; neither
    validation nor official test-engine embeddings are ever used to fit
    centroids, standardization or assignments.

    Returns:
        Verified, cached, deterministic cluster artifact.

    Raises:
        ValueError: If an upstream artifact or engine history is incompatible.
    """
    prepared_file = Path(prepared_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    prepared_sha = _sha256(prepared_file)
    if output.is_file():
        return read_cluster_artifact(
            output, seed=seed, working_clusters=working_clusters,
            failure_clusters=failure_clusters, prepared_sha256=prepared_sha,
        )
    upstream = resolve_patchformer_checkpoint("FD003", seed)
    frozen_encoder = _PatchFormerModel(30, _SENSORS_FD003, 64, 5, 2, 0.1)
    encoder_hash = _trusted_patchformer_weights(frozen_encoder, upstream)
    endpoint = Path(
        f".experiments/rul/prepared/{sequence_flow_run('FD003', seed, 'cfc')}-endpoints.pt"
    )
    endpoint_settings = read_prepared_artifact(endpoint)["settings"]
    if (
        endpoint_settings.get("subset") != "FD003"
        or endpoint_settings.get("seed") != seed
        or endpoint_settings.get("feature_normalization") != "train_endpoints_only"
        or endpoint_settings.get("frozen_patchformer_checkpoint_sha256") != encoder_hash
        or endpoint_settings.get("held_out_test_engine_labels_used") is not False
    ):
        raise ValueError("Cluster preparation requires matching train-only normalization")
    reference_mean = torch.as_tensor(endpoint_settings["feature_mean"], dtype=torch.float32)
    reference_std = torch.as_tensor(endpoint_settings["feature_std"], dtype=torch.float32)
    if (
        reference_mean.shape != (30, 128)
        or reference_std.shape != (30, 128)
        or not bool((reference_std > 0).all())
    ):
        raise ValueError("Incompatible PatchFormer endpoint standardization")
    prepared = read_rul_artifact(prepared_file)
    rows = cast(list[dict[str, Any]], prepared["rows"])
    features = cast(Tensor, prepared["features"])
    if features.ndim != 3 or features.shape[1:] != (30, _SENSORS_FD003):
        raise ValueError("Expected FD003 30-cycle 15-sensor windows")
    engines: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["split"] == "train":
            engines[str(row["asset_id"])].append((int(row["step"]), index))
    if len(engines) < max(working_clusters, failure_clusters):
        raise ValueError("Insufficient training engines for clustering")
    selected_indices: list[int] = []
    for engine in sorted(engines):
        positions = sorted(engines[engine])
        if len(positions) < endpoint_windows * 2:
            raise ValueError(f"Too few complete healthy/failure windows: {engine}")
        selected_indices.extend(index for _, index in positions[:endpoint_windows])
        selected_indices.extend(index for _, index in positions[-endpoint_windows:])
    frozen_encoder.eval().requires_grad_(False)
    # CPU preparation is reproducible across Kubernetes GPU node types.
    observed = []
    with torch.inference_mode():
        for start in range(0, len(selected_indices), encoding_batch_size):
            indices = selected_indices[start:start + encoding_batch_size]
            embeddings = frozen_encoder.forward_features(features[indices])
            pooled = ((embeddings - reference_mean) / reference_std).mean(dim=1)
            observed.append(pooled.cpu())
    vectors = torch.cat(observed)
    healthy = []
    terminal = []
    for engine_index in range(len(engines)):
        offset = engine_index * endpoint_windows * 2
        healthy.append(vectors[offset:offset + endpoint_windows].mean(dim=0))
        terminal.append(
            vectors[offset + endpoint_windows:offset + 2 * endpoint_windows].mean(dim=0)
        )
    assignments: dict[str, list[int]] = {}
    result: dict[str, Any] = {}
    for label, points, count, offset in (
        ("working", torch.stack(healthy), working_clusters, 101),
        ("failure", torch.stack(terminal), failure_clusters, 202),
    ):
        mean = points.mean(dim=0)
        std = points.std(dim=0, unbiased=False).clamp_min(1e-4)
        centers, cluster_ids = _kmeans((points - mean) / std, count, seed + offset)
        result[f"{label}_centroids"] = centers
        result[f"{label}_mean"] = mean
        result[f"{label}_std"] = std
        for name, cluster in zip(sorted(engines), cluster_ids.tolist(), strict=True):
            assignments.setdefault(name, []).append(int(cluster))
    result.update({
        "format": _CLUSTER_FORMAT,
        "train_engine_assignments": assignments,
        "settings": {
            "subset": "FD003",
            "seed": seed,
            "working_clusters": working_clusters,
            "failure_clusters": failure_clusters,
            "endpoint_windows": endpoint_windows,
            "prepared_sha256": prepared_sha,
            "encoder_sha256": encoder_hash,
            "training_engines_only": True,
            "uses_test_or_val_endpoints": False,
        },
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        return read_cluster_artifact(
            output, seed=seed, working_clusters=working_clusters,
            failure_clusters=failure_clusters, prepared_sha256=prepared_sha,
            encoder_sha256=encoder_hash,
        )
    torch.save(result, output)
    return read_cluster_artifact(
        output, seed=seed, working_clusters=working_clusters,
        failure_clusters=failure_clusters, prepared_sha256=prepared_sha,
        encoder_sha256=encoder_hash,
    )


@data_source("AutoClusterConditionedCMapSSRULWindows")
class AutoClusterConditionedCMapSSRULWindows(AutoPairedCMapSSRULWindows):
    """Provide TRAIN-only pseudo-labels alongside original RUL and predecessor."""

    cluster_artifact: str = (
        ".experiments/rul/prepared/cmapss-FD003-seed42-cfc-conditional-clusters-v1.pt"
    )
    working_clusters: int = Field(default=3, ge=2, le=12)
    failure_clusters: int = Field(default=2, ge=2, le=12)
    endpoint_windows: int = Field(default=4, ge=1, le=20)

    def build(self) -> NexuDataset:
        """Build the native engine-disjoint dataset with masked pseudo-labels.

        Returns:
            Source with oracle labels only for training-engine windows.

        Raises:
            ValueError: If the source or cluster provenance is incompatible.
        """
        if self.subset != "FD003" or self.seed != 42:
            raise ValueError("Conditional CfC clusters currently support FD003 seed 42")
        original = super().build()
        if not isinstance(original, _PairedRULRuntime):
            raise ValueError("Conditional CfC requires native paired engine windows")
        clusters = prepare_cluster_artifact(
            prepared_path=self.artifact_path,
            output_path=self.cluster_artifact,
            seed=self.seed,
            working_clusters=self.working_clusters,
            failure_clusters=self.failure_clusters,
            endpoint_windows=self.endpoint_windows,
        )
        assignments = cast(dict[str, list[int]], clusters["train_engine_assignments"])
        meta = original.meta
        if meta is None:
            raise ValueError("Paired RUL windows require engine metadata")
        work = []
        failure = []
        for _, row in meta.iterrows():
            if row["split"] == "train":
                engine = str(row["asset_id"])
                if engine not in assignments:
                    raise ValueError(f"Unclustered training engine: {engine}")
                pair = assignments[engine]
                work.append(pair[0])
                failure.append(pair[1])
            else:
                work.append(-1)
                failure.append(-1)
        return _ClusterLabeledRULRuntime(original, work, failure)


class _ClusterLabeledRULRuntime(_PairedRULRuntime):
    """Preserve predecessor alignment and per-split auxiliary target masking."""

    def __init__(
        self, original: _PairedRULRuntime, work: list[int], failure: list[int]
    ) -> None:
        if original.meta is None or len(work) != len(original.meta):
            raise ValueError("Misaligned cluster and RUL source metadata")
        super().__init__(
            features=cast(Tensor, original.data),
            targets=original._target,
            raw=original._raw,
            rows=original.meta.to_dict("records"),
        )
        self._working_cluster = torch.tensor(work, dtype=torch.long)
        self._failure_cluster = torch.tensor(failure, dtype=torch.long)

    def take(self, indices: Sequence[int]) -> Self:
        """Retain pseudo-label alignment after engine-disjoint split selection.

        Returns:
            The correctly sliced native conditioned dataset.
        """
        selected = list(indices)
        subset = super().take(selected)
        subset._working_cluster = self._working_cluster[selected]
        subset._failure_cluster = self._failure_cluster[selected]
        return subset

    def load_labels(self, idx: int, row: Any) -> TensorDict:
        """Add train-only cluster targets, masked as -1 elsewhere.

        Returns:
            RUL labels with both cluster targets shaped [1].
        """
        labels = super().load_labels(idx, row)
        labels["working_cluster"] = self._working_cluster[idx].reshape(1)
        labels["failure_cluster"] = self._failure_cluster[idx].reshape(1)
        return labels
