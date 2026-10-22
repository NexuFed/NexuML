"""Frozen PatchFormer latent-flow geometry and checkpoint validation.

Flow trajectories start at a handful of train-only healthy references. For
each new embedding, nearest-path projection gives an approximate flow time.
This is not a learned physical clock or a guaranteed monotonic degradation law.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from nexuml.core.components import LayerBuildContext
from nexuml_library.layers.model.flow import Flow, _FlowRuntime


def resolve_latent_flow_checkpoint(
    subset: Literal["FD001", "FD003"], seed: int
) -> Path:
    """Locate the validation-selected native latent flow checkpoint.

    Returns:
        One trained checkpoint belonging to the matching subset/seed.

    Raises:
        FileNotFoundError: If no flow experiment has been trained.
        ValueError: If the checkpoint selection is ambiguous or missing.
    """
    run = f"rul-{subset}-patchformer-latent-flow-seed{seed}"
    directory = Path(".experiments/rul/checkpoints") / run
    manifest = Path(".experiments/rul/audits") / f"{run}.checkpoint.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("best_model_path"), str):
            raise ValueError(f"Invalid flow checkpoint manifest: {manifest}")
        selected = Path(data["best_model_path"]).expanduser().resolve()
        if selected.parent.name != run or not selected.is_file():
            raise ValueError(f"Stale flow checkpoint manifest: {manifest}")
        return selected
    candidates = sorted(
        path.resolve() for path in directory.glob("*.ckpt")
        if path.name != "last.ckpt"
    )
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise ValueError(
            f"Ambiguous flow checkpoints in {directory}; select best by validation SWD"
        )
    raise FileNotFoundError(
        f"Train 'maintenance-rul-patchformer-flow-{subset.lower()}-s{seed}' "
        "with NexuML before running flow-time RUL calibration"
    )


def load_frozen_latent_flow(
    checkpoint: Path, *, seed: int, feature_dim: int = 128
) -> tuple[_FlowRuntime, str]:
    """Reconstruct the existing NEX-196 vector field from trusted weights.

    Args:
        checkpoint: Native local Lightning checkpoint written by NexuML.
        seed: Original flow initialization and interpolation seed.
        feature_dim: Pretrained DPAN pooled feature width.

    Returns:
        Frozen runtime with trajectory capture and SHA-256 provenance.

    Raises:
        ValueError: If the checkpoint's vector-field parameters are incomplete.
    """
    definition = Flow(
        hidden_dims=[256, 256],
        num_steps=50,
        record_trajectory=True,
        initialization_seed=seed,
    )
    context = LayerBuildContext(
        input_sizes={"flow_state": (feature_dim,), "flow_time": (1,)},
        keys_in=["flow_state", "flow_time"],
        keys_out=["predicted_velocity"],
    )
    runtime = definition.build(context)
    if not isinstance(runtime, _FlowRuntime):
        raise ValueError("Flow must build the existing NexuML runtime")
    # Only locally produced trusted Lightning files are ever read.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Flow checkpoint must be a Lightning state dictionary")
    saved = payload.get("state_dict", payload)
    if not isinstance(saved, dict):
        raise ValueError("Flow checkpoint has no model state_dict")
    matched: dict[str, Tensor] = {}
    for key, expected in runtime.state_dict().items():
        possible = [
            value for name, value in saved.items()
            if isinstance(name, str)
            and (name == key or name.endswith(f".{key}"))
        ]
        if len(possible) != 1 or not isinstance(possible[0], Tensor):
            raise ValueError(f"Missing or ambiguous frozen Flow parameter: {key}")
        if possible[0].shape != expected.shape:
            raise ValueError(f"Flow parameter shape mismatch: {key}")
        matched[key] = possible[0]
    runtime.load_state_dict(matched, strict=True)
    runtime.requires_grad_(False)
    runtime.eval()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return runtime, digest


def healthy_reference_trajectories(
    runtime: _FlowRuntime, train_healthy: Tensor, *, references: int = 8
) -> Tensor:
    """Integrate only training healthy embedding references along the flow.

    Returns:
        Frozen trajectories with shape [references,steps+1,feature_dim].

    Raises:
        ValueError: If the training healthy endpoints are missing.
    """
    if train_healthy.ndim != 2 or not len(train_healthy):
        raise ValueError("Need nonempty training healthy reference embeddings")
    count = min(references, len(train_healthy))
    indices = torch.linspace(
        0, len(train_healthy) - 1, count
    ).round().long()
    with torch.no_grad():
        runtime.sample(train_healthy[indices])
        result = runtime.integrator.last_trajectory
    if result is None or result.shape != (count, 51, train_healthy.shape[1]):
        raise ValueError("NexuML Euler integrator did not record flow trajectories")
    return result.detach().cpu()


def project_flow_time(embeddings: Tensor, trajectories: Tensor) -> Tensor:
    """Project embedded observations onto frozen healthy-to-degraded paths.

    The projection minimizes Euclidean distance to any sampled trajectory,
    yielding time in [0,1] and off-manifold nearest-path distance.
    No RUL labels are consulted.

    Args:
        embeddings: Batch of standardized DPAN embeddings [N,D].
        trajectories: [K,T,D] train-only reference trajectories.

    Returns:
        [N,2] tensor containing approximate flow time and normalized distance.

    Raises:
        ValueError: If feature dimensions or trajectory counts are invalid.
    """
    if (
        embeddings.ndim != 2
        or trajectories.ndim != 3
        or trajectories.shape[0] < 1
        or trajectories.shape[1] < 2
        or embeddings.shape[-1] != trajectories.shape[-1]
    ):
        raise ValueError("Invalid flow projection dimensions")
    reference_points = trajectories.reshape(-1, embeddings.shape[-1])
    results: list[Tensor] = []
    with torch.no_grad():
        for batch in embeddings.split(256):
            distances = torch.cdist(batch.float(), reference_points.float())
            nearest, index = distances.min(dim=-1)
            time = (index % trajectories.shape[1]).float() / (
                trajectories.shape[1] - 1
            )
            results.append(
                torch.stack((time, nearest / embeddings.shape[-1] ** 0.5), dim=1)
            )
    return torch.cat(results)


def pearson_flow_progress(times: Tensor, rul: Tensor) -> float | None:
    """Report observed validation correlation without fitting to these labels.

    Returns:
        Pearson correlation between time and capped RUL or None if constant.
    """
    a, b = times.flatten().double(), rul.flatten().double()
    if len(a) < 2:
        return None
    a, b = a - a.mean(), b - b.mean()
    denominator = a.norm() * b.norm()
    if not float(denominator) > 0:
        return None
    return float((a * b).sum() / denominator)
