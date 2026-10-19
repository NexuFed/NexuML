"""Diagnose independent acoustic endpoint coupling and learned flow velocities.

All pairing alternatives and the constant-velocity baseline are constructed
from training recordings only. Validation is diagnostic; test remains sealed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from nexuml.training.lightning import (
    RuntimeArtifacts,
    create_runtime_artifacts_from_trainer_checkpoint,
)
from nexuml_library.evaluation.maintenance.flow_transport import (
    _get_layers,
    resolve_checkpoints,
    sliced_wasserstein,
    transport_trajectory,
)
from nexuml_library.layers.model.flow import _FlowRuntime

_FIXED_TIMES = (0.25, 0.5, 0.75)


def velocity_statistics(healthy: Tensor, faulty: Tensor) -> dict[str, float]:
    """Quantify unconditional target-velocity disagreement without a model.

    The mean-velocity energy fraction is zero when the independent pairing
    velocities cancel and one when all target velocities are identical.
    It does *not* estimate the conditional mean E[v | x_t, t].

    Args:
        healthy: Samples with batch dimension first.
        faulty: Corresponding target samples, same shape.

    Returns:
        Velocity RMS, coherence and feature-scale statistics.

    Raises:
        ValueError: If endpoints are empty, incompatible or nonfinite.
    """
    if healthy.ndim < 2 or healthy.shape != faulty.shape or not healthy.shape[0]:
        raise ValueError("Healthy/faulty samples must be nonempty and equal-shaped")
    source = healthy.detach().flatten(1).to(device="cpu", dtype=torch.float32)
    target = faulty.detach().flatten(1).to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(source).all() or not torch.isfinite(target).all():
        raise ValueError("Endpoint features must be finite")
    velocity = target - source
    mean = velocity.mean(dim=0)
    total_energy = float(velocity.square().sum(dim=1).mean())
    mean_energy = float(mean.square().sum())
    fraction = mean_energy / total_energy if total_energy > 0 else 1.0
    return {
        "velocity_rms_per_feature": float(velocity.square().mean().sqrt()),
        "mean_velocity_rms_per_feature": float(mean.square().mean().sqrt()),
        "mean_velocity_energy_fraction": min(1.0, fraction),
        "unconditional_cancellation_fraction": max(0.0, 1.0 - fraction),
        "zero_velocity_mse": float(velocity.square().mean()),
        "mean_velocity_mse": float((velocity - mean).square().mean()),
        "endpoint_feature_rms": float(torch.cat((source, target)).square().mean().sqrt()),
        "pooled_feature_std_rms": float(
            torch.cat((source, target)).std(dim=0, unbiased=False).square().mean().sqrt()
        ),
    }


def neighbor_velocity_alignment(
    healthy: Tensor,
    faulty: Tensor,
    *,
    k: int = 8,
    times: tuple[float, ...] = _FIXED_TIMES,
) -> list[dict[str, float]]:
    """Estimate local target-direction consistency at fixed interpolation times.

    Nearest neighbours are chosen by training-only, pooled-scale-standardized
    interpolation states. Self-neighbours are excluded. This is a descriptive
    finite-neighbourhood diagnostic, not an unbiased conditional MI/variance
    estimator.

    Args:
        healthy: Source samples.
        faulty: Corresponding destination samples.
        k: Number of other samples in each neighbourhood.
        times: Transport positions used to compare local velocity directions.

    Returns:
        Local cosine agreements and energy ratios at each time.

    Raises:
        ValueError: If fewer than two samples are available or k is invalid.
    """
    if k < 1 or healthy.shape != faulty.shape or len(healthy) < 2:
        raise ValueError("Neighbourhood alignment needs two endpoints and k >= 1")
    source = healthy.detach().flatten(1).to(device="cpu", dtype=torch.float32)
    target = faulty.detach().flatten(1).to(device="cpu", dtype=torch.float32)
    velocity = target - source
    standard_deviation = torch.cat((source, target)).std(dim=0, unbiased=False)
    scale = standard_deviation.clamp_min(1.0)
    denom = velocity.norm(dim=1).clamp_min(1e-8)
    neighborhood_count = min(k, len(source) - 1)
    results: list[dict[str, float]] = []
    for t in times:
        state = ((1.0 - t) * source + t * target) / scale
        distances = torch.cdist(state, state)
        distances.fill_diagonal_(float("inf"))
        closest = torch.topk(distances, k=neighborhood_count, dim=1, largest=False).indices
        local_mean = velocity[closest].mean(dim=1)
        norm = local_mean.norm(dim=1).clamp_min(1e-8)
        cosines = (velocity * local_mean).sum(dim=1) / (denom * norm)
        magnitude = float(local_mean.square().sum(dim=1).mean())
        target_magnitude = float(velocity.square().sum(dim=1).mean())
        results.append(
            {
                "flow_time": float(t),
                "mean_neighbor_velocity_cosine": float(cosines.mean()),
                "local_mean_velocity_energy_fraction": (
                    magnitude / target_magnitude if target_magnitude > 0 else 1.0
                ),
                "nearest_neighbor_count": float(neighborhood_count),
            }
        )
    return results


def training_only_ot_diagnostic(
    healthy: Tensor,
    faulty: Tensor,
    machine_ids: list[str],
) -> dict[str, Any]:
    """Compare actual independent pair costs with hypothetical OT pairing.

    Only recording pairs from the *same training machine* can be reassigned;
    this comparison does not change the training dataset or train a model.
    The mean displacement is invariant to permutations of target recordings.

    Args:
        healthy: Independently paired training healthy features.
        faulty: Independently paired training faulty features.
        machine_ids: Machine identity in sample order.

    Returns:
        Independent versus minimum-cost (Hungarian) matching costs per machine.

    Raises:
        ValueError: If machine identities and samples differ in count.
        ImportError: If SciPy for the exact finite assignment is unavailable.
    """
    if len(machine_ids) != len(healthy) or healthy.shape != faulty.shape:
        raise ValueError("Machine IDs must correspond to equal-shaped training pairs")
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise ImportError(
            "Exact train-only OT diagnostic requires SciPy. Install the optional "
            "SciPy dependency to run it."
        ) from exc
    source = healthy.detach().flatten(1).to(device="cpu", dtype=torch.float32)
    target = faulty.detach().flatten(1).to(device="cpu", dtype=torch.float32)
    rows: list[dict[str, Any]] = []
    for machine in sorted(set(machine_ids)):
        indices = [index for index, label in enumerate(machine_ids) if label == machine]
        first, second = source[indices], target[indices]
        # cdist output is Euclidean distance; square to recover the MSE cost.
        costs = torch.cdist(first, second).square() / first.shape[1]
        row_indices, column_indices = linear_sum_assignment(costs.numpy())
        row_tensor = torch.as_tensor(row_indices, dtype=torch.long)
        column_tensor = torch.as_tensor(column_indices, dtype=torch.long)
        random_cost = float(costs.diagonal().mean())
        optimal_cost = float(costs[row_tensor, column_tensor].mean())
        rows.append(
            {
                "machine_id": machine,
                "num_pairs": len(indices),
                "independent_pair_mse": random_cost,
                "min_cost_assignment_mse": optimal_cost,
                "possible_assignment_mse_reduction": (
                    1.0 - optimal_cost / random_cost if random_cost > 0 else 0.0
                ),
            }
        )
    total = sum(row["num_pairs"] for row in rows)
    return {
        "by_machine": rows,
        "num_pairs": total,
        "independent_pair_mse": sum(
            row["independent_pair_mse"] * row["num_pairs"] for row in rows
        ) / total,
        "min_cost_assignment_mse": sum(
            row["min_cost_assignment_mse"] * row["num_pairs"] for row in rows
        ) / total,
        "diagnostic_only": True,
    }


@torch.no_grad()
def _collect_features(
    artifacts: RuntimeArtifacts,
    *,
    split: str,
    device: torch.device,
    batch_size: int = 16,
) -> tuple[Tensor, Tensor, list[str]]:
    """Decode and transform one entire split without shuffling pair identities.

    Args:
        artifacts: Checkpoint-reconstructed NexuML pipeline and data module.
        split: Either train or val; intentionally no test access.
        device: Device for the feature frontends.
        batch_size: Decode/transform batch size.

    Returns:
        Detached CPU healthy and faulty features, plus aligned machine IDs.
    """
    if split not in {"train", "val"}:
        raise ValueError("Diagnostics use training and validation only")
    artifacts.pipeline.to(device).eval()
    dataset = artifacts.data_module.dataset.get_split(split)
    if dataset.meta is None or "group_id" not in dataset.meta:
        raise ValueError("Acoustic diagnostics require machine-ID metadata")
    first_layer, second_layer, _ = _get_layers(artifacts)
    healthy: list[Tensor] = []
    faulty: list[Tensor] = []
    for start in range(0, len(dataset), batch_size):
        rows = [dataset[index][0] for index in range(start, min(start + batch_size, len(dataset)))]
        source_wave = torch.stack(
            [cast(Tensor, row["healthy_waveform"]) for row in rows]
        ).to(device)
        target_wave = torch.stack(
            [cast(Tensor, row["faulty_waveform"]) for row in rows]
        ).to(device)
        healthy.append(first_layer.forward_tensor(source_wave).cpu())
        faulty.append(second_layer.forward_tensor(target_wave).cpu())
    if not healthy:
        raise ValueError(f"Empty {split} split")
    return (
        torch.cat(healthy),
        torch.cat(faulty),
        dataset.meta["group_id"].astype(str).tolist(),
    )


@torch.no_grad()
def _predictor_diagnostics(
    flow: _FlowRuntime,
    healthy: Tensor,
    faulty: Tensor,
    mean_training_velocity: Tensor,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Compare learned predictor with fixed zero and train-only mean velocities.

    Returns:
        Per-time squared-error and predicted-velocity scale/alignment metrics.
    """
    source = healthy.to(device)
    target_velocity = (faulty - healthy).to(device)
    mean_training_velocity = mean_training_velocity.to(device)
    total_energy = float(target_velocity.square().mean())
    constant_energy = float((target_velocity - mean_training_velocity).square().mean())
    scores: list[dict[str, float]] = []
    for t in _FIXED_TIMES:
        time = torch.full((len(source), 1), t, device=device, dtype=source.dtype)
        shape = (len(source),) + (1,) * (source.ndim - 1)
        state = source + t * target_velocity
        prediction = flow.velocity(state, time)
        if prediction.shape != target_velocity.shape or not torch.isfinite(prediction).all():
            raise ValueError("Learned velocity is nonfinite or dimensionally incompatible")
        pred_flat = prediction.flatten(1)
        true_flat = target_velocity.flatten(1)
        cosine = torch.nn.functional.cosine_similarity(pred_flat, true_flat, dim=1)
        scores.append(
            {
                "flow_time": t,
                "learned_velocity_mse": float((prediction - target_velocity).square().mean()),
                "zero_velocity_mse": total_energy,
                "training_mean_velocity_mse": constant_energy,
                "learned_velocity_rms": float(prediction.square().mean().sqrt()),
                "target_velocity_rms": float(target_velocity.square().mean().sqrt()),
                "mean_predicted_target_cosine": float(cosine.mean()),
            }
        )
    return {"fixed_times": scores}


@torch.no_grad()
def investigate_velocity(
    artifacts: RuntimeArtifacts,
    *,
    device: torch.device,
    neighbors: int = 8,
    projections: int = 128,
    seed: int = 42,
) -> dict[str, Any]:
    """Investigate velocity disagreement on training and predictor fit on val.

    All pairing analysis, kNN fitting, OT and baseline estimation use train
    samples only. Validation is read to estimate model generalization.
    Held-out test is never opened.

    Returns:
        JSON-ready diagnostic report to guide the next experimental design.
    """
    train_healthy, train_faulty, groups = _collect_features(
        artifacts, split="train", device=device
    )
    val_healthy, val_faulty, _ = _collect_features(
        artifacts, split="val", device=device
    )
    train_velocity = train_faulty - train_healthy
    mean_train = train_velocity.mean(dim=0, keepdim=True)
    _, _, flow = _get_layers(artifacts)
    train_stats = velocity_statistics(train_healthy, train_faulty)
    val_stats = velocity_statistics(val_healthy, val_faulty)
    trajectory = transport_trajectory(flow, val_healthy.to(device)).cpu()
    baseline_shift = val_healthy + mean_train

    return {
        "training": {
            "num_pairs": len(train_healthy),
            "machine_ids": sorted(set(groups)),
            "velocity_statistics": train_stats,
            "neighbor_alignment": neighbor_velocity_alignment(
                train_healthy, train_faulty, k=neighbors
            ),
            "pairing_alternative": training_only_ot_diagnostic(
                train_healthy, train_faulty, groups
            ),
            "predictor": _predictor_diagnostics(
                flow, train_healthy, train_faulty, mean_train, device=device
            ),
        },
        "validation": {
            "num_pairs": len(val_healthy),
            "velocity_statistics": val_stats,
            "predictor": _predictor_diagnostics(
                flow, val_healthy, val_faulty, mean_train, device=device
            ),
            "transport_baselines": {
                "healthy_to_faulty_swd": sliced_wasserstein(
                    val_healthy, val_faulty, projections=projections, seed=seed
                ),
                "training_mean_shift_to_faulty_swd": sliced_wasserstein(
                    baseline_shift, val_faulty, projections=projections, seed=seed
                ),
                "flow_to_faulty_swd": sliced_wasserstein(
                    trajectory[:, -1], val_faulty, projections=projections, seed=seed
                ),
            },
        },
        "protocol": (
            "Only train and validation used. OT assignment is hypothetical and computed "
            "separately inside each training machine. No RUL inference, test selection or "
            "training modifications."
        ),
    }


def main() -> None:
    """Run NEX-233 velocity-cancellation diagnostics on an existing checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(".experiments/checkpoints/mimii-acoustic-flow"),
    )
    parser.add_argument("--last-checkpoint", type=Path)
    parser.add_argument("--best-checkpoint", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".experiments/nex233-velocity-diagnostics.json"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--projections", type=int, default=128)
    args = parser.parse_args()

    best, last = resolve_checkpoints(
        args.checkpoint_dir,
        best_checkpoint=args.best_checkpoint,
        last_checkpoint=args.last_checkpoint,
    )
    artifacts = create_runtime_artifacts_from_trainer_checkpoint(best)
    report = investigate_velocity(
        artifacts,
        device=torch.device(args.device),
        neighbors=args.neighbors,
        projections=args.projections,
    )
    report["best_checkpoint"] = str(best)
    report["last_checkpoint"] = str(last)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    train = report["training"]["velocity_statistics"]
    val = report["validation"]["transport_baselines"]
    cancellation = train["unconditional_cancellation_fraction"]
    print(f"Training unconditional cancellation: {cancellation:.2%}")
    for label, score in val.items():
        print(f"Validation {label}: {score:.5f}")
    print(f"Diagnostics written to {args.output}")


if __name__ == "__main__":
    main()
