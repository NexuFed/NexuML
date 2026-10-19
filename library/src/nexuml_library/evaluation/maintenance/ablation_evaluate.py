"""Validation-first comparison of NEX-233 prepared-feature FM checkpoints."""

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
    feature_metrics,
    resolve_checkpoints,
    sliced_wasserstein,
    transport_trajectory,
)
from nexuml_library.layers.model.flow import _FlowRuntime

_TIMES = (0.25, 0.5, 0.75)


def _flow(artifacts: RuntimeArtifacts) -> _FlowRuntime:
    """Select the sole trained flow layer.

    Returns:
        Existing trained Flow runtime.

    Raises:
        ValueError: If the checkpoint is not a prepared-feature flow.
    """
    matches = [layer for stage, _, layer in artifacts.pipeline.iter_layers() if stage == "Flow"]
    if len(matches) != 1 or not isinstance(matches[0], _FlowRuntime):
        raise ValueError("Expected one trained Flow layer")
    return matches[0]


@torch.no_grad()
def evaluate_prepared_split(
    artifacts: RuntimeArtifacts,
    *,
    split: str,
    device: torch.device,
    seed: int = 42,
    projections: int = 128,
) -> dict[str, Any]:
    """Measure held-out endpoint transport and fixed-time FM velocity accuracy.

    Args:
        artifacts: Reconstructed model and tensor-backed data module.
        split: Validation or explicitly requested held-out test.
        device: Evaluation device.
        seed: Fixed SWD projection seed.
        projections: Number of random directions for sliced Wasserstein.

    Returns:
        Independent baseline, mean-shift baseline, learned transport and
        three fixed-time FM losses.

    Raises:
        ValueError: If the split is invalid or empty.
    """
    if split not in {"val", "test"}:
        raise ValueError("Only val and explicitly selected test can be evaluated")
    artifacts.pipeline.to(device).eval()
    artifacts.data_module.setup("fit")
    flow = _flow(artifacts)
    dataset = artifacts.data_module.dataset
    training = dataset.get_split("train")
    train_features = cast(Tensor, training.data)
    train_mean_velocity = (train_features[:, 1] - train_features[:, 0]).mean(dim=0)
    loader = (
        artifacts.data_module.val_dataloader()
        if split == "val"
        else artifacts.data_module.test_dataloader()
    )
    healthy_parts: list[Tensor] = []
    faulty_parts: list[Tensor] = []
    paths: list[Tensor] = []
    losses: dict[float, float] = {time: 0.0 for time in _TIMES}
    zero_losses: dict[float, float] = {time: 0.0 for time in _TIMES}
    mean_losses: dict[float, float] = {time: 0.0 for time in _TIMES}
    n = 0
    for features, _labels in loader:
        healthy = cast(Tensor, features["healthy_spectrum"]).to(device)
        faulty = cast(Tensor, features["faulty_spectrum"]).to(device)
        velocity = faulty - healthy
        count = len(healthy)
        if not count:
            continue
        for time_value in _TIMES:
            time = torch.full((count, 1), time_value, device=device, dtype=healthy.dtype)
            intermediate = healthy + time_value * velocity
            predicted = flow.velocity(intermediate, time)
            losses[time_value] += float(
                (predicted - velocity).square().flatten(1).mean(dim=1).sum()
            )
            zero_losses[time_value] += float(velocity.square().flatten(1).mean(dim=1).sum())
            mean_losses[time_value] += float(
                (velocity - train_mean_velocity.to(device)).square().flatten(1).mean(dim=1).sum()
            )
        paths.append(transport_trajectory(flow, healthy).cpu())
        healthy_parts.append(healthy.cpu())
        faulty_parts.append(faulty.cpu())
        n += count
    if n == 0:
        raise ValueError(f"Empty {split} feature split")
    source = torch.cat(healthy_parts)
    target = torch.cat(faulty_parts)
    trajectory = torch.cat(paths)
    metrics = feature_metrics(source, target, trajectory, projections=projections, seed=seed)
    metrics["split"] = split
    metrics["fixed_time_fm_mse"] = {str(time): losses[time] / n for time in _TIMES}
    metrics["zero_velocity_mse"] = sum(zero_losses.values()) / (n * len(_TIMES))
    metrics["train_mean_velocity_mse"] = sum(mean_losses.values()) / (n * len(_TIMES))
    metrics["relative_mse_reduction_vs_zero"] = (
        1.0 - sum(losses.values()) / sum(zero_losses.values())
        if sum(zero_losses.values())
        else None
    )
    metrics["training_mean_shift_swd"] = sliced_wasserstein(
        source + train_mean_velocity,
        target,
        projections=projections,
        seed=seed,
    )
    metrics["num_samples"] = n
    return metrics


@torch.no_grad()
def evaluate_training_distribution(
    artifacts: RuntimeArtifacts,
    *,
    device: torch.device,
    seed: int = 42,
    projections: int = 128,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Measure training-distribution transport without sample pairing assumptions.

    The complete train split is evaluated both jointly and per machine ID.
    Metrics are descriptive only: training SWD never selects checkpoints.
    Validation and test samples are never read by this function.

    Args:
        artifacts: Reconstructed trained flow and prepared dataset.
        device: Device for Euler transport.
        seed: Fixed SWD projection seed, shared with validation evaluation.
        projections: Number of sliced Wasserstein projection directions.
        batch_size: Maximum number of healthy inputs per transport batch.

    Returns:
        Joint and per-machine SWD, relative improvements, and sample counts.

    Raises:
        ValueError: If the prepared training data or group metadata is invalid.
    """
    if batch_size <= 0:
        raise ValueError("Training transport batch size must be positive")
    artifacts.pipeline.to(device).eval()
    training = artifacts.data_module.dataset.get_split("train")
    endpoints = cast(Tensor, training.data)
    meta = training.meta
    if (
        endpoints.ndim != 3
        or endpoints.shape[1] != 2
        or not len(endpoints)
        or meta is None
        or "group_id" not in meta.columns
        or len(meta) != len(endpoints)
    ):
        raise ValueError("Expected nonempty train endpoints and aligned machine IDs")

    healthy = endpoints[:, 0].cpu()
    faulty = endpoints[:, 1].cpu()
    flow = _flow(artifacts)
    transported = torch.cat(
        [flow.sample(chunk.to(device)).detach().cpu() for chunk in healthy.split(batch_size)],
        dim=0,
    )

    def score(indices: list[int]) -> dict[str, float | int | None]:
        """Compute pairing-invariant SWD for one subset of training observations.

        Returns:
            Baseline SWD, transported SWD, improvement, and subset size.
        """
        before = sliced_wasserstein(
            healthy[indices], faulty[indices], projections=projections, seed=seed
        )
        after = sliced_wasserstein(
            transported[indices], faulty[indices], projections=projections, seed=seed
        )
        return {
            "swd_healthy_to_faulty": before,
            "swd_transported_to_faulty": after,
            "swd_relative_reduction": 1.0 - after / before if before > 0 else None,
            "num_samples": len(indices),
        }

    groups = cast(list[str], meta["group_id"].tolist())
    by_machine = {
        group: score([i for i, name in enumerate(groups) if name == group])
        for group in sorted(set(groups))
    }
    return {
        "split": "train",
        **score(list(range(len(healthy)))),
        "by_machine": by_machine,
    }


def evaluate_prepared_checkpoints(
    checkpoint_dir: Path,
    *,
    device: torch.device,
    last_checkpoint: Path | None = None,
    best_checkpoint: Path | None = None,
    include_test: bool = False,
    seed: int = 42,
    projections: int = 128,
) -> dict[str, Any]:
    """Evaluate validation-best and final checkpoint; test only on explicit opt-in.

    Args:
        checkpoint_dir: Directory containing one run's checkpoints.
        device: Evaluation execution device.
        last_checkpoint: Explicit final checkpoint for a multi-run directory.
        best_checkpoint: Optional explicit validation-selected checkpoint.
        include_test: Evaluate held-out test only after choosing a configuration.
        seed: SWD sampling seed.
        projections: Monte-Carlo SWD projection count.

    Returns:
        Machine-readable report with checkpoint IDs and validation metrics.
    """
    best, last = resolve_checkpoints(
        checkpoint_dir,
        best_checkpoint=best_checkpoint,
        last_checkpoint=last_checkpoint,
    )
    best_artifacts = create_runtime_artifacts_from_trainer_checkpoint(best)
    source = best_artifacts.data_module.dataset
    settings = getattr(source, "settings", None)
    report: dict[str, Any] = {
        "best_checkpoint": str(best),
        "last_checkpoint": str(last),
        "prepared_settings": settings,
        "training_best": evaluate_training_distribution(
            best_artifacts,
            device=device,
            seed=seed,
            projections=projections,
        ),
        "validation_best": evaluate_prepared_split(
            best_artifacts,
            split="val",
            device=device,
            seed=seed,
            projections=projections,
        ),
        "test_is_sealed": not include_test,
        "note": (
            "Do not compare absolute SWD or feature MSE across distinct "
            "frontends/scalings. Use dimensionless relative improvements. "
            "Flow time is not a physical degradation clock."
        ),
    }
    if last.resolve() == best.resolve():
        report["training_last"] = report["training_best"]
        report["validation_last"] = report["validation_best"]
    else:
        last_artifacts = create_runtime_artifacts_from_trainer_checkpoint(last)
        report["training_last"] = evaluate_training_distribution(
            last_artifacts,
            device=device,
            seed=seed,
            projections=projections,
        )
        report["validation_last"] = evaluate_prepared_split(
            last_artifacts,
            split="val",
            device=device,
            seed=seed,
            projections=projections,
        )
    if include_test:
        report["held_out_test_best"] = evaluate_prepared_split(
            best_artifacts,
            split="test",
            device=device,
            seed=seed,
            projections=projections,
        )
    return report


def main() -> None:
    """Evaluate one selected ablation training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--best-checkpoint", type=Path)
    parser.add_argument("--last-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--projections", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-test", action="store_true")
    args = parser.parse_args()
    report = evaluate_prepared_checkpoints(
        args.checkpoint_dir,
        device=torch.device(args.device),
        best_checkpoint=args.best_checkpoint,
        last_checkpoint=args.last_checkpoint,
        include_test=args.include_test,
        seed=args.seed,
        projections=args.projections,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "Validation relative SWD improvement: "
        f"{report['validation_best']['swd_relative_reduction']:.4%}"
    )
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
