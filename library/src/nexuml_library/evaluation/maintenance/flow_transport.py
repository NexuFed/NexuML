"""Post-training validation of acoustic healthy-to-faulty Flow Matching.

The transport starts from real healthy observations, not Gaussian noise.
Checkpoint selection uses Lightning's validation-best metadata, and the
held-out test distribution is evaluated only for that selected checkpoint.
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
from nexuml_library.layers.feature.lmbe import _LMBERuntime
from nexuml_library.layers.model.flow import _FlowRuntime


def resolve_checkpoints(
    checkpoint_dir: Path,
    *,
    best_checkpoint: Path | None = None,
    last_checkpoint: Path | None = None,
) -> tuple[Path, Path]:
    """Locate Lightning validation-best and last checkpoints without using test scores.

    Args:
        checkpoint_dir: Directory containing checkpoints for one training run.
        best_checkpoint: Optional explicit validation-selected checkpoint.
        last_checkpoint: Optional explicit final-state checkpoint.

    Returns:
        Pair of (best, last) existing checkpoint paths.

    Raises:
        FileNotFoundError: If a required checkpoint is missing.
        ValueError: If the directory contains ambiguous checkpoint candidates.
    """
    if last_checkpoint is None:
        candidates = sorted(checkpoint_dir.glob("last*.ckpt"))
        exact = checkpoint_dir / "last.ckpt"
        if exact.is_file():
            last_checkpoint = exact
        elif len(candidates) == 1:
            last_checkpoint = candidates[0]
        else:
            raise ValueError(
                f"Expected one last checkpoint in {checkpoint_dir}, got "
                f"{len(candidates)}. Supply --last-checkpoint explicitly."
            )
    last_checkpoint = last_checkpoint.expanduser().resolve()
    if not last_checkpoint.is_file():
        raise FileNotFoundError(last_checkpoint)

    if best_checkpoint is None:
        # The callback state identifies the best *validation* checkpoint.
        # Never rank checkpoints by their held-out test losses.
        state = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        callbacks = state.get("callbacks", {})
        callback_paths: set[Path] = set()
        for callback in callbacks.values():
            if not isinstance(callback, dict):
                continue
            name = callback.get("best_model_path")
            if isinstance(name, str) and name:
                saved = Path(name).expanduser()
                if saved.is_file():
                    callback_paths.add(saved.resolve())
                elif (checkpoint_dir / saved.name).is_file():
                    callback_paths.add((checkpoint_dir / saved.name).resolve())
        if len(callback_paths) == 1:
            best_checkpoint = next(iter(callback_paths))
        elif len(callback_paths) > 1:
            raise ValueError("Conflicting best-model paths; specify --best-checkpoint.")
        else:
            alternatives = sorted(
                path for path in checkpoint_dir.glob("*.ckpt")
                if not path.name.startswith("last")
            )
            if len(alternatives) != 1:
                raise ValueError(
                    f"Cannot uniquely identify best validation checkpoint in {checkpoint_dir}; "
                    "pass --best-checkpoint (and --last-checkpoint if necessary)."
                )
            best_checkpoint = alternatives[0]

    best_checkpoint = best_checkpoint.expanduser().resolve()
    if not best_checkpoint.is_file():
        raise FileNotFoundError(best_checkpoint)
    return best_checkpoint, last_checkpoint


def sliced_wasserstein(
    source: Tensor,
    target: Tensor,
    *,
    projections: int = 128,
    seed: int = 42,
) -> float:
    """Compute seeded sliced 1-Wasserstein distance for equally sized sample sets.

    A single fixed set of unit projection directions is reused across all
    trajectory times. No correspondence between source and target samples
    is assumed.

    Args:
        source: Distribution samples, with batch dimension first.
        target: Target distribution samples, of identical shape.
        projections: Number of random unit projection directions.
        seed: Projection RNG seed, independent of global randomness.

    Returns:
        Nonnegative finite mean 1D Wasserstein distance over projections.

    Raises:
        ValueError: If dimensions, sample counts or inputs are invalid.
    """
    if source.ndim < 2 or target.shape != source.shape or source.shape[0] == 0:
        raise ValueError("Sliced Wasserstein needs nonempty equal-shaped sample sets")
    if projections < 1:
        raise ValueError("projections must be positive")
    source = source.detach().reshape(source.shape[0], -1).to(
        dtype=torch.float32, device="cpu"
    )
    target = target.detach().reshape(target.shape[0], -1).to(
        dtype=torch.float32, device="cpu"
    )
    if not torch.isfinite(source).all() or not torch.isfinite(target).all():
        raise ValueError("Nonfinite features cannot be compared")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    directions = torch.randn(
        (source.shape[1], projections), generator=generator, dtype=source.dtype
    )
    directions = torch.nn.functional.normalize(directions, p=2, dim=0)
    projected_source = torch.sort(source @ directions, dim=0).values
    projected_target = torch.sort(target @ directions, dim=0).values
    return float((projected_source - projected_target).abs().mean().item())


@torch.no_grad()
def transport_trajectory(
    flow: _FlowRuntime,
    healthy: Tensor,
) -> Tensor:
    """Integrate from real healthy features with the existing Euler integrator.

    Returns:
        Trajectory of shape (batch, num_steps + 1, ...feature shape).

    Raises:
        ValueError: If the flow's recorded path does not preserve endpoints.
    """
    original_record = flow.integrator.record_trajectory
    try:
        flow.integrator.record_trajectory = True
        terminal = flow.sample(healthy)
        trajectory = flow.integrator.last_trajectory
    finally:
        flow.integrator.record_trajectory = original_record
        flow.integrator.last_trajectory = None
    if trajectory is None or not torch.equal(trajectory[:, 0], healthy):
        raise ValueError("Euler integrator did not record the healthy initial state")
    if not torch.equal(trajectory[:, -1], terminal):
        raise ValueError("Euler integrator trajectory does not end at its sampled state")
    if not torch.isfinite(trajectory).all():
        raise ValueError("Nonfinite flow trajectory")
    return trajectory


def feature_metrics(
    healthy: Tensor,
    faulty: Tensor,
    trajectory: Tensor,
    *,
    projections: int = 128,
    seed: int = 42,
) -> dict[str, Any]:
    """Score endpoint approach and a fixed five-point transport trajectory.

    Args:
        healthy: Healthy feature samples.
        faulty: Faulty feature samples.
        trajectory: Healthy-initialized trajectory with axis 1 as Euler time.
        projections: Monte-Carlo projection count.
        seed: RNG seed used consistently at all trajectory positions.

    Returns:
        Sliced Wasserstein distances and endpoint improvement statistics.

    Raises:
        ValueError: If trajectory shape or contents are inconsistent.
    """
    if (
        trajectory.shape[0] != healthy.shape[0]
        or trajectory.shape[2:] != healthy.shape[1:]
    ):
        raise ValueError("Trajectory feature shape must match healthy features")
    if trajectory.shape[1] < 2 or healthy.shape != faulty.shape:
        raise ValueError("Need at least two time points and equally shaped endpoint batches")

    indices = [round(k * (trajectory.shape[1] - 1) / 4) for k in range(5)]
    distances = [
        sliced_wasserstein(trajectory[:, index], faulty, projections=projections, seed=seed)
        for index in indices
    ]
    initial = distances[0]
    final = distances[-1]
    return {
        "swd_healthy_to_faulty": initial,
        "swd_transported_to_faulty": final,
        "swd_relative_reduction": 1.0 - final / initial if initial > 0 else None,
        "transport_mean_rms_displacement": float(
            (trajectory[:, -1] - healthy).flatten(1).square().mean(dim=1).sqrt().mean()
        ),
        "trajectory": [
            {"flow_time": k / 4, "swd_to_faulty": value}
            for k, value in enumerate(distances)
        ],
        "num_samples": int(healthy.shape[0]),
    }


def _get_layers(artifacts: RuntimeArtifacts) -> tuple[_LMBERuntime, _LMBERuntime, _FlowRuntime]:
    """Resolve trained audio frontend and flow layers from NexuML stages.

    Returns:
        Healthy feature layer, faulty feature layer and trained flow runtime.
    """
    found: dict[str, list[Any]] = {}
    for stage, _, layer in artifacts.pipeline.iter_layers():
        found.setdefault(stage, []).append(layer)
    return (
        cast(_LMBERuntime, found["AudioFeatures"][0]),
        cast(_LMBERuntime, found["AudioFeatures"][1]),
        cast(_FlowRuntime, found["Flow"][0]),
    )


@torch.no_grad()
def evaluate_split(
    artifacts: RuntimeArtifacts,
    *,
    split: str,
    device: torch.device,
    projections: int = 128,
    seed: int = 42,
) -> tuple[dict[str, Any], tuple[Tensor, Tensor, Tensor]]:
    """Evaluate fixed-time FM loss and real-healthy-initialized transport.

    Fixed times are used for checkpoint comparability: unlike normal FM
    training/validation, no new random interpolation times are sampled.
    Only the validation split should be used when comparing checkpoints.

    Args:
        artifacts: Reconstituted NexuML checkpoint and data module.
        split: Either val or test.
        device: Explicit evaluation device.
        projections: Seeded sliced Wasserstein projection count.
        seed: Reproducible projection seed.

    Returns:
        Metrics dictionary and CPU feature/trajectory tensors for plotting.

    Raises:
        ValueError: If an unsupported split is selected.
    """
    if split not in {"val", "test"}:
        raise ValueError("Only val and test splits are supported")
    artifacts.pipeline.to(device).eval()
    artifacts.data_module.setup("fit")
    loader = (
        artifacts.data_module.val_dataloader()
        if split == "val"
        else artifacts.data_module.test_dataloader()
    )
    healthy_layer, faulty_layer, flow = _get_layers(artifacts)
    healthy_samples: list[Tensor] = []
    faulty_samples: list[Tensor] = []
    flow_paths: list[Tensor] = []
    total_loss = 0.0
    total_samples = 0

    for features, _labels in loader:
        healthy_waveform = cast(Tensor, features["healthy_waveform"]).to(device)
        faulty_waveform = cast(Tensor, features["faulty_waveform"]).to(device)
        healthy = healthy_layer.forward_tensor(healthy_waveform)
        faulty = faulty_layer.forward_tensor(faulty_waveform)
        if healthy.shape != faulty.shape or not torch.isfinite(healthy).all():
            raise ValueError("Healthy and faulty features must be finite and have equal shapes")
        if not torch.isfinite(faulty).all():
            raise ValueError("Nonfinite faulty features")
        target_velocity = faulty - healthy

        for t in (0.25, 0.5, 0.75):
            time = torch.full((healthy.shape[0], 1), t, dtype=healthy.dtype, device=device)
            shape = (healthy.shape[0],) + (1,) * (healthy.ndim - 1)
            interpolated = healthy + time.reshape(shape) * target_velocity
            velocity = flow.velocity(interpolated, time)
            if not torch.isfinite(velocity).all():
                raise ValueError("Nonfinite learned vector field")
            total_loss += float((velocity - target_velocity).square().flatten(1).mean(dim=1).sum())

        flow_paths.append(transport_trajectory(flow, healthy).cpu())
        healthy_samples.append(healthy.cpu())
        faulty_samples.append(faulty.cpu())
        total_samples += healthy.shape[0]

    if not total_samples:
        raise ValueError(f"Cannot evaluate empty {split} partition")
    all_healthy = torch.cat(healthy_samples, dim=0)
    all_faulty = torch.cat(faulty_samples, dim=0)
    all_trajectory = torch.cat(flow_paths, dim=0)
    score = feature_metrics(
        all_healthy, all_faulty, all_trajectory, projections=projections, seed=seed
    )
    score["fixed_time_fm_mse"] = total_loss / (3 * total_samples)
    score["split"] = split
    score["fixed_times"] = [0.25, 0.5, 0.75]
    return score, (all_healthy, all_faulty, all_trajectory)


def _checkpoint_metadata(path: Path) -> dict[str, int | float | None]:
    """Read saved epoch and validation-best score for provenance.

    Returns:
        JSON-serializable checkpoint metadata without model parameters.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    best_score: float | None = None
    for callback in state.get("callbacks", {}).values():
        if isinstance(callback, dict) and callback.get("best_model_score") is not None:
            best_score = float(callback["best_model_score"])
            break
    return {
        "epoch_zero_indexed": int(state.get("epoch", -1)),
        "global_step": int(state.get("global_step", -1)),
        "callback_best_validation_loss": best_score,
    }


def read_tensorboard_losses(log_dir: Path) -> dict[str, list[dict[str, float]]]:
    """Read training and validation scalar histories from one TensorBoard run.

    Args:
        log_dir: Exactly one TensorBoard version directory.

    Returns:
        Available scalar histories indexed by their logged metric names.

    Raises:
        FileNotFoundError: If the supplied run has no TensorBoard event files.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    if not any(log_dir.glob("events.out.tfevents.*")):
        raise FileNotFoundError(f"No TensorBoard events in {log_dir}")
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    output: dict[str, list[dict[str, float]]] = {}
    for name in ("train/loss", "train/loss_epoch", "val/loss"):
        if name in accumulator.Tags().get("scalars", []):
            output[name] = [
                {"step": float(event.step), "value": float(event.value)}
                for event in accumulator.Scalars(name)
            ]
    return output


def _plot_report(
    output_dir: Path,
    features: tuple[Tensor, Tensor, Tensor],
    metrics: dict[str, Any],
    loss_curves: dict[str, list[dict[str, float]]],
) -> None:
    """Render fixed evaluation plots using the optional matplotlib backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    healthy, faulty, trajectory = features
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    rows = metrics["trajectory"]
    ax.plot([r["flow_time"] for r in rows], [r["swd_to_faulty"] for r in rows], marker="o")
    ax.set(
        xlabel="Flow integration time",
        ylabel="Sliced W1 to faulty",
        title="Healthy-initialized transport",
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "transport_distance.png", dpi=180)
    plt.close(fig)

    # PCA fit on validation/test observations for descriptive visualization
    # only; it is not used for model selection or quantitative scoring.
    matrices = [
        healthy.flatten(1).to(torch.float32),
        faulty.flatten(1).to(torch.float32),
        trajectory[:, -1].flatten(1).to(torch.float32),
    ]
    cloud = torch.cat(matrices, dim=0)
    center = cloud.mean(dim=0, keepdim=True)
    _, _, vt = torch.linalg.svd(cloud - center, full_matrices=False)
    directions = vt[:2].T
    projected = [(matrix - center) @ directions for matrix in matrices]
    fig, ax = plt.subplots()
    for points, label in zip(
        projected, ("Healthy", "Faulty", "Transported"), strict=True
    ):
        x = points[:, 0].numpy()
        y = points[:, 1].numpy() if points.shape[1] >= 2 else [0.0] * len(x)
        ax.scatter(x, y, s=9, alpha=0.6, label=label)
    ax.set(xlabel="PCA axis 1", ylabel="PCA axis 2", title="Endpoint distributions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "feature_pca.png", dpi=180)
    plt.close(fig)

    if loss_curves:
        fig, ax = plt.subplots()
        for label, rows in loss_curves.items():
            ax.plot([r["step"] for r in rows], [r["value"] for r in rows], label=label)
        ax.set(xlabel="Global optimizer step", ylabel="Logged loss", title="Training history")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "loss_curves.png", dpi=180)
        plt.close(fig)


def evaluate_checkpoints(
    checkpoint_dir: Path,
    output_dir: Path,
    *,
    best_checkpoint: Path | None = None,
    last_checkpoint: Path | None = None,
    tensorboard_run: Path | None = None,
    device: str = "cpu",
    projections: int = 128,
    seed: int = 42,
    plots: bool = True,
) -> dict[str, Any]:
    """Compare best/last on validation, then evaluate best once on held-out test.

    Returns:
        JSON-serializable metrics with checkpoint paths and transport diagnostics.
    """
    best, last = resolve_checkpoints(
        checkpoint_dir,
        best_checkpoint=best_checkpoint,
        last_checkpoint=last_checkpoint,
    )
    target_device = torch.device(device)
    best_artifacts = create_runtime_artifacts_from_trainer_checkpoint(best)
    best_val, _ = evaluate_split(
        best_artifacts, split="val", device=target_device, projections=projections, seed=seed
    )
    last_artifacts = create_runtime_artifacts_from_trainer_checkpoint(last)
    last_val, _ = evaluate_split(
        last_artifacts, split="val", device=target_device, projections=projections, seed=seed
    )
    # Test is used once after selecting the model by validation only.
    best_test, test_features = evaluate_split(
        best_artifacts, split="test", device=target_device, projections=projections, seed=seed
    )
    losses = read_tensorboard_losses(tensorboard_run) if tensorboard_run else {}
    report: dict[str, Any] = {
        "best_checkpoint": str(best),
        "last_checkpoint": str(last),
        "checkpoint_metadata": {
            "best": _checkpoint_metadata(best),
            "last": _checkpoint_metadata(last),
        },
        "validation_best": best_val,
        "validation_last": last_val,
        "held_out_test_best": best_test,
        "tensorboard_run": str(tensorboard_run) if tensorboard_run is not None else None,
        "loss_curves": losses,
        "method": "Healthy-seeded Euler transport; seeded sliced W1; fixed-time FM MSE",
        "note": (
            "MIMII has endpoint labels, not run-to-failure RUL annotations. "
            "Flow time is not a physical degradation-time estimate."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    if plots:
        _plot_report(output_dir, test_features, best_test, losses)
    return report


def main() -> None:
    """Run checkpoint evaluation as a standalone NexuML library command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(".experiments/checkpoints/mimii-acoustic-flow"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".experiments/nex233-mimii-evaluation"),
    )
    parser.add_argument("--best-checkpoint", type=Path)
    parser.add_argument("--last-checkpoint", type=Path)
    parser.add_argument("--tensorboard-run", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--projections", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    summary = evaluate_checkpoints(
        args.checkpoint_dir,
        args.output_dir,
        best_checkpoint=args.best_checkpoint,
        last_checkpoint=args.last_checkpoint,
        tensorboard_run=args.tensorboard_run,
        device=args.device,
        projections=args.projections,
        seed=args.seed,
        plots=not args.no_plots,
    )
    for name in ("validation_best", "validation_last", "held_out_test_best"):
        metrics = summary[name]
        print(
            f"{name}: FM MSE={metrics['fixed_time_fm_mse']:.4f}, "
            f"healthy→faulty SW1={metrics['swd_healthy_to_faulty']:.4f}, "
            f"transported→faulty SW1={metrics['swd_transported_to_faulty']:.4f}"
        )
    print(f"Evaluation written to {args.output_dir}")


if __name__ == "__main__":
    main()
