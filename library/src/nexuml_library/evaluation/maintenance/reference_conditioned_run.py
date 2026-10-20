"""Run reference-conditioned Flow Matching with machine or family holdouts.

Checkpoints are fitted and selected on training-machine validation only.
Unseen-machine assessment artifacts are first read after checkpoint selection.
Both the legacy four-fan experiment and sixteen-group protocols are supported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import lightning as L
import torch
from torch import Tensor

from nexuml.training.lightning import (
    NexuSession,
    RuntimeArtifacts,
    create_runtime_artifacts_from_trainer_checkpoint,
)
from nexuml_library.data.maintenance.healthy_reference import (
    read_healthy_reference_assessment,
    read_healthy_reference_training,
)
from nexuml_library.evaluation.maintenance.ablation_train import (
    verify_latest_checkpoint,
)
from nexuml_library.evaluation.maintenance.flow_transport import resolve_checkpoints
from nexuml_library.evaluation.maintenance.reference_conditioned_validation import (
    evaluate_reference_transport,
    evaluate_training_machine_validation,
)
from nexuml_library.layers.model.healthy_reference_flow import (
    _HealthyReferenceConditionedFlowRuntime,
)
from nexuml_library.scenarios.flow.healthy_reference_fisher import (
    healthy_reference_flow_scenario,
)


def _conditional_layer(
    artifacts: RuntimeArtifacts,
    *,
    device: torch.device,
) -> _HealthyReferenceConditionedFlowRuntime:
    """Load the sole trained conditional-flow layer from a checkpoint.

    Returns:
        Ready-to-evaluate velocity field and healthy context encoder.

    Raises:
        ValueError: If a checkpoint is not a healthy-reference flow.
    """
    artifacts.pipeline.to(device).eval()
    matches = [layer for stage, _, layer in artifacts.pipeline.iter_layers() if stage == "Flow"]
    if len(matches) != 1 or not isinstance(matches[0], _HealthyReferenceConditionedFlowRuntime):
        raise ValueError("Expected trained HealthyReferenceConditionedFlow layer")
    return matches[0]


@torch.no_grad()
def _context_impact(
    flow: _HealthyReferenceConditionedFlowRuntime,
    assessment: dict[str, Any],
    training: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Measure whether the learned field changes when healthy context changes.

    Returns:
        Mean RMS velocity change relative to other training machine supports.
        This is a sensitivity diagnostic, NOT evidence of improved transport.
    """
    features = cast(Tensor, assessment["features"])
    correct = cast(Tensor, assessment["references"]).to(device)
    references = cast(Tensor, training["references"])
    names = cast(list[str], training["machine_groups"])
    state = features[: min(32, len(features)), 0].to(device)
    time = state.new_full((len(state), 1), 0.5)
    predicted = flow.velocity(state, time, correct)
    values = {}
    for index, name in enumerate(names):
        alternative = flow.velocity(state, time, references[index].to(device))
        values[name] = float((predicted - alternative).square().mean().sqrt())
    return {"velocity_rms_difference_by_alternative_machine_context": values}


def run_reference_conditioned_folds(
    manifest_file: str | Path,
    *,
    checkpoint_root: str | Path,
    reports_dir: str | Path,
    heldout_machine: str | None = None,
    heldout_machines: tuple[str, ...] | None = None,
    context_mode: str = "reference",
    width: int = 128,
    context_dim: int = 32,
    encoder_width: int = 128,
    num_steps: int = 50,
    lr: float = 3e-4,
    batch_size: int = 16,
    max_epochs: int = 300,
    patience: int = 25,
    accelerator: str = "auto",
) -> dict[str, Any]:
    """Fit validated conditional flows then measure zero-shot held-out SWD.

    Use a fresh checkpoint root for each configuration. Evaluation reads the
    unseen-machine faulty recordings only after best checkpoint selection.

    Args:
        manifest_file: Manifest created by reference_conditioned_prepare.
        checkpoint_root: Parent directory for validation-selected checkpoints.
        reports_dir: Location for per-fold JSON and overall summary.
        heldout_machine: Optional single held-out fold.
        heldout_machines: Optional exact subset of held-out folds.
        context_mode: Healthy-reference conditioning or constant-context ablation.
        width: Width of both vector-field MLP hidden layers.
        context_dim: DeepSets output dimension.
        encoder_width: DeepSets point-wise hidden width.
        num_steps: Euler integration step count.
        lr: Adam learning rate.
        batch_size: Flow Matching train batch size.
        max_epochs: Maximum epochs per fold.
        patience: Validation-SWD early stopping patience.
        accelerator: Lightning device backend (cpu, gpu, auto).

    Returns:
        Per-fold validation-selected SWD and unseen-machine assessments.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
        ValueError: If split provenance or model configuration is inconsistent.
        FileExistsError: If output reports already exist.
    """
    manifest = json.loads(Path(manifest_file).read_text())
    protocol = manifest.get("protocol")
    if protocol not in {
        "leave_one_machine_out_healthy_reference",
        "leave_one_family_out_healthy_reference",
    }:
        raise ValueError("Expected machine- or family-held-out healthy-reference manifest")
    entries = cast(dict[str, dict[str, Any]], manifest["folds"])
    if heldout_machine is not None and heldout_machines is not None:
        raise ValueError("Choose either a single held-out machine or a fold subset")
    if heldout_machine is not None:
        selected = [heldout_machine]
    elif heldout_machines is not None:
        if not heldout_machines or len(set(heldout_machines)) != len(heldout_machines):
            raise ValueError("Held-out machine subset must be nonempty and unique")
        selected = sorted(heldout_machines)
    else:
        selected = sorted(entries)
    if any(machine not in entries for machine in selected):
        raise ValueError("Requested held-out machine not found in prepared folds")
    if context_mode not in {"reference", "constant"}:
        raise ValueError("Unknown context_mode")
    report_root = Path(reports_dir).expanduser().resolve()
    checkpoint_root = Path(checkpoint_root).expanduser().resolve()
    summary_path = report_root / f"summary-{context_mode}.json"
    report_paths = {
        machine: report_root / f"{machine.replace(':', '-')}-{context_mode}.json"
        for machine in selected
    }
    if summary_path.exists() or any(path.exists() for path in report_paths.values()):
        raise FileExistsError("Refusing to overwrite previous NEX-233 evaluation")
    report_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda"
        if accelerator == "gpu" or (accelerator == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    summary: dict[str, Any] = {
        "protocol": protocol,
        "manifest": str(Path(manifest_file).expanduser().resolve()),
        "context_mode": context_mode,
        "selected_by": "train_machine_macro_val/transport_swd",
        "original_id06_test_role": "retired_exploratory",
        "folds": {},
    }
    for machine in selected:
        entry = entries[machine]
        training_file = Path(entry["training_file"])
        if "assessment_files" in entry:
            assessment_files = {
                group: Path(filename)
                for group, filename in cast(dict[str, str], entry["assessment_files"]).items()
            }
        else:
            assessment_files = {machine: Path(entry["assessment_file"])}
        if not assessment_files:
            raise ValueError("Holdout fold has no assessment machines")
        training = read_healthy_reference_training(training_file)
        if training["settings"]["heldout_machine"] != machine:
            raise ValueError(f"Training fold has incorrect held-out ID: {machine}")
        if set(assessment_files).intersection(training["machine_groups"]):
            raise ValueError("Held-out machine may not appear in training data")
        if protocol == "leave_one_family_out_healthy_reference":
            family = machine.removeprefix("family:")
            expected_groups = set(training["settings"]["heldout_machines"])
            if (
                len(assessment_files) != 4
                or set(assessment_files) != expected_groups
                or any(group.split(":")[0] != family for group in assessment_files)
            ):
                raise ValueError("Held-out family must contain its four recorded machine IDs")
        # File existence only; never deserialize unseen features before fit.
        for assessment_file in assessment_files.values():
            if not assessment_file.is_file():
                raise FileNotFoundError(assessment_file)
        seed = int(manifest["seed"])
        spec = healthy_reference_flow_scenario(
            training_file,
            width=width,
            context_dim=context_dim,
            encoder_width=encoder_width,
            context_mode=context_mode,
            num_steps=num_steps,
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            seed=seed,
            checkpoint_root=checkpoint_root,
        )
        run_name = (
            f"{training_file.stem}-{context_mode}-mlp-{width}x{width}"
            f"-context-{context_dim}-seed{seed}"
        )
        directory = checkpoint_root / run_name
        if list(directory.glob("*.ckpt")):
            raise FileExistsError(f"Previous fold checkpoint(s) found: {directory}")
        L.seed_everything(seed + sorted(entries).index(machine), workers=True)
        session = NexuSession.from_scenario(spec, accelerator=accelerator, devices=1)
        print(f"Training context={context_mode}, unseen machine={machine}")
        session.setup().fit()
        verify_latest_checkpoint(session.trainer, directory / "last.ckpt")
        session.validate()
        best, last = resolve_checkpoints(directory)
        artifacts = create_runtime_artifacts_from_trainer_checkpoint(best)
        flow = _conditional_layer(artifacts, device=device)
        validation = evaluate_training_machine_validation(
            flow, training_file, device=device, batch_size=batch_size, seed=seed
        )
        # First access to unseen-machine assessment after validation selection.
        unseen_by_machine: dict[str, dict[str, float]] = {}
        sensitivity_by_machine: dict[str, dict[str, Any]] = {}
        assessment_recordings: dict[str, int] = {}
        for group, assessment_file in sorted(assessment_files.items()):
            assessment = read_healthy_reference_assessment(assessment_file)
            if assessment["heldout_machine"] != group:
                raise ValueError(f"Wrong unseen machine in assessment: {group}")
            if assessment["settings"] != training["settings"]:
                raise ValueError("Training/assessment scaling provenance mismatch")
            healthy = cast(Tensor, assessment["features"])[:, 0]
            faulty = cast(Tensor, assessment["features"])[:, 1]
            reference = cast(Tensor, assessment["references"])
            with torch.no_grad():
                unseen_by_machine[group] = evaluate_reference_transport(
                    flow,
                    healthy,
                    faulty,
                    reference,
                    device=device,
                    batch_size=batch_size,
                    seed=seed,
                )
                sensitivity_by_machine[group] = _context_impact(
                    flow, assessment, training, device=device
                )
            assessment_recordings[group] = len(healthy)
        unseen = {
            "initial_swd": sum(value["initial_swd"] for value in unseen_by_machine.values())
            / len(unseen_by_machine),
            "transported_swd": sum(value["transported_swd"] for value in unseen_by_machine.values())
            / len(unseen_by_machine),
            "swd_relative_reduction": sum(
                value["swd_relative_reduction"] for value in unseen_by_machine.values()
            )
            / len(unseen_by_machine),
        }
        report = {
            "heldout_machine": machine,
            "protocol": protocol,
            "known_training_machines": training["machine_groups"],
            "unseen_machine_groups": sorted(assessment_files),
            "context_mode": context_mode,
            "reference_size": int(training["references"].shape[1]),
            "assessment_recordings": (
                next(iter(assessment_recordings.values()))
                if len(assessment_recordings) == 1
                else assessment_recordings
            ),
            "assessment_recordings_by_machine": assessment_recordings,
            "best_checkpoint": str(best),
            "last_checkpoint": str(last),
            "train_machine_validation": validation,
            "unseen_machine_assessment": unseen,
            "unseen_by_machine": unseen_by_machine,
            "context_sensitivity": (
                next(iter(sensitivity_by_machine.values()))
                if len(sensitivity_by_machine) == 1
                else sensitivity_by_machine
            ),
            "context_sensitivity_by_machine": sensitivity_by_machine,
            "zero_shot_faulty_labels_accessed_during_fit": False,
            "warning": (
                "Exploratory MIMII machine/family holdout; original id_06 seal "
                "retired. No physical degradation time or RUL calibration."
            ),
        }
        report_paths[machine].write_text(json.dumps(report, indent=2) + "\n")
        summary["folds"][machine] = {
            "report": str(report_paths[machine]),
            "val_swd_mean": validation["mean_transported_swd"],
            "unseen_machine_swd_relative_reduction": unseen["swd_relative_reduction"],
            "unseen_machine_initial_swd": unseen["initial_swd"],
            "unseen_machine_transported_swd": unseen["transported_swd"],
            "unseen_by_machine": unseen_by_machine,
            "best_checkpoint": str(best),
        }
        print(f"Held out {machine}: unseen SWD reduction {unseen['swd_relative_reduction']:+.2%}")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    """Fit selected reference-conditioned folds with explicit CLI parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--heldout-machine", type=str)
    parser.add_argument("--heldout-machines", nargs="+")
    parser.add_argument("--context-mode", choices=("reference", "constant"), default="reference")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--context-dim", type=int, default=32)
    parser.add_argument("--encoder-width", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"), default="auto")
    args = parser.parse_args()
    summary = run_reference_conditioned_folds(
        args.manifest,
        checkpoint_root=args.checkpoint_root,
        reports_dir=args.reports_dir,
        heldout_machine=args.heldout_machine,
        heldout_machines=(
            tuple(args.heldout_machines) if args.heldout_machines is not None else None
        ),
        context_mode=args.context_mode,
        width=args.width,
        context_dim=args.context_dim,
        encoder_width=args.encoder_width,
        num_steps=args.num_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        accelerator=args.accelerator,
    )
    print(f"Finished {len(summary['folds'])} unseen-machine folds")


if __name__ == "__main__":
    main()
