"""Run leave-one-machine-out healthy-reference Flow Matching on frozen FISHER.

Each fold trains and selects checkpoints exclusively on three machines.
The fourth machine's assessment artifact is read only after fitting and
validation-best checkpoint selection. Original id_06 is exploratory here.
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
    matches = [
        layer for stage, _, layer in artifacts.pipeline.iter_layers() if stage == "Flow"
    ]
    if len(matches) != 1 or not isinstance(
        matches[0], _HealthyReferenceConditionedFlowRuntime
    ):
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
        heldout_machine: Optional one fold to train instead of all four.
        context_mode: Healthy-reference conditioning or constant-context ablation.
        width: Width of both vector-field MLP hidden layers.
        context_dim: DeepSets output dimension.
        encoder_width: DeepSets pointwise hidden width.
        num_steps: Euler integration step count.
        lr: Adam learning rate.
        batch_size: Flow Matching train batch size.
        max_epochs: Maximum epochs per fold.
        patience: Validation-SWD early stopping patience.
        accelerator: Lightning device backend (cpu, gpu, auto).

    Returns:
        Per-fold validation-selected SWD and unseen-machine assessments.

    Raises:
        ValueError: If split provenance or model configuration is inconsistent.
        FileExistsError: If output reports already exist.
    """
    manifest = json.loads(Path(manifest_file).read_text())
    if manifest.get("protocol") != "leave_one_machine_out_healthy_reference":
        raise ValueError("Expected leave-one-machine-out healthy-reference manifest")
    entries = cast(dict[str, dict[str, Any]], manifest["folds"])
    if heldout_machine is not None and heldout_machine not in entries:
        raise ValueError("Requested held-out machine not found in prepared folds")
    selected = [heldout_machine] if heldout_machine is not None else sorted(entries)
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
        "cuda" if accelerator == "gpu" or (
            accelerator == "auto" and torch.cuda.is_available()
        ) else "cpu"
    )
    summary: dict[str, Any] = {
        "protocol": "leave_one_machine_out_healthy_reference",
        "manifest": str(Path(manifest_file).expanduser().resolve()),
        "context_mode": context_mode,
        "selected_by": "train_machine_macro_val/transport_swd",
        "original_id06_test_role": "retired_exploratory",
        "folds": {},
    }
    for fold, machine in enumerate(selected):
        entry = entries[machine]
        training_file = Path(entry["training_file"])
        assessment_file = Path(entry["assessment_file"])
        training = read_healthy_reference_training(training_file)
        if training["settings"]["heldout_machine"] != machine:
            raise ValueError(f"Training fold has incorrect held-out ID: {machine}")
        if machine in training["machine_groups"]:
            raise ValueError("Held-out machine may not appear in training data")
        # Do not load ANY unseen-machine assessment values during fitting.
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
        L.seed_everything(seed + fold, workers=True)
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
        # First access to unseen-machine support and faulty assessment
        # observations: fitting and validation selection are already complete.
        assessment = read_healthy_reference_assessment(assessment_file)
        if assessment["heldout_machine"] != machine:
            raise ValueError("Wrong held-out machine assessment artifact")
        if assessment["settings"] != training["settings"]:
            raise ValueError("Training/assessment scaling provenance mismatch")
        healthy = cast(Tensor, assessment["features"])[:, 0]
        faulty = cast(Tensor, assessment["features"])[:, 1]
        reference = cast(Tensor, assessment["references"])
        with torch.no_grad():
            unseen = evaluate_reference_transport(
                flow, healthy, faulty, reference,
                device=device, batch_size=batch_size, seed=seed
            )
            sensitivity = _context_impact(
                flow, assessment, training, device=device
            )
        report = {
            "heldout_machine": machine,
            "known_training_machines": training["machine_groups"],
            "context_mode": context_mode,
            "reference_size": int(reference.shape[0]),
            "assessment_recordings": len(healthy),
            "best_checkpoint": str(best),
            "last_checkpoint": str(last),
            "train_machine_validation": validation,
            "unseen_machine_assessment": unseen,
            "context_sensitivity": sensitivity,
            "zero_shot_faulty_labels_accessed_during_fit": False,
            "warning": (
                "Exploratory closed-family four-machine leave-one-out; "
                "no physical degradation time or RUL calibration."
            ),
        }
        report_paths[machine].write_text(json.dumps(report, indent=2) + "\n")
        summary["folds"][machine] = {
            "report": str(report_paths[machine]),
            "val_swd_mean": validation["mean_transported_swd"],
            "unseen_machine_swd_relative_reduction": unseen["swd_relative_reduction"],
            "unseen_machine_initial_swd": unseen["initial_swd"],
            "unseen_machine_transported_swd": unseen["transported_swd"],
            "best_checkpoint": str(best),
        }
        print(
            f"Held out {machine}: unseen SWD reduction "
            f"{unseen['swd_relative_reduction']:+.2%}"
        )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    """Fit selected reference-conditioned folds with explicit CLI parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--heldout-machine", type=str)
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
