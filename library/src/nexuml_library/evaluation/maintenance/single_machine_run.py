"""Train and evaluate all prepared single-machine FISHER flow experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import torch

from nexuml_library.data.maintenance.prepared_endpoints import read_prepared_artifact
from nexuml_library.evaluation.maintenance.ablation_evaluate import (
    evaluate_prepared_checkpoints,
)
from nexuml_library.evaluation.maintenance.ablation_train import train_ablation


def run_single_machine_experiments(
    manifest_file: str | Path,
    *,
    checkpoint_root: str | Path,
    reports_dir: str | Path,
    width: int = 128,
    lr: float = 3e-4,
    batch_size: int = 16,
    max_epochs: int = 300,
    accelerator: str = "auto",
    allow_original_test_machine: bool = False,
) -> dict[str, Any]:
    """Train distinct machine-specific flows with distributional selection.

    Best and last checkpoints are evaluated on each machine's **own**
    recording-disjoint train and validation partitions. There is no test
    evaluation or checkpoint selection based on training SWD.

    Args:
        manifest_file: Single-machine preparation manifest.
        checkpoint_root: Fresh parent directory for per-machine checkpoints.
        reports_dir: Destination for individual and consolidated JSON reports.
        width: Width of both hidden layers of the vector field.
        lr: Flow Matching learning rate.
        batch_size: Flow training batch size.
        max_epochs: Training budget before validation-SWD early stopping.
        accelerator: Lightning accelerator, e.g. cpu or gpu.
        allow_original_test_machine: Explicit confirmation of test unsealing.

    Returns:
        Machine-wise train/validation SWD improvement and checkpoint provenance.

    Raises:
        ValueError: If manifest is not per-machine or test handling is unsafe.
        FileExistsError: If any output report already exists.
    """
    manifest = json.loads(Path(manifest_file).read_text())
    if manifest.get("protocol") != "within_machine_recording_holdout":
        raise ValueError("Expected a within-machine recording holdout manifest")
    if manifest.get("original_test_machine_included") and not allow_original_test_machine:
        raise ValueError("Original test machine needs explicit --allow-original-test-machine")
    entries = cast(dict[str, dict[str, Any]], manifest["machines"])
    if not entries:
        raise ValueError("Manifest contains no machines")
    output = Path(reports_dir).expanduser().resolve()
    checkpoints = Path(checkpoint_root).expanduser().resolve()
    if (output / "summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite {output / 'summary.json'}")
    targets = {
        machine: output / f"{machine.replace(':', '-')}.json"
        for machine in entries
    }
    if any(path.exists() for path in targets.values()):
        raise FileExistsError("Per-machine evaluation report already exists")
    output.mkdir(parents=True, exist_ok=True)
    evaluate_on = torch.device(
        "cuda" if accelerator == "gpu" or (
            accelerator == "auto" and torch.cuda.is_available()
        ) else "cpu"
    )
    summary: dict[str, Any] = {
        "protocol": "within_machine_recording_holdout",
        "manifest": str(Path(manifest_file).resolve()),
        "original_test_machine_included": manifest["original_test_machine_included"],
        "conditional_flow": False,
        "selected_by": "val/transport_swd",
        "machines": {},
    }
    for machine, entry in sorted(entries.items()):
        prepared_file = Path(entry["prepared_file"])
        artifact = read_prepared_artifact(prepared_file)
        settings = cast(dict[str, Any], artifact["settings"])
        if (
            settings.get("split_protocol") != "within_machine_recording_holdout"
            or settings.get("machine_group") != machine
        ):
            raise ValueError(f"Manifest and prepared artifact disagree: {machine}")
        if settings.get("original_test_machine_repurposed") and not (
            allow_original_test_machine
        ):
            raise ValueError(f"Original test machine not authorized: {machine}")
        print(f"Training single-machine FISHER flow: {machine}")
        train_ablation(
            prepared_file,
            checkpoint_root=checkpoints,
            width=width,
            lr=lr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            seed=int(settings["seed"]),
            accelerator=accelerator,
        )
        run_name = f"{prepared_file.stem}-mlp-{width}x{width}"
        report = evaluate_prepared_checkpoints(
            checkpoints / run_name,
            device=evaluate_on,
            include_test=False,
            seed=int(settings["seed"]),
        )
        targets[machine].write_text(json.dumps(report, indent=2) + "\n")
        summary["machines"][machine] = {
            "original_machine_split": settings["original_machine_split"],
            "train_recordings": settings["train_recordings"],
            "val_recordings": settings["val_recordings"],
            "best_checkpoint": report["best_checkpoint"],
            "last_checkpoint": report["last_checkpoint"],
            "training_best": report["training_best"]["swd_relative_reduction"],
            "training_last": report["training_last"]["swd_relative_reduction"],
            "validation_best": report["validation_best"]["swd_relative_reduction"],
            "validation_last": report["validation_last"]["swd_relative_reduction"],
            "beats_no_transport": report["validation_best"]["swd_relative_reduction"] > 0,
            "original_test_is_sealed": report["test_is_sealed"],
            "report": str(targets[machine]),
        }
        print(
            f"{machine}: selected validation SWD reduction "
            f"{summary['machines'][machine]['validation_best']:+.2%}"
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    """Train the prepared machine-specific experiments sequentially."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, choices=(128, 512), default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--allow-original-test-machine", action="store_true")
    args = parser.parse_args()
    result = run_single_machine_experiments(
        args.manifest,
        checkpoint_root=args.checkpoint_root,
        reports_dir=args.reports_dir,
        width=args.width,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        allow_original_test_machine=args.allow_original_test_machine,
    )
    print(f"Saved {len(result['machines'])} machine reports; no test split evaluated.")


if __name__ == "__main__":
    main()
