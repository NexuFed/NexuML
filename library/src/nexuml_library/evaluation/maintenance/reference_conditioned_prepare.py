"""Prepare strict leave-one-machine-out FISHER healthy-reference experiments.

Per fold: train/validation queries originate exclusively from training
machines, each with disjoint healthy support. An unseen machine contributes
only healthy support at inference; its assessment endpoints are persisted in
a separate artifact and never loaded by the training dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from nexuml_library.data.maintenance.healthy_reference import (
    read_healthy_reference_assessment,
    read_healthy_reference_training,
)
from nexuml_library.data.maintenance.prepared_endpoints import read_prepared_artifact


def build_reference_conditioned_fold(
    source_file: str | Path,
    *,
    heldout_machine: str,
    reference_size: int = 32,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build train/validation and fully separated unseen-machine evaluation.

    Source is the original, unnormalized FISHER-independent-pairs artifact.
    Machine-wise healthy reference sets are disjoint from query recordings.
    Scaling is fitted on healthy support and healthy training queries from
    the *training machines only*, never validation or unseen-machine data.

    Args:
        source_file: Original pretrained FISHER independent/raw preparation.
        heldout_machine: Exactly one machine missing from train/validation.
        reference_size: Number of fixed healthy reference recordings per machine.
        validation_fraction: Fraction of non-support train-machine queries for val.
        seed: Per-machine deterministic split seed.

    Returns:
        Two in-memory, separately serializable training and assessment payloads.

    Raises:
        ValueError: For incompatible source, insufficient recordings or
            invalid split and provenance settings.
    """
    source = read_prepared_artifact(source_file)
    settings = cast(dict[str, Any], source["settings"])
    if (
        settings.get("frontend") != "fisher"
        or settings.get("coupling") != "independent"
        or settings.get("scaling") != "raw"
        or settings.get("split_protocol") == "within_machine_recording_holdout"
    ):
        raise ValueError("Expected original FISHER independent/raw features")
    if reference_size < 2 or not 0 < validation_fraction < 1:
        raise ValueError("Reference size >= 2 and validation fraction in (0,1) required")
    embeddings = cast(Tensor, source["features"]).detach().float().cpu()
    rows = cast(list[dict[str, Any]], source["rows"])
    names = sorted({str(row["group_id"]) for row in rows})
    if len(names) < 3 or heldout_machine not in names:
        raise ValueError("Need at least three machines and a valid held-out machine")
    used_paths: set[str] = set()
    for row in rows:
        for key in ("healthy_file", "faulty_file"):
            filename = row.get(key)
            if not isinstance(filename, str) or not filename or filename in used_paths:
                raise ValueError("All source recording identities must be distinct")
            used_paths.add(filename)

    train_machines = [name for name in names if name != heldout_machine]
    support_indices: dict[str, list[int]] = {}
    train_indices: list[int] = []
    val_indices: list[int] = []
    heldout_indices: list[int] = []
    for index, machine in enumerate(names):
        indices = [i for i, row in enumerate(rows) if row["group_id"] == machine]
        if len(indices) < reference_size + 4:
            raise ValueError(f"Not enough recordings for support and queries: {machine}")
        generator = torch.Generator(device="cpu").manual_seed(seed + index)
        permutation = torch.randperm(len(indices), generator=generator).tolist()
        shuffled = [indices[i] for i in permutation]
        support_indices[machine] = shuffled[:reference_size]
        query = shuffled[reference_size:]
        if machine == heldout_machine:
            heldout_indices.extend(query)
        else:
            count = max(2, min(len(query) - 2, round(len(query) * validation_fraction)))
            val_indices.extend(query[:count])
            train_indices.extend(query[count:])

    # Fit normalization exclusively on training-machine healthy observations.
    fitting = torch.cat(
        [
            embeddings[train_indices, 0],
            embeddings[
                [i for machine in train_machines for i in support_indices[machine]], 0
            ],
        ]
    )
    mean = fitting.mean(0)
    std = fitting.std(0, unbiased=False).clamp_min(1e-6)
    scaled = (embeddings - mean[None, None, :]) / std[None, None, :]
    train_indices.sort()
    val_indices.sort()
    heldout_indices.sort()
    query_indices = train_indices + val_indices
    machine_index = {name: index for index, name in enumerate(train_machines)}
    query_rows = [
        {
            **rows[i],
            "split": "train" if i in train_set else "val",
            "original_split": rows[i]["split"],
            "reference_index": machine_index[str(rows[i]["group_id"])],
        }
        for train_set in [set(train_indices)]
        for i in query_indices
    ]
    support_healthy_files = [
        [str(rows[i]["healthy_file"]) for i in support_indices[name]]
        for name in train_machines
    ]
    config = {
        "split_protocol": "leave_one_machine_out_healthy_reference",
        "source_file": str(Path(source_file).expanduser().resolve()),
        "heldout_machine": heldout_machine,
        "train_machines": train_machines,
        "reference_size": reference_size,
        "validation_fraction": validation_fraction,
        "seed": seed,
        "scaling_fit": "training_machine_healthy_only",
        "coupling": "independent",
        "frontend": "fisher",
        "feature_dim": int(embeddings.shape[-1]),
        "original_machine_test_seal": "retired_exploratory",
        "train_queries": len(train_indices),
        "validation_queries": len(val_indices),
        "heldout_assessment_queries": len(heldout_indices),
    }
    training = {
        "format": "nex233-healthy-context-train-v1",
        "features": scaled[query_indices].contiguous(),
        "references": torch.stack(
            [scaled[support_indices[name], 0] for name in train_machines]
        ).contiguous(),
        "rows": query_rows,
        "machine_groups": train_machines,
        "support_healthy_files": support_healthy_files,
        "settings": config,
        "training_mean": mean,
        "training_std": std,
    }
    assessment = {
        "format": "nex233-healthy-context-eval-v1",
        "features": scaled[heldout_indices].contiguous(),
        "references": scaled[support_indices[heldout_machine], 0].contiguous(),
        "rows": [
            {
                **rows[i],
                "original_split": rows[i]["split"],
                "split": "assessment",
            }
            for i in heldout_indices
        ],
        "support_healthy_files": [
            str(rows[i]["healthy_file"]) for i in support_indices[heldout_machine]
        ],
        "heldout_machine": heldout_machine,
        "settings": config,
    }
    return training, assessment


def save_reference_conditioned_folds(
    source_file: str | Path,
    output_dir: str | Path,
    *,
    heldout_machine: str | None = None,
    reference_size: int = 32,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Save one or four distinct folds without overwriting prior artifacts.

    Returns:
        JSON manifest including per-fold training and assessment file paths.

    Raises:
        FileExistsError: If a destination already contains a fold or manifest.
    """
    source = read_prepared_artifact(source_file)
    rows = cast(list[dict[str, Any]], source["rows"])
    names = sorted({str(row["group_id"]) for row in rows})
    selected = [heldout_machine] if heldout_machine is not None else names
    destination = Path(output_dir).expanduser().resolve()
    if (destination / "manifest.json").exists():
        raise FileExistsError("Refusing to overwrite existing fold manifest")
    files = {
        name: {
            "training": destination / f"heldout-{name.replace(':', '-')}-training.pt",
            "assessment": destination / f"heldout-{name.replace(':', '-')}-assessment.pt",
        }
        for name in selected
    }
    if any(path.exists() for entry in files.values() for path in entry.values()):
        raise FileExistsError("Refusing to overwrite existing conditional fold artifacts")
    folds = {
        name: build_reference_conditioned_fold(
            source_file,
            heldout_machine=name,
            reference_size=reference_size,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        for name in selected
    }
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "protocol": "leave_one_machine_out_healthy_reference",
        "source_file": str(Path(source_file).expanduser().resolve()),
        "seed": seed,
        "reference_size": reference_size,
        "validation_fraction": validation_fraction,
        "original_test_machine_repurposed": True,
        "folds": {},
    }
    for name, (train, assessment) in folds.items():
        paths = files[name]
        torch.save(train, paths["training"])
        torch.save(assessment, paths["assessment"])
        read_healthy_reference_training(paths["training"])
        read_healthy_reference_assessment(paths["assessment"])
        manifest["folds"][name] = {
            "training_file": str(paths["training"]),
            "assessment_file": str(paths["assessment"]),
            "train_machines": train["machine_groups"],
            "train_queries": train["settings"]["train_queries"],
            "val_queries": train["settings"]["validation_queries"],
            "assessment_queries": len(assessment["features"]),
        }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    """Generate leakage-safe healthy-reference leave-one-machine-out folds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heldout-machine", type=str)
    parser.add_argument("--reference-size", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = save_reference_conditioned_folds(
        args.source_file,
        args.output_dir,
        heldout_machine=args.heldout_machine,
        reference_size=args.reference_size,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    for machine, entry in manifest["folds"].items():
        print(
            f"Held out {machine}: {entry['train_queries']} train, "
            f"{entry['val_queries']} val, "
            f"{entry['assessment_queries']} unseen-machine assessment"
        )


if __name__ == "__main__":
    main()
