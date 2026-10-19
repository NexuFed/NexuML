"""Prepare recording-disjoint single-machine FISHER flow experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor

from nexuml_library.data.maintenance.prepared_endpoints import read_prepared_artifact
from nexuml_library.evaluation.maintenance.ablation_prepare import training_normalizer

Scaling = Literal["raw", "standardized"]


def build_single_machine_artifacts(
    source_file: str | Path,
    *,
    scaling: Scaling = "standardized",
    val_fraction: float = 0.25,
    seed: int = 42,
    include_original_test_machine: bool = False,
) -> dict[str, dict[str, Any]]:
    """Split FISHER embeddings within each machine without recording leakage.

    The source must contain raw, independently paired FISHER embeddings.
    Original machine-disjoint train/val/test labels are provenance only:
    the new holdout consists of distinct recordings from the same machine.
    Per-feature normalization is refitted using that machine's new train
    partition only. Original test-machine data remain unaccessed by default.

    Args:
        source_file: Immutable FISHER-independent-raw prepared artifact.
        scaling: Raw embeddings or train-only per-machine standardization.
        val_fraction: Fraction of recordings reserved for validation.
        seed: Fixed seed for within-machine recording splits and flow training.
        include_original_test_machine: Explicit opt-in to unseal original test ID.

    Returns:
        Machine name to validated in-memory prepared artifact.

    Raises:
        ValueError: If the source or split is invalid, or recordings overlap.
    """
    if scaling not in ("raw", "standardized") or not 0 < val_fraction < 1:
        raise ValueError("Choose raw/standardized scaling and validation fraction in (0,1)")
    source = read_prepared_artifact(source_file)
    settings = cast(dict[str, Any], source["settings"])
    if (
        settings.get("frontend") != "fisher"
        or settings.get("coupling") != "independent"
        or settings.get("scaling") != "raw"
    ):
        raise ValueError("Single-machine preparation requires FISHER independent raw features")
    features = cast(Tensor, source["features"])
    rows = cast(list[dict[str, Any]], source["rows"])
    machines = sorted({str(row["group_id"]) for row in rows})
    artifacts: dict[str, dict[str, Any]] = {}
    for index, machine in enumerate(machines):
        indices = [i for i, row in enumerate(rows) if row["group_id"] == machine]
        original_splits = {rows[i]["split"] for i in indices}
        if len(original_splits) != 1:
            raise ValueError("Source has machine identity overlapping original splits")
        original_split = next(iter(original_splits))
        if original_split == "test" and not include_original_test_machine:
            continue
        if len(indices) < 4:
            raise ValueError(f"Need at least four pairs for within-machine split: {machine}")
        for key in ("healthy_file", "faulty_file"):
            filenames = [rows[i].get(key) for i in indices]
            if not all(isinstance(name, str) and name for name in filenames):
                raise ValueError(f"Missing {key} provenance for {machine}")
            if len(set(filenames)) != len(filenames):
                raise ValueError(f"Repeated {key} recordings prevent leakage-safe splitting")

        generator = torch.Generator(device="cpu").manual_seed(seed + index)
        shuffled = torch.randperm(len(indices), generator=generator).tolist()
        n_val = max(2, min(len(indices) - 2, round(len(indices) * val_fraction)))
        val_positions = set(shuffled[:n_val])
        selected = features[indices].clone()
        records = [
            {
                **rows[source_index],
                "original_split": original_split,
                "split": "val" if offset in val_positions else "train",
            }
            for offset, source_index in enumerate(indices)
        ]
        if scaling == "standardized":
            mean, std = training_normalizer(selected, records)
            selected = (selected - mean) / std
        else:
            mean = torch.zeros((1, 1, selected.shape[-1]), dtype=selected.dtype)
            std = torch.ones_like(mean)

        # The source is raw: previous machine-disjoint training statistics
        # and coupling-cost diagnostics are not valid for this new split.
        excluded = {
            "training_velocity_statistics",
            "training_neighbor_alignment",
            "independent_train_pair_mse",
            "selected_train_pair_mse",
            "relative_pair_mse_reduction",
        }
        new_settings = {k: v for k, v in settings.items() if k not in excluded}
        new_settings.update(
            {
                "coupling": "independent",
                "scaling": scaling,
                "seed": seed,
                "source_pairing_seed": settings.get("seed"),
                "source_prepared_file": str(Path(source_file).expanduser().resolve()),
                "split_protocol": "within_machine_recording_holdout",
                "machine_group": machine,
                "original_machine_split": original_split,
                "original_test_machine_repurposed": original_split == "test",
                "original_test_machine_opt_in": include_original_test_machine,
                "normalization_fit_split": "train" if scaling == "standardized" else "none",
                "ot_fit_split": "none",
                "val_fraction_requested": val_fraction,
                "train_recordings": len(indices) - n_val,
                "val_recordings": n_val,
            }
        )
        artifacts[machine] = {
            "format": "nex233-features-v1",
            "features": selected.cpu().contiguous(),
            "rows": records,
            "settings": new_settings,
            "training_mean": mean.cpu(),
            "training_std": std.cpu(),
        }
    if not artifacts:
        raise ValueError("No eligible machines found in source artifact")
    return artifacts


def save_single_machine_artifacts(
    source_file: str | Path,
    output_dir: str | Path,
    *,
    scaling: Scaling = "standardized",
    val_fraction: float = 0.25,
    seed: int = 42,
    include_original_test_machine: bool = False,
) -> dict[str, Any]:
    """Persist separate immutable artifacts and a manifest for each machine.

    Returns:
        Machine manifest with paths, recording counts and sealing provenance.

    Raises:
        FileExistsError: If any target already exists.
        ValueError: If prepared single-machine data violate their contract.
    """
    artifacts = build_single_machine_artifacts(
        source_file,
        scaling=scaling,
        val_fraction=val_fraction,
        seed=seed,
        include_original_test_machine=include_original_test_machine,
    )
    directory = Path(output_dir).expanduser().resolve()
    paths = {
        machine: directory / f"fisher-{scaling}-{machine.replace(':', '-')}-seed{seed}.pt"
        for machine in artifacts
    }
    manifest_path = directory / "manifest.json"
    for path in [*paths.values(), manifest_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    directory.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "protocol": "within_machine_recording_holdout",
        "source_file": str(Path(source_file).expanduser().resolve()),
        "seed": seed,
        "scaling": scaling,
        "val_fraction": val_fraction,
        "original_test_machine_included": any(
            artifact["settings"]["original_test_machine_repurposed"]
            for artifact in artifacts.values()
        ),
        "machines": {},
    }
    for machine, artifact in artifacts.items():
        path = paths[machine]
        torch.save(artifact, path)
        read_prepared_artifact(path)
        settings = artifact["settings"]
        manifest["machines"][machine] = {
            "prepared_file": str(path),
            "original_split": settings["original_machine_split"],
            "train_recordings": settings["train_recordings"],
            "val_recordings": settings["val_recordings"],
        }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    """Prepare all eligible single-machine FISHER experiments from one artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scaling", choices=("raw", "standardized"), default="standardized")
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-original-test-machine", action="store_true")
    args = parser.parse_args()
    manifest = save_single_machine_artifacts(
        args.source_file,
        args.output_dir,
        scaling=args.scaling,
        val_fraction=args.val_fraction,
        seed=args.seed,
        include_original_test_machine=args.include_original_test_machine,
    )
    for machine, row in manifest["machines"].items():
        print(
            f"{machine}: {row['train_recordings']} train, "
            f"{row['val_recordings']} val; {row['prepared_file']}"
        )
    if manifest["original_test_machine_included"]:
        print("Original test machine was explicitly repurposed: prior test seal is broken.")
    else:
        print("Original test machine remains sealed.")


if __name__ == "__main__":
    main()
