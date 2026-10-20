"""Prepare 16-machine and held-out-family healthy-reference FISHER protocols.

Machine holdouts reuse the existing reference-conditioned fold builder.
Family holdouts keep all four machine identities of an unseen family out of
training and checkpoint selection. Each held-out identity supplies only its
own independent healthy support set for zero-shot inference.
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
from nexuml_library.evaluation.maintenance.multifamily_fisher_prepare import (
    MIMII_FAMILIES,
    MIMII_IDS,
)
from nexuml_library.evaluation.maintenance.reference_conditioned_prepare import (
    save_reference_conditioned_folds,
)


def _validated_source(
    source_file: str | Path,
) -> tuple[Tensor, list[dict[str, Any]]]:
    """Require a complete raw FISHER experiment with sixteen unique groups.

    Returns:
        Floating FISHER embeddings and immutable source recording metadata.

    Raises:
        ValueError: If the source is not the expected sixteen-group extraction.
    """
    artifact = read_prepared_artifact(source_file)
    settings = cast(dict[str, Any], artifact["settings"])
    if (
        settings.get("frontend") != "fisher"
        or settings.get("scaling") != "raw"
        or settings.get("coupling") != "independent"
        or settings.get("split_protocol") != "multifamily_raw_for_reference_holdouts"
    ):
        raise ValueError("Expected the original multi-family independent/raw FISHER artifact")
    embeddings = cast(Tensor, artifact["features"]).detach().float().cpu()
    rows = cast(list[dict[str, Any]], artifact["rows"])
    expected = {f"{family}:{identity}" for family in MIMII_FAMILIES for identity in MIMII_IDS}
    actual = {str(row["group_id"]) for row in rows}
    if actual != expected:
        raise ValueError("Expected exactly sixteen MIMII family-qualified machine groups")
    seen: set[str] = set()
    for row in rows:
        family, identity = str(row["group_id"]).split(":")
        if row.get("machine_type", family) != family or row.get("machine_id", identity) != identity:
            raise ValueError("MIMII family and machine ID provenance disagree")
        for key in ("healthy_file", "faulty_file"):
            path = row.get(key)
            if not isinstance(path, str) or not path or path in seen:
                raise ValueError("Source files must be unique across all MIMII families")
            seen.add(path)
    return embeddings, rows


def build_family_holdout(
    source_file: str | Path,
    *,
    heldout_family: str,
    reference_size: int = 32,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Split twelve training groups and four strictly unseen assessment groups.

    A per-machine healthy support set is disjoint from that machine's training
    or assessment queries. Feature scaling fits only healthy support and
    training queries of the *twelve training groups*, never their validation
    data nor any observation from the held-out family.

    Returns:
        NexuML training artifact and one assessment artifact per unseen ID.

    Raises:
        ValueError: If families, counts or split proportions are inconsistent.
    """
    if heldout_family not in MIMII_FAMILIES:
        raise ValueError("Held-out family must be fan, pump, slider, or valve")
    if reference_size < 2 or not 0 < validation_fraction < 1:
        raise ValueError("Need >=2 support recordings and validation fraction in (0,1)")
    embeddings, rows = _validated_source(source_file)
    groups = sorted({str(row["group_id"]) for row in rows})
    train_groups = [group for group in groups if not group.startswith(f"{heldout_family}:")]
    unseen_groups = [group for group in groups if group.startswith(f"{heldout_family}:")]
    support: dict[str, list[int]] = {}
    train: list[int] = []
    validation: list[int] = []
    assessment: dict[str, list[int]] = {}
    for index, name in enumerate(groups):
        indices = [i for i, row in enumerate(rows) if row["group_id"] == name]
        if len(indices) < reference_size + 4:
            raise ValueError(f"Insufficient support and query recordings: {name}")
        permutation = torch.randperm(
            len(indices), generator=torch.Generator().manual_seed(seed + index)
        ).tolist()
        shuffled = [indices[j] for j in permutation]
        support[name] = shuffled[:reference_size]
        queries = shuffled[reference_size:]
        if name in unseen_groups:
            assessment[name] = sorted(queries)
        else:
            num_val = max(
                2, min(len(queries) - 2, round(validation_fraction * len(queries)))
            )
            validation.extend(queries[:num_val])
            train.extend(queries[num_val:])

    fitting_indices = train + [i for group in train_groups for i in support[group]]
    fitting = embeddings[fitting_indices, 0]
    mean = fitting.mean(0)
    std = fitting.std(0, unbiased=False).clamp_min(1e-6)
    scaled = (embeddings - mean[None, None, :]) / std[None, None, :]
    train_set = set(train)
    queries = sorted(train + validation)
    group_index = {name: index for index, name in enumerate(train_groups)}
    settings: dict[str, Any] = {
        "split_protocol": "leave_one_family_out_healthy_reference",
        "source_file": str(Path(source_file).expanduser().resolve()),
        "heldout_machine": f"family:{heldout_family}",
        "heldout_family": heldout_family,
        "heldout_machines": unseen_groups,
        "train_machines": train_groups,
        "reference_size": reference_size,
        "validation_fraction": validation_fraction,
        "seed": seed,
        "scaling_fit": "training_machine_healthy_only",
        "coupling": "independent",
        "frontend": "fisher",
        "feature_dim": int(embeddings.shape[-1]),
        "original_machine_test_seal": "retired_exploratory",
        "train_queries": len(train),
        "validation_queries": len(validation),
        "heldout_assessment_queries": sum(map(len, assessment.values())),
    }
    training = {
        "format": "nex233-healthy-context-train-v1",
        "features": scaled[queries].contiguous(),
        "references": torch.stack(
            [scaled[support[group], 0] for group in train_groups]
        ).contiguous(),
        "rows": [
            {
                **rows[i],
                "original_split": rows[i]["split"],
                "split": "train" if i in train_set else "val",
                "reference_index": group_index[str(rows[i]["group_id"])],
            }
            for i in queries
        ],
        "machine_groups": train_groups,
        "support_healthy_files": [
            [str(rows[i]["healthy_file"]) for i in support[group]]
            for group in train_groups
        ],
        "settings": settings,
        "training_mean": mean,
        "training_std": std,
    }
    evaluations = {
        group: {
            "format": "nex233-healthy-context-eval-v1",
            "features": scaled[indices].contiguous(),
            "references": scaled[support[group], 0].contiguous(),
            "rows": [
                {
                    **rows[i],
                    "original_split": rows[i]["split"],
                    "split": "assessment",
                }
                for i in indices
            ],
            "support_healthy_files": [
                str(rows[i]["healthy_file"]) for i in support[group]
            ],
            "heldout_machine": group,
            "settings": settings,
        }
        for group, indices in assessment.items()
    }
    return training, evaluations


def save_family_holdouts(
    source_file: str | Path,
    output_dir: str | Path,
    *,
    heldout_family: str | None = None,
    reference_size: int = 32,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Persist immutable family-held-out training and assessment artifacts.

    Returns:
        Run-compatible manifest pointing to each machine's assessment file.

    Raises:
        FileExistsError: If any prior output would be overwritten.
    """
    destination = Path(output_dir).expanduser().resolve()
    families = [heldout_family] if heldout_family is not None else list(MIMII_FAMILIES)
    if (destination / "manifest.json").exists():
        raise FileExistsError("Refusing to overwrite existing family holdout manifest")
    all_files = [
        destination / f"heldout-family-{family}-training.pt"
        for family in families
    ] + [
        destination / f"heldout-family-{family}-{identity}-assessment.pt"
        for family in families
        for identity in MIMII_IDS
    ]
    if any(path.exists() for path in all_files):
        raise FileExistsError("Refusing to overwrite existing family fold artifacts")
    folds = {
        family: build_family_holdout(
            source_file,
            heldout_family=family,
            reference_size=reference_size,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        for family in families
    }
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "protocol": "leave_one_family_out_healthy_reference",
        "source_file": str(Path(source_file).expanduser().resolve()),
        "seed": seed,
        "reference_size": reference_size,
        "validation_fraction": validation_fraction,
        "original_test_machine_repurposed": True,
        "folds": {},
    }
    for family, (training, evaluations) in folds.items():
        training_file = destination / f"heldout-family-{family}-training.pt"
        torch.save(training, training_file)
        read_healthy_reference_training(training_file)
        assessment_files: dict[str, str] = {}
        for group, payload in evaluations.items():
            identity = group.split(":")[1]
            path = destination / f"heldout-family-{family}-{identity}-assessment.pt"
            torch.save(payload, path)
            read_healthy_reference_assessment(path)
            assessment_files[group] = str(path)
        manifest["folds"][f"family:{family}"] = {
            "training_file": str(training_file),
            "assessment_files": assessment_files,
            "train_machines": training["machine_groups"],
            "train_queries": training["settings"]["train_queries"],
            "val_queries": training["settings"]["validation_queries"],
            "assessment_queries": sum(len(item["features"]) for item in evaluations.values()),
        }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare_multifamily_protocols(
    source_file: str | Path,
    output_dir: str | Path,
    *,
    protocols: str = "both",
    seeds: tuple[int, ...] = (42,),
    reference_size: int = 32,
    validation_fraction: float = 0.2,
) -> dict[str, Any]:
    """Prepare 16-machine and four-family holdouts for each split seed.

    Returns:
        Index containing one independent machine/family manifest per seed.

    Raises:
        ValueError: If complete MIMII source or requested options are invalid.
    """
    _validated_source(source_file)
    if protocols not in {"both", "machine", "family"} or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Choose machine/family/both and unique nonempty seeds")
    destination = Path(output_dir).expanduser().resolve()
    index_path = destination / "index.json"
    if index_path.exists():
        raise FileExistsError("Refusing to overwrite existing multi-family index")
    paths = {
        seed: {
            protocol: destination / f"seed{seed}" / protocol / "manifest.json"
            for protocol in ("machine", "family")
            if protocols in ("both", protocol)
        }
        for seed in seeds
    }
    if any(file.exists() for pair in paths.values() for file in pair.values()):
        raise FileExistsError("A requested seed/protocol manifest already exists")
    index: dict[str, Any] = {
        "source_file": str(Path(source_file).expanduser().resolve()),
        "seeds": list(seeds),
        "protocols": protocols,
        "manifests": {},
    }
    for seed, selected in paths.items():
        index["manifests"][str(seed)] = {}
        for protocol, manifest in selected.items():
            if protocol == "machine":
                result = save_reference_conditioned_folds(
                    source_file,
                    manifest.parent,
                    seed=seed,
                    reference_size=reference_size,
                    validation_fraction=validation_fraction,
                )
            else:
                result = save_family_holdouts(
                    source_file,
                    manifest.parent,
                    seed=seed,
                    reference_size=reference_size,
                    validation_fraction=validation_fraction,
                )
            index["manifests"][str(seed)][protocol] = {
                "manifest": str(manifest),
                "fold_count": len(result["folds"]),
            }
    destination.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    return index


def main() -> None:
    """Prepare multiple independent machine/family holdout folds and seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocols", choices=("both", "machine", "family"), default="both")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--reference-size", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    index = prepare_multifamily_protocols(
        args.source_file,
        args.output_dir,
        protocols=args.protocols,
        seeds=tuple(args.seeds),
        reference_size=args.reference_size,
        validation_fraction=args.validation_fraction,
    )
    print(json.dumps(index["manifests"], indent=2))


if __name__ == "__main__":
    main()
