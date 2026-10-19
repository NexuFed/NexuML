"""Machine-wise healthy/faulty embedding compatibility diagnostics for NEX-233.

This is a strictly descriptive analysis over prepared train/validation data.
No test samples, flow checkpoints or fault-label-informed cluster assignments
are used to learn a pseudo-machine-type.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from nexuml_library.data.maintenance.prepared_endpoints import read_prepared_artifact


def _projection_directions(dimension: int, projections: int, seed: int) -> Tensor:
    """Construct the same seeded projection basis as NEX-233 sliced Wasserstein.

    Returns:
        Matrix of unit directions with shape [dimension, projections].
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    directions = torch.randn(
        (dimension, projections), generator=generator, dtype=torch.float32
    )
    return torch.nn.functional.normalize(directions, p=2, dim=0)


def _projected_swd(source: Tensor, target: Tensor, generator: torch.Generator) -> float:
    """Compute pairing-invariant sliced W1 from cached projected observations.

    Unequal sample counts are balanced by seeded without-replacement sampling.
    When counts match, all observations are retained.

    Returns:
        Nonnegative sliced W1 across the shared projection directions.
    """
    n = min(len(source), len(target))
    if len(source) != n:
        source = source[torch.randperm(len(source), generator=generator)[:n]]
    if len(target) != n:
        target = target[torch.randperm(len(target), generator=generator)[:n]]
    return float(
        (source.sort(dim=0).values - target.sort(dim=0).values).abs().mean()
    )


def _interval(samples: list[float]) -> list[float]:
    """Compute empirical central 95 percent bootstrap interval.

    Returns:
        Lower and upper bootstrap percentiles.
    """
    return torch.quantile(torch.tensor(samples), torch.tensor([0.025, 0.975])).tolist()


def _distribution_pairs(
    embeddings: dict[str, Tensor],
    names: list[str],
    *,
    bootstrap: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    """Measure inter-machine sliced W1 and resampling uncertainty.

    Returns:
        Symmetric matrix and 95-percent bootstrap intervals for each pair.
    """
    distances = [[0.0 for _ in names] for _ in names]
    pairs: list[dict[str, Any]] = []
    for i, left in enumerate(names):
        for j in range(i + 1, len(names)):
            right = names[j]
            a, b = embeddings[left], embeddings[right]
            value = _projected_swd(a, b, generator)
            samples = [
                _projected_swd(
                    a[torch.randint(len(a), (len(a),), generator=generator)],
                    b[torch.randint(len(b), (len(b),), generator=generator)],
                    generator,
                )
                for _ in range(bootstrap)
            ]
            distances[i][j] = distances[j][i] = value
            pairs.append(
                {"left": left, "right": right, "swd": value, "bootstrap_ci95": _interval(samples)}
            )
    return {"labels": names, "matrix": distances, "pairs": pairs}


def _shift_cosine(a: Tensor, b: Tensor) -> float | None:
    """Compare two mean fault-displacement vectors without using sample pairing.

    Returns:
        Cosine similarity or None if at least one displacement is zero.
    """
    a_norm = float(torch.linalg.vector_norm(a))
    b_norm = float(torch.linalg.vector_norm(b))
    if a_norm <= 1e-12 or b_norm <= 1e-12:
        return None
    return float(torch.dot(a, b) / (a_norm * b_norm))


def _fault_direction_pairs(
    healthy: dict[str, Tensor],
    faulty: dict[str, Tensor],
    names: list[str],
    *,
    bootstrap: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    """Compare mean fault-displacement directions, with bootstrap intervals.

    Returns:
        Pairwise cosine matrix, bootstrap intervals, and displacement norms.
    """
    shifts = {name: faulty[name].mean(0) - healthy[name].mean(0) for name in names}
    matrix: list[list[float | None]] = [
        [_shift_cosine(shifts[left], shifts[right]) for right in names]
        for left in names
    ]
    pairs: list[dict[str, Any]] = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            values: list[float] = []
            for _ in range(bootstrap):
                boot_shifts: list[Tensor] = []
                for name in (left, right):
                    h, f = healthy[name], faulty[name]
                    h_sample = h[torch.randint(len(h), (len(h),), generator=generator)]
                    f_sample = f[torch.randint(len(f), (len(f),), generator=generator)]
                    boot_shifts.append(f_sample.mean(0) - h_sample.mean(0))
                cosine = _shift_cosine(boot_shifts[0], boot_shifts[1])
                if cosine is not None:
                    values.append(cosine)
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "cosine": _shift_cosine(shifts[left], shifts[right]),
                    "bootstrap_ci95": _interval(values) if values else None,
                }
            )
    return {
        "labels": names,
        "matrix": matrix,
        "pairs": pairs,
        "shift_l2_norm": {
            name: float(torch.linalg.vector_norm(shifts[name])) for name in names
        },
    }


def _split_half_reference(
    projections: dict[str, Tensor],
    *,
    bootstrap: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    """Estimate within-machine healthy variability using disjoint split halves.

    Returns:
        Per-machine median split-half SWD and 95 percent resampling interval.
    """
    output: dict[str, Any] = {}
    for name, matrix in projections.items():
        if len(matrix) < 4:
            output[name] = {"available": False, "reason": "fewer_than_four_recordings"}
            continue
        half = len(matrix) // 2
        samples = []
        for _ in range(bootstrap):
            order = torch.randperm(len(matrix), generator=generator)
            samples.append(
                _projected_swd(
                    matrix[order[:half]], matrix[order[half : 2 * half]], generator
                )
            )
        output[name] = {
            "available": True,
            "median": float(torch.tensor(samples).median()),
            "resampling_ci95": _interval(samples),
            "half_size": half,
        }
    return output


def _hierarchical_hypotheses(
    matrix: list[list[float]],
    names: list[str],
    train: list[str],
    validation: list[str],
) -> dict[str, Any]:
    """Describe average-linkage healthy clusters without fitting a model.

    Each cluster cut involving validation data is exploratory only.
    Two training identities cannot establish a transferable taxonomy.

    Returns:
        Merge history, exploratory two-group assignment, and nearest train reference.
    """
    clusters = [[name] for name in names]
    index = {name: i for i, name in enumerate(names)}
    merges: list[dict[str, Any]] = []
    k2: dict[str, int] | None = None
    while len(clusters) > 1:
        if len(clusters) == 2 and k2 is None:
            k2 = {
                name: cluster_id
                for cluster_id, members in enumerate(clusters)
                for name in members
            }
        candidates = []
        for i, left in enumerate(clusters):
            for j in range(i + 1, len(clusters)):
                right = clusters[j]
                distance = sum(
                    matrix[index[a]][index[b]] for a in left for b in right
                ) / (len(left) * len(right))
                candidates.append((distance, i, j))
        distance, i, j = min(candidates)
        first, second = clusters[i], clusters[j]
        merges.append({"left": first, "right": second, "average_linkage_swd": distance})
        clusters = [
            cluster for k, cluster in enumerate(clusters) if k not in (i, j)
        ] + [sorted(first + second)]
    nearest = {}
    for name in validation:
        ranking = sorted(
            ((machine, matrix[index[name]][index[machine]]) for machine in train),
            key=lambda pair: pair[1],
        )
        nearest[name] = {
            "nearest_training_machine": ranking[0][0],
            "healthy_swd": ranking[0][1],
            "second_nearest_swd": ranking[1][1] if len(ranking) > 1 else None,
        }
    return {
        "method": "descriptive_average_linkage_of_healthy_swd",
        "merges": merges,
        "exploratory_two_group_assignment": k2,
        "assignment_includes_validation_healthy_features": bool(validation),
        "nearest_training_reference": nearest,
        "usable_for_conditional_training": False,
        "limitation": (
            "Only two training machine identities; validation-inclusive "
            "clusters are transductive and cannot establish a transferable type."
        ),
    }


def _mean_shift_transfer(
    healthy: dict[str, Tensor],
    faulty: dict[str, Tensor],
    projected_healthy: dict[str, Tensor],
    projected_faulty: dict[str, Tensor],
    train: list[str],
    validation: list[str],
    directions: Tensor,
    generator: torch.Generator,
) -> dict[str, Any]:
    """Evaluate train-derived mean shifts on unseen healthy validation machines.

    Faulty validation samples are used only as evaluation targets; an oracle
    mean shift derived from them is reported separately as descriptive context.

    Returns:
        Per-validation-machine SWD for no shift, train shifts, and oracle shift.
    """
    pooled_h = torch.cat([healthy[name] for name in train])
    pooled_f = torch.cat([faulty[name] for name in train])
    shifts = {
        "train_pooled": pooled_f.mean(0) - pooled_h.mean(0),
        **{name: faulty[name].mean(0) - healthy[name].mean(0) for name in train},
    }
    output: dict[str, Any] = {}
    for name in validation:
        source = projected_healthy[name]
        target = projected_faulty[name]
        baseline = _projected_swd(source, target, generator)
        alternatives = {}
        for origin, shift in shifts.items():
            swd = _projected_swd(source + shift @ directions, target, generator)
            alternatives[origin] = {
                "transported_swd": swd,
                "relative_reduction": 1.0 - swd / baseline if baseline > 0 else None,
            }
        oracle_shift = faulty[name].mean(0) - healthy[name].mean(0)
        oracle_swd = _projected_swd(
            source + oracle_shift @ directions, target, generator
        )
        output[name] = {
            "no_transport_swd": baseline,
            "train_derived_shifts": alternatives,
            "oracle_validation_mean_shift_swd_diagnostic_only": oracle_swd,
        }
    return output


@torch.no_grad()
def diagnose_machine_compatibility(
    prepared_file: str | Path,
    *,
    projections: int = 128,
    bootstrap: int = 64,
    seed: int = 42,
) -> dict[str, Any]:
    """Analyze machine-specific healthy/faulty marginals without reading test features.

    Fit no representation or conditioning model; use only prepared training
    features plus validation features for held-out descriptive comparisons.
    Test features are never indexed; no checkpoint selection uses this report.

    Args:
        prepared_file: Previously frozen independent/OT feature artifact.
        projections: Shared random projection count across all machines.
        bootstrap: Resampling repetitions for descriptive uncertainty.
        seed: Fixed seed for projections, subsampling, and bootstrap.

    Returns:
        Pairwise healthy/faulty SWD, shift alignment, healthy cluster
        hypotheses and train-derived mean-shift validation diagnostics.

    Raises:
        ValueError: If no training/validation machines, invalid controls,
            or nonfinite prepared embeddings are encountered.
    """
    if projections <= 0 or bootstrap < 2:
        raise ValueError("Positive projections and at least two bootstrap draws required")
    payload = read_prepared_artifact(prepared_file)
    rows = cast(list[dict[str, Any]], payload["rows"])
    features = cast(Tensor, payload["features"])
    grouped: dict[str, dict[str, Tensor]] = {}
    # Explicitly index only train/val; test observations never enter diagnostics.
    for split in ("train", "val"):
        for name in sorted({
            str(row["group_id"]) for row in rows if row["split"] == split
        }):
            indices = [
                i for i, row in enumerate(rows)
                if row["split"] == split and row["group_id"] == name
            ]
            grouped[name] = {
                "healthy": features[indices, 0].float().cpu(),
                "faulty": features[indices, 1].float().cpu(),
            }
    training = sorted({
        str(row["group_id"]) for row in rows if row["split"] == "train"
    })
    validation = sorted({
        str(row["group_id"]) for row in rows if row["split"] == "val"
    })
    if not training or not validation or set(training) & set(validation):
        raise ValueError("Expected disjoint nonempty training/validation machine groups")
    names = training + validation
    directions = _projection_directions(features.shape[-1], projections, seed)
    # Canonicalize each marginal independently: seeded bootstrap estimates
    # must not change when endpoint pair assignments are permuted.
    axis = directions[:, 0]
    healthy = {
        name: grouped[name]["healthy"][
            torch.argsort(grouped[name]["healthy"] @ axis, stable=True)
        ]
        for name in names
    }
    faulty = {
        name: grouped[name]["faulty"][
            torch.argsort(grouped[name]["faulty"] @ axis, stable=True)
        ]
        for name in names
    }
    projected_h = {name: healthy[name] @ directions for name in names}
    projected_f = {name: faulty[name] @ directions for name in names}
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)

    h_pairs = _distribution_pairs(
        projected_h, names, bootstrap=bootstrap, generator=generator
    )
    f_pairs = _distribution_pairs(
        projected_f, names, bootstrap=bootstrap, generator=generator
    )
    settings = cast(dict[str, Any], payload["settings"])
    return {
        "format": "nex233-machine-compatibility-v1",
        "source_artifact": str(Path(prepared_file).expanduser().resolve()),
        "feature_settings": {
            key: settings.get(key)
            for key in (
                "frontend",
                "coupling",
                "scaling",
                "seed",
                "feature_dim",
                "model_name",
                "revision",
                "machine_type",
                "noise_db",
                "normalization_fit_split",
                "ot_fit_split",
            )
        },
        "projections": projections,
        "bootstrap_repetitions": bootstrap,
        "seed": seed,
        "machines": {
            name: {"split": "train" if name in training else "val",
                   "num_recordings": len(healthy[name])}
            for name in names
        },
        "healthy_swd": h_pairs,
        "faulty_swd": f_pairs,
        "healthy_split_half_swd": _split_half_reference(
            projected_h, bootstrap=bootstrap, generator=generator
        ),
        "mean_fault_direction_cosine": _fault_direction_pairs(
            healthy, faulty, names, bootstrap=bootstrap, generator=generator
        ),
        "healthy_cluster_hypotheses": _hierarchical_hypotheses(
            h_pairs["matrix"], names, training, validation
        ),
        "mean_shift_transfer_to_validation": _mean_shift_transfer(
            healthy, faulty, projected_h, projected_f, training, validation,
            directions, generator
        ),
        "test_is_sealed": True,
        "interpretation": (
            "Descriptive diagnostics, not validated pseudo-machine types. "
            "Validation fault labels are used only for diagnosing shift and "
            "mean-shift transfer; cluster hypotheses use healthy data only. "
            "No test observations are scored."
        ),
    }


def main() -> None:
    """Run machine-compatibility diagnostics from an immutable feature artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--projections", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plots-dir", type=Path)
    args = parser.parse_args()
    report = diagnose_machine_compatibility(
        args.prepared_file,
        projections=args.projections,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.plots_dir is not None:
        from nexuml_library.evaluation.maintenance.machine_compatibility_plots import (
            plot_machine_compatibility,
        )

        plot_machine_compatibility(report, args.prepared_file, args.plots_dir)
    print(f"Diagnostics written to {args.output}")
    print("Test split is sealed; pseudo-machine-type clusters remain exploratory.")


if __name__ == "__main__":
    main()
