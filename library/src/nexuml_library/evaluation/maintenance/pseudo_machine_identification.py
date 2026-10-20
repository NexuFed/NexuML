"""Identify FISHER machine identities and exploratory healthy-derived pseudo-types.

NEX-233 diagnostic: measure healthy-to-faulty routing robustness before
implementing conditional Flow Matching. Known machine IDs are proxy labels,
not proven transferable pseudo-machine-types. The original id_06 test
identity is deliberately retired for this exploratory experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from nexuml_library.data.maintenance.prepared_endpoints import read_prepared_artifact

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _load_fisher(source_file: str | Path) -> tuple[Tensor, list[str], list[str]]:
    """Read raw, independently paired FISHER data and validate recording identity.

    Returns:
        Raw endpoints, per-pair machine groups and sorted machine identities.

    Raises:
        ValueError: If the artifact is unsuitable for this diagnostic.
    """
    payload = read_prepared_artifact(source_file)
    settings = cast(dict[str, Any], payload["settings"])
    if (
        settings.get("frontend") != "fisher"
        or settings.get("coupling") != "independent"
        or settings.get("scaling") != "raw"
        or settings.get("split_protocol") == "within_machine_recording_holdout"
    ):
        raise ValueError("Expected original FISHER-independent-raw prepared artifact")
    rows = cast(list[dict[str, Any]], payload["rows"])
    endpoints = cast(Tensor, payload["features"]).float().cpu()
    files: list[str] = []
    for row in rows:
        for key in ("healthy_file", "faulty_file"):
            path = row.get(key)
            if not isinstance(path, str) or not path:
                raise ValueError(f"Missing original recording provenance: {key}")
            files.append(path)
    if len(files) != len(set(files)):
        raise ValueError("Duplicate recording paths would leak across splits")
    groups = [str(row["group_id"]) for row in rows]
    names = sorted(set(groups))
    if len(names) < 3:
        raise ValueError("Need at least three machine identities for pseudo-type K=2")
    if not any(row["split"] == "test" for row in rows):
        raise ValueError("Expected original machine-disjoint dataset with retired test ID")
    return endpoints, groups, names


def _machine_partitions(
    groups: list[str],
    names: list[str],
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, list[int]]:
    """Stratify paired recording IDs, never individual endpoint observations.

    Returns:
        Train, validation and assessment row indices without overlap.

    Raises:
        ValueError: If fractions or per-machine recording counts are invalid.
    """
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1):
        raise ValueError("Fractions must be strictly between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Training and validation fractions leave no assessment data")
    generator = np.random.default_rng(seed)
    partitions: dict[str, list[int]] = {"train": [], "validation": [], "assessment": []}
    for name in names:
        rows = np.asarray([i for i, group in enumerate(groups) if group == name])
        if len(rows) < 10:
            raise ValueError(f"At least ten pairs required for {name}")
        indices = generator.permutation(rows).tolist()
        n_train = round(len(rows) * train_fraction)
        n_val = round(len(rows) * validation_fraction)
        if min(n_train, n_val, len(rows) - n_train - n_val) < 2:
            raise ValueError(f"Each partition needs at least two recordings: {name}")
        partitions["train"].extend(indices[:n_train])
        partitions["validation"].extend(indices[n_train : n_train + n_val])
        partitions["assessment"].extend(indices[n_train + n_val :])
    return partitions


def _swd_matrix(
    projected: dict[str, FloatArray], names: list[str]
) -> list[list[float]]:
    """Compute machine-wise sliced W1 from healthy training projections.

    Assumes the same projection basis across all machine distributions.
    Every source/target projection is sorted independently, so assignments
    between healthy and faulty recordings play no role.

    Returns:
        Symmetric pairwise sliced-W1 matrix.
    """
    matrix = [[0.0 for _ in names] for _ in names]
    for i, left in enumerate(names):
        for j in range(i + 1, len(names)):
            a, b = projected[left], projected[names[j]]
            n = min(len(a), len(b))
            # All machines have the same number of train recordings in the
            # standard NEX-233 protocol. Reject unequal counts instead of
            # introducing stochastic subsampling into clustering.
            if len(a) != len(b):
                raise ValueError("Healthy SWD clustering requires balanced machine samples")
            distance = float(np.mean(np.abs(np.sort(a, axis=0) - np.sort(b, axis=0))))
            if n == 0:
                raise ValueError("Empty machine distribution")
            matrix[i][j] = matrix[j][i] = distance
    return matrix


def _clusters(matrix: list[list[float]], names: list[str], count: int) -> dict[str, int]:
    """Derive deterministic average-linkage pseudo-types from healthy SWD.

    Returns:
        Machine identity to pseudo-type index, canonically ordered by names.

    Raises:
        ValueError: If the number of pseudo-types is outside 2..N-1.
    """
    if not 2 <= count < len(names):
        raise ValueError("Pseudo-type count must be between 2 and machines minus 1")
    clusters = [[name] for name in names]
    indices = {name: i for i, name in enumerate(names)}
    while len(clusters) > count:
        candidates: list[tuple[float, int, int]] = []
        for i, left in enumerate(clusters):
            for j in range(i + 1, len(clusters)):
                right = clusters[j]
                d = sum(matrix[indices[a]][indices[b]] for a in left for b in right)
                d /= len(left) * len(right)
                candidates.append((d, i, j))
        _, first, second = min(candidates)
        merged = sorted(clusters[first] + clusters[second])
        clusters = [
            cluster for index, cluster in enumerate(clusters)
            if index not in (first, second)
        ] + [merged]
    canonical = sorted(clusters, key=lambda members: members[0])
    return {name: i for i, members in enumerate(canonical) for name in members}


def _score(
    truth: IntArray,
    predicted: IntArray,
    probabilities: FloatArray,
    labels: list[int],
) -> dict[str, Any]:
    """Summarize recognition and routing confidence on unseen recordings.

    Returns:
        Accuracy, balanced accuracy, macro F1, per-class recall, confusion
        matrix, multiclass Brier score and expected calibration error.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
    )

    matrix = confusion_matrix(truth, predicted, labels=labels)
    confidence = probabilities.max(axis=1)
    correct = predicted == truth
    expected_calibration_error = 0.0
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if np.any(selected):
            expected_calibration_error += float(np.mean(selected)) * abs(
                float(np.mean(correct[selected])) - float(np.mean(confidence[selected]))
            )
    one_hot = np.eye(len(labels))[truth]
    return {
        "num_recordings": int(len(truth)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=labels, zero_division=0)),
        "recall_by_label": {
            str(label): float(matrix[i, i] / matrix[i].sum())
            if matrix[i].sum() else None
            for i, label in enumerate(labels)
        },
        "confusion_matrix": matrix.tolist(),
        "mean_max_probability": float(confidence.mean()),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece_10_bins": expected_calibration_error,
    }


def _logistic_report(
    train: FloatArray,
    train_labels: IntArray,
    eval_sets: dict[str, tuple[FloatArray, IntArray]],
    labels: list[int],
) -> dict[str, Any]:
    """Train a fixed linear routing classifier and evaluate held-out conditions.

    Returns:
        Condition-specific metrics and class prior with no parameter tuning.
    """
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    model.fit(train, train_labels)
    result: dict[str, Any] = {}
    for key, (features, truth) in eval_sets.items():
        predicted = cast(IntArray, model.predict(features))
        raw_probs = cast(FloatArray, model.predict_proba(features))
        probabilities = np.zeros((len(features), len(labels)), dtype=np.float64)
        for column, cls in enumerate(model.classes_):
            probabilities[:, int(cls)] = raw_probs[:, column]
        result[key] = _score(truth, predicted, probabilities, labels)
    return result


def _prototype_report(
    healthy_train: FloatArray,
    y_train: IntArray,
    eval_sets: dict[str, tuple[FloatArray, IntArray]],
    labels: list[int],
) -> dict[str, Any]:
    """Provide a no-optimization machine-identity centroid baseline.

    Returns:
        Evaluation metrics on healthy and faulty held-out recordings.
    """
    centroids = np.stack([healthy_train[y_train == cls].mean(axis=0) for cls in labels])
    result: dict[str, Any] = {}
    for key, (features, truth) in eval_sets.items():
        distances = ((features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        predicted = np.argmin(distances, axis=1).astype(np.int64)
        # Distances are not calibrated: report only accuracy/F1/confusion.
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

        result[key] = {
            "num_recordings": int(len(truth)),
            "accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, labels=labels, zero_division=0)),
            "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
        }
    return result


def _one_seed(
    embeddings: Tensor,
    groups: list[str],
    names: list[str],
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    pca_components: int,
    projections: int,
    bootstrap: int,
) -> dict[str, Any]:
    """Fit healthy-only preprocessing and report closed-set routing diagnostics.

    Returns:
        Reproducible split, cluster stability and predictive metrics per seed.
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import StandardScaler

    partition = _machine_partitions(
        groups, names, seed=seed,
        train_fraction=train_fraction, validation_fraction=validation_fraction
    )
    ids = {name: i for i, name in enumerate(names)}
    data = embeddings.numpy().astype(np.float64)
    machine_labels = np.asarray([ids[name] for name in groups], dtype=np.int64)
    training_ids = np.asarray(partition["train"], dtype=np.int64)
    healthy_train = data[training_ids, 0]
    standardizer = StandardScaler().fit(healthy_train)
    # No validation/assessment or faulty observations fit preprocessing.
    normalized = standardizer.transform(data.reshape(-1, data.shape[-1]))
    normalized = normalized.reshape(data.shape)
    components = min(pca_components, len(healthy_train) - 1, embeddings.shape[-1])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    pca.fit(normalized[training_ids, 0])
    latent = pca.transform(normalized.reshape(-1, embeddings.shape[-1])).reshape(
        len(data), 2, components
    )
    generator = np.random.default_rng(seed + 10_000)
    directions = generator.normal(size=(embeddings.shape[-1], projections))
    directions /= np.maximum(np.linalg.norm(directions, axis=0, keepdims=True), 1e-12)
    healthy_projected = normalized[training_ids, 0] @ directions
    per_machine = {
        name: healthy_projected[machine_labels[training_ids] == ids[name]]
        for name in names
    }
    matrix = _swd_matrix(per_machine, names)

    cluster_hypotheses: dict[str, Any] = {}
    for count in (2, 3):
        if count >= len(names):
            continue
        original = _clusters(matrix, names, count)
        assignment = np.asarray([original[name] for name in names], dtype=np.int64)
        agreement: list[float] = []
        partitions_equal = 0
        for _ in range(bootstrap):
            samples = {
                name: x[generator.integers(0, len(x), size=len(x))]
                for name, x in per_machine.items()
            }
            labels = _clusters(_swd_matrix(samples, names), names, count)
            trial = np.asarray([labels[name] for name in names], dtype=np.int64)
            agreement.append(float(adjusted_rand_score(assignment, trial)))
            partitions_equal += int(
                all((assignment[i] == assignment[j]) == (trial[i] == trial[j])
                    for i in range(len(names)) for j in range(len(names)))
            )
        cluster_hypotheses[f"k{count}"] = {
            "machine_to_pseudo_type": original,
            "pseudo_type_sizes_in_machine_identities": [
                sum(value == cls for value in original.values()) for cls in range(count)
            ],
            "bootstrap_mean_ari": float(np.mean(agreement)),
            "bootstrap_exact_partition_fraction": partitions_equal / bootstrap,
            "bootstrap_repetitions": bootstrap,
            "model_selection_uses_faulty_embeddings": False,
        }

    representations = {"machine_identity": machine_labels}
    for key, row in cluster_hypotheses.items():
        assignment = cast(dict[str, int], row["machine_to_pseudo_type"])
        representations[f"pseudo_{key}"] = np.asarray(
            [assignment[name] for name in groups], dtype=np.int64
        )
    results: dict[str, Any] = {}
    for target_name, y in representations.items():
        classes = sorted(set(y.tolist()))
        result: dict[str, Any] = {
            "class_labels": [
                names[label] if target_name == "machine_identity" else str(label)
                for label in classes
            ],
            "chance_accuracy_balanced": 1.0 / len(classes),
            "classifiers": {},
        }
        for training_condition in ("healthy_only", "faulty_only", "mixed"):
            training_parts = (
                (0,) if training_condition == "healthy_only" else
                (1,) if training_condition == "faulty_only" else (0, 1)
            )
            train_x = np.concatenate([latent[training_ids, state] for state in training_parts])
            train_y = np.concatenate([y[training_ids] for _ in training_parts])
            evaluations = {
                f"{split}_{state_name}": (
                    latent[partition[split], state], y[partition[split]]
                )
                for split in ("validation", "assessment")
                for state, state_name in ((0, "healthy"), (1, "faulty"))
            }
            result["classifiers"][training_condition] = _logistic_report(
                train_x, train_y, evaluations, classes
            )
        if target_name == "machine_identity":
            evaluations = {
                f"{split}_{state_name}": (
                    latent[partition[split], state], y[partition[split]]
                )
                for split in ("validation", "assessment")
                for state, state_name in ((0, "healthy"), (1, "faulty"))
            }
            result["centroid_healthy_only"] = _prototype_report(
                latent[training_ids, 0], y[training_ids], evaluations, classes
            )
        results[target_name] = result

    return {
        "seed": seed,
        "counts": {
            split: {
                name: sum(groups[index] == name for index in indices)
                for name in names
            }
            for split, indices in partition.items()
        },
        "preprocessing": {
            "standardizer_fit": "training_healthy_only",
            "pca_fit": "training_healthy_only",
            "pca_components": components,
            "pca_explained_variance_ratio_sum": float(
                np.sum(pca.explained_variance_ratio_)
            ),
            "source_fisher_dim": int(embeddings.shape[-1]),
        },
        "healthy_train_swd": {"labels": names, "matrix": matrix},
        "pseudo_type_hypotheses": cluster_hypotheses,
        "targets": results,
    }


def identify_pseudo_machine_types(
    source_file: str | Path,
    *,
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    pca_components: int = 32,
    projections: int = 128,
    bootstrap: int = 32,
) -> dict[str, Any]:
    """Compare known-machine and healthy-cluster identification in FISHER space.

    Classifiers are fit separately on healthy, faulty, and combined
    **training** recordings. Healthy-only reference classifiers are the
    principal deployment-relevant condition. All evaluations are on
    disjoint validation/assessment recordings, by acoustic state.

    Healthy training embeddings alone determine the pseudo-type clustering.
    This is a known-machine, closed-set diagnostic, NOT proof of transferable
    pseudo-types or of classification of unseen degradation severities.

    Args:
        source_file: Original FISHER independent raw prepared dataset.
        seeds: Repeatable independent within-machine train/val/assessment splits.
        train_fraction: Per-machine fraction for training.
        validation_fraction: Per-machine fraction for validation.
        pca_components: Dimensionality fitted on healthy training embeddings.
        projections: Random directions for healthy machine SWD.
        bootstrap: Healthy-training bootstrap draws for cluster stability.

    Returns:
        Per-seed reports with confusion matrices and aggregate accuracies.

    Raises:
        ValueError: If configuration is invalid or source provenance is missing.
    """
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and unique")
    if pca_components <= 0 or projections <= 0 or bootstrap < 2:
        raise ValueError("PCA, projections and bootstrap counts must be positive")
    endpoints, groups, names = _load_fisher(source_file)
    runs = [
        _one_seed(
            endpoints, groups, names, seed=seed, train_fraction=train_fraction,
            validation_fraction=validation_fraction, pca_components=pca_components,
            projections=projections, bootstrap=bootstrap
        )
        for seed in seeds
    ]
    averages: dict[str, Any] = {}
    for target, target_report in runs[0]["targets"].items():
        averages[target] = {}
        for regime, evaluations in target_report["classifiers"].items():
            averages[target][regime] = {}
            for split_state in evaluations:
                per_seed = [
                    run["targets"][target]["classifiers"][regime][split_state]
                    for run in runs
                ]
                averages[target][regime][split_state] = {
                    key: {
                        "mean": float(np.mean([row[key] for row in per_seed])),
                        "std_across_splits": float(
                            np.std([row[key] for row in per_seed], ddof=1)
                        ) if len(per_seed) > 1 else None,
                    }
                    for key in ("accuracy", "balanced_accuracy", "macro_f1",
                                "multiclass_brier", "ece_10_bins")
                }
        if target == "machine_identity":
            averages[target]["centroid_healthy_only"] = {
                split_state: {
                    key: float(np.mean([
                        run["targets"][target]["centroid_healthy_only"][split_state][key]
                        for run in runs
                    ]))
                    for key in ("accuracy", "macro_f1")
                }
                for split_state in runs[0]["targets"][target]["centroid_healthy_only"]
            }
    return {
        "format": "nex233-pseudo-machine-identification-v1",
        "source_artifact": str(Path(source_file).expanduser().resolve()),
        "frontend": "fisher",
        "machine_groups": names,
        "original_test_machine_repurposed": True,
        "splits": "recording_disjoint_within_each_known_machine",
        "preprocessing_fit": "healthy_training_only",
        "seeds": list(seeds),
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "assessment_fraction": 1.0 - train_fraction - validation_fraction,
        "bootstrap_repetitions": bootstrap,
        "runs": runs,
        "aggregate": averages,
        "caveats": [
            "MIMII contains real machine IDs but no ground-truth pseudo-machine-type classes.",
            "Pseudo-type labels are train-healthy-only hierarchical cluster hypotheses.",
            "Closed-set classification: all evaluated machine identities appear in training.",
            "Faulty endpoint does not specify a degradation stage or RUL.",
            "Faulty-only and mixed training are diagnostic upper-bound comparisons.",
            "Cluster-label predictability does not demonstrate shared fault mechanisms.",
            "id_06 was previously a test machine and is now exploratory development data.",
        ],
    }


def main() -> None:
    """Write seeded machine-identity and pseudo-type inference report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--projections", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=32)
    args = parser.parse_args()
    report = identify_pseudo_machine_types(
        args.source_file,
        seeds=tuple(args.seeds),
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        pca_components=args.pca_components,
        projections=args.projections,
        bootstrap=args.bootstrap,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for regime in ("healthy_only", "faulty_only", "mixed"):
        target = report["aggregate"]["machine_identity"][regime]
        healthy = target["assessment_healthy"]["accuracy"]["mean"]
        faulty = target["assessment_faulty"]["accuracy"]["mean"]
        print(f"Identity routing {regime}: healthy={healthy:.2%}, faulty={faulty:.2%}")
    print(f"Saved exploratory identification diagnostics to {args.output}")


if __name__ == "__main__":
    main()
