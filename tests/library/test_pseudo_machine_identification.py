"""Tests for closed-set machine recognition and healthy-derived pseudo-types."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from nexuml_library.evaluation.maintenance.pseudo_machine_identification import (
    _clusters,
    _load_fisher,
    _machine_partitions,
    identify_pseudo_machine_types,
)


def _source(tmp_path: Path) -> Path:
    """Generate independent raw FISHER endpoints with four machine identities.

    Returns:
        Complete synthetic machine-disjoint prepared feature artifact.
    """
    generator = torch.Generator().manual_seed(7)
    points: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for index, (group, split) in enumerate(
        (
            ("fan:id_00", "train"),
            ("fan:id_02", "train"),
            ("fan:id_04", "val"),
            ("fan:id_06", "test"),
        )
    ):
        center = torch.zeros(8)
        center[index] = 4.0
        for recording in range(20):
            healthy = center + 0.1 * torch.randn(8, generator=generator)
            faulty = healthy + 0.7 * torch.randn(8, generator=generator)
            points.append(torch.stack((healthy, faulty)))
            rows.append(
                {
                    "group_id": group,
                    "split": split,
                    "paired": 1,
                    "healthy_file": f"{group}/normal/{recording}.wav",
                    "faulty_file": f"{group}/abnormal/{recording}.wav",
                }
            )
    path = tmp_path / "fisher-independent-raw-seed42.pt"
    torch.save(
        {
            "format": "nex233-features-v1",
            "features": torch.stack(points),
            "rows": rows,
            "settings": {
                "frontend": "fisher",
                "coupling": "independent",
                "scaling": "raw",
                "seed": 42,
                "feature_dim": 8,
            },
        },
        path,
    )
    return path


def test_identification_evaluates_healthy_and_faulty_with_separate_splits(
    tmp_path: Path,
) -> None:
    """Closed-set evaluation uses all four machines without mixing recordings."""
    pytest.importorskip("sklearn")
    report = identify_pseudo_machine_types(
        _source(tmp_path), seeds=(42, 43),
        pca_components=4, projections=16, bootstrap=4
    )
    assert report["original_test_machine_repurposed"]
    assert len(report["machine_groups"]) == 4
    assert set(report["aggregate"]) == {
        "machine_identity", "pseudo_k2", "pseudo_k3"
    }
    for run in report["runs"]:
        for machine in report["machine_groups"]:
            assert [
                run["counts"][split][machine]
                for split in ("train", "validation", "assessment")
            ] == [12, 4, 4]
        assert run["preprocessing"]["standardizer_fit"] == "training_healthy_only"
        assert run["preprocessing"]["pca_fit"] == "training_healthy_only"
        assert len(run["pseudo_type_hypotheses"]["k2"]["machine_to_pseudo_type"]) == 4
        for key in ("k2", "k3"):
            compatibility = run["pseudo_type_hypotheses"][key][
                "held_out_fault_compatibility"
            ]
            assert len(compatibility["pairs"]) == 6
            assert not compatibility["assessment_faulty_used_to_assign_pseudo_types"]
            assert all(
                "assessment_mean_fault_direction_cosine" in pair
                for pair in compatibility["pairs"]
            )
        for target, data in run["targets"].items():
            classes = len(data["class_labels"])
            assert classes == (4 if target == "machine_identity" else int(target[-1]))
            for regime in ("healthy_only", "faulty_only", "mixed"):
                for split in ("validation", "assessment"):
                    for state in ("healthy", "faulty"):
                        score = data["classifiers"][regime][f"{split}_{state}"]
                        assert score["num_recordings"] == 16
                        assert len(score["confusion_matrix"]) == classes
                        assert all(len(row) == classes for row in score["confusion_matrix"])
                        assert sum(sum(row) for row in score["confusion_matrix"]) == 16
                        assert 0 <= score["accuracy"] <= 1
                        assert 0 <= score["ece_10_bins"] <= 1
                        assert score["multiclass_brier"] >= 0


def test_healthy_pseudo_clusters_do_not_depend_on_faulty_embeddings(
    tmp_path: Path,
) -> None:
    """Changing all faulty inputs cannot change healthy-inferred clusters."""
    pytest.importorskip("sklearn")
    source = _source(tmp_path)
    before = identify_pseudo_machine_types(
        source, seeds=(42,), pca_components=4, projections=8, bootstrap=4
    )
    data = torch.load(source, map_location="cpu", weights_only=True)
    data["features"][:, 1] = 100.0 * torch.randn(
        data["features"][:, 1].shape, generator=torch.Generator().manual_seed(99)
    )
    torch.save(data, source)
    after = identify_pseudo_machine_types(
        source, seeds=(42,), pca_components=4, projections=8, bootstrap=4
    )
    assert before["runs"][0]["healthy_train_swd"] == after["runs"][0]["healthy_train_swd"]
    for key in ("k2", "k3"):
        old = before["runs"][0]["pseudo_type_hypotheses"][key]
        new = after["runs"][0]["pseudo_type_hypotheses"][key]
        for field in (
            "machine_to_pseudo_type",
            "pseudo_type_sizes_in_machine_identities",
            "bootstrap_mean_ari",
            "bootstrap_exact_partition_fraction",
        ):
            assert old[field] == new[field]
    for target in ("machine_identity", "pseudo_k2", "pseudo_k3"):
        old = before["runs"][0]["targets"][target]["classifiers"]["healthy_only"]
        new = after["runs"][0]["targets"][target]["classifiers"]["healthy_only"]
        for split in ("validation_healthy", "assessment_healthy"):
            assert old[split] == new[split]


def test_machine_splits_disjoint_and_deterministic(tmp_path: Path) -> None:
    """The same original file is never represented in two split partitions."""
    embeddings, groups, names = _load_fisher(_source(tmp_path))
    first = _machine_partitions(
        groups, names, seed=42, train_fraction=0.6, validation_fraction=0.2
    )
    second = _machine_partitions(
        groups, names, seed=42, train_fraction=0.6, validation_fraction=0.2
    )
    assert first == second
    assert embeddings.shape[0] == 80
    assert sorted(sum(first.values(), [])) == list(range(80))
    assert all(
        set(first[left]).isdisjoint(first[right])
        for left, right in (("train", "validation"), ("train", "assessment"),
                            ("validation", "assessment"))
    )


def test_cluster_labels_follow_healthy_distances_not_fault_annotations() -> None:
    """Average-linkage respects healthy SWD and has canonical cluster labels."""
    names = ["a", "b", "c", "d"]
    distances = [
        [0.0, 0.1, 4.0, 4.0],
        [0.1, 0.0, 4.0, 4.0],
        [4.0, 4.0, 0.0, 0.2],
        [4.0, 4.0, 0.2, 0.0],
    ]
    assert _clusters(distances, names, 2) == {"a": 0, "b": 0, "c": 1, "d": 1}
    with pytest.raises(ValueError, match="Pseudo-type count"):
        _clusters(distances, names, 4)


def test_invalid_or_duplicated_source_and_splits_are_rejected(tmp_path: Path) -> None:
    """Prevent shared-recording leakage and invalid split controls."""
    source = _source(tmp_path)
    data = torch.load(source, map_location="cpu", weights_only=True)
    data["rows"][0]["faulty_file"] = data["rows"][0]["healthy_file"]
    torch.save(data, source)
    with pytest.raises(ValueError, match="Duplicate recording"):
        _load_fisher(source)
    groups = [str(i % 4) for i in range(80)]
    with pytest.raises(ValueError, match="leave no assessment"):
        _machine_partitions(
            groups, sorted(set(groups)), seed=42,
            train_fraction=0.8, validation_fraction=0.2
        )
