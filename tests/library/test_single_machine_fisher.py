"""Leakage-safe within-machine FISHER preparation and isolated training tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from nexuml.training.lightning import NexuSession
from nexuml_library.data.maintenance.prepared_endpoints import read_prepared_artifact
from nexuml_library.evaluation.maintenance.ablation_evaluate import (
    evaluate_prepared_checkpoints,
)
from nexuml_library.evaluation.maintenance.single_machine_prepare import (
    build_single_machine_artifacts,
    save_single_machine_artifacts,
)
from nexuml_library.evaluation.maintenance.single_machine_run import (
    run_single_machine_experiments,
)
from nexuml_library.scenarios.flow.mimii_acoustic_ablation import mimii_acoustic_ablation


def _source(tmp_path: Path) -> Path:
    """Create a raw, independently paired FISHER artifact with four machines.

    Returns:
        Original machine-disjoint source file with unique recording identities.
    """
    features: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for group, split, offset in (
        ("fan:id_00", "train", 0.0),
        ("fan:id_02", "train", 2.0),
        ("fan:id_04", "val", 4.0),
        ("fan:id_06", "test", 6.0),
    ):
        for i in range(12):
            healthy = torch.tensor(
                [offset + 0.1 * i, float(i % 3), float(i % 4), float(i % 5)]
            )
            faulty = healthy + torch.tensor([0.3, 0.2, -0.1, 0.5])
            features.append(torch.stack([healthy, faulty]))
            rows.append(
                {
                    "group_id": group,
                    "split": split,
                    "healthy_file": f"{group}-healthy-{i}.wav",
                    "faulty_file": f"{group}-faulty-{i}.wav",
                    "pair_id": f"{group}:{i}",
                    "paired": 1,
                }
            )
    file = tmp_path / "fisher-independent-raw-seed42.pt"
    torch.save(
        {
            "format": "nex233-features-v1",
            "features": torch.stack(features),
            "rows": rows,
            "settings": {
                "frontend": "fisher",
                "coupling": "independent",
                "scaling": "raw",
                "seed": 42,
                "feature_dim": 4,
                "normalization_fit_split": "train",
            },
        },
        file,
    )
    return file


def test_single_machine_holdouts_are_disjoint_and_train_normalized(
    tmp_path: Path,
) -> None:
    """Every machine gets isolated recordings and local train-only scaling."""
    file = _source(tmp_path)
    before = read_prepared_artifact(file)
    artifacts = build_single_machine_artifacts(file, seed=42)
    assert set(artifacts) == {"fan:id_00", "fan:id_02", "fan:id_04"}

    for machine, artifact in artifacts.items():
        rows = artifact["rows"]
        features = artifact["features"]
        settings = artifact["settings"]
        assert len(rows) == 12
        assert sum(row["split"] == "train" for row in rows) == 9
        assert sum(row["split"] == "val" for row in rows) == 3
        assert {row["group_id"] for row in rows} == {machine}
        assert not settings["original_test_machine_repurposed"]
        train = features[[i for i, row in enumerate(rows) if row["split"] == "train"]]
        assert torch.allclose(train.reshape(-1, 4).mean(0), torch.zeros(4), atol=1e-5)
        assert torch.allclose(
            train.reshape(-1, 4).std(0, unbiased=False), torch.ones(4), atol=1e-5
        )
        for key in ("healthy_file", "faulty_file"):
            training = {row[key] for row in rows if row["split"] == "train"}
            validation = {row[key] for row in rows if row["split"] == "val"}
            assert training.isdisjoint(validation)

    again = build_single_machine_artifacts(file, seed=42)
    for machine in artifacts:
        assert torch.equal(artifacts[machine]["features"], again[machine]["features"])
        assert artifacts[machine]["rows"] == again[machine]["rows"]
    # The original (including its sealed test rows) was not modified.
    assert torch.equal(before["features"], read_prepared_artifact(file)["features"])


def test_single_machine_preparation_writes_immutable_artifacts(tmp_path: Path) -> None:
    """Emit machine manifests with no original test machine by default."""
    file = _source(tmp_path)
    root = tmp_path / "within-machine"
    manifest = save_single_machine_artifacts(file, root, seed=42)
    assert not manifest["original_test_machine_included"]
    assert len(manifest["machines"]) == 3
    for machine, row in manifest["machines"].items():
        payload = read_prepared_artifact(row["prepared_file"])
        assert payload["settings"]["machine_group"] == machine
        assert {entry["split"] for entry in payload["rows"]} == {"train", "val"}
    with pytest.raises(FileExistsError, match="overwrite"):
        save_single_machine_artifacts(file, root, seed=42)


def test_original_test_machine_requires_explicit_opt_in(tmp_path: Path) -> None:
    """Opting in unseals id_06 and permanently changes its scientific role."""
    file = _source(tmp_path)
    default = build_single_machine_artifacts(file)
    assert "fan:id_06" not in default
    enabled = build_single_machine_artifacts(file, include_original_test_machine=True)
    assert "fan:id_06" in enabled
    assert enabled["fan:id_06"]["settings"]["original_test_machine_repurposed"]


def test_can_prepare_only_original_test_machine_without_retraining_others(
    tmp_path: Path,
) -> None:
    """Selecting one machine retains the all-machine deterministic split."""
    file = _source(tmp_path)
    all_machines = build_single_machine_artifacts(
        file, include_original_test_machine=True
    )
    selected = build_single_machine_artifacts(
        file,
        include_original_test_machine=True,
        machine_group="fan:id_06",
    )
    assert set(selected) == {"fan:id_06"}
    assert torch.equal(
        selected["fan:id_06"]["features"],
        all_machines["fan:id_06"]["features"],
    )
    assert selected["fan:id_06"]["rows"] == all_machines["fan:id_06"]["rows"]

    root = tmp_path / "id06-only"
    manifest = save_single_machine_artifacts(
        file,
        root,
        include_original_test_machine=True,
        machine_group="fan:id_06",
    )
    assert set(manifest["machines"]) == {"fan:id_06"}
    assert manifest["original_test_machine_included"]
    assert read_prepared_artifact(
        manifest["machines"]["fan:id_06"]["prepared_file"]
    )["settings"]["original_test_machine_repurposed"]

    with pytest.raises(ValueError, match="No eligible machines"):
        build_single_machine_artifacts(file, machine_group="fan:id_06")
    with pytest.raises(ValueError, match="absent"):
        build_single_machine_artifacts(
            file, include_original_test_machine=True, machine_group="fan:id_08"
        )


def test_single_machine_rejects_reused_recordings(tmp_path: Path) -> None:
    """Reject source file identities repeated across machine holdout subsets."""
    file = _source(tmp_path)
    data = torch.load(file, map_location="cpu", weights_only=True)
    data["rows"][1]["healthy_file"] = data["rows"][0]["healthy_file"]
    torch.save(data, file)
    with pytest.raises(ValueError, match="Repeated healthy_file"):
        build_single_machine_artifacts(file)


def test_single_machine_end_to_end_swd_and_checkpoint(tmp_path: Path) -> None:
    """One machine can train through the original pipeline without test access."""
    file = _source(tmp_path)
    manifest = save_single_machine_artifacts(file, tmp_path / "prepared")
    prepared = Path(manifest["machines"]["fan:id_00"]["prepared_file"])
    spec = mimii_acoustic_ablation(
        prepared,
        hidden_dims=[8, 8],
        num_steps=4,
        batch_size=3,
        max_epochs=2,
        patience=3,
        seed=42,
        checkpoint_root=tmp_path / "checkpoints",
    )
    session = NexuSession.from_scenario(
        spec,
        accelerator="cpu",
        devices=1,
        log_dir=tmp_path / "logs",
        enable_progress_bar=False,
        enable_loggers=False,
    )
    session.setup().fit()
    validation = session.validate()
    assert validation and "val/transport_swd" in validation[0]
    directory = tmp_path / "checkpoints" / f"{prepared.stem}-mlp-8x8"
    report = evaluate_prepared_checkpoints(
        directory, device=torch.device("cpu"), projections=8, seed=42
    )
    assert report["test_is_sealed"]
    assert "held_out_test_best" not in report
    assert report["training_best"]["num_samples"] == 9
    assert report["validation_best"]["num_samples"] == 3
    with pytest.raises(ValueError, match="no test split"):
        evaluate_prepared_checkpoints(
            directory, device=torch.device("cpu"), include_test=True
        )


def test_single_machine_runner_rejects_unsealed_manifest(tmp_path: Path) -> None:
    """A second explicit opt-in is required to train original test-machine data."""
    file = _source(tmp_path)
    manifest = save_single_machine_artifacts(
        file,
        tmp_path / "prepared",
        include_original_test_machine=True,
    )
    manifest_file = tmp_path / "prepared" / "manifest.json"
    assert manifest["original_test_machine_included"]
    assert json.loads(manifest_file.read_text())["original_test_machine_included"]
    with pytest.raises(ValueError, match="explicit"):
        run_single_machine_experiments(
            manifest_file,
            checkpoint_root=tmp_path / "checkpoints",
            reports_dir=tmp_path / "reports",
            accelerator="cpu",
        )
