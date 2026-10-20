"""Sixteen-group MIMII FISHER extraction and zero-shot holdout contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
from torch import Tensor

from nexuml_library.data.maintenance.healthy_reference import (
    read_healthy_reference_assessment,
    read_healthy_reference_training,
)
from nexuml_library.data.maintenance.prepared_endpoints import read_prepared_artifact
from nexuml_library.evaluation.maintenance import multifamily_fisher_prepare
from nexuml_library.evaluation.maintenance.multifamily_reference_prepare import (
    build_family_holdout,
    prepare_multifamily_protocols,
    save_family_holdouts,
)
from nexuml_library.evaluation.maintenance.multifamily_reference_run import (
    summarize_multifamily_experiments,
)
from nexuml_library.evaluation.maintenance.reference_conditioned_run import (
    run_reference_conditioned_folds,
)


def _source(tmp_path: Path, *, recordings: int = 12) -> Path:
    """Create original-format raw independent FISHER artifacts for 16 groups.

    Returns:
        Recording-unique source with original MIMII split provenance.
    """
    generator = torch.Generator().manual_seed(27)
    rows: list[dict[str, Any]] = []
    features: list[Tensor] = []
    for family_index, family in enumerate(("fan", "pump", "slider", "valve")):
        for index, identity in enumerate(("id_00", "id_02", "id_04", "id_06")):
            for sample in range(recordings):
                healthy = torch.randn(8, generator=generator) * 0.3
                healthy[family_index] += 2.0
                healthy[4 + index] += 1.0
                faulty = healthy + torch.randn(8, generator=generator) * 0.4
                features.append(torch.stack((healthy, faulty)))
                rows.append(
                    {
                        "group_id": f"{family}:{identity}",
                        "machine_type": family,
                        "machine_id": identity,
                        "healthy_file": f"{family}/{identity}/normal/{sample}.wav",
                        "faulty_file": f"{family}/{identity}/abnormal/{sample}.wav",
                        "paired": 1,
                        "split": "train" if index < 2 else "val" if index == 2 else "test",
                    }
                )
    source = tmp_path / "multifamily-fisher-raw.pt"
    torch.save(
        {
            "format": "nex233-features-v1",
            "features": torch.stack(features),
            "rows": rows,
            "settings": {
                "frontend": "fisher",
                "scaling": "raw",
                "coupling": "independent",
                "seed": 42,
                "split_protocol": "multifamily_raw_for_reference_holdouts",
            },
        },
        source,
    )
    return source


def test_one_frozen_fisher_extracts_all_sixteen_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All four machine families share exactly the same encoder instance."""

    class FakeFisher(torch.nn.Module):
        initialized = 0

        def __init__(self, **kwargs: Any) -> None:
            super().__init__()
            FakeFisher.initialized += 1

        def forward(self, x: Tensor) -> Tensor:
            return x[:, :8]

    class FakeDataset:
        def __init__(self, family: str) -> None:
            records: list[dict[str, Any]] = []
            for index, identity in enumerate(("id_00", "id_02", "id_04", "id_06")):
                for sample in range(3):
                    records.append(
                        {
                            "group_id": f"{family}:{identity}",
                            "machine_type": family,
                            "machine_id": identity,
                            "healthy_file": f"{family}/{identity}/normal/{sample}.wav",
                            "faulty_file": f"{family}/{identity}/abnormal/{sample}.wav",
                            "paired": 1,
                            "split": (
                                "train" if index < 2 else "val" if index == 2 else "test"
                            ),
                        }
                    )
            self.meta = pd.DataFrame.from_records(records)

        def __len__(self) -> int:
            return len(self.meta)

        def __getitem__(self, idx: int) -> tuple[dict[str, Tensor], None]:
            value = torch.full((16,), float(idx))
            return {
                "healthy_waveform": value,
                "faulty_waveform": value + 1.0,
            }, None

    class FakeMIMII:
        def __init__(self, **kwargs: Any) -> None:
            self.family = str(kwargs["machine_type"])

        def build(self) -> FakeDataset:
            return FakeDataset(self.family)

    monkeypatch.setattr(multifamily_fisher_prepare, "FISHEREncoder", FakeFisher)
    monkeypatch.setattr(
        multifamily_fisher_prepare, "MIMIIAcousticEndpointPairs", FakeMIMII
    )
    output = tmp_path / "fisher-16.pt"
    summary = multifamily_fisher_prepare.save_multifamily_fisher(
        output, device=torch.device("cpu"), batch_size=5, max_pairs_per_machine=3
    )
    assert FakeFisher.initialized == 1
    assert summary["pairs"] == 48
    assert len(summary["machine_groups"]) == 16
    artifact = read_prepared_artifact(output)
    assert artifact["features"].shape == (48, 2, 8)
    assert torch.all(artifact["features"][:, 1] > artifact["features"][:, 0])
    assert (tmp_path / "fisher-16.json").is_file()
    with pytest.raises(FileExistsError, match="overwrite"):
        multifamily_fisher_prepare.save_multifamily_fisher(output)


def test_family_holdout_has_twelve_train_and_four_unseen_groups(tmp_path: Path) -> None:
    """Support/query split and held-out-family semantics exclude every family ID."""
    source = _source(tmp_path)
    training, assessments = build_family_holdout(
        source, heldout_family="pump", reference_size=3, validation_fraction=0.25
    )
    assert len(training["machine_groups"]) == 12
    assert training["references"].shape == (12, 3, 8)
    assert training["features"].shape == (108, 2, 8)
    assert sum(row["split"] == "train" for row in training["rows"]) == 84
    assert sum(row["split"] == "val" for row in training["rows"]) == 24
    assert len(assessments) == 4
    assert all(name.startswith("pump:") for name in assessments)
    assert all(not row["group_id"].startswith("pump:") for row in training["rows"])
    assert all(row["split"] in {"train", "val"} for row in training["rows"])
    query_ids = {
        row[key] for row in training["rows"] for key in ("healthy_file", "faulty_file")
    }
    support_ids = {
        name for group in training["support_healthy_files"] for name in group
    }
    assert query_ids.isdisjoint(support_ids)
    for name, payload in assessments.items():
        assert payload["heldout_machine"] == name
        assert payload["features"].shape == (9, 2, 8)
        assert payload["references"].shape == (3, 8)
        ids = {
            row[key] for row in payload["rows"]
            for key in ("healthy_file", "faulty_file")
        }
        assert all(row["group_id"] == name for row in payload["rows"])
        assert set(payload["support_healthy_files"]).isdisjoint(ids)
        assert ids.isdisjoint(query_ids | support_ids)


def test_heldout_family_faults_cannot_modify_train_or_normalizer(
    tmp_path: Path,
) -> None:
    """No faulty endpoint or healthy input from the held-out family fits model."""
    source = _source(tmp_path)
    first, _ = build_family_holdout(
        source, heldout_family="valve", reference_size=3
    )
    payload = read_prepared_artifact(source)
    for index, row in enumerate(payload["rows"]):
        if row["group_id"].startswith("valve:"):
            payload["features"][index] = 10000.0
    torch.save(payload, source)
    second, _ = build_family_holdout(
        source, heldout_family="valve", reference_size=3
    )
    for key in ("features", "references", "training_mean", "training_std"):
        assert torch.equal(first[key], second[key])


def test_multi_seed_machine_and_family_manifest_roundtrip(tmp_path: Path) -> None:
    """All 16 machine folds and four family folds per seed stay isolated."""
    source = _source(tmp_path)
    destination = tmp_path / "folds"
    result = prepare_multifamily_protocols(
        source,
        destination,
        seeds=(42, 43),
        protocols="both",
        reference_size=3,
        validation_fraction=0.25,
    )
    assert set(result["manifests"]) == {"42", "43"}
    for seed in ("42", "43"):
        machine_manifest = json.loads(
            Path(result["manifests"][seed]["machine"]["manifest"]).read_text()
        )
        family_manifest = json.loads(
            Path(result["manifests"][seed]["family"]["manifest"]).read_text()
        )
        assert len(machine_manifest["folds"]) == 16
        assert len(family_manifest["folds"]) == 4
        training_file = family_manifest["folds"]["family:fan"]["training_file"]
        training = read_healthy_reference_training(training_file)
        assert all(not name.startswith("fan:") for name in training["machine_groups"])
        for assessment_file in family_manifest["folds"]["family:fan"]["assessment_files"].values():
            assessment = read_healthy_reference_assessment(assessment_file)
            assert assessment["heldout_machine"].startswith("fan:")
    with pytest.raises(FileExistsError, match="overwrite"):
        prepare_multifamily_protocols(source, destination, reference_size=3)


def test_machine_family_validation_rejects_incomplete_groups(tmp_path: Path) -> None:
    """The expanded benchmark never silently evaluates fewer than 16 IDs."""
    source = _source(tmp_path)
    payload = read_prepared_artifact(source)
    payload["rows"] = payload["rows"][:-12]
    payload["features"] = payload["features"][:-12]
    torch.save(payload, source)
    with pytest.raises(ValueError, match="sixteen"):
        build_family_holdout(source, heldout_family="fan", reference_size=3)


def test_family_runner_checks_all_unseen_id_distributions(tmp_path: Path) -> None:
    """CPU checkpoint selection precedes each of four unseen-family assessments."""
    source = _source(tmp_path)
    manifest = save_family_holdouts(
        source, tmp_path / "folds", heldout_family="fan", reference_size=3
    )
    summary = run_reference_conditioned_folds(
        tmp_path / "folds" / "manifest.json",
        checkpoint_root=tmp_path / "checkpoints",
        reports_dir=tmp_path / "reports",
        heldout_machine="family:fan",
        context_mode="reference",
        width=12,
        context_dim=5,
        encoder_width=10,
        num_steps=2,
        batch_size=16,
        max_epochs=1,
        patience=2,
        accelerator="cpu",
    )
    assert set(manifest["folds"]) == {"family:fan"}
    output = summary["folds"]["family:fan"]
    assert len(output["unseen_by_machine"]) == 4
    assert all(group.startswith("fan:") for group in output["unseen_by_machine"])
    assert Path(output["best_checkpoint"]).is_file()
    assert (tmp_path / "reports" / "summary-reference.json").is_file()


def test_multi_seed_summary_macro_and_paired_comparison(tmp_path: Path) -> None:
    """Report summaries preserve machine-group granularity and paired deltas."""
    root = tmp_path / "runs"
    for seed in (42, 43):
        for mode, value in (("reference", 0.2), ("constant", -0.1)):
            directory = root / f"seed{seed}" / "family" / mode
            directory.mkdir(parents=True)
            summary = {
                "protocol": "leave_one_family_out_healthy_reference",
                "context_mode": mode,
                "folds": {
                    "family:fan": {
                        "unseen_by_machine": {
                            f"fan:id_{number:02d}": {"swd_relative_reduction": value}
                            for number in (0, 2, 4, 6)
                        }
                    }
                },
            }
            (directory / f"summary-{mode}.json").write_text(
                json.dumps(summary) + "\n"
            )
    result = summarize_multifamily_experiments(
        root, protocol="family", seeds=(42, 43)
    )
    assert result["models"]["reference"]["machine_assessment_count"] == 8
    assert result["models"]["reference"]["mean_over_seed_machine_macros"] == pytest.approx(0.2)
    assert result["paired_comparison"]["reference_wins"] == 8
    assert all(
        pair["conditioning_advantage_pp"] == pytest.approx(30.0)
        for pair in result["paired_comparison"]["pairs"]
    )
