"""Tests for matched known-family LOMO vs unseen-family LOFO evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from nexuml_library.evaluation.maintenance import multifamily_reference_run
from nexuml_library.evaluation.maintenance.machine_family_comparison import (
    compare_machine_family_generalization,
)
from nexuml_library.evaluation.maintenance.multifamily_reference_run import (
    run_multifamily_experiments,
)


def _aggregate(
    destination: Path,
    *,
    protocol: str,
    seeds: tuple[int, ...] = (42, 43),
    families: tuple[str, ...] = ("fan",),
    offset: float = 0.0,
) -> Path:
    """Create four-group-per-family macro results for both context modes.

    Returns:
        A completed aggregate JSON file for the selected protocol.
    """
    modes: dict[str, dict[str, Any]] = {}
    for mode, bonus in (("reference", 0.12), ("constant", 0.0)):
        records = []
        for seed in seeds:
            for family in families:
                for machine in ("id_00", "id_02", "id_04", "id_06"):
                    group = f"{family}:{machine}"
                    records.append(
                        {
                            "seed": seed,
                            "fold": (
                                group if protocol == "machine" else f"family:{family}"
                            ),
                            "machine": group,
                            "family": family,
                            "relative_reduction": offset + bonus,
                        }
                    )
        modes[mode] = {"records": records}
    destination.write_text(
        json.dumps({"protocol": protocol, "seeds": seeds, "models": modes}) + "\n"
    )
    return destination


def test_paired_known_family_is_more_transferable(tmp_path: Path) -> None:
    """The comparison keeps each physical machine and split seed paired."""
    machine = _aggregate(tmp_path / "known.json", protocol="machine", offset=0.1)
    family = _aggregate(tmp_path / "unseen.json", protocol="family", offset=-0.3)
    result = compare_machine_family_generalization(machine, family, family="fan")
    assert result["families"] == ["fan"]
    assert result["overall"]["machine_assessments"] == 8
    assert result["overall"]["mean_relative_reduction"]["known_family_reference"] == pytest.approx(
        0.22
    )
    assert result["overall"]["mean_relative_reduction"]["unseen_family_reference"] == pytest.approx(
        -0.18
    )
    assert result["overall"]["mean_reference_transfer_advantage_pp"] == pytest.approx(40.0)
    assert result["overall"]["positive_transport_count"]["known_family_reference"] == 8
    assert result["overall"]["positive_transport_count"]["unseen_family_reference"] == 0
    assert all(
        item["known_family_conditioning_advantage_pp"] == pytest.approx(12.0)
        for item in result["per_machine_seed"]
    )


def test_comparison_requires_matching_seeds_and_identities(tmp_path: Path) -> None:
    """Different held-out sample IDs or split seeds are not compared."""
    machine = _aggregate(tmp_path / "known.json", protocol="machine", seeds=(42,))
    family = _aggregate(tmp_path / "unseen.json", protocol="family", seeds=(43,))
    with pytest.raises(ValueError, match="seeds"):
        compare_machine_family_generalization(machine, family)
    _aggregate(family, protocol="family", seeds=(42,), families=("pump",))
    with pytest.raises(ValueError, match="same machine"):
        compare_machine_family_generalization(machine, family)


def test_comparison_rejects_duplicate_machine_records(tmp_path: Path) -> None:
    """A repeated machine result cannot alter the macro score silently."""
    machine = _aggregate(
        tmp_path / "known.json", protocol="machine", seeds=(42,)
    )
    family = _aggregate(tmp_path / "unseen.json", protocol="family", seeds=(42,))
    payload = json.loads(machine.read_text())
    payload["models"]["reference"]["records"].append(
        dict(payload["models"]["reference"]["records"][0])
    )
    machine.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        compare_machine_family_generalization(machine, family)


def test_comparison_rejects_incomplete_family(tmp_path: Path) -> None:
    """A three-machine subset must not be described as a family holdout."""
    machine = _aggregate(
        tmp_path / "known.json", protocol="machine", seeds=(42,)
    )
    family = _aggregate(
        tmp_path / "unseen.json", protocol="family", seeds=(42,)
    )
    for path in (machine, family):
        payload = json.loads(path.read_text())
        for mode in ("reference", "constant"):
            payload["models"][mode]["records"] = [
                item for item in payload["models"][mode]["records"]
                if item["machine"] != "fan:id_06"
            ]
        path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="four-machine family"):
        compare_machine_family_generalization(machine, family)


def test_known_family_runner_selects_four_machine_folds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Family filter means 4 distinct LOMO models, each trained on 15 groups."""
    selected: list[tuple[str, ...] | None] = []

    def fake_run(
        manifest_file: str | Path,
        *,
        reports_dir: str | Path,
        context_mode: str,
        heldout_machines: tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        selected.append(heldout_machines)
        assert heldout_machines is not None and len(heldout_machines) == 4
        assert all(group.startswith("fan:") for group in heldout_machines)
        directory = Path(reports_dir)
        directory.mkdir(parents=True)
        result = {
            "protocol": "leave_one_machine_out_healthy_reference",
            "context_mode": context_mode,
            "folds": {
                group: {
                    "unseen_by_machine": {
                        group: {
                            "swd_relative_reduction": (
                                0.2 if context_mode == "reference" else 0.1
                            )
                        }
                    }
                }
                for group in heldout_machines
            },
        }
        (directory / f"summary-{context_mode}.json").write_text(
            json.dumps(result) + "\n"
        )
        return result

    monkeypatch.setattr(
        multifamily_reference_run, "run_reference_conditioned_folds", fake_run
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "folds": {
                    f"{family}:{identity}": {}
                    for family in ("fan", "pump", "slider", "valve")
                    for identity in ("id_00", "id_02", "id_04", "id_06")
                }
            }
        )
    )
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps(
            {
                "seeds": [42],
                "manifests": {"42": {"machine": {"manifest": str(manifest)}}},
            }
        )
    )
    result = run_multifamily_experiments(
        index,
        output_root=tmp_path / "out",
        protocol="machine",
        seeds=(42,),
        heldout_family="fan",
    )
    assert selected == [
        ("fan:id_00", "fan:id_02", "fan:id_04", "fan:id_06"),
        ("fan:id_00", "fan:id_02", "fan:id_04", "fan:id_06"),
    ]
    assert result["models"]["reference"]["machine_assessment_count"] == 4
    assert result["paired_comparison"]["reference_wins"] == 4
    with pytest.raises(ValueError, match="machine folds only"):
        run_multifamily_experiments(
            index,
            output_root=tmp_path / "invalid",
            protocol="family",
            heldout_family="fan",
        )
