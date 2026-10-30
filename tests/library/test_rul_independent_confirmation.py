"""Regression contracts for native, isolated FD003 CfC confirmation trials."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
from typing import cast

import pytest

from nexuml_library.evaluation.maintenance.rul_independent_confirmation import (
    CONFIGURATIONS,
    SEEDS,
    summarize_independent_confirmation,
)
from nexuml_library.data.maintenance.rul_enhanced import AutoPairedCMapSSRULWindows
from nexuml_library.evaluation.maintenance.rul_engine_audit import RULPerEngineAudit
from nexuml_library.layers.model.rul_enhanced_cfc import EnhancedCfCRULHead
from nexuml_library.scenarios.maintenance import (
    rul_enhanced_cfc_confirmation as confirm,
)


LAUNCHER = (
    Path(__file__).resolve().parents[2]
    / ".k8s/scenarios/submit_rul_enhanced_cfc_confirm_fd003.py"
)


def load_launcher(monkeypatch: pytest.MonkeyPatch):
    """Load the standalone Kubernetes launcher with its sibling module visible."""
    monkeypatch.syspath_prepend(str(LAUNCHER.parent))
    spec = importlib.util.spec_from_file_location("nex233_confirm_k8s", LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
@pytest.mark.parametrize("seed", SEEDS)
def test_independent_scenario_is_separately_selected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configuration: str,
    seed: int,
) -> None:
    """Each trial has its own checkpoint, optimizer and output artifacts."""
    monkeypatch.chdir(tmp_path)
    scenario = confirm.independent_cfc_confirmation(
        cast(confirm.Confirmation, configuration), seed
    )
    run = f"rul-FD003-enhanced-cfc-confirm-{configuration}-s{seed}"
    assert scenario.name == run
    assert len(scenario.pipeline.stages["RULModel"]) == 1
    head = scenario.pipeline.stages["RULModel"][0].component
    assert isinstance(head, EnhancedCfCRULHead)
    assert head.initialization_seed == seed
    assert head.upstream_seed == 42
    assert head.train_encoder == (configuration != "baseline")
    assert head.trend is False
    assert head.sensor_graph == (configuration == "all_no_trend")
    source = scenario.data.datasets[0].source
    assert isinstance(source, AutoPairedCMapSSRULWindows)
    assert source.seed == 42
    assert scenario.training.max_epochs == 150
    assert scenario.training.batch_size == 64
    assert scenario.training.loss_keys == {"rul_loss": 1.0}
    assert len(scenario.callbacks) == 5
    checkpoint_dir = scenario.callbacks[2].kwargs["dirpath"]
    assert isinstance(checkpoint_dir, str)
    assert Path(checkpoint_dir).name == run
    assert scenario.callbacks[2].kwargs["monitor"] == "val/loss"
    assert scenario.callbacks[3].kwargs["patience"] == 20
    audit_file = scenario.callbacks[4].kwargs["output_file"]
    curves_file = scenario.callbacks[1].kwargs["output_file"]
    assert isinstance(audit_file, str)
    assert isinstance(curves_file, str)
    assert audit_file.endswith(f"/{run}.checkpoint.json")
    assert curves_file.endswith(f"/{run}.curves.csv")
    evaluator = scenario.evaluation.algorithms[-1].algorithm
    assert isinstance(evaluator, RULPerEngineAudit)
    assert evaluator.output_file.endswith(f"/{run}.csv")
    assert scenario.logging is not None and scenario.logging.run_name == run


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
@pytest.mark.parametrize("seed", SEEDS)
def test_all_independent_scenarios_are_registered(
    configuration: str, seed: int
) -> None:
    """Native resolve must discover nine distinct decorator registrations."""
    function = getattr(confirm, f"independent_{configuration}_s{seed}")
    assert function.__nexuml_discovered__["key"] == (
        f"maintenance-rul-enhanced-cfc-confirm-fd003-{configuration}-s{seed}"
    )


def test_k8s_render_isolated_jobs_and_forbidden_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep all Job paths distinct without overwriting composite artifacts."""
    launcher = load_launcher(monkeypatch)
    seen: set[str] = set()
    for configuration in CONFIGURATIONS:
        for seed in SEEDS:
            name = launcher.confirmation_job_name(configuration, seed)
            assert name not in seen
            seen.add(name)
            source = "job-staging/nexuml-nex233/" + "a" * 64
            manifest = launcher.render_confirmation_manifest(
                configuration, seed, "a" * 64, source
            )
            assert f"name: {name}" in manifest
            assert f"subPath: {source}" in manifest
            assert "student-worker-1" in manifest
            assert "operator: NotIn" in manifest
            assert 'nvidia.com/gpu: "1"' in manifest
            assert "subPath: models/nexuml/.experiments" in manifest
            assert "mountPath: /mnt/nexuml-source" in manifest
            assert "mountPath: /tmp/nexuml-job-venv" in manifest
            assert "harbor.ika.rub.de/library/nexuml:latest" in manifest
            expected = (
                f"maintenance-rul-enhanced-cfc-confirm-fd003-{configuration}-s{seed}"
            )
            assert expected in manifest
            slug = configuration.replace("_", "-")
            assert f"confirmation/{slug}-s{seed}.yaml" in manifest
            assert f"confirmation/{slug}-s{seed}.execution.json" in manifest
            assert "__SCENARIO__" not in manifest
            assert "rul-fd003-cfc-six" not in manifest
    assert len(seen) == 9
    with pytest.raises(ValueError, match="Unexpected"):
        launcher.confirmation_job_name("all", 42)


def populate_runs(root: Path) -> None:
    """Build synthetic own-checkpoint evidence without loading test metrics."""
    rul_root = root.parents[2]
    for seed in SEEDS:
        for configuration in CONFIGURATIONS:
            run = f"rul-FD003-enhanced-cfc-confirm-{configuration}-s{seed}"
            checkpoint = rul_root / "checkpoints" / run / "epoch=1.ckpt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"fake checkpoint for report-only test")
            audit = rul_root / "audits" / f"{run}.checkpoint.json"
            audit.parent.mkdir(parents=True, exist_ok=True)
            audit.write_text(json.dumps({
                "selection": "best minimum val/loss",
                "best_model_path": (
                    f"/mnt/nexuml-experiments/rul/checkpoints/{run}/epoch=1.ckpt"
                ),
                "checkpoint_epoch": 1,
                "best_model_score": 200,
                "loaded_strict_model_state": True,
            }))
            curves = root / f"{run}.curves.csv"
            curves.parent.mkdir(parents=True, exist_ok=True)
            with curves.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=(
                        "epoch", "train_windows", "val_windows",
                        "val_engine_count", "train_eval_mse", "val_eval_mse",
                        "val_group_max_mse",
                    ),
                )
                writer.writeheader()
                for epoch in (0, 1):
                    writer.writerow({
                        "epoch": epoch, "train_windows": 500,
                        "val_windows": 100, "val_engine_count": 20,
                        "train_eval_mse": 80,
                        "val_eval_mse": (
                            170 if epoch == 0
                            else 160 + (0 if configuration == "baseline" else -10)
                        ),
                        "val_group_max_mse": 240,
                    })
            slug = configuration.replace("_", "-")
            marker = root / f"{slug}-s{seed}.execution.json"
            marker.write_text(json.dumps({
                "configuration": configuration,
                "seed": seed,
                "source_sha256": "a" * 64,
                "train_command_wall_seconds": 120,
                "kubernetes_node": "student-worker-2",
                "timing_includes_fit_validate_test": True,
            }))


def test_summary_uses_own_checkpoint_epoch_not_best_observed(
    tmp_path: Path,
) -> None:
    """Rank only selected epoch curves, never test files or composite metrics."""
    root = tmp_path / ".experiments/rul/enhanced-cfc/fd003/confirmation"
    populate_runs(root)
    report = summarize_independent_confirmation(root)
    assert len(report["runs"]) == 9
    assert report["independent_best_validation_checkpoints"] is True
    assert report["test_predictions_read"] is False
    assert report["ranking"][0]["selected_checkpoint_val_eval_mse_mean"] == 150
    assert report["paired_effects"]["finetune"]["vs_baseline_mean"] == -10


def test_summary_refuses_partial_campaign(tmp_path: Path) -> None:
    """A missing individual checkpoint cannot be silently ignored."""
    root = tmp_path / ".experiments/rul/enhanced-cfc/fd003/confirmation"
    populate_runs(root)
    run = "rul-FD003-enhanced-cfc-confirm-finetune-s44"
    (root.parents[2] / "audits" / f"{run}.checkpoint.json").unlink()
    with pytest.raises(FileNotFoundError):
        summarize_independent_confirmation(root)


def test_summary_refuses_stale_completion_marker(tmp_path: Path) -> None:
    """A missing success marker blocks accidental use of an interrupted run."""
    root = tmp_path / ".experiments/rul/enhanced-cfc/fd003/confirmation"
    populate_runs(root)
    (root / "finetune-s44.execution.json").unlink()
    with pytest.raises(FileNotFoundError):
        summarize_independent_confirmation(root)


def test_summary_refuses_inconsistent_source_digest(tmp_path: Path) -> None:
    """Never pool replicas trained from different source snapshots."""
    root = tmp_path / ".experiments/rul/enhanced-cfc/fd003/confirmation"
    populate_runs(root)
    marker = root / "finetune-s44.execution.json"
    data = json.loads(marker.read_text(encoding="utf-8"))
    data["source_sha256"] = "b" * 64
    marker.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="source snapshots"):
        summarize_independent_confirmation(root)
