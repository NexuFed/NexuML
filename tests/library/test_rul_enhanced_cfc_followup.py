"""Regression contracts for NEX-233 CfC 10-seed and cross-benchmark follow-up."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
from typing import Literal, cast

import pytest

from nexuml_library.data.maintenance.rul_enhanced import AutoPairedCMapSSRULWindows
from nexuml_library.evaluation.maintenance.rul_followup_summary import (
    summarize_followup,
)
from nexuml_library.layers.model.rul_enhanced_cfc import EnhancedCfCRULHead
from nexuml_library.layers.model.rul_trajectory_loss import (
    RULTrajectoryConsistencyLoss,
)
from nexuml_library.scenarios.maintenance import rul_enhanced_cfc_followup as followup

LAUNCHER = (
    Path(__file__).resolve().parents[2]
    / ".k8s/scenarios/submit_rul_enhanced_cfc_followup.py"
)


def launcher_module(monkeypatch: pytest.MonkeyPatch):
    """Load the standalone GPU Job launcher for manifest contract tests."""
    monkeypatch.syspath_prepend(str(LAUNCHER.parent))
    spec = importlib.util.spec_from_file_location("nex233_followup_launcher", LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("subset", "configuration", "seed"),
    [
        *[
            ("FD003", c, s)
            for c in followup.VALIDATION_ARMS
            for s in followup.RELIABILITY_SEEDS
        ],
        *[
            ("FD003", c, s)
            for c in followup.LEAVE_ONE_OUT
            for s in followup.ABLATION_SEEDS
        ],
        *[
            ("FD001", c, s)
            for c in followup.VALIDATION_ARMS
            for s in followup.FD001_SEEDS
        ],
    ],
)
def test_independent_run_registration_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    subset: str,
    configuration: str,
    seed: int,
) -> None:
    """All 41 native scenarios have their own selected checkpoint and seed."""
    monkeypatch.chdir(tmp_path)
    native = getattr(followup, f"followup_{subset.lower()}_{configuration}_s{seed}")
    assert native.__nexuml_discovered__["key"] == (
        f"maintenance-rul-enhanced-cfc-followup-"
        f"{subset.lower()}-{configuration}-s{seed}"
    )
    scenario = followup.followup_study(
        cast(Literal["FD001", "FD003"], subset),
        cast(followup.FollowupConfiguration, configuration), seed,
    )
    run = f"rul-{subset}-enhanced-cfc-followup-{configuration}-s{seed}"
    assert scenario.name == run
    assert scenario.training.max_epochs == 150
    assert scenario.training.batch_size == 64
    assert scenario.training.loss_keys == {"rul_loss": 1.0}
    source = scenario.data.datasets[0].source
    assert isinstance(source, AutoPairedCMapSSRULWindows)
    assert source.subset == subset and source.seed == 42
    model = scenario.pipeline.stages["RULModel"][0].component
    assert isinstance(model, EnhancedCfCRULHead)
    assert model.subset == subset and model.upstream_seed == 42
    assert model.initialization_seed == seed
    assert model.train_encoder == (
        configuration not in ("baseline", "without_finetune")
    )
    assert model.sensor_graph == (
        configuration not in ("baseline", "without_graph")
    )
    assert not model.trend
    assert model.multiscale == (
        configuration not in ("baseline", "without_multiscale")
    )
    assert model.segmentation == (
        configuration not in ("baseline", "without_segmentation")
    )
    assert model.trajectory == (
        configuration not in ("baseline", "without_trajectory")
    )
    loss = scenario.pipeline.stages["RULLoss"][0].component
    assert isinstance(loss, RULTrajectoryConsistencyLoss)
    assert loss.consistency_weight == (0.1 if model.trajectory else 0.0)
    assert scenario.callbacks[0].kwargs["seed"] == seed
    assert scenario.callbacks[2].kwargs["monitor"] == "val/loss"
    assert scenario.callbacks[3].kwargs["patience"] == 20
    checkpoint_dir = scenario.callbacks[2].kwargs["dirpath"]
    assert isinstance(checkpoint_dir, str)
    assert Path(checkpoint_dir).name == run
    checkpoint_audit = scenario.callbacks[4].kwargs["output_file"]
    curves = scenario.callbacks[1].kwargs["output_file"]
    assert isinstance(checkpoint_audit, str)
    assert isinstance(curves, str)
    assert checkpoint_audit.endswith(f"/{run}.checkpoint.json")
    assert curves.endswith(f"/{run}.curves.csv")


def test_reject_unregistered_followup_combinations() -> None:
    """No accidental test-split or unpaired configuration extensions."""
    with pytest.raises(ValueError, match="Unsupported"):
        followup.followup_study("FD001", "without_graph", 42)
    with pytest.raises(ValueError, match="Unsupported"):
        followup.followup_study("FD001", "baseline", 51)
    with pytest.raises(ValueError, match="Unsupported"):
        followup.followup_study("FD003", "without_graph", 50)
    with pytest.raises(ValueError, match="Unsupported"):
        followup.followup_study("FD003", "baseline", 52)
    with pytest.raises(ValueError, match="Unsupported"):
        followup.followup_study(
            cast(Literal["FD001", "FD003"], "FD002"), "baseline", 42
        )


def test_followup_launcher_matrix_and_immutable_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Job manifests remain native, isolated, GPU-limited and GH200-free."""
    launcher = launcher_module(monkeypatch)
    assert len(launcher.trials("reliability")) == 20
    assert len(launcher.trials("ablation")) == 15
    assert len(launcher.trials("fd001")) == 6
    jobs = launcher.trials("all")
    assert len(jobs) == 41
    digest = "a" * 64
    source = f"job-staging/nexuml-nex233/{digest}"
    job_names = set()
    for subset, config, seed in jobs:
        job = launcher.followup_job_name(subset, config, seed)
        assert job not in job_names
        job_names.add(job)
        manifest = launcher.render_followup_manifest(
            subset, config, seed, digest, source, ("gh200-1", "student-worker-1")
        )
        assert f"name: {job}" in manifest
        assert f"subPath: {source}" in manifest
        assert "ghcr.io/nexufed/nexuml:sha-f8d5490-cuda12.8.1" in manifest
        assert "values: [gh200-1, student-worker-1]" in manifest
        assert "nvidia.com/cuda.runtime.major" not in manifest
        assert "nvidia.com/gpu: \"1\"" in manifest
        assert "kueue.x-k8s.io/queue-name: jobs" in manifest
        assert "subPath: models/nexuml/.experiments" in manifest
        assert f'rul-{subset}-controlled-patchformer-seed42' in manifest
        assert "__SUBSET__" not in manifest
        assert "__EXCLUDED_NODES__" not in manifest
        assert "nexuml-study: rul-cfc-followup" in manifest
        assert f"maintenance-rul-enhanced-cfc-followup-{subset.lower()}-" in manifest
    assert len(job_names) == 41


def populate_followup(root: Path) -> None:
    """Create synthetic checkpoint-bound validation observations for 41 runs."""
    specs = [
        *[("FD003", c, s, "reliability") for c in ("baseline", "all_no_trend")
          for s in range(42, 52)],
        *[("FD003", c, s, "ablation") for c in (
            "without_finetune", "without_graph", "without_multiscale",
            "without_segmentation", "without_trajectory",
        ) for s in (42, 43, 44)],
        *[("FD001", c, s, "fd001") for c in ("baseline", "all_no_trend")
          for s in (42, 43, 44)],
    ]
    for subset, config, seed, campaign in specs:
        run = f"rul-{subset}-enhanced-cfc-followup-{config}-s{seed}"
        rul_root = root.parents[1]
        checkpoint = rul_root / "checkpoints" / run / "epoch=1.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"synthetic checkpoint")
        score = (
            160.0 if config == "baseline" else
            150.0 if config == "all_no_trend" else 155.0
        )
        if subset == "FD001":
            score -= 40
        score += seed - 42
        audit = rul_root / "audits" / f"{run}.checkpoint.json"
        audit.parent.mkdir(parents=True, exist_ok=True)
        audit.write_text(json.dumps({
            "selection": "best minimum val/loss",
            "loaded_strict_model_state": True,
            "best_model_path": f"/mnt/nexuml-experiments/rul/checkpoints/{run}/epoch=1.ckpt",
            "best_model_score": score,
            "checkpoint_epoch": 1,
        }))
        split_root = root / subset.lower()
        split_root.mkdir(parents=True, exist_ok=True)
        with (split_root / f"{run}.curves.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "epoch", "train_windows", "val_windows", "val_engine_count",
                    "train_eval_mse", "val_eval_mse", "val_group_max_mse",
                ),
            )
            writer.writeheader()
            writer.writerow({
                "epoch": 1,
                "train_windows": 17785 if subset == "FD003" else 12000,
                "val_windows": 4035 if subset == "FD003" else 3000,
                "val_engine_count": 20,
                "train_eval_mse": score - 20,
                "val_eval_mse": score,
                "val_group_max_mse": score + 20,
            })
        marker = split_root / f"{config.replace('_', '-')}-s{seed}.execution.json"
        marker.write_text(json.dumps({
            "subset": subset, "configuration": config, "campaign": campaign,
            "seed": seed, "source_sha256": "a" * 64,
            "container_image": "ghcr.io/nexufed/nexuml:sha-f8d5490-cuda12.8.1",
            "kubernetes_node": "cip-gpu-9",
            "train_command_wall_seconds": 600,
            "timing_includes_fit_validate_test": True,
        }))


def test_validation_only_41_run_summary(tmp_path: Path) -> None:
    """All three parts use aligned seeds and selected checkpoints only."""
    root = tmp_path / ".experiments/rul/enhanced-cfc/followup"
    populate_followup(root)
    report = summarize_followup(root, "all")
    assert len(report["runs"]) == 41
    assert report["test_predictions_read"] is False
    assert report["source_sha256"] == "a" * 64
    reliability = report["analyses"]["reliability"]
    assert reliability["paired_effect"]["candidate_minus_reference_mean"] == -10
    assert reliability["paired_effect"]["improved_seeds"] == 10
    assert len(reliability["paired_effect"]["paired_mean_95pct_t_interval"]) == 2
    ablation = report["analyses"]["ablation"]
    assert len(ablation["leave_one_out"]) == 5
    assert all(
        effect["candidate_minus_reference_mean"] == 5.0
        for effect in ablation["leave_one_out"].values()
    )
    assert report["analyses"]["fd001"]["paired_effect"][
        "candidate_minus_reference_mean"
    ] == -10


@pytest.mark.parametrize("campaign", ("reliability", "ablation", "fd001"))
def test_independent_campaign_reports(tmp_path: Path, campaign: str) -> None:
    """Campaigns can be reported before all other trials finish."""
    root = tmp_path / ".experiments/rul/enhanced-cfc/followup"
    populate_followup(root)
    report = summarize_followup(root, campaign)
    assert tuple(report["analyses"]) == (campaign,)


def test_incomplete_and_mixed_source_rejected(tmp_path: Path) -> None:
    """Missing markers and mixed source digests must not yield a ranking."""
    root = tmp_path / ".experiments/rul/enhanced-cfc/followup"
    populate_followup(root)
    marker = root / "fd003/baseline-s50.execution.json"
    marker.unlink()
    with pytest.raises(FileNotFoundError):
        summarize_followup(root, "reliability")
    populate_followup(root)
    data = json.loads(marker.read_text())
    data["source_sha256"] = "b" * 64
    marker.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="different staged sources"):
        summarize_followup(root, "reliability")


def test_wrong_image_and_split_rejected(tmp_path: Path) -> None:
    """Prevent mixing engine splits or CUDA runtime image environments."""
    root = tmp_path / ".experiments/rul/enhanced-cfc/followup"
    populate_followup(root)
    marker = root / "fd003/all-no-trend-s44.execution.json"
    data = json.loads(marker.read_text())
    data["container_image"] = "ghcr.io/nexufed/nexuml:latest"
    marker.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Inconsistent completion marker"):
        summarize_followup(root, "reliability")
    populate_followup(root)
    run = "rul-FD003-enhanced-cfc-followup-all_no_trend-s44"
    curve = root / "fd003" / f"{run}.curves.csv"
    content = curve.read_text()
    curve.write_text(content.replace(",4035,", ",4036,"))
    with pytest.raises(ValueError, match="Different training/validation engine splits"):
        summarize_followup(root, "reliability")
