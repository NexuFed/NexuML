"""Controlled RUL replication and frozen PatchFormer latent Flow Matching tests."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from nexuml.core.compiler import compile as compile_pipeline
from nexuml.core.config import ResolvedConfig
from nexuml_library.data.maintenance.patchformer_flow import (
    AutoPatchFormerFlowEndpoints,
    _trusted_patchformer_weights,
    resolve_patchformer_checkpoint,
)
from nexuml_library.data.maintenance.prepared_endpoints import read_prepared_artifact
from nexuml_library.data.maintenance.rul import (
    AutoCMapSSRULWindows,
    read_rul_artifact,
)
from nexuml_library.evaluation.maintenance.rul_controlled_summary import (
    summarize_controlled_study,
)
from nexuml_library.layers.model.flow import Flow
from nexuml_library.layers.model.rul_advanced import (
    AdvancedRULForecaster,
    _PatchFormerModel,
)
from nexuml_library.layers.model.rul_safety import RULAsymmetricSquaredLoss
from nexuml_library.scenarios.maintenance.rul_controlled import (
    _StudyKind,
    controlled_rul_study,
)
from nexuml_library.scenarios.maintenance.rul_patchformer_flow import (
    patchformer_embedding_flow,
)


@pytest.mark.parametrize("subset", ["FD001", "FD003"])
@pytest.mark.parametrize("seed", [42, 43, 44])
def test_controlled_study_preserves_pairing_and_protocol(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    subset: Literal["FD001", "FD003"], seed: int,
) -> None:
    """All architectures share an identical source and optimization schedule."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NEXUML_RUL_PREPARED_FILE", raising=False)
    kinds: tuple[_StudyKind, ...] = (
        "patchformer", "dmm_ja", "safety_lstm", "rul_mamba"
    )
    variants = {
        kind: controlled_rul_study(kind, subset, seed) for kind in kinds
    }
    sources = [variant.data.datasets[0].source for variant in variants.values()]
    assert all(isinstance(source, AutoCMapSSRULWindows) for source in sources)
    assert all(source == sources[0] for source in sources)
    source = sources[0]
    assert isinstance(source, AutoCMapSSRULWindows)
    assert source.subset == subset
    assert source.seed == seed
    assert source.window == 30
    assert source.rul_cap == 130
    assert source.normalize_targets is False
    assert source.train_full_windows_only is True
    assert source.padding_mode == "zero"
    assert len(source.sensor_ids or []) == (14 if subset == "FD001" else 15)
    assert all(
        model.data.input_shapes == {"rul_history": [30, len(source.sensor_ids or [])]}
        for model in variants.values()
    )
    assert len({model.name for model in variants.values()}) == 4
    assert len({
        model.callbacks[1].kwargs["dirpath"]
        for model in variants.values()
    }) == 4
    for name, model in variants.items():
        assert model.training.optimizer == variants["patchformer"].training.optimizer
        assert model.training.batch_size == variants["patchformer"].training.batch_size
        assert model.callbacks[0].kwargs["seed"] == seed
        assert model.evaluation.algorithms[0].label_key == "rul"
        assert model.evaluation.algorithms[1].label_key == "rul_raw"
        if name == "safety_lstm":
            assert isinstance(
                model.pipeline.stages["RULLoss"][0].component,
                RULAsymmetricSquaredLoss,
            )
        else:
            assert isinstance(
                model.pipeline.stages["RULModel"][0].component,
                AdvancedRULForecaster,
            )
    compiled = compile_pipeline(variants["patchformer"])
    config = tmp_path / "controlled.yaml"
    compiled.resolved_config.save(config)
    assert ResolvedConfig.load(config).to_scenario().name == (
        variants["patchformer"].name
    )


def _fake_cmapss(root: Path, subset: str) -> None:
    """Supply valid run-to-failure windows including both health endpoints."""
    train = [
        [engine, cycle, *([float(engine) + cycle * 0.002 * sensor
                          for sensor in range(1, 25)])]
        for engine in range(1, 9)
        for cycle in range(1, 171)
    ]
    test = [
        [engine, cycle, *([float(engine + cycle)] * 24)]
        for engine in (1, 2)
        for cycle in range(1, 42)
    ]
    np.savetxt(root / f"train_{subset}.txt", np.asarray(train))
    np.savetxt(root / f"test_{subset}.txt", np.asarray(test))
    np.savetxt(root / f"RUL_{subset}.txt", [12, 150])


def _save_fake_patchformer_checkpoint(path: Path, channels: int) -> None:
    """Use a valid standalone Lightning-like backbone parameter mapping."""
    model = _PatchFormerModel(
        length=30, channels=channels, width=64, patch_size=5,
        heads=2, dropout=0.1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": {
            f"pipeline.RULModel.0.backbone.{key}": value
            for key, value in model.state_dict().items()
        }},
        path,
    )


def test_frozen_patchformer_endpoints_exclude_official_test_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test observations cannot change learned endpoint scaling or validation."""
    monkeypatch.chdir(tmp_path)
    _fake_cmapss(tmp_path, "FD001")
    run = "rul-FD001-controlled-patchformer-seed42"
    checkpoint = tmp_path / run / "best.ckpt"
    _save_fake_patchformer_checkpoint(checkpoint, 14)
    manifest = tmp_path / ".experiments/rul/audits" / f"{run}.checkpoint.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"best_model_path": str(checkpoint)}) + "\n",
        encoding="utf-8",
    )
    source = AutoPatchFormerFlowEndpoints(
        root=str(tmp_path),
        prepared_rul_path=str(tmp_path / "rul.pt"),
        output_file=str(tmp_path / "endpoints.pt"),
        train_pairs=6,
        val_pairs=4,
        embedding_batch_size=4,
    )
    assert source.checkpoint_path is None
    dataset = source.build()
    assert len(dataset.get_split("train")) == 6
    assert len(dataset.get_split("val")) == 4
    result = read_prepared_artifact(source.output_file)
    assert result["features"].shape == (10, 2, 128)
    assert result["settings"]["feature_normalization"] == "train_endpoints_only"
    assert result["settings"]["held_out_test_engine_labels_used"] is False
    baseline = result["features"].clone()
    before = result["settings"]["feature_mean"]
    prepared = read_rul_artifact(source.prepared_rul_path)
    mutated = copy.deepcopy(prepared)
    for i, row in enumerate(mutated["rows"]):
        if row["split"] == "test":
            mutated["features"][i] = 12345.0
            mutated["targets"][i] = 0.0
            mutated["raw_targets"][i] = 0.0
    torch.save(mutated, source.prepared_rul_path)
    source.build()
    changed = read_prepared_artifact(source.output_file)
    assert torch.equal(changed["features"], baseline)
    assert changed["settings"]["feature_mean"] == before



def _write_best_manifest(root: Path, run: str) -> Path:
    """Write the native callback's minimal checkpoint manifest for a run.

    Returns:
        Path to the corresponding test checkpoint.
    """
    checkpoint = root / ".experiments/rul/checkpoints" / run / "epoch=0.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"trusted placeholder")
    manifest = root / ".experiments/rul/audits" / f"{run}.checkpoint.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"best_model_path": str(checkpoint)}) + "\n",
        encoding="utf-8",
    )
    return checkpoint


def test_flow_finds_existing_original_patchformer_without_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FD001 seed 42 reuses the original run when controlled run is absent."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "NEXUML_PATCHFORMER_CHECKPOINT", "/unused/obsolete/configuration.ckpt"
    )
    original = (
        "rul-cmapss-FD001-patchformer-direct-rul-adaptation-seed42"
    )
    expected = _write_best_manifest(tmp_path, original)
    assert resolve_patchformer_checkpoint("FD001", 42) == expected
    flow = patchformer_embedding_flow("FD001", 42)
    source = flow.data.datasets[0].source
    assert isinstance(source, AutoPatchFormerFlowEndpoints)
    assert source.checkpoint_path is None

    controlled = "rul-FD001-controlled-patchformer-seed42"
    preferred = _write_best_manifest(tmp_path, controlled)
    assert resolve_patchformer_checkpoint("FD001", 42) == preferred


def test_flow_checkpoint_missing_has_native_training_instruction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Resolve never needs checkpoint; dataset build reports the upstream step."""
    monkeypatch.chdir(tmp_path)
    flow = patchformer_embedding_flow("FD003", 43)
    assert isinstance(flow.data.datasets[0].source, AutoPatchFormerFlowEndpoints)
    with pytest.raises(
        FileNotFoundError, match="maintenance-rul-controlled-fd003-patchformer-s43"
    ):
        resolve_patchformer_checkpoint("FD003", 43)


def test_flow_rejects_wrong_seed_checkpoint_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reject mislabeled manifests, avoiding silent cross-seed model reuse."""
    monkeypatch.chdir(tmp_path)
    controlled = "rul-FD003-controlled-patchformer-seed44"
    bad = _write_best_manifest(tmp_path, controlled)
    requested = tmp_path / ".experiments/rul/audits" / (
        "rul-FD003-controlled-patchformer-seed43.checkpoint.json"
    )
    requested.write_text(
        json.dumps({"best_model_path": str(bad)}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Stale or mismatched"):
        resolve_patchformer_checkpoint("FD003", 43)


def test_checkpoint_embedding_weights_match_upstream_model(tmp_path: Path) -> None:
    """Reject missing checkpoint backbone parameters rather than random weights."""
    model = _PatchFormerModel(30, 14, 64, 5, 2, 0.1)
    checkpoint = tmp_path / "incomplete.ckpt"
    torch.save({"state_dict": {}}, checkpoint)
    with pytest.raises(ValueError, match="missing or ambiguous"):
        _trusted_patchformer_weights(model, checkpoint)


@pytest.mark.parametrize("subset,seed", [("FD001", 42), ("FD003", 43)])
def test_patchformer_embedding_flow_reuses_nexuml_fm_and_swd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    subset: Literal["FD001", "FD003"], seed: int,
) -> None:
    """Flow stages and callbacks match the existing feature-ablation contract."""
    monkeypatch.chdir(tmp_path)
    scenario = patchformer_embedding_flow(subset, seed)
    assert list(scenario.pipeline.stages) == ["FlowPath", "Flow", "Loss"]
    source = scenario.data.datasets[0].source
    assert isinstance(source, AutoPatchFormerFlowEndpoints)
    assert source.seed == seed
    assert source.subset == subset
    assert source.prepared_rul_path.endswith(
        f"cmapss-{subset}-seed{seed}-controlled-w30-s{seed}.pt"
    )
    assert scenario.data.input_shapes == {
        "healthy_spectrum": [128],
        "faulty_spectrum": [128],
    }
    assert any(
        "DistributionalTransportValidation" in callback_spec.factory
        for callback_spec in scenario.callbacks
    )
    assert scenario.callbacks[0].kwargs["seed"] == seed
    assert scenario.callbacks[2].kwargs["monitor"] == "val/transport_swd"
    flow_component = scenario.pipeline.stages["Flow"][0].component
    assert isinstance(flow_component, Flow)
    assert flow_component.initialization_seed == seed
    compiled = compile_pipeline(scenario)
    x = TensorDict(
        {
            "healthy_spectrum": torch.randn(3, 128),
            "faulty_spectrum": torch.randn(3, 128),
        },
        batch_size=[3],
    )
    result, _ = compiled(x, None)
    assert isinstance(result, TensorDict)
    velocity = cast(torch.Tensor, result["predicted_velocity"])
    assert velocity.shape == (3, 128)
    cast(torch.Tensor, result["flow_matching_loss"]).mean().backward()
    assert any(param.grad is not None for param in compiled.parameters())
    output = tmp_path / "flow.yaml"
    compiled.resolved_config.save(output)
    assert ResolvedConfig.load(output).to_scenario().name == scenario.name


def test_controlled_summary_rejects_mismatched_engine_labels(
    tmp_path: Path,
) -> None:
    """Never summarize pairs trained against different official test engines."""
    for seed in (42, 43, 44):
        for model in ("patchformer", "dmm_ja", "safety_lstm", "rul_mamba"):
            filename = (
                f"rul-FD001-controlled-{model.replace('_', '-')}-seed{seed}.csv"
            )
            with (tmp_path / filename).open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["asset_id", "predicted_rul", "true_rul_capped", "true_rul_raw"]
                )
                for engine in range(1, 101):
                    truth = float(40 + engine % 20)
                    writer.writerow(
                        [f"test:FD001:{engine}", truth + 1, truth, truth]
                    )
    report = summarize_controlled_study(
        tmp_path, subsets=("FD001",), seeds=(42, 43, 44)
    )
    comparisons = report["comparisons"]
    assert len(comparisons) == 2
    assert all(row["per_seed"][0]["seed"] == 42 for row in comparisons)
    assert all(
        row["aggregates"]["patchformer"]["rmse"][
            "mean_difference_vs_safety_lstm"
        ] == pytest.approx(0.0)
        for row in comparisons
    )
    wrong = tmp_path / "rul-FD001-controlled-dmm-ja-seed44.csv"
    with wrong.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["asset_id", "predicted_rul", "true_rul_capped", "true_rul_raw"]
        )
        for engine in range(1, 101):
            truth = float(41 + engine % 20)
            writer.writerow([f"test:FD001:{engine}", truth + 1, truth, truth])
    with pytest.raises(ValueError, match="different test targets"):
        summarize_controlled_study(
            tmp_path, subsets=("FD001",), seeds=(42, 43, 44)
        )
