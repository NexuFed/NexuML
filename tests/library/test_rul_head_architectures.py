"""Matched Transformer, Mamba-style SSM and CfC RUL head tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Literal

import pytest
import torch

from nexuml.core.compiler import compile as compile_pipeline
from nexuml.core.components import LayerBuildContext
from nexuml_library.data.maintenance.rul_sequence_flow import (
    AutoPatchFormerSequenceFlowRULFeatures,
)
from nexuml_library.evaluation.maintenance.rul_engine_audit import RULPerEngineAudit
from nexuml_library.evaluation.maintenance.rul_head_architecture_summary import (
    summarize_rul_head_architectures,
)
from nexuml_library.layers.model.rul_sequence_flow import (
    PatchFormerSequenceFlowRULHead,
)
from nexuml_library.scenarios.maintenance.rul_head_architectures import (
    cfc_rul_head_architecture_study,
)

Head = Literal["transformer", "mamba", "cfc"]
Arm = Literal["encoder", "flow"]


@pytest.mark.parametrize("architecture", ["transformer", "mamba", "cfc"])
def test_rul_architecture_consumes_all_tokens_and_has_gradients(
    architecture: Head,
) -> None:
    """Every readout predicts bounded RUL from both 128-feature channels."""
    head = PatchFormerSequenceFlowRULHead(
        architecture=architecture,
        hidden_dim=64,
        attention_heads=4,
        dropout=0.0,
        initialization_seed=42,
    )
    layer = head.build(
        LayerBuildContext(
            input_sizes={"sequence_rul_features": (30, 256)},
            keys_in=["sequence_rul_features"],
            keys_out=["predicted_rul"],
        )
    )
    features = torch.randn(2, 30, 256, requires_grad=True)
    output = layer.forward_tensor(features)
    assert output.shape == (2, 1)
    assert bool(((output > 0) & (output < 130)).all())
    output.sum().backward()
    assert features.grad is not None
    assert bool((features.grad[:, :, :128] != 0).any())
    assert bool((features.grad[:, :, 128:] != 0).any())
    assert all(
        parameter.grad is not None
        for parameter in layer.parameters()
        if parameter.requires_grad
    )


@pytest.mark.parametrize("subset", ["FD001", "FD003"])
@pytest.mark.parametrize("architecture", ["transformer", "mamba", "cfc"])
@pytest.mark.parametrize("arm", ["encoder", "flow"])
@pytest.mark.parametrize("seed", [42, 43, 44])
def test_head_study_keeps_upstream_and_native_training_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    architecture: Head,
    arm: Arm,
    seed: int,
    subset: Literal["FD001", "FD003"],
) -> None:
    """Six cells × three head seeds hold the same frozen data and MSE protocol."""
    monkeypatch.chdir(tmp_path)
    study = cfc_rul_head_architecture_study(architecture, arm, seed, subset)
    source = study.data.datasets[0].source
    assert isinstance(source, AutoPatchFormerSequenceFlowRULFeatures)
    assert source.subset == subset
    assert source.seed == 42
    assert source.architecture == "cfc"
    assert source.use_flow == (arm == "flow")
    assert source.engine_balanced is False
    assert f"cmapss-{subset}" in source.prepared_rul_path
    assert f"rul-{subset}-patchformer-sequence-cfc" in source.endpoint_artifact
    layer = study.pipeline.stages["RULModel"][0].component
    assert isinstance(layer, PatchFormerSequenceFlowRULHead)
    assert layer.architecture == architecture
    assert layer.hidden_dim == 64
    assert layer.dropout == pytest.approx(0.1)
    assert layer.initialization_seed == seed
    assert study.training.lr == pytest.approx(1e-3)
    assert study.training.loss_keys == {"rul_loss": 1.0}
    assert study.name == f"rul-{subset}-cfc-head-{architecture}-{arm}-s{seed}"
    assert study.callbacks[0].kwargs["seed"] == seed
    checkpoint_dir = study.callbacks[2].kwargs["dirpath"]
    assert isinstance(checkpoint_dir, str)
    assert f"rul-{subset}" in checkpoint_dir
    assert study.callbacks[2].kwargs["monitor"] == "val/loss"
    assert study.data.input_shapes == {"sequence_rul_features": [30, 256]}
    if seed == 42:
        compiled = compile_pipeline(study)
        assert compiled.resolved_config.to_scenario().name == study.name


def _synthetic_run(
    root: Path, audits: Path, architecture: str, arm: str,
    subset: Literal["FD001", "FD003"] = "FD001",
) -> None:
    """Write coherent fitting-epoch, provenance and checkpoint fixture files."""
    name = f"rul-{subset}-cfc-head-{architecture}-{arm}-s42"
    score = (
        {"transformer": 220.0, "mamba": 190.0, "cfc": 205.0}[architecture]
        + (25.0 if arm == "encoder" else 0.0)
    )
    with (root / f"{name}.curves.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("epoch", "train_eval_mse", "val_eval_mse", "mse_gap"),
        )
        writer.writeheader()
        writer.writerow({
            "epoch": 0, "train_eval_mse": 140.0, "val_eval_mse": score,
            "mse_gap": score - 140.0,
        })
        writer.writerow({
            "epoch": 1, "train_eval_mse": 60.0, "val_eval_mse": score + 25.0,
            "mse_gap": score - 35.0,
        })
    (root / f"{name}.curves.validation_engines.json").write_text(
        json.dumps({
            "best_epoch": 0,
            "test_labels_used": False,
            "validation_engines": [
                {"asset_id": f"val:{index}", "windows": 10, "mse": score + index}
                for index in range(8)
            ],
            "validation_engine_groups": [
                {"group": index, "engines": 2, "window_mse": score + index}
                for index in range(4)
            ],
        }),
        encoding="utf-8",
    )
    (audits / f"{name}.geometry.json").write_text(
        json.dumps({
            "subset": subset,
            "seed": 42,
            "architecture": "cfc",
            "use_flow": arm == "flow",
            "engine_balanced": False,
            "test_labels_used_in_feature_fitting": False,
            "patchformer_checkpoint_sha256": "shared-encoder",
            "endpoint_artifact_sha256": "shared-endpoint",
            "flow_checkpoint_sha256": "shared-flow" if arm == "flow" else None,
        }),
        encoding="utf-8",
    )
    (audits / f"{name}.checkpoint.json").write_text(
        json.dumps({
            "checkpoint_epoch": 0,
            "loaded_strict_model_state": True,
        }),
        encoding="utf-8",
    )


@pytest.mark.parametrize("subset", ["FD001", "FD003"])
def test_head_summary_is_validation_only_and_checks_frozen_provenance(
    tmp_path: Path,
    subset: Literal["FD001", "FD003"],
) -> None:
    """Rank complete matched head grid without looking at any test audit CSV."""
    root = tmp_path / "head-architectures"
    audits = tmp_path / "audits"
    root.mkdir()
    audits.mkdir()
    for architecture in ("transformer", "mamba", "cfc"):
        for arm in ("encoder", "flow"):
            _synthetic_run(root, audits, architecture, arm, subset)
    summary = summarize_rul_head_architectures(root, audits, (42,), subset)
    assert summary["subset"] == subset
    assert summary["test_predictions_read"] is False
    assert summary["best_validation_head_for_flow"] == "mamba"
    assert len(summary["ranking"]) == 3
    assert summary["ranking"][0]["paired_flow_minus_encoder_val_mse_mean"] == -25
    assert all(
        entry["trainable_parameters"] > 0 for entry in summary["ranking"]
    )
    assert len({
        entry["trainable_parameters"] for entry in summary["ranking"]
    }) == 3
    corrupted = audits / f"rul-{subset}-cfc-head-mamba-flow-s42.geometry.json"
    payload = json.loads(corrupted.read_text(encoding="utf-8"))
    payload["flow_checkpoint_sha256"] = "different-flow"
    corrupted.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent frozen upstream"):
        summarize_rul_head_architectures(root, audits, (42,), subset)


def test_fd003_scenarios_do_not_reuse_fd001_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FD003 source, test audit and curves must not reference FD001 artifacts."""
    monkeypatch.chdir(tmp_path)
    fd001 = cfc_rul_head_architecture_study("cfc", "flow", 42, "FD001")
    fd003 = cfc_rul_head_architecture_study("cfc", "flow", 42, "FD003")
    source001 = fd001.data.datasets[0].source
    source003 = fd003.data.datasets[0].source
    assert isinstance(source001, AutoPatchFormerSequenceFlowRULFeatures)
    assert isinstance(source003, AutoPatchFormerSequenceFlowRULFeatures)
    assert source001.prepared_rul_path != source003.prepared_rul_path
    assert source001.endpoint_artifact != source003.endpoint_artifact
    audit001 = fd001.evaluation.algorithms[-1].algorithm
    audit003 = fd003.evaluation.algorithms[-1].algorithm
    assert isinstance(audit001, RULPerEngineAudit)
    assert isinstance(audit003, RULPerEngineAudit)
    assert audit001.output_file != audit003.output_file
    curve001 = fd001.callbacks[1].kwargs["output_file"]
    curve003 = fd003.callbacks[1].kwargs["output_file"]
    assert isinstance(curve001, str)
    assert isinstance(curve003, str)
    assert "fd003" in curve003
    assert "fd003" not in curve001
