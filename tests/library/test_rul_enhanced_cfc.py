"""Tests for real predecessor pairs and toggled CfC RUL improvements."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import lightning as L
from lightning.pytorch.trainer.states import TrainerFn

import pytest
import torch
from tensordict import TensorDict
from torch import Tensor

from nexuml.core.components import LayerBuildContext
from nexuml_library.data.maintenance.rul_enhanced import (
    _PairedRULRuntime,
    AutoPairedCMapSSRULWindows,
)
from nexuml_library.evaluation.maintenance.rul_composite_comparison import (
    RULCompositeAblationComparison,
    summarize_composite_seeds,
)
from nexuml_library.evaluation.maintenance.rul_enhanced_summary import (
    summarize_enhanced_cfc,
)
from nexuml_library.evaluation.maintenance.rul_mc_conformal import (
    _quantile,
    _selected_calibration_windows,
)
from nexuml_library.layers.model.rul_enhanced_cfc import (
    EnhancedCfCRULHead,
    _EnhancedCfCRuntime,
)
from nexuml_library.layers.model.rul_trajectory_loss import (
    RULTrajectoryConsistencyLoss,
)
from nexuml_library.scenarios.maintenance.rul_enhanced_cfc import (
    Ablation,
    CONFIGURATIONS,
    OPTIONS,
    enabled_options,
    enhanced_cfc_rul_study,
    enhanced_cfc_native_comparison,
)


def _context(sensors: int = 15) -> LayerBuildContext:
    """Create exact raw C-MAPSS pair dimensions.

    Returns:
        Native compiler context for two shared-weight RUL predictions.
    """
    return LayerBuildContext(
        input_sizes={
            "rul_history": (30, sensors),
            "previous_rul_history": (30, sensors),
            "pair_mask": (1,),
        },
        keys_in=["rul_history"],
        keys_out=["predicted_rul", "previous_predicted_rul"],
    )


def test_paired_windows_observe_exact_engine_and_split_boundaries() -> None:
    """Predecessors must be actual adjacent samples from the same asset."""
    rows = [
        {"split": "train", "asset_id": "train:1", "step": 30},
        {"split": "train", "asset_id": "train:1", "step": 31},
        {"split": "val", "asset_id": "val:1", "step": 30},
        {"split": "val", "asset_id": "val:1", "step": 31},
        {"split": "test", "asset_id": "test:1", "step": 20},
    ]
    features = torch.arange(5.0).view(5, 1, 1).expand(5, 30, 15).clone()
    target = torch.arange(5.0).reshape(5, 1)
    raw = target + 20
    dataset = _PairedRULRuntime(
        features=features, targets=target, raw=raw, rows=rows
    )
    train = dataset.get_split("train")
    val = dataset.get_split("val")
    test = dataset.get_split("test")
    first, _ = train[0]
    second, labels = train[1]
    assert cast(Tensor, first["pair_mask"]).item() == 0
    assert cast(Tensor, second["pair_mask"]).item() == 1
    assert cast(Tensor, second["previous_rul_history"])[0, 0].item() == 0
    assert cast(Tensor, second["rul_history"])[0, 0].item() == 1
    assert labels is not None
    assert cast(Tensor, labels["rul_raw"]).item() == 21
    prev_val, _ = val[1]
    assert cast(Tensor, prev_val["previous_rul_history"])[0, 0].item() == 2
    one_test, _ = test[0]
    assert cast(Tensor, one_test["pair_mask"]).item() == 0


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
def test_enhanced_scenarios_expose_each_predeclared_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configuration: str,
) -> None:
    """On/off configurations keep seed and upstream data exactly fixed."""
    monkeypatch.chdir(tmp_path)
    variant = cast(Ablation, configuration)
    scenario = enhanced_cfc_rul_study(variant, 43)
    assert scenario.name == f"rul-FD003-enhanced-cfc-{variant}-s43"
    source = scenario.data.datasets[0].source
    assert isinstance(source, AutoPairedCMapSSRULWindows)
    assert source.subset == "FD003"
    assert source.seed == 42
    assert scenario.data.input_shapes["rul_history"] == [30, 15]
    head = scenario.pipeline.stages["RULModel"][0].component
    assert isinstance(head, EnhancedCfCRULHead)
    options = enabled_options(variant)
    assert head.train_encoder == ("finetune" in options)
    assert head.sensor_graph == ("graph" in options)
    assert head.trend == ("trend" in options)
    assert head.multiscale == ("multiscale" in options)
    assert head.segmentation == ("segmentation" in options)
    assert head.trajectory == ("trajectory" in options)
    assert head.initialization_seed == 43
    assert scenario.training.lr == pytest.approx(1e-3)
    assert scenario.training.loss_keys == {"rul_loss": 1.0}
    assert any(
        "RULMCConformalDiagnostics" in str(item.factory)
        for item in scenario.callbacks
    ) is (configuration == "all")


def test_every_effect_has_a_single_and_leave_one_out() -> None:
    """Only the chosen mechanism differs in paired ablation comparisons."""
    baseline = enabled_options("baseline")
    all_on = enabled_options("all")
    assert baseline == frozenset()
    assert all_on == frozenset(OPTIONS)
    for option in OPTIONS:
        assert enabled_options(option) == {option}
        variant = cast(
            Literal[
                "all_no_finetune", "all_no_graph", "all_no_trend",
                "all_no_multiscale", "all_no_segmentation",
                "all_no_trajectory",
            ],
            f"all_no_{option}",
        )
        assert all_on - enabled_options(variant) == {option}


def test_enhanced_head_uses_pretrained_encoder_and_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same code path permits frozen or trainable PatchFormer with all options."""
    import nexuml_library.layers.model.rul_enhanced_cfc as module

    monkeypatch.setattr(
        module, "resolve_patchformer_checkpoint", lambda *_: Path("dummy.ckpt")
    )
    monkeypatch.setattr(module, "_trusted_patchformer_weights", lambda *_: "digest")
    monkeypatch.setattr(
        module,
        "read_prepared_artifact",
        lambda *_: {
            "settings": {
                "subset": "FD003",
                "seed": 42,
                "feature_normalization": "train_endpoints_only",
                "frozen_patchformer_checkpoint_sha256": "digest",
                "held_out_test_engine_labels_used": False,
                "feature_mean": torch.zeros(30, 128).tolist(),
                "feature_std": torch.ones(30, 128).tolist(),
            }
        },
    )
    head = EnhancedCfCRULHead(
        train_encoder=True,
        sensor_graph=True,
        trend=True,
        multiscale=True,
        segmentation=True,
        trajectory=True,
        initialization_seed=42,
    ).build(_context())
    assert isinstance(head, _EnhancedCfCRuntime)
    head.train()
    x = TensorDict(
        {
            "rul_history": torch.randn(2, 30, 15),
            "previous_rul_history": torch.randn(2, 30, 15),
            "pair_mask": torch.tensor([[1.0], [0.0]]),
        },
        batch_size=[2],
    )
    y = TensorDict({"rul": torch.tensor([[70.0], [30.0]])}, batch_size=[2])
    result, _ = head(x, y)
    assert isinstance(result, TensorDict)
    assert cast(Tensor, result["predicted_rul"]).shape == (2, 1)
    assert cast(Tensor, result["previous_predicted_rul"]).shape == (2, 1)
    cast(Tensor, result["predicted_rul"]).sum().backward()
    assert any(parameter.grad is not None for parameter in head.encoder.parameters())

    # Optional additions must never change the common readout initialization.
    reference = EnhancedCfCRULHead(initialization_seed=42).build(_context())
    graph_only = EnhancedCfCRULHead(
        initialization_seed=42, sensor_graph=True
    ).build(_context())
    assert isinstance(reference, _EnhancedCfCRuntime)
    assert isinstance(graph_only, _EnhancedCfCRuntime)
    for key in (
        "input_projection.weight",
        "input_projection.bias",
        "temporal.cells.0.first.weight",
        "regressor.0.weight",
    ):
        assert torch.equal(
            reference.state_dict()[key], graph_only.state_dict()[key]
        )
        assert torch.equal(
            reference.state_dict()[key], head.state_dict()[key]
        )
    assert graph_only.sensor_graph is not None
    assert head.sensor_graph is not None
    for key, parameter in graph_only.sensor_graph.state_dict().items():
        assert torch.equal(parameter, head.sensor_graph.state_dict()[key])


def test_trajectory_objective_uses_true_pairs_only_and_train_only() -> None:
    """Penalty cannot contaminate validation loss or nonexistent engine pairs."""
    context = LayerBuildContext(
        input_sizes={
            "predicted_rul": (1,), "previous_predicted_rul": (1,),
            "pair_mask": (1,),
        },
        keys_in=["predicted_rul"],
        keys_out=["rul_loss"],
    )
    layer = RULTrajectoryConsistencyLoss(consistency_weight=0.5).build(context)
    x = TensorDict(
        {
            "predicted_rul": torch.tensor([[70.0], [80.0]]),
            "previous_predicted_rul": torch.tensor([[73.0], [200.0]]),
            "pair_mask": torch.tensor([[1.0], [0.0]]),
        },
        batch_size=[2],
    )
    y = TensorDict({"rul": torch.tensor([[70.0], [80.0]])}, batch_size=[2])
    layer.train()
    trained, _ = layer(x.clone(), y)
    assert isinstance(trained, TensorDict)
    assert cast(Tensor, trained["rul_loss"]).mean().item() == pytest.approx(2.0)
    layer.eval()
    validated, _ = layer(x.clone(), y)
    assert isinstance(validated, TensorDict)
    assert cast(Tensor, validated["rul_loss"]).mean().item() == pytest.approx(0.0)


def test_uncertainty_quantile_is_engine_level_and_finite_sample() -> None:
    """Conformal diagnostic rank explicitly rejects insufficient support."""
    assert _quantile([1.0] * 20, 0.1) == 1.0
    assert _quantile([1.0, 2.0], 0.1) == float("inf")
    with pytest.raises(ValueError, match="finite"):
        _quantile([float("nan")], 0.1)
    import pandas as pd

    metadata = pd.DataFrame({
        "split": ["val"] * 4,
        "asset_id": ["v:1", "v:1", "v:2", "v:2"],
    })
    assert len(_selected_calibration_windows(metadata)) == 2


def test_enhanced_summary_never_reads_test_audits(tmp_path: Path) -> None:
    """Complete fixed 16-cell synthetic validation grid and paired effects."""
    root = tmp_path / "curves"
    audit = tmp_path / "audits"
    root.mkdir()
    audit.mkdir()
    for index, configuration in enumerate(CONFIGURATIONS):
        name = f"rul-FD003-enhanced-cfc-{configuration}-s42"
        with (root / f"{name}.curves.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("epoch", "train_eval_mse", "val_eval_mse", "mse_gap")
            )
            writer.writeheader()
            writer.writerow({
                "epoch": 0, "train_eval_mse": 90, "val_eval_mse": 150 + index,
                "mse_gap": 60 + index,
            })
        (root / f"{name}.curves.validation_engines.json").write_text(
            json.dumps({
                "best_epoch": 0,
                "test_labels_used": False,
                "validation_engines": [
                    {"asset_id": f"val:{j}", "windows": 8, "mse": 150 + j}
                    for j in range(5)
                ],
            }),
            encoding="utf-8",
        )
        (audit / f"{name}.checkpoint.json").write_text(
            json.dumps({
                "checkpoint_epoch": 0, "loaded_strict_model_state": True
            }),
            encoding="utf-8",
        )
    report = summarize_enhanced_cfc(root, audit, (42,))
    assert report["test_predictions_read"] is False
    assert len(report["ranking"]) == 14
    assert len(report["effects"]) == 6
    assert report["effects"]["finetune"]["single_minus_baseline_mean"] == 1


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_native_composite_trains_independent_named_heads(seed: int) -> None:
    """One ordinary scenario carries all 16 on/off RUL model/loss branches."""
    scenario = enhanced_cfc_native_comparison(seed)
    assert scenario.name == f"rul-FD003-enhanced-cfc-compare-six-s{seed}"
    assert scenario.training.batch_size == 16
    assert len(scenario.pipeline.stages["RULModels"]) == len(CONFIGURATIONS)
    assert len(scenario.pipeline.stages["RULLoss"]) == len(CONFIGURATIONS)
    assert len(scenario.training.loss_keys) == len(CONFIGURATIONS)
    assert len(scenario.evaluation.algorithms) == 0
    assert sum(scenario.training.loss_keys.values()) == pytest.approx(1.0)
    for conf, model, loss in zip(
        CONFIGURATIONS,
        scenario.pipeline.stages["RULModels"],
        scenario.pipeline.stages["RULLoss"],
    ):
        assert isinstance(model.component, EnhancedCfCRULHead)
        assert isinstance(loss.component, RULTrajectoryConsistencyLoss)
        assert model.keys_out == [
            f"predicted_rul_{conf}", f"previous_predicted_rul_{conf}"
        ]
        assert loss.keys_in == [f"predicted_rul_{conf}"]
        assert loss.keys_out == [f"rul_loss_{conf}"]
        assert loss.component.previous_key == f"previous_predicted_rul_{conf}"
        assert (
            loss.component.consistency_weight > 0
        ) is ("trajectory" in enabled_options(conf))
        assert model.component.initialization_seed == seed
        assert scenario.training.loss_keys[f"rul_loss_{conf}"] == pytest.approx(
            1.0 / len(CONFIGURATIONS)
        )
    callbacks = [cb.factory for cb in scenario.callbacks]
    assert any("RULCompositeAblationComparison" in name for name in callbacks)
    assert any("ModelCheckpoint" in name for name in callbacks)


def test_native_comparison_records_only_fitting_validation(
    tmp_path: Path,
) -> None:
    """The composite result ranks recorded per-arm validation MSE, not test."""
    output = tmp_path / "comparison.validation.csv"
    callback = RULCompositeAblationComparison(str(output), seed=42)
    metrics = {
        f"val/rul_loss_{conf}": torch.tensor(100.0 + index)
        for index, conf in enumerate(CONFIGURATIONS)
    }
    trainer = cast(
        L.Trainer,
        SimpleNamespace(
            sanity_checking=False,
            state=SimpleNamespace(fn=TrainerFn.FITTING),
            current_epoch=0,
            callback_metrics=metrics,
        ),
    )
    module = cast(L.LightningModule, SimpleNamespace())
    callback.on_fit_start(trainer, module)
    callback.on_validation_end(trainer, module)
    callback.on_fit_end(trainer, module)
    report = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert report["test_predictions_read"] is False
    assert report["optimizer_shared"] is True
    assert report["independently_restored_arm_checkpoints"] is False
    assert len(report["ranking"]) == len(CONFIGURATIONS)
    assert len(report["effects"]) == len(OPTIONS)
    assert report["ranking"][0]["configuration"] == "baseline"


def test_uncertainty_not_a_point_estimator_factor() -> None:
    """Calibration observes existing fitted all-on weights; it is not an arm."""
    assert len(OPTIONS) == 6
    assert len(CONFIGURATIONS) == 14
    assert "uncertainty" not in CONFIGURATIONS
    assert "all_no_uncertainty" not in CONFIGURATIONS
    for configuration in CONFIGURATIONS:
        spec = enhanced_cfc_rul_study(configuration, 42)
        calibration = [
            cb for cb in spec.callbacks
            if "RULMCConformalDiagnostics" in cb.factory
        ]
        assert len(calibration) == int(configuration == "all")
        if calibration:
            assert "uncertainty.json" in str(calibration[0].kwargs["output_file"])
            factories = [cb.factory for cb in spec.callbacks]
            calibrator_index = next(
                i for i, factory in enumerate(factories)
                if "RULMCConformalDiagnostics" in factory
            )
            restoration_index = next(
                i for i, factory in enumerate(factories)
                if "Goel2026RestoreBestWeights" in factory
            )
            assert calibrator_index > restoration_index
    assert all("uncertainty" not in opt for opt in OPTIONS)


def test_three_seed_composite_summary_ignores_legacy_and_test(
    tmp_path: Path,
) -> None:
    """Only matching six-factor v2 seed reports may be pooled."""
    for seed in (42, 43, 44):
        ranking = [
            {
                "configuration": configuration,
                "best_observed_val_mse": float(100 + index + seed),
                "best_epoch": 7,
            }
            for index, configuration in enumerate(CONFIGURATIONS)
        ]
        report = {
            "study_version": "six-model-factors-v2",
            "subset": "FD003",
            "head_seed": seed,
            "configurations": list(CONFIGURATIONS),
            "optimizer_shared": True,
            "test_predictions_read": False,
            "ranking": ranking,
        }
        file = (
            tmp_path / f"rul-FD003-enhanced-cfc-compare-six-s{seed}.validation.json"
        )
        file.write_text(json.dumps(report), encoding="utf-8")
    summary = summarize_composite_seeds(tmp_path)
    assert summary["test_predictions_read"] is False
    assert summary["head_seeds"] == [42, 43, 44]
    assert len(summary["ranking"]) == 14
    assert len(summary["paired_effects"]) == 6
    assert summary["ranking"][0]["configuration"] == "baseline"
    assert summary["ranking"][0]["validation_mse_std"] == pytest.approx(1.0)
    assert summary["paired_effects"]["finetune"][
        "single_minus_baseline_mean"
    ] == pytest.approx(1.0)
    bad = tmp_path / "rul-FD003-enhanced-cfc-compare-six-s42.validation.json"
    invalid = json.loads(bad.read_text(encoding="utf-8"))
    invalid["configurations"].append("uncertainty")
    bad.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="Incompatible"):
        summarize_composite_seeds(tmp_path)
