"""Tests for real predecessor pairs and toggled CfC RUL improvements."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Literal, cast

import pytest
import torch
from tensordict import TensorDict
from torch import Tensor

from nexuml.core.components import LayerBuildContext
from nexuml_library.data.maintenance.rul_enhanced import (
    _PairedRULRuntime,
    AutoPairedCMapSSRULWindows,
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
    CONFIGURATIONS,
    OPTIONS,
    enabled_options,
    enhanced_cfc_rul_study,
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
    assert first["pair_mask"].item() == 0
    assert second["pair_mask"].item() == 1
    assert second["previous_rul_history"][0, 0].item() == 0
    assert second["rul_history"][0, 0].item() == 1
    assert labels is not None
    assert labels["rul_raw"].item() == 21
    prev_val, _ = val[1]
    assert prev_val["previous_rul_history"][0, 0].item() == 2
    one_test, _ = test[0]
    assert one_test["pair_mask"].item() == 0


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
def test_enhanced_scenarios_expose_each_predeclared_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configuration: str,
) -> None:
    """On/off configurations keep seed and upstream data exactly fixed."""
    monkeypatch.chdir(tmp_path)
    variant = cast(
        Literal[
            "baseline", "finetune", "graph", "trend", "multiscale",
            "segmentation", "trajectory", "uncertainty", "all",
            "all_no_finetune", "all_no_graph", "all_no_trend",
            "all_no_multiscale", "all_no_segmentation",
            "all_no_trajectory", "all_no_uncertainty",
        ],
        configuration,
    )
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
    ) is ("uncertainty" in options)


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
                "all_no_trajectory", "all_no_uncertainty",
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
    assert len(report["ranking"]) == 16
    assert len(report["effects"]) == 7
    assert report["effects"]["finetune"]["single_minus_baseline_mean"] == 1
