"""NEX-233 run-to-failure data, architecture and NexuML CLI contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from tensordict import TensorDict

from nexuml.core.compiler import compile as compile_pipeline
from nexuml.core.config import ResolvedConfig
from nexuml_library.data.maintenance.rul import (
    PreparedRULWindows,
    read_rul_artifact,
    validate_rul_payload,
)
from nexuml_library.evaluation.maintenance.rul_configs import resolve_rul_configs
from nexuml_library.evaluation.maintenance.rul_evaluator import _RULBenchmarkRuntime
from nexuml_library.evaluation.maintenance.rul_prepare import (
    prepare_cmapss,
    prepare_xjtu_sy,
    save_rul_artifact,
)
from nexuml_library.layers.model.rul_forecaster import (
    _RULObjectiveRuntime,
    _TemporalRULRuntime,
)
from nexuml_library.layers.model.rul_metrics import _RULMetricsRuntime
from nexuml_library.scenarios.maintenance.rul import maintenance_rul_benchmark


def _cmapss_files(folder: Path, *, extra_test: float = 0) -> None:
    """Write minimal official-layout 26-column FD001 fixtures."""
    folder.mkdir(parents=True, exist_ok=True)
    for origin, count in (("train", 8), ("test", 2)):
        rows = []
        for engine in range(1, count + 1):
            for cycle in range(1, 6):
                channels = [engine * 0.1 + cycle * 0.01 + index * 0.001
                            for index in range(24)]
                if origin == "test":
                    channels = [v + extra_test for v in channels]
                rows.append([engine, cycle, *channels])
        np.savetxt(folder / f"{origin}_FD001.txt", np.asarray(rows))
    np.savetxt(folder / "RUL_FD001.txt", np.array([3, 5]))


def _xjtu_files(folder: Path, *, recordings: int = 5) -> None:
    """Write all fifteen bearing folders with chronological two-axis vibrations."""
    for condition in (1, 2, 3):
        for bearing in range(1, 6):
            target = folder / f"35Hz{condition}" / f"Bearing{condition}_{bearing}"
            target.mkdir(parents=True)
            for i in range(1, recordings + 1):
                wave = np.linspace(-1, 1, 40) * (1 + 0.1 * i + 0.01 * bearing)
                pd.DataFrame({
                    "Horizontal_vibration": wave,
                    "Vertical_vibration": wave[::-1] * 0.7,
                }).to_csv(target / f"{i}.csv", index=False)


def test_cmapss_asset_disjoint_and_official_rul(tmp_path: Path) -> None:
    """No engine crosses splits; official test offsets provide terminal RUL."""
    root = tmp_path / "cmapss"
    _cmapss_files(root)
    payload = prepare_cmapss(root, window=3, seed=42)
    validate_rul_payload(payload)
    saved = save_rul_artifact(payload, tmp_path / "prepared.pt")
    assert saved["num_windows"] > 0
    disk = read_rul_artifact(tmp_path / "prepared.pt")
    assert disk["features"].shape[1:] == (3, 24)
    split_assets = disk["settings"]["split_assets"]
    assert all(split_assets[key] for key in ("train", "val", "test"))
    assert not set(split_assets["train"]) & set(split_assets["val"])
    assert len(split_assets["test"]) == 2
    test_labels = {
        row["asset_id"]: float(disk["targets"][i, 0])
        for i, row in enumerate(disk["rows"]) if row["split"] == "test"
    }
    assert test_labels["test:FD001:1"] == pytest.approx(3 / 125)
    assert test_labels["test:FD001:2"] == pytest.approx(5 / 125)
    with pytest.raises(FileExistsError):
        save_rul_artifact(payload, tmp_path / "prepared.pt")


def test_cmapss_scaler_ignores_test_values(tmp_path: Path) -> None:
    """Replacing every test input cannot affect train-only fitted normalization."""
    root = tmp_path / "cmapss"
    _cmapss_files(root)
    baseline = prepare_cmapss(root, window=3)
    _cmapss_files(root, extra_test=10000)
    altered = prepare_cmapss(root, window=3)
    assert baseline["settings"]["scaler"] == altered["settings"]["scaler"]
    before = baseline["features"][
        [i for i, row in enumerate(baseline["rows"]) if row["split"] == "train"]
    ]
    after = altered["features"][
        [i for i, row in enumerate(altered["rows"]) if row["split"] == "train"]
    ]
    assert torch.equal(before, after)


def test_xjtu_bearing_disjoint_and_causal_windows(tmp_path: Path) -> None:
    """Each condition has 3 train, 1 val, 1 test bearing, without future context."""
    _xjtu_files(tmp_path)
    payload = prepare_xjtu_sy(tmp_path, window=3, seed=43)
    validate_rul_payload(payload)
    assert payload["features"].shape == (75, 3, 12)
    split_assets = payload["settings"]["split_assets"]
    assert len(split_assets["train"]) == 9
    assert len(split_assets["val"]) == 3
    assert len(split_assets["test"]) == 3
    assert payload["settings"]["target_unit"] == "recordings"
    for i, row in enumerate(payload["rows"]):
        if row["step"] == 4:
            assert float(payload["targets"][i, 0]) == pytest.approx(0)


def test_rul_dataset_split_preserves_targets(tmp_path: Path) -> None:
    """Normal NexuML get_split/take does not silently detach RUL labels."""
    root = tmp_path / "cmapss"
    _cmapss_files(root)
    save_rul_artifact(prepare_cmapss(root, window=3), tmp_path / "prepared.pt")
    dataset = PreparedRULWindows(path=str(tmp_path / "prepared.pt")).build()
    for split in ("train", "val", "test"):
        subset = dataset.get_split(split)
        assert len(subset) > 0
        for index in range(min(len(subset), 3)):
            x, y = subset[index]
            assert x["rul_history"].shape == (3, 24)
            assert y is not None
            assert y["rul"].shape == (1,)
    x, y = dataset[0]
    assert y is not None and x["rul_history"].shape == (3, 24)


@pytest.mark.parametrize(
    "architecture", ["feature_mlp", "lstm", "tcn", "transformer", "tsmixer"]
)
def test_temporal_models_forward_and_backward(architecture: str) -> None:
    """Every baseline consumes causal windows and supports regression gradients."""
    runtime = _TemporalRULRuntime(
        input_sizes={"rul_history": (8, 12)},
        keys_in=["rul_history"],
        keys_out=["predicted_rul"],
        architecture=architecture,
        hidden_dim=16,
        depth=2,
        attention_heads=4,
        output_dim=1,
        dropout=0.0,
    )
    prediction = runtime.forward_tensor(torch.randn(3, 8, 12))
    assert prediction.shape == (3, 1)
    prediction.square().mean().backward()
    assert any(p.grad is not None for p in runtime.parameters())


def test_quantile_loss_and_original_unit_metrics() -> None:
    """Quantile objective uses y.rul and metric output restores original units."""
    x = TensorDict({"predicted_rul": torch.tensor([[0.1, 0.5, 0.9]])}, batch_size=[1])
    y = TensorDict({"rul": torch.tensor([[0.6]])}, batch_size=[1])
    objective = _RULObjectiveRuntime(
        input_sizes={"predicted_rul": (3,)}, keys_in=["predicted_rul"],
        keys_out=["rul_loss"], objective="quantile",
        quantiles=[0.1, 0.5, 0.9], label_key="rul",
    )
    result, _ = objective(x, y)
    assert result["rul_loss"].item() >= 0
    metric = _RULMetricsRuntime(
        input_sizes={"predicted_rul": (3,)}, keys_in=["predicted_rul"],
        keys_out=["rul_mae", "rul_rmse", "rul_late_error"],
        target_scale=125, prediction_index=1, label_key="rul",
    )
    measured, _ = metric(x, y)
    assert float(measured["rul_mae"][0]) == pytest.approx(12.5)
    assert float(measured["rul_rmse"][0]) == pytest.approx(12.5)
    assert float(measured["rul_late_error"][0]) == 0.0


def test_exact_rul_evaluator_and_quantile_coverage() -> None:
    """Exact held-out aggregation differs from an unweighted mean of batch RMSEs."""
    runtime = _RULBenchmarkRuntime(
        prediction_key="predicted_rul", label_key="rul",
        target_scale=100.0, point_index=1, lower_index=0, upper_index=2,
    )
    first = TensorDict({
        "predicted_rul": torch.tensor([[0.1, 0.2, 0.3], [0.1, 0.8, 0.9]])
    }, batch_size=[2])
    y = TensorDict({"rul": torch.tensor([[0.2], [0.4]])}, batch_size=[2])
    runtime.eval_batch(first, y)
    second = TensorDict({
        "predicted_rul": torch.tensor([[0.2, 0.5, 0.7]])
    }, batch_size=[1])
    y2 = TensorDict({"rul": torch.tensor([[0.6]])}, batch_size=[1])
    runtime.eval_batch(second, y2)
    result = runtime.results()
    assert result["rul_test_count"] == 3
    assert result["rul_test_mae"] == pytest.approx(50 / 3)
    assert result["rul_test_rmse"] == pytest.approx((1700 / 3) ** 0.5)
    assert result["rul_test_interval_coverage"] == pytest.approx(2 / 3)
    assert result["rul_test_quantile_crossing_rate"] == 0


def test_resolved_scenario_roundtrip_and_no_custom_training(tmp_path: Path) -> None:
    """Standard ScenarioSpec YAML resolution/build supports native NexuML CLI."""
    root = tmp_path / "cmapss"
    _cmapss_files(root)
    artifact = tmp_path / "prepared.pt"
    save_rul_artifact(prepare_cmapss(root, window=3), artifact)
    scenario = maintenance_rul_benchmark(
        artifact, architecture="tcn", hidden_dim=16, depth=2,
        max_epochs=2, batch_size=4, seed=42,
    )
    config = compile_pipeline(scenario).resolved_config
    path = tmp_path / "rul.yaml"
    config.save(path)
    restored = ResolvedConfig.load(path).to_scenario()
    assert restored.name == scenario.name
    assert restored.data.input_shapes["rul_history"] == [3, 24]
    assert restored.training.loss_keys == {"rul_loss": 1.0}
    assert "rul_mae" in restored.training.metric_keys
    compile_pipeline(restored)
    index = resolve_rul_configs(
        [artifact], tmp_path / "configs",
        architectures=("lstm", "tcn"), max_epochs=2, depth=2,
        hidden_dim=16,
    )
    assert len(index["configs"]) == 2
    assert all("nexuml train --config " in x["train_command"] for x in index["configs"])
    saved = json.loads((tmp_path / "configs" / "manifest.json").read_text())
    assert len(saved["configs"]) == 2
    with pytest.raises(FileExistsError):
        resolve_rul_configs([artifact], tmp_path / "configs")


def test_rul_rejects_window_level_cross_asset_leakage(tmp_path: Path) -> None:
    """Prepared metadata cannot relabel a training asset as an evaluation asset."""
    root = tmp_path / "cmapss"
    _cmapss_files(root)
    payload = prepare_cmapss(root, window=3)
    first = next(i for i, row in enumerate(payload["rows"]) if row["split"] == "train")
    payload["rows"][first]["split"] = "test"
    with pytest.raises(ValueError, match="span multiple splits"):
        validate_rul_payload(payload)
