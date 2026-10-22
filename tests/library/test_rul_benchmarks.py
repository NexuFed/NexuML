"""NEX-233 run-to-failure data, architecture and NexuML CLI contract tests."""

from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from tensordict import TensorDict

from nexuml.core.compiler import compile as compile_pipeline
from nexuml.core.config import ResolvedConfig
from nexuml_library.data.maintenance import rul_download
from nexuml_library.data.maintenance.rul import (
    AutoCMapSSRULWindows,
    PreparedRULWindows,
    read_rul_artifact,
    validate_rul_payload,
)
from nexuml_library.evaluation.maintenance.rul_configs import resolve_rul_configs
from nexuml_library.evaluation.maintenance.rul_engine_audit import (
    RULPerEngineAudit,
    _RULPerEngineAuditRuntime,
)
from nexuml_library.evaluation.maintenance.rul_evaluator import (
    RULBenchmarkEvaluator,
    _RULBenchmarkRuntime,
)
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
from nexuml_library.scenarios.maintenance.rul import (
    maintenance_rul_all,
    maintenance_rul_benchmark,
    maintenance_rul_cen2024_tsmixer_proxy,
    maintenance_rul_goel2026_cnn_protocol,
    maintenance_rul_goel2026_lstm_protocol,
    maintenance_rul_lu2025_tcn_protocol,
)


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
    assert result["rul_test_interval_coverage"] == pytest.approx(1.0)
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


def test_default_rul_scenario_resolves_without_prepared_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registry scenarios compile before the default FD001 artifact is prepared."""
    monkeypatch.delenv("NEXUML_RUL_PREPARED_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    scenario = maintenance_rul_benchmark()
    assert scenario.data.input_shapes["rul_history"] == [30, 24]
    assert scenario.data.datasets[0].source is not None
    compile_pipeline(scenario)
    with pytest.raises(FileNotFoundError, match="Prepared RUL artifact"):
        maintenance_rul_benchmark(tmp_path / "missing.pt")


@pytest.mark.parametrize("override", ["", "   "])
def test_blank_prepared_file_environment_enables_auto_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override: str,
) -> None:
    """An empty override must not resolve the current directory as a dataset."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEXUML_RUL_PREPARED_FILE", override)
    scenario = maintenance_rul_all()
    source = scenario.data.datasets[0].source
    assert isinstance(source, AutoCMapSSRULWindows)
    assert source.download is True
    assert source.artifact_path == str(
        (tmp_path / ".experiments/rul/prepared/cmapss-FD001-seed42.pt").resolve()
    )
    compile_pipeline(scenario)


def test_seed_specific_auto_artifact_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Independent NexuML seeds never reuse the same prepared split cache."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NEXUML_RUL_PREPARED_FILE", raising=False)
    scenario42 = maintenance_rul_benchmark(seed=42)
    scenario43 = maintenance_rul_benchmark(seed=43)
    source42 = scenario42.data.datasets[0].source
    source43 = scenario43.data.datasets[0].source
    assert isinstance(source42, AutoCMapSSRULWindows)
    assert isinstance(source43, AutoCMapSSRULWindows)
    assert source42.artifact_path.endswith("seed42.pt")
    assert source43.artifact_path.endswith("seed43.pt")
    assert source43.seed == 43
    compile_pipeline(scenario43)


def test_composite_rul_scenario_has_six_distinct_heads(tmp_path: Path) -> None:
    """A single NexuML scenario exposes independent outputs and metrics per head."""
    root = tmp_path / "cmapss"
    _cmapss_files(root)
    artifact = tmp_path / "prepared.pt"
    save_rul_artifact(prepare_cmapss(root, window=3), artifact)
    scenario = maintenance_rul_all(
        artifact, hidden_dim=16, depth=1, batch_size=4, max_epochs=1, patience=1
    )
    assert len(scenario.pipeline.stages["RULModels"]) == 6
    assert len(scenario.pipeline.stages["RULLoss"]) == 12
    assert len(scenario.training.loss_keys) == 6
    assert len(scenario.training.metric_keys) == 18
    assert len(scenario.evaluation.algorithms) == 6
    assert len({spec.name for spec in scenario.evaluation.algorithms}) == 6
    assert sum(scenario.training.loss_keys.values()) == pytest.approx(1.0)

    compiled = compile_pipeline(scenario)
    x = TensorDict({"rul_history": torch.randn(2, 3, 24)}, batch_size=[2])
    y = TensorDict({"rul": torch.tensor([[0.2], [0.4]])}, batch_size=[2])
    result, labels = compiled(x, y)
    assert labels is y
    for name in ("feature_mlp", "lstm", "tcn", "transformer", "tsmixer", "quantile_tcn"):
        prediction = result[f"predicted_rul_{name}"]
        assert prediction.shape == (2, 3 if name == "quantile_tcn" else 1)
        assert f"rul_loss_{name}" in result.keys()
        assert f"rul_mae_{name}" in result.keys()
    total_loss = sum(
        result[key].mean() * weight for key, weight in scenario.training.loss_keys.items()
    )
    total_loss.backward()
    assert any(parameter.grad is not None for parameter in compiled.parameters())
    yaml_path = tmp_path / "composite.yaml"
    compiled.resolved_config.save(yaml_path)
    restored = ResolvedConfig.load(yaml_path).to_scenario()
    assert len(restored.training.loss_keys) == 6
    compile_pipeline(restored)


def test_auto_cmapss_source_download_and_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NASA ZIP retrieval is lazy, safe, and reused on repeated datasource builds."""
    fixture = tmp_path / "fixture"
    _cmapss_files(fixture)
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w") as archive:
        for filename in ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"):
            archive.write(fixture / filename, arcname=f"CMAPSSData/{filename}")
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as archive:
        archive.writestr("NASA/CMAPSSData.zip", nested.getvalue())
    archive_bytes = outer.getvalue()
    requested = 0

    def mocked_urlopen(*args: object, **kwargs: object) -> io.BytesIO:
        nonlocal requested
        requested += 1
        return io.BytesIO(archive_bytes)

    monkeypatch.setattr(rul_download, "urlopen", mocked_urlopen)
    root = tmp_path / "raw"
    prepared_path = tmp_path / "prepared" / "cmapss.pt"
    source = AutoCMapSSRULWindows(root=str(root), artifact_path=str(prepared_path))
    assert not prepared_path.exists()
    dataset = source.build()
    assert len(dataset) > 0
    assert prepared_path.exists()
    assert requested == 1
    payload = read_rul_artifact(prepared_path)
    assert payload["settings"]["source_url"] == rul_download.CMAPSS_ARCHIVE_URL
    assert len(payload["settings"]["source_archive_sha256"]) == 64
    assert set(payload["settings"]["split_assets"]) == {"train", "val", "test"}

    cached = source.build()
    assert len(cached) == len(dataset)
    assert requested == 1
    assert (root / "train_FD001.txt").is_file()


def test_auto_cmapss_source_offline_and_unmatched_cache(tmp_path: Path) -> None:
    """Offline mode never downloads, and prepared artifacts are immutable."""
    source = AutoCMapSSRULWindows(
        root=str(tmp_path / "raw"),
        artifact_path=str(tmp_path / "prepared.pt"),
        download=False,
    )
    with pytest.raises(FileNotFoundError, match="C-MAPSS FD001"):
        source.build()
    root = tmp_path / "raw"
    _cmapss_files(root)
    source.build()
    incompatible = source.model_copy(update={"seed": 43})
    with pytest.raises(ValueError, match="does not match the scenario"):
        incompatible.build()


def test_default_rul_all_uses_lazy_auto_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native scenario resolve requires no manual download or preprocessing."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NEXUML_RUL_PREPARED_FILE", raising=False)
    scenario = maintenance_rul_all(
        download=False, dataset_root=tmp_path / "source-cache"
    )
    dataset_source = scenario.data.datasets[0].source
    assert isinstance(dataset_source, AutoCMapSSRULWindows)
    assert dataset_source.download is False
    assert dataset_source.root == str(tmp_path / "source-cache")
    assert not (tmp_path / "data" / "cmapss").exists()
    compiled = compile_pipeline(scenario)
    assert len(compiled.loss_keys) == 6
    assert not (tmp_path / "data" / "cmapss").exists()


def test_lu2025_preprocessing_sensor_minmax_and_cache_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Paper settings select 14 exact sensors and fit minmax on training assets."""
    root = tmp_path / "cmapss"
    _cmapss_files(root)
    baseline = prepare_cmapss(root, window=40, sensor_ids=[
        2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21
    ], normalization="minmax")
    assert baseline["features"].shape[1:] == (40, 14)
    assert baseline["settings"]["feature_names"] == [
        f"sensor_{index}" for index in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
    ]
    assert baseline["settings"]["normalization"] == "minmax"
    _cmapss_files(root, extra_test=1000.0)
    altered = prepare_cmapss(root, window=40, sensor_ids=[
        2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21
    ], normalization="minmax")
    assert baseline["settings"]["scaler"] == altered["settings"]["scaler"]

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NEXUML_RUL_PREPARED_FILE", raising=False)
    lu = maintenance_rul_lu2025_tcn_protocol()
    cen = maintenance_rul_cen2024_tsmixer_proxy()
    lu_source = lu.data.datasets[0].source
    cen_source = cen.data.datasets[0].source
    assert isinstance(lu_source, AutoCMapSSRULWindows)
    assert isinstance(cen_source, AutoCMapSSRULWindows)
    assert lu_source.window == 40
    assert lu_source.normalization == "minmax"
    assert lu_source.sensor_ids == [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
    assert lu.training.batch_size == 256
    assert lu.training.lr == pytest.approx(1e-3)
    assert lu.training.max_epochs == 300
    assert lu.data.input_shapes["rul_history"] == [40, 14]
    assert cen_source.window == 28
    assert cen_source.rul_cap == 120
    assert cen.training.lr == pytest.approx(0.01)
    assert cen_source.artifact_path != lu_source.artifact_path
    for scenario in (lu, cen):
        compiled = compile_pipeline(scenario)
        path = tmp_path / f"{scenario.name}.yaml"
        compiled.resolved_config.save(path)
        restored = ResolvedConfig.load(path).to_scenario()
        compile_pipeline(restored)


def test_goel2026_lstm_architecture_and_training_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolved scenario reproduces Goel's LSTM head and known optimizer settings."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NEXUML_RUL_PREPARED_FILE", raising=False)
    scenario = maintenance_rul_goel2026_lstm_protocol()
    source = scenario.data.datasets[0].source
    assert isinstance(source, AutoCMapSSRULWindows)
    assert source.window == 30
    assert source.rul_cap == 130
    assert source.normalization == "zscore"
    assert source.padding_mode == "zero"
    assert source.train_full_windows_only is True
    assert source.normalize_targets is False
    assert len(source.sensor_ids or []) == 14
    assert scenario.training.optimizer.factory == "torch.optim.rmsprop:RMSprop"
    assert scenario.training.optimizer.kwargs["weight_decay"] == pytest.approx(1e-5)
    assert scenario.training.lr == pytest.approx(1e-3)
    assert scenario.training.batch_size == 64
    assert scenario.training.max_epochs == 200
    assert scenario.training.loss_keys == {"rul_loss": 1.0}
    evaluator = scenario.evaluation.algorithms[0].algorithm
    assert isinstance(evaluator, RULBenchmarkEvaluator)
    assert evaluator.include_nasa_score is True
    assert len(scenario.evaluation.algorithms) == 2
    assert scenario.evaluation.algorithms[1].label_key == "rul_raw"
    assert isinstance(scenario.evaluation.algorithms[2].algorithm, RULPerEngineAudit)
    assert any("Goel2026PlateauLR" in item.factory for item in scenario.callbacks)
    assert scenario.data.input_shapes["rul_history"] == [30, 14]

    compiled = compile_pipeline(scenario)
    x = TensorDict({"rul_history": torch.randn(2, 30, 14)}, batch_size=[2])
    y = TensorDict({"rul": torch.tensor([[0.2], [0.4]])}, batch_size=[2])
    output, _ = compiled(x, y)
    assert output["predicted_rul"].shape == (2, 1)
    output["rul_loss"].mean().backward()
    assert any(parameter.grad is not None for parameter in compiled.parameters())
    config = tmp_path / "goel2026.yaml"
    compiled.resolved_config.save(config)
    restored = ResolvedConfig.load(config).to_scenario()
    assert restored.training.optimizer.factory == "torch.optim.rmsprop:RMSprop"
    compile_pipeline(restored)


def test_goel2026_cnn_paper_model_and_independent_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two Goel paper models share data but not model/checkpoint artifacts."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NEXUML_RUL_PREPARED_FILE", raising=False)
    lstm = maintenance_rul_goel2026_lstm_protocol()
    cnn = maintenance_rul_goel2026_cnn_protocol()
    assert lstm.name != cnn.name
    assert lstm.data.datasets[0].source == cnn.data.datasets[0].source
    assert cnn.training.optimizer.factory == "torch.optim.rmsprop:RMSprop"
    assert cnn.training.loss_keys == {"rul_loss": 1.0}
    assert cnn.callbacks[1].kwargs["dirpath"] != lstm.callbacks[1].kwargs["dirpath"]
    compiled = compile_pipeline(cnn)
    x = TensorDict({"rul_history": torch.randn(2, 30, 14)}, batch_size=[2])
    y = TensorDict({"rul": torch.tensor([[30.0], [50.0]])}, batch_size=[2])
    result, _ = compiled(x, y)
    assert result["predicted_rul"].shape == (2, 1)
    result["rul_loss"].mean().backward()
    assert any(param.grad is not None for param in compiled.parameters())


def test_asymmetric_nasa_score_for_paper_protocol() -> None:
    """NexuML evaluates positive errors more harshly than negative errors."""
    evaluator = _RULBenchmarkRuntime(
        prediction_key="predicted_rul",
        label_key="rul",
        target_scale=1.0,
        point_index=0,
        lower_index=None,
        upper_index=None,
        include_nasa_score=True,
    )
    x = TensorDict(
        {"predicted_rul": torch.tensor([[10.0], [20.0]])},
        batch_size=[2],
    )
    y = TensorDict({"rul": torch.tensor([[0.0], [30.0]])}, batch_size=[2])
    evaluator.eval_batch(x, y)
    expected = math.expm1(10.0 / 10.0) + math.expm1(10.0 / 13.0)
    assert evaluator.results()["rul_test_nasa_score"] == pytest.approx(expected)


def test_goel_restores_best_val_weights_before_native_testing(tmp_path: Path) -> None:
    """Paper callback must replace final-epoch weights with the saved best."""
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint

    from nexuml_library.evaluation.maintenance.rul_paper_callbacks import (
        Goel2026RestoreBestWeights,
    )

    model = L.LightningModule()
    model.linear = torch.nn.Linear(2, 1)
    best_weights = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    checkpoint = tmp_path / "best.ckpt"
    torch.save({"state_dict": best_weights}, checkpoint)
    with torch.no_grad():
        model.linear.weight.fill_(3.0)
        model.linear.bias.fill_(-5.0)
    callback_model = ModelCheckpoint(monitor="val/loss", mode="min")
    callback_model.best_model_path = str(checkpoint)
    trainer = L.Trainer(
        callbacks=[callback_model],
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    audit_path = tmp_path / "best-checkpoint.json"
    Goel2026RestoreBestWeights(output_file=str(audit_path)).on_fit_end(trainer, model)
    assert all(
        torch.equal(model.state_dict()[name], original)
        for name, original in best_weights.items()
    )
    metadata = json.loads(audit_path.read_text())
    assert metadata["best_model_path"] == str(checkpoint)
    assert metadata["loaded_strict_model_state"] is True


def test_goel_native_per_engine_audit_exports_provenance(
    tmp_path: Path,
) -> None:
    """Test indices resolve to distinct engines even for shuffled test batches."""
    _cmapss_files(tmp_path)
    np.savetxt(tmp_path / "RUL_FD001.txt", [160.0, 7.0])
    payload = prepare_cmapss(tmp_path, window=3, rul_cap=130, normalize_targets=False)
    artifact = tmp_path / "prepared.pt"
    save_rul_artifact(payload, artifact)
    report = tmp_path / "audit" / "engines.csv"
    runtime = _RULPerEngineAuditRuntime(
        artifact_path=str(artifact),
        output_file=str(report),
        prediction_key="predicted_rul",
        target_scale=1.0,
    )
    x = TensorDict(
        {
            "sample_index": torch.tensor([1, 0]),
            "predicted_rul": torch.tensor([[7.0], [130.0]]),
        },
        batch_size=[2],
    )
    y = TensorDict(
        {
            "rul": torch.tensor([[7.0], [130.0]]),
            "rul_raw": torch.tensor([[7.0], [160.0]]),
        },
        batch_size=[2],
    )
    runtime.eval_batch(x, y)
    runtime.eval_end()
    records = pd.read_csv(report)
    assert records["asset_id"].tolist() == ["test:FD001:1", "test:FD001:2"]
    assert records["error_capped"].tolist() == pytest.approx([0.0, 0.0])
    assert records["error_raw"].tolist() == pytest.approx([-30.0, 0.0])
    assert records["target_was_capped"].tolist() == [1, 0]
    manifest = json.loads(report.with_suffix(".json").read_text())
    assert manifest["test_engines"] == 2
    assert manifest["test_engines_above_cap"] == 1
    assert not set(manifest["split_assets"]["train"]) & set(
        manifest["split_assets"]["val"]
    )
    assert runtime.results()["rul_audit_test_count"] == 2.0
    assert runtime.results()["rul_audit_capped_engine_count"] == 1.0


def test_goel_native_per_engine_audit_rejects_duplicate_observation(
    tmp_path: Path,
) -> None:
    """Engine identity must be unique, not silently deduplicated."""
    _cmapss_files(tmp_path)
    payload = prepare_cmapss(tmp_path, window=3)
    artifact = tmp_path / "prepared.pt"
    save_rul_artifact(payload, artifact)
    runtime = _RULPerEngineAuditRuntime(
        artifact_path=str(artifact),
        output_file=str(tmp_path / "residuals.csv"),
        prediction_key="predicted_rul",
        target_scale=125.0,
    )
    index = next(i for i, row in enumerate(payload["rows"]) if row["split"] == "test")
    record = payload["rows"][index]
    local_index = next(
        i for i, row in enumerate(
            row for row in payload["rows"] if row["split"] == "test"
        ) if row["asset_id"] == record["asset_id"]
    )
    x = TensorDict(
        {
            "sample_index": torch.tensor([local_index]),
            "predicted_rul": payload["targets"][index:index + 1],
        },
        batch_size=[1],
    )
    y = TensorDict(
        {
            "rul": payload["targets"][index:index + 1],
            "rul_raw": payload["raw_targets"][index:index + 1],
        },
        batch_size=[1],
    )
    runtime.eval_batch(x, y)
    with pytest.raises(ValueError, match="Duplicate engine"):
        runtime.eval_batch(x, y)


def test_official_raw_and_capped_nasa_evaluators_disagree() -> None:
    """Both published target conventions remain separately measurable."""
    x = TensorDict({"predicted_rul": torch.tensor([[130.0]])}, batch_size=[1])
    y = TensorDict(
        {"rul": torch.tensor([[130.0]]), "rul_raw": torch.tensor([[160.0]])},
        batch_size=[1],
    )
    capped = _RULBenchmarkRuntime(
        prediction_key="predicted_rul",
        label_key="rul",
        target_scale=1.0,
        point_index=0,
        lower_index=None,
        upper_index=None,
        include_nasa_score=True,
    )
    raw = _RULBenchmarkRuntime(
        prediction_key="predicted_rul",
        label_key="rul_raw",
        target_scale=1.0,
        point_index=0,
        lower_index=None,
        upper_index=None,
        include_nasa_score=True,
    )
    for evaluator in (capped, raw):
        evaluator.eval_batch(x, y)
    assert capped.results()["rul_test_rmse"] == pytest.approx(0.0)
    assert raw.results()["rul_test_rmse"] == pytest.approx(30.0)
    assert capped.results()["rul_test_nasa_score"] == pytest.approx(0.0)
    assert raw.results()["rul_test_nasa_score"] == pytest.approx(math.expm1(30.0 / 13))


def test_goel2026_padded_test_and_cycle_targets(tmp_path: Path) -> None:
    """Short test histories receive zeros; outputs retain cycles, not fractions."""
    rows = []
    for engine in range(1, 9):
        for cycle in range(1, 41):
            rows.append([engine, cycle, *([float(engine + cycle)] * 24)])
    np.savetxt(tmp_path / "train_FD001.txt", np.asarray(rows))
    test_rows = [
        [engine, cycle, *([float(engine + cycle)] * 24)]
        for engine in (1, 2) for cycle in range(1, 11)
    ]
    np.savetxt(tmp_path / "test_FD001.txt", np.asarray(test_rows))
    np.savetxt(tmp_path / "RUL_FD001.txt", [160.0, 7.0])
    payload = prepare_cmapss(
        tmp_path,
        window=30,
        seed=42,
        rul_cap=130,
        sensor_ids=[2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21],
        normalization="zscore",
        padding_mode="zero",
        train_full_windows_only=True,
        normalize_targets=False,
    )
    assert payload["features"].shape[1:] == (30, 14)
    assert payload["settings"]["target_scale"] == pytest.approx(1.0)
    test_indices = [
        i for i, row in enumerate(payload["rows"]) if row["split"] == "test"
    ]
    assert len(test_indices) == 2
    assert all(torch.all(payload["features"][i, :20] == 0) for i in test_indices)
    assert sorted(float(payload["targets"][i, 0]) for i in test_indices) == [7.0, 130.0]
    assert sorted(float(payload["raw_targets"][i, 0]) for i in test_indices) == [
        7.0, 160.0
    ]
    artifact = tmp_path / "goel-cycle-targets.pt"
    save_rul_artifact(payload, artifact)
    dataset = PreparedRULWindows(path=str(artifact)).build().get_split("test")
    reported = sorted(
        (float(dataset[i][1]["rul"][0]), float(dataset[i][1]["rul_raw"][0]))
        for i in range(len(dataset))
    )
    assert reported == [(7.0, 7.0), (130.0, 160.0)]
    for i, row in enumerate(payload["rows"]):
        if row["split"] != "test":
            assert row["step"] >= 29


def test_rul_rejects_window_level_cross_asset_leakage(tmp_path: Path) -> None:
    """Prepared metadata cannot relabel a training asset as an evaluation asset."""
    root = tmp_path / "cmapss"
    _cmapss_files(root)
    payload = prepare_cmapss(root, window=3)
    first = next(i for i, row in enumerate(payload["rows"]) if row["split"] == "train")
    payload["rows"][first]["split"] = "test"
    with pytest.raises(ValueError, match="span multiple splits"):
        validate_rul_payload(payload)
