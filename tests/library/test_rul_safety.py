"""Isolation, gradient, and native-resolver tests for cost-sensitive FD001 RUL."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from nexuml.core.compiler import compile as compile_pipeline
from nexuml.core.config import ResolvedConfig
from nexuml.core.types import ScenarioSpec
from nexuml_library.evaluation.maintenance.rul_prepare import prepare_cmapss
from nexuml_library.evaluation.maintenance.rul_engine_audit import RULPerEngineAudit
from nexuml_library.evaluation.maintenance.rul_safety_diagnostics import (
    summarize_rul_sensor_stages,
)
from nexuml_library.layers.model.rul_safety import (
    RULAsymmetricSquaredLoss,
    _RULAsymmetricRuntime,
)
from nexuml_library.scenarios.maintenance.rul import (
    maintenance_rul_goel2026_cnn_protocol,
    maintenance_rul_goel2026_lstm_protocol,
    maintenance_rul_safety_cnn,
    maintenance_rul_safety_lstm,
)


def _synthetic_fd001(root: Path) -> None:
    """Create engine-disjoint synthetic trajectories without external downloads."""
    rows = []
    for engine in range(1, 9):
        for cycle in range(1, 121):
            values = [
                float(engine) * 0.1 + float(cycle) * float(sensor) * 0.001
                for sensor in range(1, 25)
            ]
            rows.append([engine, cycle, *values])
    np.savetxt(root / "train_FD001.txt", np.asarray(rows))
    test_rows = [
        [engine, cycle, *([float(engine + cycle)] * 24)]
        for engine in (1, 2) for cycle in range(1, 12)
    ]
    np.savetxt(root / "test_FD001.txt", np.asarray(test_rows))
    np.savetxt(root / "RUL_FD001.txt", [10, 140])


def test_asymmetric_rul_loss_weights_optimistic_predictions() -> None:
    """Optimistic error receives twice the MSE gradient at equal magnitude."""
    runtime = _RULAsymmetricRuntime(
        input_sizes={"predicted_rul": (1,)},
        keys_in=["predicted_rul"],
        keys_out=["rul_loss"],
        label_key="rul",
        overestimate_weight=2.0,
    )
    predicted = torch.tensor([[20.0], [10.0]], requires_grad=True)
    x = TensorDict({"predicted_rul": predicted}, batch_size=[2])
    y = TensorDict({"rul": torch.tensor([[10.0], [20.0]])}, batch_size=[2])
    result, _ = runtime(x, y)
    assert isinstance(result, TensorDict)
    loss = cast(torch.Tensor, result["rul_loss"]).mean()
    assert loss.item() == pytest.approx(150.0)
    loss.backward()
    assert predicted.grad is not None
    assert predicted.grad[:, 0].tolist() == pytest.approx([20.0, -10.0])

    with pytest.raises(ValueError):
        RULAsymmetricSquaredLoss(overestimate_weight=0.5)


def test_train_val_sensor_stage_audit_does_not_consult_test_data(
    tmp_path: Path,
) -> None:
    """Neither test trajectory values nor test RUL may affect stage reports."""
    _synthetic_fd001(tmp_path)
    payload = prepare_cmapss(
        tmp_path,
        window=30,
        rul_cap=130.0,
        sensor_ids=[2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21],
        normalize_targets=False,
        train_full_windows_only=True,
    )
    report = summarize_rul_sensor_stages(payload)
    assert report["splits"]["train"]["0-30"]["count"] > 0
    assert report["splits"]["train"]["60-90"]["count"] > 0
    assert report["splits"]["val"]["0-30"]["count"] > 0
    assert len(report["train_midlife_late_sensor_contrast"]) == 14
    assert report["splits"]["train"]["0-30"]["mean"] != (
        report["splits"]["train"]["60-90"]["mean"]
    )
    altered = copy.deepcopy(payload)
    for i, row in enumerate(altered["rows"]):
        if row["split"] == "test":
            altered["features"][i] = 10_000.0
            altered["targets"][i] = 0.0
            altered["raw_targets"][i] = 0.0
    assert summarize_rul_sensor_stages(altered) == report
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    ("scenario_fn", "baseline_fn"),
    [
        (maintenance_rul_safety_lstm, maintenance_rul_goel2026_lstm_protocol),
        (maintenance_rul_safety_cnn, maintenance_rul_goel2026_cnn_protocol),
    ],
)
def test_safety_scenario_isolated_from_paper_reproduction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario_fn: Callable[[], ScenarioSpec],
    baseline_fn: Callable[[], ScenarioSpec],
) -> None:
    """Cost-sensitive runs have independent checkpoints and unchanged Goel data."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NEXUML_RUL_PREPARED_FILE", raising=False)
    safety = scenario_fn()
    paper = baseline_fn()
    assert safety.name != paper.name
    assert safety.data == paper.data
    assert safety.training.optimizer == paper.training.optimizer
    assert safety.pipeline.stages["RULModel"] == paper.pipeline.stages["RULModel"]
    safety_component = safety.pipeline.stages["RULLoss"][0].component
    assert isinstance(safety_component, RULAsymmetricSquaredLoss)
    assert safety_component.overestimate_weight == 2.0
    assert safety.training.loss_keys == {"rul_loss": 1.0}
    assert safety.callbacks[1].kwargs["dirpath"] != paper.callbacks[1].kwargs["dirpath"]
    assert any("RULSensorStageAudit" in item.factory for item in safety.callbacks)
    assert not any("RULSensorStageAudit" in item.factory for item in paper.callbacks)
    safety_audit = safety.evaluation.algorithms[2].algorithm
    paper_audit = paper.evaluation.algorithms[2].algorithm
    assert isinstance(safety_audit, RULPerEngineAudit)
    assert isinstance(paper_audit, RULPerEngineAudit)
    assert safety_audit.output_file != paper_audit.output_file
    compiled = compile_pipeline(safety)
    dims = safety.data.input_shapes["rul_history"]
    batch = TensorDict({"rul_history": torch.randn(2, *dims)}, batch_size=[2])
    labels = TensorDict({"rul": torch.tensor([[30.0], [80.0]])}, batch_size=[2])
    predictions, _ = compiled(batch, labels)
    predictions["rul_loss"].mean().backward()
    assert any(param.grad is not None for param in compiled.parameters())
    yaml_path = tmp_path / "safety.yaml"
    compiled.resolved_config.save(yaml_path)
    restored = ResolvedConfig.load(yaml_path).to_scenario()
    assert any("RULSensorStageAudit" in item.factory for item in restored.callbacks)
    compile_pipeline(restored)
