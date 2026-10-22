"""Native battery-capacity data and loss protocol tests for two 2025 papers."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch
from tensordict import TensorDict

from nexuml.core.compiler import compile as compile_pipeline
from nexuml.core.config import ResolvedConfig
from nexuml.core.types import ScenarioSpec
from nexuml_library.data.maintenance.battery_capacity import BatteryCapacityWindows
from nexuml_library.evaluation.maintenance.battery_capacity import (
    _BatteryCapacityEval,
    _BatterySMAPERuntime,
)
from nexuml_library.evaluation.maintenance.battery_eol_rollout import (
    rollout_capacity_to_eol,
)
from nexuml_library.scenarios.maintenance.battery_capacity import (
    maintenance_battery_patchformer_nasa,
    maintenance_battery_rul_mamba_nasa,
)


def _capacity_csvs(root: Path) -> None:
    """Write complete synthetic battery traces with nonidentical ageing rates."""
    for offset, cell in enumerate(("B0005", "B0006", "B0007", "B0018")):
        cycles = np.arange(1, 67)
        capacity = 2.0 - 0.007 * cycles - 0.0005 * offset * cycles
        pd.DataFrame({"cycle": cycles, "capacity": capacity}).to_csv(
            root / f"{cell}.csv", index=False
        )


def test_battery_capacity_windows_use_real_next_cycle_ah_and_chronological_split(
    tmp_path: Path,
) -> None:
    """No made-up RUL: all labels are observed following-cycle capacity."""
    _capacity_csvs(tmp_path)
    normalizer = tmp_path / "scaler.json"
    definition = BatteryCapacityWindows(
        root=str(tmp_path), lookback=16, normalizer_file=str(normalizer)
    )
    source = definition.build()
    saved = json.loads(normalizer.read_text())
    assert saved["fit_scope"] == "training_cycles_only"
    assert saved["held_out_cell"] == "B0005"
    assert saved["capacity_scale_ah"] > 0
    train = source.get_split("train")
    val = source.get_split("val")
    test = source.get_split("test")
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    assert set(train.meta["asset_id"]) == {"B0006", "B0007", "B0018"}
    assert set(val.meta["asset_id"]) == {"B0006", "B0007", "B0018"}
    assert set(test.meta["asset_id"]) == {"B0005"}
    assert max(train.meta.loc[train.meta["asset_id"] == "B0006", "step"]) < (
        min(val.meta.loc[val.meta["asset_id"] == "B0006", "step"])
    )
    x, labels = test[0]
    assert labels is not None
    assert cast(torch.Tensor, x["capacity_history"]).shape == (16, 1)
    predicted = labels.get("future_capacity")
    assert isinstance(predicted, torch.Tensor)
    assert float(predicted.item()) == pytest.approx(2.0 - 0.007 * 17)


def test_battery_smape_matches_paper_formula_and_has_gradients() -> None:
    """Normalized 2*absolute error / sum absolute capacities is differentiable."""
    runtime = _BatterySMAPERuntime(
        input_sizes={"predicted_capacity": (1,)},
        keys_in=["predicted_capacity"],
        keys_out=["capacity_loss"],
        label_key="future_capacity",
        epsilon=1e-6,
    )
    predicted = torch.tensor([[1.0], [1.5]], requires_grad=True)
    x = TensorDict({"predicted_capacity": predicted}, batch_size=[2])
    y = TensorDict(
        {"future_capacity": torch.tensor([[1.5], [1.5]])},
        batch_size=[2],
    )
    result, _ = runtime(x, y)
    assert isinstance(result, TensorDict)
    loss = cast(torch.Tensor, result["capacity_loss"]).mean()
    assert float(loss.item()) == pytest.approx(0.2)
    loss.backward()
    assert predicted.grad is not None

    evaluation = _BatteryCapacityEval(
        prediction_key="predicted_capacity", label_key="future_capacity"
    )
    evaluation.eval_batch(x, y)
    metrics = evaluation.results()
    assert metrics["capacity_test_mae_ah"] == pytest.approx(0.25)
    assert metrics["capacity_test_rmse_ah"] == pytest.approx((0.25 / 2) ** 0.5)
    assert metrics["capacity_test_smape"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("scenario_fn", "window", "arch"),
    [
        (maintenance_battery_patchformer_nasa, 30, "patchformer"),
        (maintenance_battery_rul_mamba_nasa, 16, "rul_mamba"),
    ],
)
def test_battery_paper_models_compile_with_native_capacity_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario_fn: Callable[[], ScenarioSpec],
    window: int,
    arch: str,
) -> None:
    """Paper-derived model compiles with real capacity targets and SMAPE."""
    _capacity_csvs(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEXUML_BATTERY_CAPACITY_ROOT", str(tmp_path))
    scenario = scenario_fn()
    assert scenario.data.input_shapes == {"capacity_history": [window, 1]}
    source = scenario.data.datasets[0].source
    assert isinstance(source, BatteryCapacityWindows)
    assert source.end_of_life_ah == pytest.approx(1.4)
    assert source.test_cell == "B0005"
    assert scenario.training.loss_keys == {"capacity_loss": 1.0}
    assert scenario.evaluation.algorithms[0].label_key == "future_capacity"
    compiled = compile_pipeline(scenario)
    examples = TensorDict(
        {"capacity_history": torch.randn(2, window, 1)}, batch_size=[2]
    )
    labels = TensorDict(
        {"future_capacity": torch.tensor([[1.4], [1.6]])}, batch_size=[2]
    )
    result, _ = compiled(examples, labels)
    prediction = cast(torch.Tensor, result["predicted_capacity"])
    assert prediction.shape == (2, 1)
    cast(torch.Tensor, result["capacity_loss"]).mean().backward()
    assert any(parameter.grad is not None for parameter in compiled.parameters())
    yaml_path = tmp_path / f"{arch}.yaml"
    compiled.resolved_config.save(yaml_path)
    restored = ResolvedConfig.load(yaml_path).to_scenario()
    assert restored.name == scenario.name
    compile_pipeline(restored)


def test_autoregressive_battery_eol_rollout_and_censoring() -> None:
    """Predict EOL only from the fitted forecaster, never the true future."""
    history = torch.tensor([[[0.7], [0.6]], [[0.9], [0.8]]])

    def decay(x: torch.Tensor) -> torch.Tensor:
        """Return a decreasing physical Ah estimate for each sequence."""
        return x[:, -1] - 0.1

    result = rollout_capacity_to_eol(
        decay,
        history,
        capacity_offset_ah=0.0,
        capacity_scale_ah=1.0,
        end_of_life_ah=0.35,
        max_cycles=4,
    )
    assert result.tolist() == [3, -1]
    with pytest.raises(ValueError):
        rollout_capacity_to_eol(
            decay,
            history,
            capacity_offset_ah=0.0,
            capacity_scale_ah=0.0,
            end_of_life_ah=0.35,
            max_cycles=4,
        )
