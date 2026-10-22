"""Leakage-safe real two-modality DMM-JA data and native scenario tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch
from tensordict import TensorDict

from nexuml.core.compiler import compile as compile_pipeline
from nexuml.core.config import ResolvedConfig
from nexuml_library.data.maintenance.bimodal_rul import LabeledBimodalRULWindows
from nexuml_library.scenarios.maintenance.rul_dmm_ja_bimodal import (
    maintenance_rul_dmm_ja_bimodal,
)


def _write_bimodal_csv(path: Path, test_offset: float = 0.0) -> None:
    """Write explicitly observed RUL with three nonoverlapping assets."""
    rows = []
    for asset, split, offset in (
        ("train-1", "train", 0.0),
        ("validate-1", "val", 0.5),
        ("heldout-1", "test", test_offset),
    ):
        for cycle in range(1, 36):
            rows.append(
                {
                    "asset_id": asset,
                    "cycle": cycle,
                    "split": split,
                    "temperature": 30.0 + cycle * 0.2 + offset,
                    "load": 20.0 + cycle * 0.3 + offset,
                    "rul": float(35 - cycle),
                }
            )
    pd.DataFrame.from_records(rows).to_csv(path, index=False)


def test_bimodal_source_uses_only_train_modalities_for_scaling(
    tmp_path: Path,
) -> None:
    """Changing held-out measurements cannot alter training observations."""
    path = tmp_path / "bimodal.csv"
    normalizer = tmp_path / "normalizer.json"
    _write_bimodal_csv(path)
    definition = LabeledBimodalRULWindows(
        path=str(path), window=24, normalizer_file=str(normalizer)
    )
    original = definition.build()
    train = original.get_split("train")
    test = original.get_split("test")
    assert len(train) == 35 and len(test) == 35
    original_x, _ = train[20]
    train_values = cast(torch.Tensor, original_x["rul_history"]).clone()
    original_stats = json.loads(normalizer.read_text())
    assert original_stats["fit_scope"] == "training_assets_only"
    assert set(original_stats["split_assets"]["test"]) == {"heldout-1"}

    _write_bimodal_csv(path, test_offset=5000.0)
    modified = definition.build()
    modified_x, _ = modified.get_split("train")[20]
    assert torch.equal(
        train_values, cast(torch.Tensor, modified_x["rul_history"])
    )
    assert json.loads(normalizer.read_text()) == original_stats
    _, labels = modified.get_split("test")[3]
    assert labels is not None
    target = labels.get("rul")
    assert isinstance(target, torch.Tensor)
    assert target.item() == pytest.approx(31.0)


def test_bimodal_source_rejects_missing_real_labels(tmp_path: Path) -> None:
    """No ETDataset or industrial time-series RUL labels are manufactured."""
    path = tmp_path / "bimodal.csv"
    _write_bimodal_csv(path)
    frame = pd.read_csv(path).drop(columns=["rul"])
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing bimodal RUL"):
        LabeledBimodalRULWindows(path=str(path)).build()


def test_native_dmm_ja_bimodal_scenario_compiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Native DMM-JA accepts true two-feature windows and supervised RUL."""
    path = tmp_path / "bimodal.csv"
    _write_bimodal_csv(path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEXUML_RUL_BIMODAL_CSV", str(path))
    scenario = maintenance_rul_dmm_ja_bimodal()
    assert scenario.data.input_shapes["rul_history"] == [24, 2]
    source = scenario.data.datasets[0].source
    assert isinstance(source, LabeledBimodalRULWindows)
    assert source.path == str(path)
    assert scenario.training.loss_keys == {"rul_loss": 1.0}
    compiled = compile_pipeline(scenario)
    x = TensorDict({"rul_history": torch.randn(2, 24, 2)}, batch_size=[2])
    y = TensorDict({"rul": torch.tensor([[10.0], [20.0]])}, batch_size=[2])
    result, _ = compiled(x, y)
    predicted = cast(torch.Tensor, result["predicted_rul"])
    assert predicted.shape == (2, 1)
    cast(torch.Tensor, result["rul_loss"]).mean().backward()
    assert any(param.grad is not None for param in compiled.parameters())
    config = tmp_path / "dmm-ja-bimodal.yaml"
    compiled.resolved_config.save(config)
    assert ResolvedConfig.load(config).to_scenario().name == scenario.name
