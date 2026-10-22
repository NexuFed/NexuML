"""Regression tests for PatchFormer-Flow RUL geometry and native calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import pytest
import torch
from tensordict import TensorDict

from nexuml.core.compiler import compile as compile_pipeline
from nexuml.core.config import ResolvedConfig
from nexuml_library.data.maintenance.rul_flow_calibration import (
    AutoPatchFormerFlowRULFeatures,
)
from nexuml_library.evaluation.maintenance.rul_engine_audit import RULPerEngineAudit
from nexuml_library.evaluation.maintenance.rul_flow_geometry import (
    pearson_flow_progress,
    project_flow_time,
    resolve_latent_flow_checkpoint,
)
from nexuml_library.layers.model.rul_flow_calibration import PatchFormerFlowRULHead
from nexuml_library.scenarios.maintenance.rul_flow_calibration import (
    patchformer_flow_rul_calibration,
)


def test_flow_progress_projection_uses_nearest_training_trajectory() -> None:
    """Exact path points recover their discretized transport-time positions."""
    trajectories = torch.zeros(2, 6, 2)
    trajectories[0, :, 0] = torch.linspace(0, 1, 6)
    trajectories[1, :, 0] = torch.linspace(0, 1, 6)
    trajectories[1, :, 1] = 10
    samples = torch.tensor([[0.0, 0.0], [0.6, 0.0], [1.0, 0.0]])
    coordinates = project_flow_time(samples, trajectories)
    assert coordinates.shape == (3, 2)
    assert torch.allclose(
        coordinates[:, 0], torch.tensor([0.0, 0.6, 1.0])
    )
    assert torch.equal(coordinates[:, 1], torch.zeros(3))
    assert pearson_flow_progress(
        coordinates[:, 0], torch.tensor([100.0, 60.0, 20.0])
    ) == pytest.approx(-1.0)


def test_no_false_flow_clock_when_correlation_is_undefined() -> None:
    """Constant inferred time yields missing descriptive correlation."""
    assert pearson_flow_progress(
        torch.ones(4), torch.arange(4.0)
    ) is None
    with pytest.raises(ValueError, match="Invalid flow projection"):
        project_flow_time(torch.randn(3, 3), torch.randn(4, 5, 2))


@pytest.mark.parametrize("mode", ["flow_time", "hybrid"])
def test_native_flow_rul_heads_learn_without_updating_pretrained_weights(
    mode: Literal["flow_time", "hybrid"],
) -> None:
    """Calibrator gradients flow through its own MLP only."""
    from nexuml.core.components import LayerBuildContext

    definition = PatchFormerFlowRULHead(mode=mode, initialization_seed=42)
    runtime = definition.build(
        LayerBuildContext(
            input_sizes={"flow_rul_features": (3,)},
            keys_in=["flow_rul_features"],
            keys_out=["predicted_rul"],
        )
    )
    inputs = torch.tensor(
        [[0.2, 0.3, 80.0], [0.8, 0.2, 20.0]],
        dtype=torch.float32,
    )
    predictions = runtime.forward_tensor(inputs)
    assert predictions.shape == (2, 1)
    assert bool(torch.isfinite(predictions).all())
    predictions.square().mean().backward()
    assert any(parameter.grad is not None for parameter in runtime.parameters())
    if mode == "flow_time":
        assert bool(((predictions >= 0) & (predictions <= 130)).all())
    else:
        assert bool(((predictions - inputs[:, 2:3]).abs() <= 30).all())


@pytest.mark.parametrize("subset", ["FD001", "FD003"])
@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("mode", ["flow_time", "hybrid"])
def test_calibration_registered_with_native_audited_split_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    subset: Literal["FD001", "FD003"],
    seed: int, mode: Literal["flow_time", "hybrid"],
) -> None:
    """All calibrated modes preserve per-engine baseline and label protocols."""
    monkeypatch.chdir(tmp_path)
    scenario = patchformer_flow_rul_calibration(subset, seed, mode)
    source = scenario.data.datasets[0].source
    assert isinstance(source, AutoPatchFormerFlowRULFeatures)
    assert source.subset == subset
    assert source.seed == seed
    assert scenario.data.input_shapes == {"flow_rul_features": [3]}
    assert source.prepared_rul_path.endswith(
        f"cmapss-{subset}-seed{seed}-controlled-w30-s{seed}.pt"
    )
    assert len(scenario.evaluation.algorithms) == 3
    assert scenario.evaluation.algorithms[0].label_key == "rul"
    assert scenario.evaluation.algorithms[1].label_key == "rul_raw"
    assert scenario.evaluation.algorithms[2].name == "engine_audit"
    audit = scenario.evaluation.algorithms[2].algorithm
    assert isinstance(audit, RULPerEngineAudit)
    assert audit.output_file.endswith(
        f"rul-{subset}-patchformer-flow-{mode.replace('_', '-')}-seed{seed}.csv"
    )
    head = scenario.pipeline.stages["RULModel"][0].component
    assert isinstance(head, PatchFormerFlowRULHead)
    assert head.mode == mode
    assert scenario.callbacks[1].kwargs["monitor"] == "val/loss"
    compiled = compile_pipeline(scenario)
    features = TensorDict(
        {"flow_rul_features": torch.randn(2, 3)}, batch_size=[2]
    )
    targets = TensorDict(
        {"rul": torch.tensor([[30.0], [80.0]])}, batch_size=[2]
    )
    output, _ = compiled(features, targets)
    assert isinstance(output, TensorDict)
    assert cast(torch.Tensor, output["predicted_rul"]).shape == (2, 1)
    cast(torch.Tensor, output["rul_loss"]).mean().backward()
    assert any(p.grad is not None for p in compiled.parameters())
    file = tmp_path / f"{mode}-{subset}-{seed}.yaml"
    compiled.resolved_config.save(file)
    assert ResolvedConfig.load(file).to_scenario().name == scenario.name


def test_frozen_flow_prerequisite_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An untrained flow cannot be silently replaced with a random model."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="patchformer-flow-fd001-s42"):
        resolve_latent_flow_checkpoint("FD001", 42)
