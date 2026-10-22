"""Architecture and native scenario contracts for three recent RUL models."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
import torch
from tensordict import TensorDict

from nexuml.core.compiler import compile as compile_pipeline
from nexuml.core.config import ResolvedConfig
from nexuml.core.types import ScenarioSpec
from nexuml_library.data.maintenance.rul import AutoCMapSSRULWindows
from nexuml_library.evaluation.maintenance.rul_engine_audit import RULPerEngineAudit
from nexuml_library.layers.model.rul_advanced import (
    AdvancedRULForecaster,
    _DualPatchAttention,
    _JumpAwareFusion,
    _RULMambaModel,
)
from nexuml_library.layers.model.rul_selective_ssm import SelectiveSSMBlock
from nexuml_library.scenarios.maintenance.rul_advanced import (
    maintenance_rul_dmm_ja_adapted,
    maintenance_rul_mamba_adapted,
    maintenance_rul_patchformer_adapted,
)


@pytest.mark.parametrize("length", [2, 16, 30])
def test_selective_ssm_is_causal_and_differentiable(length: int) -> None:
    """Future changes cannot influence earlier causal SSM outputs."""
    torch.manual_seed(42)
    block = SelectiveSSMBlock(8, state_dim=4, conv_kernel=3)
    x = torch.randn(2, length, 8, requires_grad=True)
    out = block(x)
    assert out.shape == x.shape
    out.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    prefix = length // 2
    altered = x.detach().clone()
    altered[:, prefix:] += 25
    with torch.no_grad():
        baseline = block(x.detach())
        changed = block(altered)
    assert torch.allclose(baseline[:, :prefix], changed[:, :prefix], atol=1e-5)


@pytest.mark.parametrize("length,patch", [(30, 5), (29, 5), (16, 3)])
def test_patchformer_dual_patch_preserves_sequence(length: int, patch: int) -> None:
    """Intra/inter attention and replication padding preserve time alignment."""
    block = _DualPatchAttention(length, 8, patch)
    x = torch.randn(3, length, 8, requires_grad=True)
    result = block(x)
    assert result.shape == (3, length, 16)
    result.sum().backward()
    assert x.grad is not None


def test_jump_aware_bimodal_fusion_and_mamba_decoder_gradients() -> None:
    """Bidirectional jump-aware fusion and FAN-GRN decoder backpropagate."""
    x = torch.randn(2, 16, 14)
    dmm = _JumpAwareFusion(16, 14, 16, 0.1, 2.5)
    dmm(x).mean().backward()
    assert all(parameter.grad is not None for parameter in dmm.parameters())

    mamba = _RULMambaModel(14, 16, 2, 0.1)
    mamba(x).mean().backward()
    assert all(parameter.grad is not None for parameter in mamba.parameters())


@pytest.mark.parametrize(
    ("scenario_fn", "architecture", "window", "width"),
    [
        (maintenance_rul_dmm_ja_adapted, "dmm_ja", 24, 64),
        (maintenance_rul_patchformer_adapted, "patchformer", 30, 64),
        (maintenance_rul_mamba_adapted, "rul_mamba", 16, 48),
    ],
)
def test_advanced_paper_architectures_compile_in_native_nexuml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario_fn: Callable[[], ScenarioSpec],
    architecture: str,
    window: int,
    width: int,
) -> None:
    """All architectures compile with FD001 provenance and independent audits."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NEXUML_RUL_PREPARED_FILE", raising=False)
    selected = scenario_fn()
    assert selected.training.loss_keys == {"rul_loss": 1.0}
    assert selected.data.input_shapes["rul_history"] == [window, 14]
    source = selected.data.datasets[0].source
    assert isinstance(source, AutoCMapSSRULWindows)
    assert source.download is True
    assert source.rul_cap == 130
    assert source.train_full_windows_only is True
    profile = f"advanced-{architecture.replace('_', '-')}-fd001"
    assert source.artifact_path.endswith(f"-{profile}.pt")
    model = selected.pipeline.stages["RULModel"][0].component
    assert isinstance(model, AdvancedRULForecaster)
    assert model.architecture == architecture
    assert model.hidden_dim == width
    assert selected.evaluation.algorithms[0].label_key == "rul"
    assert selected.evaluation.algorithms[1].label_key == "rul_raw"
    assert isinstance(selected.evaluation.algorithms[2].algorithm, RULPerEngineAudit)
    compiled = compile_pipeline(selected)
    x = TensorDict(
        {"rul_history": torch.randn(2, window, 14)}, batch_size=[2]
    )
    y = TensorDict({"rul": torch.tensor([[20.0], [70.0]])}, batch_size=[2])
    result, _ = compiled(x, y)
    prediction = cast(torch.Tensor, result["predicted_rul"])
    assert prediction.shape == (2, 1)
    assert torch.isfinite(prediction).all()
    loss = cast(torch.Tensor, result["rul_loss"]).mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in compiled.parameters())
    config = tmp_path / f"{architecture}.yaml"
    compiled.resolved_config.save(config)
    restored = ResolvedConfig.load(config).to_scenario()
    assert restored.name == selected.name
    compile_pipeline(restored)


def test_advanced_rul_rejects_invalid_dimensions() -> None:
    """Typed configuration rejects invalid model choices and patch FAN heads."""
    with pytest.raises(ValueError):
        AdvancedRULForecaster(architecture="not_a_model")
    with pytest.raises(ValueError):
        SelectiveSSMBlock(0)
