"""Offline contract tests for the public FISHER waveform encoder."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from nexuml import nn_module
from nexuml.core.components import LayerBuildContext
from nexuml.core.config import ResolvedConfig
from nexuml.core.types import LayerSpec, PipelineSpec, ScenarioSpec
from nexuml_library.layers.model.fisher import FISHEREncoder


class _FakeFISHER(nn.Module):
    """Small deterministic stand-in for the external checkpoint."""

    def __init__(self, band_width: int = 50) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(band_width=band_width)
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.last_spectrogram: torch.Tensor | None = None

    def extract_features(self, spec: torch.Tensor) -> torch.Tensor:
        self.last_spectrogram = spec.detach().clone()
        signal = spec.mean(dim=(2, 3))
        return torch.cat((signal, signal * self.weight), dim=1)

    def forward(
        self,
        spec: torch.Tensor,
        *,
        mode: str,
        mask: bool,
        features_only: bool,
    ) -> dict[str, torch.Tensor]:
        assert mode == "IMAGE"
        assert mask is False
        assert features_only is True
        self.last_spectrogram = spec.detach().clone()
        signal = spec.mean(dim=(2, 3)).reshape(-1, 1, 1)
        tokens = torch.cat(
            (
                (signal * self.weight).expand(-1, 3, 2),
                torch.ones((len(signal), 3, 2), device=signal.device),
            ),
            dim=-1,
        )
        return {"x": tokens}


@pytest.fixture
def fake_huggingface(monkeypatch: pytest.MonkeyPatch) -> tuple[list[dict[str, Any]], list[_FakeFISHER]]:
    """Replace Transformers loading with an offline fake checkpoint.

    Returns:
        Captured loading arguments and the created mock backbones.
    """
    options: list[dict[str, Any]] = []
    backbones: list[_FakeFISHER] = []

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs: Any) -> _FakeFISHER:
            options.append({"name": model_name, **kwargs})
            model = _FakeFISHER()
            backbones.append(model)
            return model

    stub = ModuleType("transformers")
    stub.AutoModel = FakeAutoModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", stub)
    return options, backbones


def test_encoder_is_frozen_and_uses_official_input_preprocessing(
    fake_huggingface: tuple[list[dict[str, Any]], list[_FakeFISHER]],
) -> None:
    options, backbones = fake_huggingface
    encoder = FISHEREncoder(
        model_name="jiangab/FISHER-small-0723",
        sample_rate=16000,
        revision="tested-commit",
        local_files_only=True,
    )
    assert options[0] == {
        "name": "jiangab/FISHER-small-0723",
        "trust_remote_code": True,
        "revision": "tested-commit",
        "local_files_only": True,
    }
    encoder.train()
    assert not backbones[0].training
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    waveform = torch.randn(2, 1600)
    result = encoder(waveform)
    assert result.shape == (2, 2)
    assert not result.requires_grad
    spec = backbones[0].last_spectrogram
    assert spec is not None
    assert spec.shape == (2, 1, 8, 201)

    window = torch.hann_window(400)
    raw = waveform - waveform.mean(dim=1, keepdim=True)
    reference = torch.stft(
        raw, n_fft=400, hop_length=160, window=window, center=False, return_complex=True
    )
    reference_spec = torch.log(reference.abs() + 1e-10).transpose(1, 2)
    reference_spec = (reference_spec + 3.017344307886898) / (2.1531635155379805 * 2)
    assert torch.allclose(spec[:, 0], reference_spec, atol=1e-6)


@pytest.mark.parametrize("pooling,expected_dim", [("mean", 4), ("mean_std", 8)])
def test_encoder_pooled_tokens_support_unfrozen_gradients(
    fake_huggingface: tuple[list[dict[str, Any]], list[_FakeFISHER]],
    pooling: str,
    expected_dim: int,
) -> None:
    _, backbones = fake_huggingface
    encoder = FISHEREncoder(pooling=cast(Any, pooling), frozen=False)
    result = encoder(torch.randn(2, 2000))
    assert result.shape == (2, expected_dim)
    result.square().mean().backward()
    assert backbones[0].weight.grad is not None
    assert torch.isfinite(backbones[0].weight.grad)


def test_encoder_handles_multichannel_input_and_short_sample_rejection(
    fake_huggingface: tuple[list[dict[str, Any]], list[_FakeFISHER]],
) -> None:
    _ = fake_huggingface
    encoder = FISHEREncoder(sample_rate=16000)
    waveform = torch.randn(2, 2, 1600)
    assert torch.equal(encoder(waveform), encoder(waveform.mean(dim=1)))
    with pytest.raises(ValueError, match="requires at least"):
        encoder(torch.zeros(2, 399))
    with pytest.raises(ValueError, match="expects"):
        encoder(torch.zeros(1600))
    with pytest.raises(TypeError, match="floating-point"):
        encoder(torch.ones((2, 1600), dtype=torch.long))
    with pytest.raises(ValueError, match="finite"):
        encoder(torch.full((2, 1600), float("nan")))


def test_nexuml_nn_module_tensordict_and_yaml_roundtrip(
    fake_huggingface: tuple[list[dict[str, Any]], list[_FakeFISHER]],
) -> None:
    _ = fake_huggingface
    definition = nn_module(
        FISHEREncoder, sample_rate=16000, pooling="mean_std", frozen=True
    )
    runtime = definition.build(
        LayerBuildContext(
            input_sizes={"waveform": (1600,)},
            keys_in=["waveform"],
            keys_out=["embedding"],
        )
    )
    waveform = torch.randn(2, 1600)
    out, _ = runtime(TensorDict({"waveform": waveform}, batch_size=[2]))
    assert out["embedding"].shape == (2, 8)
    scenario = ScenarioSpec(
        name="fisher-roundtrip",
        pipeline=PipelineSpec(
            stages={
                "Pretrained": [
                    LayerSpec(
                        component=definition,
                        keys_in=["waveform"],
                        keys_out=["embedding"],
                    )
                ]
            }
        ),
    )
    yaml_text = ResolvedConfig.from_scenario(scenario).to_yaml()
    restored = ResolvedConfig.from_yaml(yaml_text)
    assert restored.pipeline.stages["Pretrained"][0].component == definition
    assert "FISHEREncoder" in yaml_text
    assert "NnModule" in yaml_text
