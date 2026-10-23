"""Tests of full-DPAN token RUL prediction with and without frozen sequence flow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from nexuml.core.compiler import compile as compile_pipeline
from nexuml.core.components import LayerBuildContext
from nexuml.core.config import ResolvedConfig
from nexuml_library.data.maintenance.prepared_endpoints import read_prepared_artifact
from nexuml_library.data.maintenance.rul_sequence_flow import (
    AutoPatchFormerSequenceFlowRULFeatures,
)
from nexuml_library.evaluation.maintenance.rul_sequence_flow_checkpoint import (
    load_frozen_sequence_flow,
    resolve_sequence_flow_checkpoint,
    sequence_flow_run,
)
from nexuml_library.layers.model.flow import SequenceFlow, _FlowRuntime
from nexuml_library.layers.model.rul_sequence_flow import (
    PatchFormerSequenceFlowRULHead,
)
from nexuml_library.scenarios.maintenance.rul_sequence_flow import (
    patchformer_sequence_flow_rul,
)


@pytest.mark.parametrize("architecture", ["transformer", "conformer", "cfc"])
@pytest.mark.parametrize("subset", ["FD001", "FD003"])
@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("use_flow", [True, False])
def test_matched_sequence_rul_scenario_resolves_and_compiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    architecture: Literal["transformer", "conformer", "cfc"],
    subset: Literal["FD001", "FD003"],
    seed: int,
    use_flow: bool,
) -> None:
    """Every scenario uses a shared trained readout and official engine audits."""
    monkeypatch.chdir(tmp_path)
    spec = patchformer_sequence_flow_rul(subset, seed, architecture, use_flow)
    source = spec.data.datasets[0].source
    assert isinstance(source, AutoPatchFormerSequenceFlowRULFeatures)
    assert source.architecture == architecture
    assert source.use_flow is use_flow
    assert source.subset == subset and source.seed == seed
    assert spec.data.input_shapes == {"sequence_rul_features": [30, 256]}
    assert spec.training.loss_keys == {"rul_loss": 1.0}
    assert spec.pipeline.stages["RULModel"][0].keys_in == ["sequence_rul_features"]
    assert spec.pipeline.stages["RULModel"][0].component == (
        PatchFormerSequenceFlowRULHead(
            hidden_dim=128,
            attention_heads=4,
            dropout=0.1,
            initialization_seed=seed,
        )
    )
    assert [algorithm.label_key for algorithm in spec.evaluation.algorithms] == [
        "rul", "rul_raw", "rul"
    ]
    assert spec.evaluation.algorithms[2].name == "engine_audit"
    assert spec.callbacks[1].kwargs["monitor"] == "val/loss"
    compiled = compile_pipeline(spec)
    assert compiled.resolved_config.to_scenario().name == spec.name


def test_temporal_readout_consumes_full_sequence_and_preserves_gradients() -> None:
    """Both RUL arms retain every token with learnable attention weighting."""
    head = PatchFormerSequenceFlowRULHead(
        hidden_dim=32, attention_heads=4, dropout=0.0
    )
    runtime = head.build(
        LayerBuildContext(
            input_sizes={"sequence_rul_features": (30, 256)},
            keys_in=["sequence_rul_features"],
            keys_out=["predicted_rul"],
        )
    )
    x = torch.randn(3, 30, 256, requires_grad=True)
    predictions = runtime.forward_tensor(x)
    assert predictions.shape == (3, 1)
    assert bool(((predictions >= 0) & (predictions <= 130)).all())
    predictions.square().mean().backward()
    assert x.grad is not None
    assert torch.count_nonzero(x.grad[:, :, :128]).item() > 0
    assert torch.count_nonzero(x.grad[:, :, 128:]).item() > 0
    assert len(runtime.token_scores.weight.grad.shape) == 2
    with pytest.raises(ValueError, match="30,256"):
        runtime.forward_tensor(torch.randn(3, 256))


@pytest.mark.parametrize("architecture", ["transformer", "conformer", "cfc"])
def test_sequence_checkpoint_is_strict_and_frozen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    architecture: Literal["transformer", "conformer", "cfc"],
) -> None:
    """Restore exact 30-token state and reject missing or modified parameters."""
    monkeypatch.chdir(tmp_path)
    run = sequence_flow_run("FD001", 42, architecture)
    directory = tmp_path / ".experiments/rul/checkpoints" / run
    directory.mkdir(parents=True)
    runtime = SequenceFlow(
        architecture=architecture,
        model_dim=128, depth=2, attention_heads=4,
        conv_kernel=7, dropout=0.0, num_steps=50,
        initialization_seed=42,
    ).build(
        LayerBuildContext(
            input_sizes={"flow_state": (30, 128), "flow_time": (1,)},
            keys_in=["flow_state", "flow_time"],
            keys_out=["predicted_velocity"],
        )
    )
    assert isinstance(runtime, _FlowRuntime)
    state = {
        f"pipeline.Flow.0.{name}": value
        for name, value in runtime.state_dict().items()
    }
    selected = directory / "epoch=4-step=100.ckpt"
    torch.save({"state_dict": state}, selected)
    torch.save({"state_dict": {}}, directory / "last.ckpt")
    assert resolve_sequence_flow_checkpoint("FD001", 42, architecture) == selected
    restored, digest = load_frozen_sequence_flow(selected, architecture, 42)
    assert len(digest) == 64
    assert not any(param.requires_grad for param in restored.parameters())
    assert restored.vector_field.training is False
    for key, value in restored.state_dict().items():
        assert torch.equal(value, runtime.state_dict()[key])
    second = directory / "epoch=5-step=110.ckpt"
    torch.save({"state_dict": state}, second)
    with pytest.raises(ValueError, match="Multiple best candidates"):
        resolve_sequence_flow_checkpoint("FD001", 42, architecture)
    second.unlink()
    state.pop(next(iter(state)))
    torch.save({"state_dict": state}, selected)
    with pytest.raises(ValueError, match="Missing, ambiguous"):
        load_frozen_sequence_flow(selected, architecture, 42)


def _write_cmapss(path: Path) -> None:
    """Provide a small legal NASA FD001 trio for frozen-source tests."""
    train = [
        [engine, cycle, *([float(engine + cycle) / 100] * 24)]
        for engine in range(1, 9) for cycle in range(1, 65)
    ]
    test = [
        [engine, cycle, *([float(engine + cycle) / 100] * 24)]
        for engine in (1, 2) for cycle in range(1, 32)
    ]
    np.savetxt(path / "train_FD001.txt", np.asarray(train))
    np.savetxt(path / "test_FD001.txt", np.asarray(test))
    np.savetxt(path / "RUL_FD001.txt", [12, 40])


class _TestEncoder(nn.Module):
    """Frozen DPAN stand-in whose former RUL head is forbidden."""

    def forward_features(self, x: Tensor) -> Tensor:
        """Produce full DPAN-shaped outputs from all causal sensor positions.

        Returns:
            Feature tensor with shape [B,30,128].
        """
        return x[:, :, :1].expand(-1, -1, 128)

    def forward(self, x: Tensor) -> Tensor:
        """Forbid use of the original PatchFormer prediction.

        Raises:
            AssertionError: When a caller attempts the forbidden RUL head.
        """
        raise AssertionError("PatchFormer original RUL head must not be called")


class _TestFlow(nn.Module):
    """Frozen integration stub with known nonzero per-token displacement."""

    def sample(self, x: Tensor) -> Tensor:
        """Generate a deterministic change in all sequence tokens.

        Returns:
            Transported embeddings of identical shape.
        """
        return x + 0.25


@pytest.mark.parametrize("use_flow", [True, False])
def test_paired_sequence_sources_never_call_old_rul_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_flow: bool,
) -> None:
    """Both sources retain identical DPAN channels and train-only normalization."""
    import nexuml_library.data.maintenance.rul_sequence_flow as source_module

    monkeypatch.chdir(tmp_path)
    _write_cmapss(tmp_path)
    checkpoint = tmp_path / "rul-FD001-controlled-patchformer-seed42" / "a.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"mock-trusted-checkpoint")
    endpoint = tmp_path / "matching-endpoints.pt"
    prepared = tmp_path / "rul-data.pt"
    encoder = _TestEncoder()
    monkeypatch.setattr(source_module, "_PatchFormerModel", lambda *_a: encoder)
    monkeypatch.setattr(
        source_module, "resolve_patchformer_checkpoint",
        lambda *_a, **_kw: checkpoint
    )
    monkeypatch.setattr(
        source_module, "_trusted_patchformer_weights", lambda *_a: "same"
    )
    settings = {
        "subset": "FD001",
        "seed": 42,
        "feature_layout": "sequence",
        "sequence_length": 30,
        "embedding_dim": 128,
        "feature_normalization": "train_endpoints_only",
        "held_out_test_engine_labels_used": False,
        "source_rul_artifact": str(prepared),
        "frozen_patchformer_checkpoint_path": str(checkpoint),
        "frozen_patchformer_checkpoint_sha256": "same",
        "feature_mean": [[0.0] * 128 for _ in range(30)],
        "feature_std": [[1.0] * 128 for _ in range(30)],
    }
    torch.save(
        {
            "format": "nex233-features-v1",
            "features": torch.zeros(2, 2, 30, 128),
            "rows": [
                {"split": "train", "group_id": "reference"},
                {"split": "val", "group_id": "reference-val"},
            ],
            "settings": settings,
        },
        endpoint,
    )
    assert read_prepared_artifact(endpoint)["features"].shape == (2, 2, 30, 128)
    if use_flow:
        monkeypatch.setattr(
            source_module,
            "resolve_sequence_flow_checkpoint",
            lambda *_a: checkpoint,
        )
        monkeypatch.setattr(
            source_module,
            "load_frozen_sequence_flow",
            lambda *_a: (_TestFlow(), "flow-checksum"),
        )
    else:
        def forbid_flow(*args: object) -> None:
            """Prohibit loading any Flow checkpoint in the encoder-only control.

            Raises:
                AssertionError: If the encoder-only branch loads a Flow.
            """
            raise AssertionError("encoder-only must not load frozen Flow")
        monkeypatch.setattr(
            source_module, "resolve_sequence_flow_checkpoint", forbid_flow
        )
    source = AutoPatchFormerSequenceFlowRULFeatures(
        subset="FD001",
        seed=42,
        root=str(tmp_path),
        prepared_rul_path=str(prepared),
        endpoint_artifact=str(endpoint),
        diagnostics_file=str(tmp_path / "geometry.json"),
        use_flow=use_flow,
        encoding_device="cpu",
        encoding_batch_size=32,
    )
    dataset = source.build()
    features = cast(Tensor, dataset.data)
    assert features.shape[-2:] == (30, 256)
    assert bool(torch.isfinite(features).all())
    if use_flow:
        assert torch.allclose(features[:, :, 128:], torch.full_like(
            features[:, :, 128:], 0.25
        ))
    else:
        assert torch.count_nonzero(features[:, :, 128:]).item() == 0
    assert bool((features[:, :, :128].abs().sum() > 0))
    diagnostic = json.loads((tmp_path / "geometry.json").read_text())
    assert diagnostic["uses_patchformer_regression_head"] is False
    assert diagnostic["test_labels_used_in_feature_fitting"] is False
    assert diagnostic["flow_checkpoint_sha256"] == (
        "flow-checksum" if use_flow else None
    )
    assert len(dataset.get_split("test")) == 2
    settings["held_out_test_engine_labels_used"] = True
    torch.save(
        {
            "format": "nex233-features-v1",
            "features": torch.zeros(2, 2, 30, 128),
            "rows": [
                {"split": "train", "group_id": "reference"},
                {"split": "val", "group_id": "reference-val"},
            ],
            "settings": settings,
        },
        endpoint,
    )
    with pytest.raises(ValueError, match="official RUL protocol"):
        source.build()
