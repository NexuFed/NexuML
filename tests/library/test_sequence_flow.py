"""Regression tests for full-PatchFormer-sequence Transformer/Conformer/CfC flows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from nexuml.core.compiler import compile as compile_pipeline
from nexuml.core.components import LayerBuildContext
from nexuml.core.config import ResolvedConfig
from nexuml_library.data.maintenance.patchformer_flow import (
    AutoPatchFormerFlowEndpoints,
)
from nexuml_library.data.maintenance.prepared_endpoints import read_prepared_artifact
from nexuml_library.evaluation.maintenance.ablation_validation import (
    score_distributional_transport,
)
from nexuml_library.layers.generative.flow.sequence_vector_field import (
    SequenceTimeConditionedVectorField,
    _SequenceVectorFieldRuntime,
)
from nexuml_library.layers.model.flow import SequenceFlow, _FlowRuntime
from nexuml_library.layers.model.rul_advanced import _PatchFormerModel
from nexuml_library.scenarios.maintenance.rul_patchformer_sequence_flow import (
    patchformer_sequence_flow,
)


def _sequence_context(length: int = 6, channels: int = 8) -> LayerBuildContext:
    """Create a native flow-state/time context for [B,L,C] testing.

    Returns:
        Conventional NexuML state/time input and velocity output keys.
    """
    return LayerBuildContext(
        input_sizes={"flow_state": (length, channels), "flow_time": (1,)},
        keys_in=["flow_state", "flow_time"],
        keys_out=["predicted_velocity"],
    )


@pytest.mark.parametrize("architecture", ["transformer", "conformer", "cfc"])
def test_sequence_field_preserves_time_positions_and_learns_gradients(
    architecture: Literal["transformer", "conformer", "cfc"],
) -> None:
    """Trainable per-position velocity retains all tokens and feature channels."""
    definition = SequenceTimeConditionedVectorField(
        architecture=architecture,
        model_dim=16,
        depth=1,
        attention_heads=4,
        conv_kernel=5,
        dropout=0.0,
    )
    runtime = definition.build(_sequence_context())
    assert isinstance(runtime, _SequenceVectorFieldRuntime)
    state = torch.randn(3, 6, 8)
    x = TensorDict(
        {"flow_state": state, "flow_time": torch.full((3, 1), 0.25)},
        batch_size=[3],
    )
    out, _ = runtime(x)
    prediction = out["predicted_velocity"]
    assert prediction.shape == state.shape
    assert bool(torch.isfinite(prediction).all())
    prediction.square().mean().backward()
    assert all(parameter.grad is not None for parameter in runtime.parameters())
    assert any(
        parameter.grad is not None
        and bool(torch.count_nonzero(parameter.grad))
        for parameter in runtime.parameters()
    )
    with torch.no_grad():
        earlier = runtime.velocity(state, torch.zeros(3, 1))
        later = runtime.velocity(state, torch.ones(3, 1))
    assert not torch.allclose(earlier, later)
    with pytest.raises(ValueError, match="Sequence flow expects"):
        runtime.velocity(torch.randn(3, 8), torch.zeros(3, 1))


@pytest.mark.parametrize("architecture", ["transformer", "conformer", "cfc"])
def test_sequence_flow_euler_preserves_shape_and_backpropagates(
    architecture: Literal["transformer", "conformer", "cfc"],
) -> None:
    """Existing NexuML Euler integrator transports complete sequences."""
    definition = SequenceFlow(
        architecture=architecture,
        model_dim=16,
        depth=1,
        attention_heads=4,
        conv_kernel=5,
        num_steps=2,
        record_trajectory=True,
        initialization_seed=42,
    )
    runtime = definition.build(_sequence_context())
    assert isinstance(runtime, _FlowRuntime)
    initial = torch.randn(2, 6, 8)
    transported = runtime.sample(initial)
    assert transported.shape == initial.shape
    assert bool(torch.isfinite(transported).all())
    trajectory = runtime.integrator.last_trajectory
    assert trajectory is not None and trajectory.shape == (2, 3, 6, 8)


@pytest.mark.parametrize("architecture", ["transformer", "conformer", "cfc"])
def test_sequence_flow_swd_scores_full_shape(
    architecture: Literal["transformer", "conformer", "cfc"],
) -> None:
    """SWD compares flattened sequences but Euler acts on [B,L,C] states."""
    runtime = SequenceFlow(
        architecture=architecture,
        model_dim=16,
        depth=1,
        attention_heads=4,
        conv_kernel=5,
        num_steps=2,
        initialization_seed=42,
    ).build(_sequence_context())
    assert isinstance(runtime, _FlowRuntime)
    runtime.eval()
    with torch.no_grad():
        result = score_distributional_transport(
            runtime,
            torch.zeros(4, 6, 8),
            torch.ones(4, 6, 8),
            device=torch.device("cpu"),
            batch_size=2,
            projections=8,
        )
    assert result["val/healthy_to_faulty_swd"] > 0
    assert "val/transport_swd" in result
    assert all(np.isfinite(value) for value in result.values())


@pytest.mark.parametrize("architecture", ["transformer", "conformer", "cfc"])
@pytest.mark.parametrize("subset", ["FD001", "FD003"])
@pytest.mark.parametrize("seed", [42, 43, 44])
def test_full_sequence_flow_registered_and_compiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    architecture: Literal["transformer", "conformer", "cfc"],
    subset: Literal["FD001", "FD003"],
    seed: int,
) -> None:
    """All study combinations retain matching splits and SWD checkpointing."""
    monkeypatch.chdir(tmp_path)
    study = patchformer_sequence_flow(subset, seed, architecture)
    source = study.data.datasets[0].source
    assert isinstance(source, AutoPatchFormerFlowEndpoints)
    assert source.sequence_embeddings is True
    assert source.seed == seed and source.subset == subset
    assert study.data.input_shapes == {
        "healthy_spectrum": [30, 128],
        "faulty_spectrum": [30, 128],
    }
    assert "sequence" in source.output_file
    flow = study.pipeline.stages["Flow"][0].component
    assert isinstance(flow, SequenceFlow)
    assert flow.architecture == architecture
    assert study.callbacks[2].kwargs["monitor"] == "val/transport_swd"
    assert study.callbacks[3].kwargs["monitor"] == "val/transport_swd"
    compiled = compile_pipeline(study)
    assert "flow_matching_loss" in compiled.loss_keys
    config = tmp_path / f"{architecture}-{subset}-{seed}.yaml"
    compiled.resolved_config.save(config)
    assert ResolvedConfig.load(config).to_scenario().name == study.name


def test_sequence_conformer_requires_odd_kernel() -> None:
    """Conformer depthwise convolution must preserve sequence length."""
    definition = SequenceTimeConditionedVectorField(
        architecture="conformer",
        model_dim=16,
        attention_heads=4,
        conv_kernel=4,
    )
    with pytest.raises(ValueError, match="odd"):
        definition.build(_sequence_context())


def test_sequence_transformer_attention_heads_must_divide_width() -> None:
    """Invalid attention geometry fails at build, not during Euler sampling."""
    definition = SequenceTimeConditionedVectorField(
        architecture="transformer",
        model_dim=15,
        attention_heads=4,
    )
    with pytest.raises(ValueError, match="divide"):
        definition.build(_sequence_context())


@pytest.mark.parametrize("architecture", ["transformer", "conformer", "cfc"])
def test_invalid_sequence_model_parameters_are_rejected(
    architecture: Literal["transformer", "conformer", "cfc"],
) -> None:
    """Invalid input shapes cannot silently trigger flat-MLP behavior."""
    definition = SequenceTimeConditionedVectorField(
        architecture=architecture,
        model_dim=16,
        depth=1,
        attention_heads=4,
        conv_kernel=5,
    )
    with pytest.raises(ValueError, match="requires"):
        definition.build(
            LayerBuildContext(
                input_sizes={"flow_state": (8,), "flow_time": (1,)},
                keys_in=["flow_state", "flow_time"],
                keys_out=["predicted_velocity"],
            )
        )


def test_full_patchformer_endpoint_artifact_preserves_tokens_without_fan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Endpoint preparation returns [N,2,30,128] and never calls RUL head."""
    monkeypatch.chdir(tmp_path)
    train = [
        [engine, cycle, *([float(engine) + cycle * 0.002 * sensor
                          for sensor in range(1, 25)])]
        for engine in range(1, 9)
        for cycle in range(1, 171)
    ]
    test = [
        [engine, cycle, *([float(engine + cycle)] * 24)]
        for engine in (1, 2)
        for cycle in range(1, 42)
    ]
    np.savetxt(tmp_path / "train_FD001.txt", np.asarray(train))
    np.savetxt(tmp_path / "test_FD001.txt", np.asarray(test))
    np.savetxt(tmp_path / "RUL_FD001.txt", [12, 150])
    run = "rul-FD001-controlled-patchformer-seed42"
    checkpoint = (
        tmp_path / ".experiments/rul/checkpoints" / run / "epoch=0.ckpt"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    frozen = _PatchFormerModel(30, 14, 64, 5, 2, 0.1)
    torch.save(
        {
            "state_dict": {
                f"pipeline.RULModel.0.backbone.{key}": value
                for key, value in frozen.state_dict().items()
            }
        },
        checkpoint,
    )
    manifest = tmp_path / ".experiments/rul/audits" / f"{run}.checkpoint.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"best_model_path": str(checkpoint)}) + "\n",
        encoding="utf-8",
    )

    def reject_rul_head(*args: object, **kwargs: object) -> torch.Tensor:
        """Raise if feature preparation accidentally invokes FAN regression.

        Raises:
            AssertionError: If an old RUL head is called.
        """
        raise AssertionError(
            "Sequence endpoint preparation cannot call PatchFormer FAN"
        )

    monkeypatch.setattr(type(frozen.fan), "forward", reject_rul_head)
    prepared = tmp_path / "full-sequence-endpoints.pt"
    dataset = AutoPatchFormerFlowEndpoints(
        root=str(tmp_path),
        prepared_rul_path=str(tmp_path / "rul.pt"),
        output_file=str(prepared),
        sequence_embeddings=True,
        train_pairs=4,
        val_pairs=2,
        embedding_batch_size=4,
    ).build()
    payload = read_prepared_artifact(prepared)
    assert payload["features"].shape == (6, 2, 30, 128)
    assert payload["settings"]["feature_layout"] == "sequence"
    assert payload["settings"]["sequence_length"] == 30
    assert payload["settings"]["embedding_dim"] == 128
    assert payload["settings"]["held_out_test_engine_labels_used"] is False
    assert len(dataset.get_split("train")) == 4
    assert len(dataset.get_split("val")) == 2
    invalid = dict(payload)
    invalid["settings"] = {
        **payload["settings"], "sequence_length": 29
    }
    torch.save(invalid, prepared)
    with pytest.raises(ValueError, match="Sequence endpoint shape"):
        read_prepared_artifact(prepared)
