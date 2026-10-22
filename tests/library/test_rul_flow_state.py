"""NexuML tests for standalone Flow-only RUL and encoder-only control."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest
import torch
from torch import Tensor

from nexuml.core.compiler import compile as compile_pipeline
from nexuml.core.config import ResolvedConfig
from nexuml_library.data.maintenance.rul import (
    AutoCMapSSRULWindows,
    read_rul_artifact,
)
from nexuml_library.data.maintenance.rul_flow_calibration import (
    AutoPatchFormerFlowRULFeatures,
)
from nexuml_library.layers.model.rul_flow_state import PatchFormerFlowStateRULHead
from nexuml_library.scenarios.maintenance.rul_flow_state import (
    patchformer_flow_state_rul,
)


@pytest.mark.parametrize("kind", ["flow_state", "encoder_only"])
@pytest.mark.parametrize("subset", ["FD001", "FD003"])
@pytest.mark.parametrize("seed", [42, 43, 44])
def test_native_flow_state_scenarios_have_no_prediction_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: Literal["flow_state", "encoder_only"],
    subset: Literal["FD001", "FD003"],
    seed: int,
) -> None:
    """RUL head receives only encoder and flow-state-derived features."""
    monkeypatch.chdir(tmp_path)
    scenario = patchformer_flow_state_rul(subset, seed, kind)
    source = scenario.data.datasets[0].source
    assert isinstance(source, AutoPatchFormerFlowRULFeatures)
    assert source.feature_mode == ("flow_only" if kind == "flow_state" else kind)
    assert source.seed == seed
    assert source.subset == subset
    assert scenario.data.input_shapes == {"flow_rul_features": [258]}
    assert len(scenario.evaluation.algorithms) == 3
    assert scenario.evaluation.algorithms[0].label_key == "rul"
    assert scenario.evaluation.algorithms[1].label_key == "rul_raw"
    assert scenario.evaluation.algorithms[2].name == "engine_audit"
    layer = scenario.pipeline.stages["RULModel"][0]
    assert isinstance(layer.component, PatchFormerFlowStateRULHead)
    assert layer.keys_in == ["flow_rul_features"]
    assert layer.keys_out == ["predicted_rul"]
    assert scenario.callbacks[1].kwargs["monitor"] == "val/loss"
    compiled = compile_pipeline(scenario)
    file = tmp_path / f"{kind}-{subset}-{seed}.yaml"
    compiled.resolved_config.save(file)
    assert ResolvedConfig.load(file).to_scenario().name == scenario.name


def test_standalone_flow_state_rul_head_is_trainable() -> None:
    """Flow-only calibrated RUL can learn directly from frozen flow states."""
    from nexuml.core.components import LayerBuildContext

    head = PatchFormerFlowStateRULHead(hidden_dim=64)
    runtime = head.build(
        LayerBuildContext(
            input_sizes={"flow_rul_features": (258,)},
            keys_in=["flow_rul_features"],
            keys_out=["predicted_rul"],
        )
    )
    features = torch.randn(4, 258, requires_grad=True)
    values = runtime.forward_tensor(features)
    assert values.shape == (4, 1)
    assert bool(((values >= 0) & (values <= 130)).all())
    values.square().mean().backward()
    assert features.grad is not None
    assert any(param.grad is not None for param in runtime.parameters())
    with pytest.raises(ValueError, match="258"):
        runtime.forward_tensor(torch.randn(4, 3))


def _synthetic_cmapss(root: Path) -> None:
    """Supply a local official-format C-MAPSS subset with valid health phases."""
    train = [
        [engine, cycle, *([float(engine) + cycle * 0.01 * sensor
                          for sensor in range(1, 25)])]
        for engine in range(1, 9)
        for cycle in range(1, 91)
    ]
    test = [
        [engine, cycle, *([float(engine + cycle)] * 24)]
        for engine in (1, 2)
        for cycle in range(1, 35)
    ]
    np.savetxt(root / "train_FD001.txt", np.asarray(train))
    np.savetxt(root / "test_FD001.txt", np.asarray(test))
    np.savetxt(root / "RUL_FD001.txt", [10, 15])


@pytest.mark.parametrize("mode", ["encoder_only", "flow_only"])
def test_no_patchformer_rul_predictor_is_called_during_feature_preparation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    mode: Literal["encoder_only", "flow_only"],
) -> None:
    """Even a deliberately failing FAN/RUL head cannot affect these features."""
    import nexuml_library.data.maintenance.rul_flow_calibration as data_module
    from nexuml_library.layers.model.rul_advanced import _PatchFormerModel

    monkeypatch.chdir(tmp_path)
    _synthetic_cmapss(tmp_path)
    prepared = tmp_path / ".experiments/rul/prepared/original.pt"
    AutoCMapSSRULWindows(
        root=str(tmp_path),
        artifact_path=str(prepared),
        subset="FD001",
        seed=42,
        window=30,
        stride=1,
        rul_cap=130.0,
        sensor_ids=[2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21],
        normalization="zscore",
        padding_mode="zero",
        train_full_windows_only=True,
        normalize_targets=False,
        download=True,
    ).build()
    assert read_rul_artifact(prepared)["settings"]["target_unit"] == "cycles"

    def construct_encoder(
        length: int, channels: int, width: int, patch_size: int,
        heads: int, dropout: float,
    ) -> _PatchFormerModel:
        """Disable all FAN regression inference while keeping actual DPAN."""
        model = _PatchFormerModel(length, channels, width, patch_size, heads, dropout)

        def forbid_predictor(*args: object, **kwargs: object) -> Tensor:
            """Never invoke the original supervised PatchFormer RUL predictor."""
            raise AssertionError("PatchFormer RUL predictor must not be called")

        monkeypatch.setattr(model.fan, "forward", forbid_predictor)
        return model

    monkeypatch.setattr(data_module, "_PatchFormerModel", construct_encoder)
    monkeypatch.setattr(data_module, "_trusted_patchformer_weights", lambda *_: "same")
    checkpoint = tmp_path / "rul-FD001-controlled-patchformer-seed42" / "best.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"trusted test placeholder")

    endpoint_file = tmp_path / ".experiments/rul/prepared/endpoint.pt"
    if mode == "flow_only":
        endpoint_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "nex233-features-v1",
                "features": torch.zeros(2, 2, 128),
                "rows": [
                    {"split": "train", "group_id": "train-reference"},
                    {"split": "val", "group_id": "validation-reference"},
                ],
                "settings": {
                    "subset": "FD001",
                    "seed": 42,
                    "feature_normalization": "train_endpoints_only",
                    "held_out_test_engine_labels_used": False,
                    "source_rul_artifact": str(prepared),
                    "frozen_patchformer_checkpoint_path": str(checkpoint),
                    "frozen_patchformer_checkpoint_sha256": "same",
                    "feature_mean": [0.0] * 128,
                    "feature_std": [1.0] * 128,
                },
            },
            endpoint_file,
        )
        class _ToyFlow:
            """Small frozen flow substitute to test the feature isolation contract."""

            def sample(self, embeddings: Tensor) -> Tensor:
                """Apply a nonzero but deterministic latent flow displacement."""
                return embeddings + 0.2

        monkeypatch.setattr(
            data_module, "load_frozen_latent_flow", lambda *_a, **_kw: (_ToyFlow(), "flow")
        )
        monkeypatch.setattr(
            data_module, "resolve_latent_flow_checkpoint", lambda *_a: checkpoint
        )
        monkeypatch.setattr(
            data_module,
            "healthy_reference_trajectories",
            lambda *_a, **_kw: torch.zeros(1, 51, 128),
        )
    else:
        # The encoder-only baseline must succeed without any trained Flow.
        monkeypatch.setattr(
            data_module, "resolve_patchformer_checkpoint",
            lambda *_a, **_kw: checkpoint,
        )
    source = AutoPatchFormerFlowRULFeatures(
        subset="FD001",
        seed=42,
        root=str(tmp_path),
        prepared_rul_path=str(prepared),
        endpoint_artifact=str(endpoint_file),
        diagnostics_file=str(tmp_path / f"{mode}.json"),
        encoding_batch_size=32,
        feature_mode=mode,
    )
    dataset = source.build()
    features = cast(Tensor, dataset.data)
    assert features.ndim == 2 and features.shape[-1] == 258
    assert bool(torch.isfinite(features).all())
    if mode == "encoder_only":
        assert not bool((features[:, 128:].abs() > 0).any())
    else:
        assert bool((features[:, 128:256].abs() > 0).all())
        assert bool((features[:, 256] >= 0).all())
        assert bool((features[:, 256] <= 1).all())
    diagnostic = json.loads((tmp_path / f"{mode}.json").read_text())
    assert diagnostic["uses_patchformer_regression_head"] is False
    assert diagnostic["test_labels_used_in_feature_fitting"] is False
