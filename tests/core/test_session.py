"""Tests for nexuml.training.lightning.NexuSession."""

from __future__ import annotations

import torch
import pytest
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)

from nexuml.core.types import (
    CheckpointLoadSpec,
    LayerSpec,
    LoaderSpec,
    LoggingSpec,
    PipelineSpec,
    ScenarioSpec,
    TrainingSpec,
)
from nexuml.core.compiler import compile
from nexuml.core.export import export_package, load_package, load_weights
from nexuml.data.loaders.definitions import TorchLoader
from nexuml.training.lightning import NexuSession
from nexuml_library.layers.model.linear_encoder import LinearEncoder
from nexuml_library.scenarios.data.synthetic import synthetic_vector_data
from nexuml_library.scenarios.training.defaults import default_callbacks


def _make_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        name="test_session",
        pipeline=PipelineSpec(
            stages={
                "encode": [
                    LayerSpec(
                        component=LinearEncoder(hidden_dims=[8], output_dim=4),
                        keys_in=["features"],
                        keys_out=["latent"],
                    ),
                ],
                "decode": [
                    LayerSpec(
                        component=LinearEncoder(hidden_dims=[8], output_dim=16),
                        keys_in=["latent"],
                        keys_out=["reconstructed"],
                    ),
                ],
            }
        ),
        training=TrainingSpec(
            max_epochs=1,
            batch_size=4,
            loss_keys={"reconstruction_loss": 1.0},
        ),
        data=synthetic_vector_data(feature_shape=(16,), num_samples=32),
    )


@pytest.mark.slow
def test_session_run():
    scenario = _make_scenario()
    session = NexuSession.from_scenario(scenario, enable_progress_bar=False)
    result = session.run()
    assert result.pipeline is not None
    assert result.trainer is not None


def test_session_setup_orchestrates_without_training(tmp_path):
    """Exercise NexuSession orchestration (build_runtime + build_trainer) without
    running an actual fit loop, so the default suite has training-entry-point
    coverage independent of the slow-marked end-to-end test above."""
    scenario = _make_scenario()
    scenario.data.loader = LoaderSpec(backend=TorchLoader(), batch_size=4, num_workers=0)

    session = NexuSession.from_scenario(scenario, enable_progress_bar=False, log_dir=tmp_path)
    session.setup()

    data_module = session.data_module
    data_module.setup()
    batch = next(iter(data_module.train_dataloader()))

    loss = session.lightning_module.training_step(batch, 0)

    assert session.trainer is not None
    assert torch.is_tensor(loss)
    assert loss.requires_grad


@pytest.mark.parametrize("logging", [None, LoggingSpec()], ids=["none", "empty-spec"])
def test_session_fit_without_loggers_keeps_checkpoint_callbacks(tmp_path, logging):
    """Fit with no loggers while retaining non-logger-dependent callbacks."""
    base = _make_scenario()
    callback_specs = [
        spec.model_copy(update={"kwargs": {**spec.kwargs, "dirpath": str(tmp_path)}})
        if spec.factory.endswith(":ModelCheckpoint")
        else spec
        for spec in default_callbacks()
    ]
    scenario = base.model_copy(
        update={
            "training": base.training.model_copy(update={"accelerator": "cpu", "devices": 1}),
            "data": synthetic_vector_data(feature_shape=(16,), num_samples=12),
            "logging": logging,
            "callbacks": callback_specs,
        }
    )
    session = NexuSession.from_scenario(
        scenario,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=True,
        log_dir=tmp_path,
    )

    callbacks = session.trainer_callbacks
    assert not any(isinstance(cb, (DeviceStatsMonitor, LearningRateMonitor)) for cb in callbacks)
    assert any(isinstance(cb, EarlyStopping) for cb in callbacks)
    assert any(isinstance(cb, ModelCheckpoint) for cb in callbacks)
    assert any(isinstance(cb, RichProgressBar) for cb in callbacks)

    session.fit()

    checkpoint_path = tmp_path / "last.ckpt"
    assert checkpoint_path.exists()

    strict_checkpoint = CheckpointLoadSpec(
        allow_missing=False,
        allow_shape_mismatch=False,
    )
    target_pipeline = compile(scenario)
    report = load_weights(target_pipeline, checkpoint_path, checkpoint=strict_checkpoint)
    assert sorted(report.matched) == sorted(target_pipeline.state_dict())
    assert not report.missing
    assert not report.unexpected
    assert not report.shape_mismatched

    export_dir = tmp_path / "exported"
    export_package(session.pipeline, export_dir, checkpoint_path=checkpoint_path)
    loaded_pipeline, _config, _metadata = load_package(export_dir)
    sidecar_report = load_weights(
        loaded_pipeline,
        export_dir / "lightning.ckpt",
        checkpoint=strict_checkpoint,
    )
    assert sorted(sidecar_report.matched) == sorted(loaded_pipeline.state_dict())
    assert not sidecar_report.missing
    assert not sidecar_report.unexpected
    assert not sidecar_report.shape_mismatched


def test_session_with_loggers_keeps_logger_dependent_callbacks(monkeypatch):
    import nexuml.tracking.logger as logger_module
    from lightning.pytorch.loggers.logger import DummyLogger

    monkeypatch.setattr(logger_module, "create_loggers", lambda *_args, **_kwargs: [DummyLogger()])

    scenario = _make_scenario().model_copy(update={"callbacks": default_callbacks()})
    session = NexuSession.from_scenario(scenario, enable_progress_bar=True)

    callbacks = session.trainer_callbacks
    assert any(isinstance(cb, DeviceStatsMonitor) for cb in callbacks)
    assert any(isinstance(cb, LearningRateMonitor) for cb in callbacks)
