"""Tests for nexuml.training.lightning.NexuSession."""

from __future__ import annotations

import torch
import pytest

from nexuml.core.types import LayerSpec, LoaderSpec, PipelineSpec, ScenarioSpec, TrainingSpec
from nexuml.training.lightning import NexuSession
from nexuml_library.scenarios.data.synthetic import synthetic_vector_data


def _make_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        name="test_session",
        pipeline=PipelineSpec(
            stages={
                "encode": [
                    LayerSpec(
                        type_key="LinearEncoder",
                        keys_in=["features"],
                        keys_out=["latent"],
                        params={"hidden_dims": [8], "output_dim": 4},
                    ),
                ],
                "decode": [
                    LayerSpec(
                        type_key="LinearEncoder",
                        keys_in=["latent"],
                        keys_out=["reconstructed"],
                        params={"hidden_dims": [8], "output_dim": 16},
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
    scenario.data.loader = LoaderSpec(backend="torch", batch_size=4, num_workers=0)

    session = NexuSession.from_scenario(scenario, enable_progress_bar=False, log_dir=tmp_path)
    session.setup()

    data_module = session.data_module
    data_module.setup()
    batch = next(iter(data_module.train_dataloader()))

    loss = session.lightning_module.training_step(batch, 0)

    assert session.trainer is not None
    assert torch.is_tensor(loss)
    assert loss.requires_grad
