"""Synthetic and fixture-only tests for held-out acoustic flow evaluation."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from torch import Tensor
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from nexuml.training.lightning import create_runtime_artifacts
from nexuml_library.evaluation.maintenance.flow_transport import (
    evaluate_checkpoints,
    evaluate_split,
    feature_metrics,
    read_tensorboard_losses,
    resolve_checkpoints,
    sliced_wasserstein,
    transport_trajectory,
)
from nexuml_library.layers.generative.flow.euler import EulerIntegrator
from nexuml_library.layers.model.flow import _FlowRuntime
from nexuml_library.scenarios.flow.mimii_acoustic_flow import mimii_acoustic_flow


class _TranslationFlow(torch.nn.Module):
    def __init__(self, translation: float) -> None:
        super().__init__()
        self.translation = translation
        self.integrator = EulerIntegrator(
            velocity=self.velocity,
            num_steps=4,
            input_sizes={"features": (2,)},
            keys_in=["features"],
            keys_out=["out"],
        )

    def velocity(self, state: Tensor, time: Tensor) -> Tensor:
        return torch.full_like(state, self.translation)

    def sample(self, healthy: Tensor) -> Tensor:
        return self.integrator.forward_tensor(healthy)


def test_sliced_wasserstein_identical_and_translated_distributions() -> None:
    samples = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert sliced_wasserstein(samples, samples) == 0.0
    shifted = samples + torch.tensor([2.0, 0.0])
    distance = sliced_wasserstein(samples, shifted, seed=19)
    assert distance > 0
    assert distance == sliced_wasserstein(samples, shifted, seed=19)
    assert abs(distance - sliced_wasserstein(shifted, samples, seed=19)) < 1e-6


def test_transport_records_healthy_initial_and_exact_integrated_terminal() -> None:
    healthy = torch.tensor([[0.0, 1.0], [1.0, 2.0]])
    flow = _TranslationFlow(2.0)
    trajectory = transport_trajectory(cast(_FlowRuntime, flow), healthy)
    assert trajectory.shape == (2, 5, 2)
    assert torch.equal(trajectory[:, 0], healthy)
    assert torch.allclose(trajectory[:, -1], healthy + 2.0)
    assert flow.integrator.record_trajectory is False
    assert flow.integrator.last_trajectory is None


def test_distribution_transport_diagnostics_reward_correct_flow() -> None:
    healthy = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    flow = _TranslationFlow(2.0)
    trajectory = transport_trajectory(cast(_FlowRuntime, flow), healthy)
    score = feature_metrics(healthy, healthy + 2.0, trajectory, projections=64)
    assert score["num_samples"] == 4
    assert score["swd_healthy_to_faulty"] > 0.0
    assert score["swd_transported_to_faulty"] < 1e-6
    assert score["swd_relative_reduction"] == pytest.approx(1.0)
    assert len(score["trajectory"]) == 5
    assert score["trajectory"][0]["flow_time"] == 0.0
    assert score["trajectory"][-1]["flow_time"] == 1.0


def test_distribution_metrics_report_when_transport_makes_things_worse() -> None:
    healthy = torch.zeros(5, 2)
    faulty = torch.ones(5, 2)
    flow = _TranslationFlow(-2.0)
    trajectory = transport_trajectory(cast(_FlowRuntime, flow), healthy)
    score = feature_metrics(healthy, faulty, trajectory)
    assert score["swd_relative_reduction"] < 0.0


def test_distribution_metrics_reject_nonfinite_or_incompatible_samples() -> None:
    samples = torch.randn(8, 2)
    with pytest.raises(ValueError, match="equal-shaped"):
        sliced_wasserstein(samples, samples[:4])
    with pytest.raises(ValueError, match="Nonfinite"):
        sliced_wasserstein(samples, torch.full_like(samples, float("nan")))
    with pytest.raises(ValueError, match="projections"):
        sliced_wasserstein(samples, samples, projections=0)
    with pytest.raises(ValueError, match="at least two time points"):
        feature_metrics(samples, samples, samples[:, None])


def test_checkpoint_resolver_uses_lightning_best_callback_not_file_order(
    tmp_path: Path,
) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    best = checkpoints / "epoch=03-step=00020.ckpt"
    misleading = checkpoints / "epoch=10-step=00090.ckpt"
    last = checkpoints / "last.ckpt"
    best.touch()
    misleading.touch()
    torch.save(
        {
            "callbacks": {
                "ModelCheckpoint": {"best_model_path": str(best), "best_model_score": 0.1}
            }
        },
        last,
    )
    selected, final = resolve_checkpoints(checkpoints)
    assert selected == best.resolve()
    assert final == last.resolve()


def test_checkpoint_resolver_rejects_ambiguous_best_without_metadata(
    tmp_path: Path,
) -> None:
    (tmp_path / "epoch=01.ckpt").touch()
    (tmp_path / "epoch=02.ckpt").touch()
    torch.save({"callbacks": {}}, tmp_path / "last.ckpt")
    with pytest.raises(ValueError, match="Cannot uniquely"):
        resolve_checkpoints(tmp_path)


def _write_acoustic_wave(path: Path, amplitude: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(512, dtype=np.float32)
    values = (amplitude * np.sin(2.0 * np.pi * 260 * time / 16000) * 32767).astype("<i2")
    stereo = np.repeat(values[:, None], 8, axis=1)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(8)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(stereo.tobytes())


def _machine_fixture(root: Path) -> Path:
    for machine in ("id_00", "id_02", "id_04", "id_06"):
        for condition, amplitude in (("normal", 0.2), ("abnormal", 0.65)):
            for index in range(2):
                _write_acoustic_wave(
                    root / "fan" / machine / condition / f"{index:08d}.wav",
                    amplitude,
                )
    return root


def test_checkpoint_independent_real_acoustic_validation_and_test(
    tmp_path: Path,
) -> None:
    root = _machine_fixture(tmp_path / "data")
    scenario = mimii_acoustic_flow(
        root=root,
        download=False,
        clip_num_samples=512,
        n_fft=128,
        hop_length=64,
        n_mels=8,
        hidden_dims=[8],
        num_steps=3,
        batch_size=2,
        max_epochs=1,
        num_workers=0,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    artifacts = create_runtime_artifacts(scenario)
    val, _ = evaluate_split(artifacts, split="val", device=torch.device("cpu"), projections=8)
    test, _ = evaluate_split(artifacts, split="test", device=torch.device("cpu"), projections=8)
    assert val["num_samples"] == 2
    assert test["num_samples"] == 2
    assert np.isfinite(val["fixed_time_fm_mse"])
    assert np.isfinite(test["swd_transported_to_faulty"])
    assert len(val["trajectory"]) == 5
    with pytest.raises(ValueError, match="Only val and test"):
        evaluate_split(artifacts, split="train", device=torch.device("cpu"))


def test_trained_checkpoint_evaluation_restores_and_scores_real_format_audio(
    tmp_path: Path,
) -> None:
    root = _machine_fixture(tmp_path / "data")
    checkpoint_dir = tmp_path / "ckpt"
    output_dir = tmp_path / "evaluation"
    scenario = mimii_acoustic_flow(
        root=root,
        download=False,
        clip_num_samples=512,
        n_fft=128,
        hop_length=64,
        n_mels=8,
        hidden_dims=[8],
        num_steps=2,
        batch_size=2,
        max_epochs=1,
        num_workers=0,
        checkpoint_dir=str(checkpoint_dir),
    )
    runtime = create_runtime_artifacts(scenario)
    trainer = Trainer(
        max_epochs=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                save_last=True,
            )
        ],
    )
    trainer.fit(runtime.lightning_module, datamodule=runtime.data_module)
    report = evaluate_checkpoints(
        checkpoint_dir,
        output_dir,
        device="cpu",
        projections=8,
        plots=False,
    )
    assert (output_dir / "metrics.json").is_file()
    assert report["validation_best"]["num_samples"] == 2
    assert report["validation_last"]["num_samples"] == 2
    assert report["held_out_test_best"]["num_samples"] == 2
    assert np.isfinite(report["held_out_test_best"]["fixed_time_fm_mse"])


def test_tensorboard_reader_accepts_one_explicit_run(tmp_path: Path) -> None:
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(str(tmp_path))
    writer.add_scalar("train/loss", 2.0, global_step=10)
    writer.add_scalar("val/loss", 1.0, global_step=10)
    writer.close()
    losses = read_tensorboard_losses(tmp_path)
    assert losses["train/loss"][0] == {"step": 10.0, "value": 2.0}
    assert losses["val/loss"][0] == {"step": 10.0, "value": 1.0}
