"""No-download diagnostics for conditional velocity cancellation in NEX-233."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from nexuml.training.lightning import create_runtime_artifacts
from nexuml_library.evaluation.maintenance.velocity_diagnostics import (
    investigate_velocity,
    neighbor_velocity_alignment,
    training_only_ot_diagnostic,
    velocity_statistics,
)
from nexuml_library.scenarios.flow.mimii_acoustic_flow import mimii_acoustic_flow


def test_same_velocity_has_no_unconditional_cancellation() -> None:
    source = torch.tensor([[0.0, 2.0], [1.0, 3.0], [5.0, 1.0]])
    target = source + torch.tensor([2.0, -1.0])
    stats = velocity_statistics(source, target)
    assert stats["mean_velocity_energy_fraction"] == pytest.approx(1.0)
    assert stats["unconditional_cancellation_fraction"] == pytest.approx(0.0)
    assert stats["mean_velocity_mse"] < 1e-12


def test_opposing_target_velocities_cancel_globally() -> None:
    source = torch.zeros(2, 2)
    target = torch.tensor([[2.0, 0.0], [-2.0, 0.0]])
    stats = velocity_statistics(source, target)
    assert stats["unconditional_cancellation_fraction"] == pytest.approx(1.0)
    assert stats["zero_velocity_mse"] > 0.0
    assert stats["mean_velocity_mse"] == pytest.approx(stats["zero_velocity_mse"])


def test_conditional_neighborhood_detects_conflicting_targets() -> None:
    source = torch.tensor([[-1.0], [1.0], [-1.0], [1.0]])
    target = -source
    # Every interpolation state is zero at t=0.5; target velocities oppose.
    scores = neighbor_velocity_alignment(source, target, k=3, times=(0.5,))
    assert scores[0]["flow_time"] == 0.5
    assert scores[0]["mean_neighbor_velocity_cosine"] < 0.0
    assert scores[0]["local_mean_velocity_energy_fraction"] < 0.2


def test_reassignment_reduces_pair_cost_without_changing_the_global_mean() -> None:
    pytest.importorskip("scipy.optimize")
    source = torch.tensor([[0.0], [10.0], [1.0], [11.0]])
    target = torch.tensor([[11.0], [1.0], [12.0], [2.0]])
    groups = ["id_00", "id_00", "id_02", "id_02"]
    original_mean = (target - source).mean(0)
    results = training_only_ot_diagnostic(source, target, groups)
    assert results["independent_pair_mse"] > results["min_cost_assignment_mse"]
    assert all(
        row["possible_assignment_mse_reduction"] > 0
        for row in results["by_machine"]
    )
    assert original_mean.item() == pytest.approx(1.0)


def test_velocity_diagnostics_reject_invalid_endpoints_and_neighborhoods() -> None:
    source = torch.randn(4, 2)
    with pytest.raises(ValueError, match="nonempty and equal-shaped"):
        velocity_statistics(source, source[:2])
    with pytest.raises(ValueError, match="finite"):
        velocity_statistics(source, torch.full_like(source, float("nan")))
    with pytest.raises(ValueError, match="k >= 1"):
        neighbor_velocity_alignment(source, source, k=0)
    with pytest.raises(ValueError, match="Machine IDs"):
        training_only_ot_diagnostic(source, source, ["id_00"])


def _write_wav(path: Path, amplitude: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(512, dtype=np.float32)
    waveform = (
        amplitude * np.sin(2 * np.pi * 260 * time / 16000) * 32767
    ).astype("<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(8)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(np.repeat(waveform[:, None], 8, axis=1).tobytes())


def test_end_to_end_train_and_validation_without_opening_test_data(tmp_path: Path) -> None:
    pytest.importorskip("scipy.optimize")
    root = tmp_path / "fan"
    for machine in ("id_00", "id_02", "id_04", "id_06"):
        for condition, amplitude in (("normal", 0.2), ("abnormal", 0.6)):
            for index in range(2):
                _write_wav(
                    root / machine / condition / f"{index}.wav",
                    amplitude=amplitude + 0.02 * index,
                )
    spec = mimii_acoustic_flow(
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
        checkpoint_dir=str(tmp_path / "checkpoint"),
    )
    artifacts = create_runtime_artifacts(spec)
    results = investigate_velocity(
        artifacts,
        device=torch.device("cpu"),
        neighbors=1,
        projections=8,
    )
    assert results["training"]["num_pairs"] == 4
    assert results["validation"]["num_pairs"] == 2
    assert len(results["training"]["neighbor_alignment"]) == 3
    assert len(results["training"]["pairing_alternative"]["by_machine"]) == 2
    assert len(results["validation"]["predictor"]["fixed_times"]) == 3
    assert np.isfinite(
        results["validation"]["transport_baselines"]["flow_to_faulty_swd"]
    )
    assert "test" not in results
