"""Numerical, integration and leakage tests for acoustic endpoint Flow Matching."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from nexuml.core.components import LayerBuildContext
from nexuml.core.compiler import compile
from nexuml.core.config import ResolvedConfig
from nexuml.data.creator import NexuDataCreator
from nexuml.training.lightning import create_runtime_artifacts
from nexuml_library.data.maintenance.audio_endpoints import AudioEndpointPairs
from nexuml_library.layers.feature.lmbe import LMBE
from nexuml_library.layers.generative.flow.paired_path import PairedLinearFlowPath
from nexuml_library.layers.loss.flow_matching_loss import FlowMatchingLoss
from nexuml_library.layers.model.flow import Flow
from nexuml_library.scenarios.data.audio_endpoints import audio_endpoint_data
from nexuml_library.scenarios.flow.audio_health_to_fault import audio_health_to_fault_flow


def _write_audio(file: Path, *, amplitude: float, sample_rate: int = 16000) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    values = np.arange(512, dtype=np.float64)
    samples = np.sin(2 * np.pi * 400 * values / sample_rate) * amplitude
    samples = (samples * 32767).astype("<i2")
    with wave.open(str(file), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())


def _fixture_root(root: Path, *, test: bool = False) -> Path:
    for split in (("train", "val", "test") if test else ("train", "val")):
        for condition in ("healthy", "faulty"):
            for index in range(2 if split == "train" else 1):
                group = f"{split}_{condition}_{index}"
                _write_audio(
                    root / split / condition / group / f"recording_{index}.wav",
                    amplitude=0.1 if condition == "healthy" else 0.7,
                )
    return root


def test_audio_pairs_are_real_acoustic_endpoints_and_exclude_test(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path, test=True)
    source = AudioEndpointPairs(root=root, clip_num_samples=256)
    runtime = source.build()

    assert len(runtime) == 3
    assert runtime.modality == "audio"
    assert runtime.meta is not None
    assert runtime.meta["split"].tolist().count("train") == 2
    assert set(runtime.meta["split"]) == {"train", "val"}
    assert runtime.meta["healthy_file"].str.contains("/healthy/").all()
    assert runtime.meta["faulty_file"].str.contains("/faulty/").all()
    assert not runtime.meta["healthy_file"].str.contains("/test/").any()
    assert not runtime.meta["faulty_file"].str.contains("/test/").any()
    assert runtime.meta["paired"].eq(1).all()

    x, y = runtime[0]
    assert y is not None
    assert x["healthy_waveform"].shape == (256,)
    assert x["faulty_waveform"].shape == (256,)
    assert x["healthy_waveform"].dtype == torch.float32
    assert x["faulty_waveform"].abs().mean() > x["healthy_waveform"].abs().mean()
    assert y["paired"] == 1


def test_group_overlap_across_splits_is_rejected(tmp_path: Path) -> None:
    _fixture_root(tmp_path)
    _write_audio(tmp_path / "val/healthy/train_healthy_0/extra.wav", amplitude=0.1)
    with pytest.raises(ValueError, match="group.*spans"):
        AudioEndpointPairs(root=tmp_path).build()


def test_missing_faulty_endpoints_are_not_silently_fabricated(tmp_path: Path) -> None:
    _fixture_root(tmp_path)
    for file in (tmp_path / "train/faulty").rglob("*.wav"):
        file.unlink()
    with pytest.raises(FileNotFoundError, match="faulty"):
        AudioEndpointPairs(root=tmp_path).build()


def test_held_out_test_requires_explicit_opt_in(tmp_path: Path) -> None:
    _fixture_root(tmp_path, test=True)
    runtime = AudioEndpointPairs(root=tmp_path, include_test=True).build()
    assert runtime.meta is not None
    assert set(runtime.meta["split"]) == {"train", "val", "test"}


def test_sample_rate_mismatch_is_rejected(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    _write_audio(root / "train/healthy/extra/other.wav", amplitude=0.1, sample_rate=8000)
    runtime = AudioEndpointPairs(root=root).build()
    assert runtime.meta is not None
    wrong = runtime.meta.index[
        runtime.meta["healthy_file"].str.endswith("other.wav")
    ].tolist()
    assert wrong
    with pytest.raises(ValueError, match="Sample-rate mismatch"):
        runtime[int(wrong[0])]


def test_independent_pairing_is_deterministic_and_preserves_group_splits(
    tmp_path: Path,
) -> None:
    _fixture_root(tmp_path)
    definition = AudioEndpointPairs(root=tmp_path, seed=123)
    first = definition.build()
    second = definition.build()
    assert first.meta is not None and second.meta is not None
    assert first.meta.equals(second.meta)

    data = audio_endpoint_data([tmp_path], batch_size=2, clip_num_samples=256)
    assert data.datasets[0].modality == "audio"
    assert data.datasets[0].split_type == "keep"
    dataset = NexuDataCreator().build_dataset(data)
    assert dataset.meta is not None
    assert set(dataset.meta["split"]) == {"train", "val"}
    assert dataset.meta["split"].value_counts().to_dict() == {"train": 2, "val": 1}


def test_two_endpoint_path_uses_observed_source_not_noise() -> None:
    path = PairedLinearFlowPath().build(
        LayerBuildContext(
            input_sizes={"healthy": (2,), "faulty": (2,)},
            keys_in=["healthy", "faulty"],
            keys_out=["state", "time", "velocity"],
        )
    )
    healthy = torch.tensor([[1.0, -3.0], [2.0, 4.0]])
    faulty = torch.tensor([[5.0, -1.0], [7.0, 6.0]])
    x = TensorDict({"healthy": healthy, "faulty": faulty}, batch_size=[2])
    result, _ = path(x)
    assert isinstance(result, TensorDict)

    time = result["time"]
    expected = (1 - time) * healthy + time * faulty
    assert torch.allclose(result["state"], expected)
    assert torch.equal(result["velocity"], faulty - healthy)
    assert torch.all((time >= 0) & (time <= 1))
    assert time.shape == (2, 1)


def test_two_endpoint_path_validates_contracts() -> None:
    with pytest.raises(ValueError, match="two inputs"):
        PairedLinearFlowPath().build(
            LayerBuildContext(
                input_sizes={"healthy": (4,)},
                keys_in=["healthy"],
                keys_out=["state", "time", "velocity"],
            )
        )

    with pytest.raises(ValueError, match="shapes must match"):
        PairedLinearFlowPath().build(
            LayerBuildContext(
                input_sizes={"healthy": (4,), "faulty": (2,)},
                keys_in=["healthy", "faulty"],
                keys_out=["state", "time", "velocity"],
            )
        )


def test_audio_scenario_round_trips_and_compiles(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    scenario = audio_health_to_fault_flow(
        data_roots=[root],
        clip_num_samples=512,
        n_fft=128,
        hop_length=64,
        n_mels=8,
        hidden_dims=[16],
        batch_size=2,
        max_epochs=1,
        num_steps=4,
    )
    restored = ResolvedConfig.from_yaml(ResolvedConfig.from_scenario(scenario).to_yaml())
    assert isinstance(restored.data.datasets[0].source, AudioEndpointPairs)
    assert restored.data.datasets[0].split_type == "keep"
    assert all(ds.modality == "audio" for ds in restored.data.datasets)
    assert isinstance(restored.pipeline.stages["AudioFeatures"][0].component, LMBE)
    assert isinstance(restored.pipeline.stages["FlowPath"][0].component, PairedLinearFlowPath)
    assert isinstance(restored.pipeline.stages["Flow"][0].component, Flow)
    assert isinstance(restored.pipeline.stages["Loss"][0].component, FlowMatchingLoss)
    pipeline = compile(scenario)
    assert "flow_matching_loss" in pipeline.loss_keys


def test_end_to_end_audio_training_step_gradients_and_sampling(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    scenario = audio_health_to_fault_flow(
        data_roots=[root],
        clip_num_samples=512,
        n_fft=128,
        hop_length=64,
        n_mels=8,
        hidden_dims=[16],
        batch_size=2,
        max_epochs=1,
        num_steps=4,
    )
    artifacts = create_runtime_artifacts(scenario)
    artifacts.data_module.setup("fit")
    x, y = next(iter(artifacts.data_module.train_dataloader()))
    x_out, _ = artifacts.lightning_module.pipeline(x, y)
    loss, _ = artifacts.lightning_module._compute_loss(x_out)
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert x_out["healthy_spectrum"].shape == x_out["faulty_spectrum"].shape
    assert x_out["flow_target_velocity"].shape == x_out["healthy_spectrum"].shape
    assert torch.allclose(
        x_out["flow_target_velocity"],
        x_out["faulty_spectrum"] - x_out["healthy_spectrum"],
    )
    flow = next(
        layer
        for stage_name, _name, layer in artifacts.pipeline.iter_layers()
        if stage_name == "Flow"
    )
    assert isinstance(flow, torch.nn.Module)
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in flow.parameters()
    )
    samples = cast(torch.Tensor, getattr(flow, "sample")(x_out["healthy_spectrum"]))
    assert samples.shape == x_out["healthy_spectrum"].shape
    assert torch.isfinite(samples).all()


def test_multiple_acoustic_roots_compose_without_shape_changes(tmp_path: Path) -> None:
    a = _fixture_root(tmp_path / "one")
    b = _fixture_root(tmp_path / "two")
    data = audio_endpoint_data([a, b], clip_num_samples=512, batch_size=2)
    merged = NexuDataCreator().build_dataset(data)
    assert merged.meta is not None
    assert len(merged) == 6
    assert merged.meta["source_dataset"].nunique() == 2
    assert set(merged.meta["split"]) == {"train", "val"}
    x, _ = merged[0]
    assert x["healthy_waveform"].shape == (512,)


@pytest.mark.parametrize("sample_width", [1, 2, 3, 4])
def test_pcm_wav_decodes_without_torchcodec(
    tmp_path: Path, sample_width: int
) -> None:
    root = _fixture_root(tmp_path)
    file = root / "train/healthy/train_healthy_0/recording_0.wav"
    amplitude = 1 << (8 * sample_width - 2)
    with wave.open(str(file), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(sample_width)
        output.setframerate(16000)
        if sample_width == 1:
            samples = bytes([128 + amplitude, 128 - amplitude]) * 4
        else:
            pair = (
                amplitude.to_bytes(sample_width, byteorder="little", signed=True)
                + (-amplitude).to_bytes(sample_width, byteorder="little", signed=True)
            )
            samples = pair * 4
        output.writeframes(samples)

    runtime = AudioEndpointPairs(root=root, clip_num_samples=16).build()
    assert runtime.meta is not None
    matching = runtime.meta.index[
        runtime.meta["healthy_file"].eq(str(file))
    ].tolist()
    assert matching
    x, _ = runtime[int(matching[0])]
    assert x["healthy_waveform"].dtype == torch.float32
    assert x["healthy_waveform"].shape == (16,)
    assert torch.allclose(x["healthy_waveform"], torch.zeros(16))


def test_wav_decoder_rejects_unsupported_encoding(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    file = root / "train/healthy/train_healthy_0/recording_0.wav"
    # Unsupported 40-bit sample width must fail rather than silently misdecode.
    with wave.open(str(file), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(5)
        output.setframerate(16000)
        output.writeframes(b"\x00" * 5)
    runtime = AudioEndpointPairs(root=root).build()
    assert runtime.meta is not None
    matching = runtime.meta.index[
        runtime.meta["healthy_file"].eq(str(file))
    ].tolist()
    assert matching
    with pytest.raises(ValueError, match="Unsupported PCM WAV encoding"):
        runtime[int(matching[0])]
