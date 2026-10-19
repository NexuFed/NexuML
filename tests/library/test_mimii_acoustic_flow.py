"""MIMII real-acoustic dataset and NexuML Flow Matching scenario contracts."""

from __future__ import annotations

import hashlib
import io
import wave
import zipfile
from pathlib import Path
from urllib.request import Request

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from tensordict import TensorDict

from nexuml.core.compiler import compile
from nexuml.core.config import ResolvedConfig
from nexuml.data.creator import NexuDataCreator
from nexuml.training.lightning import create_runtime_artifacts
from nexuml_library.data.maintenance import mimii_download
from nexuml_library.data.maintenance.mimii import MIMIIAcousticEndpointPairs
from nexuml_library.layers.generative.flow.paired_path import PairedLinearFlowPath
from nexuml_library.layers.model.flow import Flow
from nexuml_library.scenarios.flow.mimii_acoustic_flow import mimii_acoustic_flow


def _wav(path: Path, value: float = 0.2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(1024, dtype=np.float64)
    audio = (np.sin(2.0 * np.pi * 330.0 * t / 16000.0) * value * 32767.0).astype("<i2")
    eight_channels = np.repeat(audio[:, None], 8, axis=1)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(8)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(eight_channels.tobytes())


def _mimii_fixture(root: Path) -> Path:
    for machine in ("id_00", "id_02", "id_04", "id_06"):
        for condition in ("normal", "abnormal"):
            for i in range(3):
                _wav(
                    root / "fan" / machine / condition / f"{i:08d}.wav",
                    value=0.2 if condition == "normal" else 0.7,
                )
    return root


def test_mimii_indexes_real_recordings_without_copying(tmp_path: Path) -> None:
    root = _mimii_fixture(tmp_path)
    definition = MIMIIAcousticEndpointPairs(root=root, clip_num_samples=512)
    runtime = definition.build()

    assert runtime.modality == "audio"
    assert runtime.meta is not None
    assert len(runtime) == 12
    assert runtime.meta["split"].value_counts().to_dict() == {
        "train": 6,
        "val": 3,
        "test": 3,
    }
    assert runtime.meta.groupby("group_id")["split"].nunique().eq(1).all()
    assert set(runtime.meta["dataset_name"]) == {"MIMII"}
    assert runtime.meta["healthy_file"].str.contains("/normal/").all()
    assert runtime.meta["faulty_file"].str.contains("/abnormal/").all()
    assert runtime.meta["pair_id"].is_unique
    for split, ids in {
        "train": {"id_00", "id_02"},
        "val": {"id_04"},
        "test": {"id_06"},
    }.items():
        assert set(runtime.meta[runtime.meta["split"] == split]["machine_id"]) == ids

    x, y = runtime[0]
    assert isinstance(x, TensorDict)
    assert y is not None
    assert x["healthy_waveform"].shape == (512,)
    assert x["faulty_waveform"].shape == (512,)
    assert x["healthy_waveform"].dtype == torch.float32
    assert x["faulty_waveform"].abs().mean() > x["healthy_waveform"].abs().mean()


def test_mimii_pair_cap_does_not_duplicate_recordings(tmp_path: Path) -> None:
    root = _mimii_fixture(tmp_path)
    a = MIMIIAcousticEndpointPairs(
        root=root,
        clip_num_samples=512,
        max_pairs_per_machine=2,
        seed=17,
    ).build()
    b = MIMIIAcousticEndpointPairs(
        root=root,
        clip_num_samples=512,
        max_pairs_per_machine=2,
        seed=17,
    ).build()
    assert a.meta is not None and b.meta is not None
    assert a.meta.equals(b.meta)
    assert len(a) == 8
    for _, group in a.meta.groupby("group_id"):
        assert group["healthy_file"].is_unique
        assert group["faulty_file"].is_unique


def test_mimii_rejects_cross_split_machine_overlap() -> None:
    with pytest.raises(ValidationError, match="overlaps"):
        MIMIIAcousticEndpointPairs(
            train_machine_ids=["id_00", "id_02"],
            val_machine_ids=["id_02"],
        )


def test_mimii_missing_faults_produce_explicit_error(tmp_path: Path) -> None:
    root = _mimii_fixture(tmp_path)
    for file in (root / "fan" / "id_02" / "abnormal").glob("*.wav"):
        file.unlink()
    with pytest.raises(FileNotFoundError, match="abnormal"):
        MIMIIAcousticEndpointPairs(root=root).build()


def test_mimii_can_exclude_held_out_test_index(tmp_path: Path) -> None:
    root = _mimii_fixture(tmp_path)
    runtime = MIMIIAcousticEndpointPairs(root=root, include_test=False).build()
    assert runtime.meta is not None
    assert set(runtime.meta["split"]) == {"train", "val"}
    assert "id_06" not in runtime.meta["machine_id"].to_list()


def test_mimii_training_scenario_compiles_round_trips_and_preserves_splits(
    tmp_path: Path,
) -> None:
    root = _mimii_fixture(tmp_path)
    spec = mimii_acoustic_flow(
        root=root,
        clip_num_samples=512,
        n_fft=128,
        hop_length=64,
        n_mels=8,
        hidden_dims=[16],
        num_steps=3,
        batch_size=2,
        max_epochs=1,
        num_workers=0,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    restored = ResolvedConfig.from_yaml(ResolvedConfig.from_scenario(spec).to_yaml())
    source = restored.data.datasets[0].source
    assert isinstance(source, MIMIIAcousticEndpointPairs)
    assert source.machine_type == "fan"
    assert restored.data.datasets[0].split_type == "keep"
    assert restored.data.datasets[0].modality == "audio"
    assert len(restored.callbacks) == 2
    assert restored.logging is not None and restored.logging.mlflow is None
    assert isinstance(restored.pipeline.stages["FlowPath"][0].component, PairedLinearFlowPath)
    assert isinstance(restored.pipeline.stages["Flow"][0].component, Flow)
    pipeline = compile(spec)
    assert "flow_matching_loss" in pipeline.loss_keys

    dataset = NexuDataCreator().build_dataset(spec.data)
    assert dataset.meta is not None
    assert dataset.meta.groupby("group_id")["split"].nunique().eq(1).all()
    assert dataset.meta["split"].value_counts().to_dict() == {
        "train": 6,
        "val": 3,
        "test": 3,
    }


def test_mimii_real_format_cpu_training_step(tmp_path: Path) -> None:
    root = _mimii_fixture(tmp_path)
    spec = mimii_acoustic_flow(
        root=root,
        clip_num_samples=512,
        n_fft=128,
        hop_length=64,
        n_mels=8,
        hidden_dims=[16],
        num_steps=3,
        batch_size=2,
        max_epochs=1,
        num_workers=0,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    artifacts = create_runtime_artifacts(spec)
    artifacts.data_module.setup("fit")
    x, y = next(iter(artifacts.data_module.train_dataloader()))
    out, _ = artifacts.lightning_module.pipeline(x, y)
    loss, _ = artifacts.lightning_module._compute_loss(out)
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() > 0
    target_velocity = out["flow_target_velocity"]
    assert isinstance(target_velocity, torch.Tensor)
    assert torch.isfinite(target_velocity).all()
    assert out["healthy_spectrum"].shape == out["faulty_spectrum"].shape
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in artifacts.pipeline.parameters()
    )


def _zip_mimii_fixture(root: Path) -> bytes:
    """Create an in-memory ZIP with a valid original MIMII layout.

    Returns:
        ZIP data suitable for mocking the official source.
    """
    _mimii_fixture(root)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for wav in sorted((root / "fan").rglob("*.wav")):
            archive.write(wav, arcname=wav.relative_to(root))
    return buffer.getvalue()


def test_mimii_auto_download_extracts_official_archive_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_bytes = _zip_mimii_fixture(tmp_path / "source")
    checksum = hashlib.md5(archive_bytes, usedforsecurity=False).hexdigest()
    monkeypatch.setitem(mimii_download._MD5, "6_dB_fan.zip", checksum)
    requests: list[str] = []

    def urlopen_stub(request: object, timeout: int) -> io.BytesIO:
        requests.append(str(request))
        assert timeout == 120
        return io.BytesIO(archive_bytes)

    monkeypatch.setattr(mimii_download, "urlopen", urlopen_stub)
    destination = tmp_path / "cache" / "6_dB_fan"
    definition = MIMIIAcousticEndpointPairs(root=destination)
    first = definition.build()
    assert len(first) == 12
    assert len(requests) == 1
    assert (destination / "fan" / "id_00" / "normal").is_dir()
    assert (destination.parent / "6_dB_fan.zip").is_file()
    assert not (destination.parent / "6_dB_fan.zip.part").exists()

    second = definition.build()
    assert len(second) == len(first)
    assert len(requests) == 1


def test_mimii_download_false_is_explicitly_offline(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download=True"):
        MIMIIAcousticEndpointPairs(root=tmp_path / "missing", download=False).build()


def test_mimii_auto_download_rejects_corrupt_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        mimii_download,
        "urlopen",
        lambda request, timeout: io.BytesIO(b"corrupt archive"),
    )
    root = tmp_path / "6_dB_fan"
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        MIMIIAcousticEndpointPairs(root=root).build()
    assert not root.exists()
    assert not (tmp_path / "6_dB_fan.zip").exists()
    assert not (tmp_path / "6_dB_fan.zip.part").exists()


def test_mimii_auto_download_rejects_zip_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("../escape.txt", "unsafe")
        archive.writestr("fan/id_00/normal/sample.wav", b"RIFF")
    zipped = data.getvalue()
    checksum = hashlib.md5(zipped, usedforsecurity=False).hexdigest()
    monkeypatch.setitem(mimii_download._MD5, "6_dB_fan.zip", checksum)
    monkeypatch.setattr(mimii_download, "urlopen", lambda request, timeout: io.BytesIO(zipped))
    root = tmp_path / "6_dB_fan"
    with pytest.raises(ValueError, match="Unsafe ZIP entry"):
        MIMIIAcousticEndpointPairs(root=root).build()
    assert not root.exists()
    assert not (tmp_path / "escape.txt").exists()


def test_mimii_auto_download_resumes_partial_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_bytes = _zip_mimii_fixture(tmp_path / "source")
    checksum = hashlib.md5(archive_bytes, usedforsecurity=False).hexdigest()
    monkeypatch.setitem(mimii_download._MD5, "6_dB_fan.zip", checksum)

    root = tmp_path / "cache" / "6_dB_fan"
    root.parent.mkdir(parents=True)
    offset = len(archive_bytes) // 3
    partial = root.parent / "6_dB_fan.zip.part"
    partial.write_bytes(archive_bytes[:offset])
    requests: list[Request] = []

    class RangedResponse(io.BytesIO):
        status = 206

        def __init__(self, data: bytes) -> None:
            super().__init__(data)
            self.headers = {
                "Content-Range": (
                    f"bytes {offset}-{len(archive_bytes) - 1}/{len(archive_bytes)}"
                ),
                "Content-Length": str(len(archive_bytes) - offset),
            }

    def ranged_urlopen(request: Request, timeout: int) -> io.BytesIO:
        requests.append(request)
        assert timeout == 120
        return RangedResponse(archive_bytes[offset:])

    monkeypatch.setattr(mimii_download, "urlopen", ranged_urlopen)
    runtime = MIMIIAcousticEndpointPairs(root=root).build()
    assert len(runtime) == 12
    assert len(requests) == 1
    assert requests[0].get_header("Range") == f"bytes={offset}-"
    assert (root.parent / "6_dB_fan.zip").read_bytes() == archive_bytes
    assert not partial.exists()


def test_mimii_auto_download_restarts_when_range_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_bytes = _zip_mimii_fixture(tmp_path / "source")
    checksum = hashlib.md5(archive_bytes, usedforsecurity=False).hexdigest()
    monkeypatch.setitem(mimii_download._MD5, "6_dB_fan.zip", checksum)

    root = tmp_path / "cache" / "6_dB_fan"
    root.parent.mkdir(parents=True)
    (root.parent / "6_dB_fan.zip.part").write_bytes(b"partial")
    monkeypatch.setattr(
        mimii_download, "urlopen", lambda request, timeout: io.BytesIO(archive_bytes)
    )
    assert len(MIMIIAcousticEndpointPairs(root=root).build()) == 12
    assert (root.parent / "6_dB_fan.zip").read_bytes() == archive_bytes


def test_mimii_download_displays_transferred_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    archive_bytes = _zip_mimii_fixture(tmp_path / "source")
    checksum = hashlib.md5(archive_bytes, usedforsecurity=False).hexdigest()
    monkeypatch.setitem(mimii_download._MD5, "6_dB_fan.zip", checksum)
    monkeypatch.setattr(
        mimii_download, "urlopen", lambda request, timeout: io.BytesIO(archive_bytes)
    )
    root = tmp_path / "cache" / "6_dB_fan"
    assert len(MIMIIAcousticEndpointPairs(root=root).build()) == 12
    output = capsys.readouterr().err
    assert "[MIMII] Connecting to Zenodo" in output
    assert "[MIMII] Downloading" in output
    assert "Transfer finished; verifying checksum" in output
    assert "[MIMII] Extracting verified" in output
    assert "[MIMII] Ready:" in output
