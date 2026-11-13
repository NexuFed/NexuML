"""Read-only and recovery-safety contracts for the FD003 machine-flow inputs."""

import io
from pathlib import Path

import numpy as np
import pytest

from nexuml_library.data.maintenance import rul_download
from nexuml_library.data.maintenance.patchformer_flow import AutoPatchFormerFlowEndpoints
from nexuml_library.evaluation.maintenance import rul_machine_flow_recover as recovery
from nexuml_library.evaluation.maintenance.rul_prepare import _windowize


def test_recovery_status_is_read_only(tmp_path: Path) -> None:
    """Inspect an absent experiment without creating its directory."""
    root = tmp_path / "nexuml" / ".experiments"
    status = recovery.recovery_status(root)
    assert not root.exists()
    assert status["prepared"]["exists"] is False
    assert status["audit"]["exists"] is False
    assert status["endpoint"]["exists"] is False
    assert status["checkpoint"]["available"] is False
    assert not any(status["official_fd003_tables"].values())


def test_recovery_requires_real_prisma_mount(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never create files on the local mountpoint when JuiceFS is absent."""
    monkeypatch.setattr(Path, "is_mount", lambda self: False)
    with pytest.raises(RuntimeError, match="not mounted"):
        recovery._check_mount(Path("/mnt/juicefs_prisma/models/nexuml/.experiments"))


def test_orphaned_checkpoint_audit_cannot_be_replaced(tmp_path: Path) -> None:
    """Preserve any existing audit when its checkpoint is no longer available."""
    root = tmp_path / ".experiments"
    locations = recovery._locations(root)
    audit = locations["audit"]
    audit.parent.mkdir(parents=True)
    audit.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="audit exists"):
        recovery._train_patchformer(locations)


def test_training_uses_native_runner_and_shared_checkpoint_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage a dedicated config and run the existing scenario without overrides."""
    locations = recovery._locations(tmp_path / ".experiments")
    calls: list[tuple[list[str], Path, dict[str, str]]] = []

    def record(command: list[str], *, cwd: Path, env: dict[str, str], check: bool) -> None:
        assert check
        calls.append((command, cwd, env))

    monkeypatch.setattr(recovery.subprocess, "run", record)
    recovery._train_patchformer(locations)
    assert len(calls) == 2
    assert calls[0][0][:3] == ["nexuml", "resolve", recovery.SCENARIO]
    assert calls[1][0][:3] == ["nexuml", "train", "--config"]
    assert calls[1][0][3] == calls[0][0][4]
    assert all(cwd == tmp_path for _, cwd, _ in calls)
    assert all(
        env["NEXUML_RUL_CHECKPOINT_ROOT"]
        == str(locations["experiment_root"] / "rul" / "checkpoints")
        for _, _, env in calls
    )


def test_download_reports_bytes_without_changing_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Byte-level progress must preserve the downloaded archive exactly."""
    content = b"official-nasa-test-zip"

    class Response(io.BytesIO):
        headers = {"Content-Length": str(len(content))}

    monkeypatch.setattr(rul_download, "urlopen", lambda request, timeout: Response(content))
    monkeypatch.setattr(rul_download.zipfile, "is_zipfile", lambda file: True)
    destination = tmp_path / "nasa.zip"
    rul_download._download_archive(destination, show_progress=True)
    assert destination.read_bytes() == content


def test_window_progress_does_not_change_prepared_values() -> None:
    """Progress reporting must have no effect on chronological sensor windows."""
    observations = np.arange(12, dtype=np.float32).reshape(4, 3)
    rul = np.array([3, 2, 1, 0], dtype=np.float32)
    sequences = [("train:FD003:1", "train", observations, rul)]
    expected = _windowize(sequences, window=2, stride=1, final_only_test=True)
    actual = _windowize(
        sequences, window=2, stride=1, final_only_test=True, show_progress=True
    )
    assert expected[0].equal(actual[0])
    assert expected[1].equal(actual[1])
    assert expected[2] == actual[2]


def test_patchformer_embedding_progress_is_opt_in() -> None:
    """Existing flow scenarios retain silent embedding generation by default."""
    assert AutoPatchFormerFlowEndpoints().show_progress is False
    assert AutoPatchFormerFlowEndpoints(show_progress=True).show_progress is True
