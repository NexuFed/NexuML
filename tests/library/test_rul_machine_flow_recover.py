"""Read-only and recovery-safety contracts for the FD003 machine-flow inputs."""

from pathlib import Path

import pytest

from nexuml_library.evaluation.maintenance import rul_machine_flow_recover as recovery


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
