"""Read-only durable artifact admission; no Ray, application or native runtime."""

import hashlib
import json
import os
from pathlib import Path
import runpy

import pytest


@pytest.fixture(params=["recipe", "logging"])
def recovery(tmp_path, request):
    script = Path(__file__).resolve().parents[1] / "fixtures/physical_ai_acceptance.py"
    function = runpy.run_path(str(script))["_recovery"]
    function.__globals__["ATTEMPT_BASE"] = tmp_path
    source = tmp_path / ("a" * 32) / (request.param + "-" + "b" * 32)
    source.mkdir(parents=True)
    artifact = source / "policy.pt"
    artifact.write_bytes(b"retained actor")
    previous = {
        "attempt_root": str(source),
        "status": "PASS",
        "exit_code": 0,
        "source_tree_sha256": "c" * 64,
        "allocation": {"hostname": "old-worker"},
        "artifacts": [
            {
                "path": str(artifact),
                "bytes": artifact.stat().st_size,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        ],
    }
    guard = {"exit_code": 0, "child_returncode": 0, "errors": []}
    (source / "guard.json").write_text(json.dumps(guard))
    record = {
        "checks": {},
        "gate_scope": {},
        "gates": {"R": "UNVERIFIED"},
        "allocation": {"hostname": "new-worker"},
    }

    def run():
        (source / "case.json").write_text(json.dumps(previous))
        options = {
            "source_attempt": str(source),
            **{
                key: hashlib.sha256((source / name).read_bytes()).hexdigest()
                for key, name in [
                    ("source_case_sha256", "case.json"),
                    ("source_guard_sha256", "guard.json"),
                ]
            },
        }
        function(record, tmp_path / "reader", options, {})
        return record

    return source, artifact, previous, run


def test_recovery_verifies_retained_bytes_without_rewriting_them(recovery):
    source, artifact, _, run = recovery
    before = artifact.read_bytes(), (source / "guard.json").read_bytes()
    result = run()
    assert result["checks"]["durable_retrieval"] == "PASS"
    assert result["recovery"]["source_worker"] != result["recovery"]["reader_worker"]
    assert result["recovery"]["verified_artifacts"] == 1
    assert result["gates"]["R"] == "UNVERIFIED"
    assert before == (artifact.read_bytes(), (source / "guard.json").read_bytes())


def test_recovery_parses_only_authenticated_receipt_bytes(recovery, monkeypatch):
    source, _, previous, run = recovery
    original = Path.read_text
    reopened = []

    def replaced(path, *args, **kwargs):
        if path == source / "case.json":
            reopened.append(path)
            return json.dumps({**previous, "allocation": {"hostname": "forged-worker"}})
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", replaced)
    result = run()
    assert result["recovery"]["source_worker"] == "old-worker"
    assert not reopened


def test_recovery_refuses_artifact_symlink_replacement_at_open(recovery, monkeypatch):
    source, artifact, _, run = recovery
    original = os.open

    def replaced(path, flags, *args, **kwargs):
        if path == artifact.name:
            target = source.parent / "foreign"
            artifact.rename(target)
            artifact.symlink_to(target)
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replaced)
    with pytest.raises(OSError):
        run()


@pytest.mark.parametrize("change", ["bytes", "length", "outside", "symlink", "empty", "failed"])
def test_recovery_rejects_bad_or_foreign_retained_artifacts(recovery, change):
    source, artifact, previous, run = recovery
    if change == "bytes":
        artifact.write_bytes(b"modified actor")
    elif change == "length":
        previous["artifacts"][0]["bytes"] += 1
    elif change == "outside":
        previous["artifacts"][0]["path"] = str(source.parent / "foreign")
    elif change == "symlink":
        target = source.parent / "foreign"
        artifact.rename(target)
        artifact.symlink_to(target)
    elif change == "empty":
        previous["artifacts"] = []
    else:
        previous["status"] = "FAIL"
    with pytest.raises(AssertionError):
        run()
