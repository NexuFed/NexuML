"""Safety contracts for selective FD001 CfC follow-up recovery."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / ".k8s/scenarios/recover_rul_enhanced_cfc_followup_fd001.py"
)


def recovery_module():
    """Import the standalone recovery utility."""
    spec = importlib.util.spec_from_file_location("nex233_fd001_recovery_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def original_manifest(recovery: Any, name: str, digest: str = "a" * 64) -> str:
    """Construct a representative previously rendered Job manifest."""
    return (
        "apiVersion: batch/v1\nkind: Job\nmetadata:\n"
        f"  name: {name}\n"
        "  labels:\n    nexuml-study: rul-cfc-followup\n"
        f'  annotations:\n    nexuml.ika.rub.de/source-sha256: "{digest}"\n'
        "spec:\n"
        f"  image: {recovery.IMAGE}\n"
        '  nvidia.com/gpu: "1"\n'
        f"  subPath: job-staging/nexuml-nex233/{digest}\n"
        "  # NEX233_SOURCE_DIGEST.txt\n"
        f'  run = "{recovery.CONTROLLED_RUN}"\n'
        "  command: nexuml train\n"
    )


def test_fd001_only_manifests_preserve_staged_source() -> None:
    """Only six exact failed Jobs can have their preflight updated."""
    recovery = recovery_module()
    assert len(recovery.job_names()) == 6
    assert len(set(recovery.job_names())) == 6
    assert all("followup-fd001-" in name for name in recovery.job_names())
    for job in recovery.job_names():
        original = original_manifest(recovery, job)
        patched = recovery.patched_manifest(job, original)
        assert patched == original.replace(
            f'run = "{recovery.CONTROLLED_RUN}"',
            f'run = "{recovery.DIRECT_RUN}"',
        )
        assert recovery._source_digest(patched) == "a" * 64
    with pytest.raises(ValueError, match="Refusing"):
        recovery.patched_manifest(
            "nexuml-rul-cfc-followup-fd003-baseline-s42",
            original_manifest(recovery, recovery.job_names()[0]),
        )


def test_refuse_missing_or_mixed_original_manifests(tmp_path: Path) -> None:
    """All six recoveries must come from the same immutable source snapshot."""
    recovery = recovery_module()
    for name in recovery.job_names():
        (tmp_path / f"{name}.yaml").write_text(
            original_manifest(recovery, name), encoding="utf-8"
        )
    assert len(recovery.plan_manifests(tmp_path)) == 6
    first = recovery.job_names()[0]
    (tmp_path / f"{first}.yaml").write_text(
        original_manifest(recovery, first, "b" * 64), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="same source snapshot"):
        recovery.plan_manifests(tmp_path)
    (tmp_path / f"{first}.yaml").unlink()
    with pytest.raises(FileNotFoundError):
        recovery.plan_manifests(tmp_path)


def make_local(
    local: Path, monkeypatch: pytest.MonkeyPatch, recovery: Any,
) -> tuple[Path, dict[str, Any]]:
    """Generate test-sized checkpoint, audit, endpoint and C-MAPSS artifact."""
    rul = local / "rul"
    checkpoints = rul / "checkpoints" / recovery.DIRECT_RUN
    audit_dir = rul / "audits"
    prepared = rul / "prepared"
    for path in (checkpoints, audit_dir, prepared):
        path.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoints / "epoch=13-step=12292.ckpt"
    checkpoint.write_bytes(b"trusted local checkpoint")
    prepared_source = prepared / recovery.PREPARED_FILE
    prepared_source.write_bytes(b"existing exact C-MAPSS preparation")
    endpoint = prepared / recovery.ENDPOINT_FILE
    endpoint.write_bytes(b"existing endpoint contents")
    manifest = audit_dir / f"{recovery.DIRECT_RUN}.checkpoint.json"
    manifest.write_text(
        json.dumps({
            "selection": "best minimum val/loss",
            "loaded_strict_model_state": True,
            "best_model_path": str(checkpoint),
            "best_model_score": 120.0,
            "checkpoint_epoch": 13,
        }),
        encoding="utf-8",
    )
    provenance: dict[str, Any] = {
        "settings": {
            "subset": "FD001",
            "seed": 42,
            "feature_normalization": "train_endpoints_only",
            "frozen_patchformer_upstream_run": recovery.DIRECT_RUN,
            "frozen_patchformer_checkpoint_sha256": recovery.file_hash(checkpoint),
            "held_out_test_engine_labels_used": False,
            "source_rul_artifact": str(prepared_source),
        }
    }
    monkeypatch.setattr(recovery, "read_prepared_artifact", lambda path: provenance)
    return checkpoint, provenance


def test_hash_matching_artifacts_migrate_with_audit_rebind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Copy only proven artifacts; use the exact Kubernetes Pod checkpoint path."""
    recovery = recovery_module()
    local, shared = tmp_path / "local", tmp_path / "shared"
    local.mkdir()
    shared.mkdir()
    checkpoint, provenance = make_local(local, monkeypatch, recovery)
    pairs, audit = recovery.checked_artifacts(local, shared)
    assert len(pairs) == 3
    assert checkpoint in [source for source, _ in pairs]
    assert audit["best_model_path"] == (
        "/mnt/nexuml-experiments/rul/checkpoints/"
        f"{recovery.DIRECT_RUN}/{checkpoint.name}"
    )
    for source, destination in pairs:
        recovery.publish_file(source, destination)
    recovery.publish_audit(shared, audit)
    assert recovery.checked_artifacts(local, shared)[1] == audit
    assert all(destination.is_file() for _, destination in pairs)
    assert json.loads(
        (
            shared / "rul" / "audits" / f"{recovery.DIRECT_RUN}.checkpoint.json"
        ).read_text()
    ) == audit
    provenance["settings"]["frozen_patchformer_checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="provenance"):
        recovery.checked_artifacts(local, shared)


def test_refuse_conflicts_and_shadowed_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never overwrite inconsistent Prisma files or shadow an exact checkpoint."""
    recovery = recovery_module()
    local, shared = tmp_path / "local", tmp_path / "shared"
    local.mkdir()
    shared.mkdir()
    make_local(local, monkeypatch, recovery)
    pairs, _ = recovery.checked_artifacts(local, shared)
    destination = pairs[0][1]
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"different source checkpoint")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        recovery.checked_artifacts(local, shared)
    destination.unlink()
    controlled = shared / "rul" / "audits" / f"{recovery.CONTROLLED_RUN}.checkpoint.json"
    controlled.parent.mkdir(parents=True)
    controlled.write_text("{}")
    with pytest.raises(ValueError, match="shadow"):
        recovery.checked_artifacts(local, shared)


def test_refuse_healthy_job_before_deleting_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single completed FD001 Job prevents every deletion or replacement."""
    recovery = recovery_module()
    for name in recovery.job_names():
        (tmp_path / f"{name}.yaml").write_text(
            recovery.patched_manifest(name, original_manifest(recovery, name)),
            encoding="utf-8",
        )
    calls: list[tuple[str, ...]] = []

    def fake_kubectl(namespace: str, *args: str) -> Any:
        calls.append(args)
        if args[:2] == ("get", "job"):
            job = args[2]
            status = (
                {"conditions": [{"type": "Complete", "status": "True"}], "succeeded": 1}
                if job == recovery.job_names()[-1]
                else {"conditions": [{"type": "Failed", "status": "True"}]}
            )
            return type("Result", (), {
                "returncode": 0, "stderr": "",
                "stdout": json.dumps({
                    "metadata": {
                        "name": job,
                        "labels": {"nexuml-study": "rul-cfc-followup"},
                        "annotations": {
                            "nexuml.ika.rub.de/source-sha256": "a" * 64
                        },
                    },
                    "status": status,
                }),
            })()
        raise AssertionError(f"Unexpected kubectl call: {args}")

    monkeypatch.setattr(recovery, "_kubectl", fake_kubectl)
    with pytest.raises(RuntimeError, match="non-failed Job"):
        recovery.restart_jobs("lbecker", tmp_path)
    assert all(args[0] == "get" for args in calls)
