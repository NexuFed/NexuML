"""Contract checks for the GH200 FD003 recovery submission manifest."""

from pathlib import Path

import pytest

from nexuml_library.evaluation.maintenance.rul_machine_flow_submit_k8s import (
    DEFAULT_STAGING_ROOT,
    HOST_PVC_ROOT,
    _host_pvc_path,
    _runtime_command,
    _stage_subpath,
    build_manifest,
)


def test_recovery_job_is_pinned_to_gh200_and_has_prisma_source_mounts() -> None:
    """Keep the recovery command on GH200 and the correct persistent volume."""
    manifest = build_manifest(
        job="nexuml-rul-fd003-recovery-gh200",
        subpath="job-staging/nexuml/recovery/example",
        experiment_root=Path("/mnt/juicefs_prisma/models/nexuml/.experiments"),
    )
    assert manifest["metadata"]["namespace"] == "lbecker"
    assert manifest["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "jobs"
    pod = manifest["spec"]["template"]["spec"]
    assert pod["nodeSelector"] == {"kubernetes.io/hostname": "gh200"}
    assert pod["tolerations"] == [{
        "key": "gpu-recovery", "operator": "Equal", "value": "true",
        "effect": "NoSchedule",
    }]
    container = pod["containers"][0]
    assert container["resources"] == {
        "requests": {"nvidia.com/gpu": "1"},
        "limits": {"nvidia.com/gpu": "1"},
    }
    assert container["volumeMounts"][0]["mountPath"] == "/mnt/juicefs_prisma"
    assert container["volumeMounts"][1]["subPath"].startswith("job-staging/")
    assert pod["volumes"][0]["persistentVolumeClaim"]["claimName"] == "juicefs-prisma"


def test_runtime_runs_exact_recovery_command_with_main_and_feature() -> None:
    """Recover in the target environment using both staged project revisions."""
    command = _runtime_command(Path("/mnt/juicefs_prisma/models/nexuml/.experiments"))
    assert "-e /workspace/core -e /workspace/feature/library" in command
    assert "--rebuild --retrain-missing" in command
    assert "--experiment-root /mnt/juicefs_prisma/models/nexuml/.experiments" in command
    assert "assert torch.cuda.is_available()" in command
    assert "ERROR: source marker missing from Prisma PVC" in command
    assert "PYTHONUNBUFFERED=1" in command


def test_prisma_pv_subdirectory_is_not_staged_as_filesystem_root() -> None:
    """Map workstation paths through the PV's shared/prisma subdirectory."""
    assert HOST_PVC_ROOT == Path("/mnt/juicefs_prisma/shared/prisma")
    assert DEFAULT_STAGING_ROOT == (
        HOST_PVC_ROOT / "job-staging/nexuml/within-machine-recovery"
    )
    assert _stage_subpath(DEFAULT_STAGING_ROOT, "sha-job") == (
        "job-staging/nexuml/within-machine-recovery/sha-job"
    )
    assert _host_pvc_path(Path("/mnt/juicefs_prisma/models/nexuml/.experiments")) == (
        HOST_PVC_ROOT / "models/nexuml/.experiments"
    )
    with pytest.raises(ValueError, match="Staging root"):
        _stage_subpath(
            Path("/mnt/juicefs_prisma/job-staging/nexuml/within-machine-recovery"),
            "sha-job",
        )


def test_recovery_rejects_external_output_paths_and_unsafe_subpaths() -> None:
    """Restrict the staging and experiment output to the intended PVC."""
    with pytest.raises(ValueError, match="Experiment root"):
        build_manifest(
            job="nexuml-rul-fd003-recovery-gh200",
            subpath="job-staging/nexuml/recovery/example",
            experiment_root=Path("/tmp/experiments"),
        )
    with pytest.raises(ValueError, match="Unsafe staged source"):
        build_manifest(
            job="nexuml-rul-fd003-recovery-gh200",
            subpath="job-staging/../outside",
            experiment_root=Path("/mnt/juicefs_prisma/models/nexuml/.experiments"),
        )
    with pytest.raises(ValueError, match="PVC name"):
        build_manifest(
            job="nexuml-rul-fd003-recovery-gh200",
            subpath="job-staging/nexuml/recovery/example",
            experiment_root=Path("/mnt/juicefs_prisma/models/nexuml/.experiments"),
            pvc="juicefs_prisma",
        )
