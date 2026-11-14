"""Contract checks for the GH200 FD003 recovery submission manifest."""

from pathlib import Path

import pytest

from nexuml_library.evaluation.maintenance.rul_machine_flow_submit_k8s import (
    DEFAULT_DATA_ROOT,
    DEFAULT_STAGING_ROOT,
    STAGING_MOUNT,
    _runtime_command,
    _stage_subpath,
    build_manifest,
)


def test_recovery_job_is_pinned_to_gh200_with_separate_persistent_volumes() -> None:
    """Use the proven stage PVC and Prisma solely for datasets and models."""
    manifest = build_manifest(
        job="nexuml-rul-fd003-recovery-gh200",
        subpath="job-staging/NexuML-predictive/example",
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
    assert container["volumeMounts"][0] == {
        "name": "staging", "mountPath": "/mnt/juicefs",
    }
    assert container["volumeMounts"][1] == {
        "name": "staging", "mountPath": "/workspace",
        "subPath": "job-staging/NexuML-predictive/example",
    }
    assert container["volumeMounts"][2] == {
        "name": "prisma", "mountPath": "/mnt/juicefs_prisma",
    }
    assert pod["volumes"][0]["persistentVolumeClaim"]["claimName"] == "juicefs"
    assert pod["volumes"][1]["persistentVolumeClaim"]["claimName"] == "juicefs-prisma"
    assert container["volumeMounts"][4] == {
        "name": "temporary", "mountPath": "/tmp",
    }
    assert "emptyDir" in pod["volumes"][3]


def test_runtime_runs_exact_recovery_command_with_main_and_feature() -> None:
    """Recover in the target environment using both staged project revisions."""
    command = _runtime_command(Path("/mnt/juicefs_prisma/models/nexuml/.experiments"))
    assert "-e /workspace/core -e /workspace/feature/library" in command
    assert "--rebuild --retrain-missing" in command
    assert "--experiment-root /mnt/juicefs_prisma/models/nexuml/.experiments" in command
    assert "--data-root /mnt/juicefs_prisma/data" in command
    assert "assert torch.cuda.is_available()" in command
    assert "ERROR: source marker missing from staging PVC" in command
    assert "PYTHONUNBUFFERED=1" in command


def test_runtime_installs_all_packages_ephemerally() -> None:
    """Use the rebuttal-style uv bootstrap without persistent package installs."""
    command = _runtime_command(Path("/mnt/juicefs_prisma/models/nexuml/.experiments"))
    assert "export PYTHONUSERBASE=/tmp/nexuml-userbase" in command
    assert "export UV_CACHE_DIR=/tmp/uv-cache" in command
    assert "export UV_PYTHON_INSTALL_DIR=/tmp/nexuml-python" in command
    assert "python3 -m pip install --user --no-cache-dir uv" in command
    assert "uv venv --python 3.12 \"$VENV_DIR\"" in command
    assert "uv pip install --python \"$VENV_DIR/bin/python\" " in command
    assert "/workspace/core /workspace/feature/library" in command
    assert "--system-site-packages" not in command
    assert "-e /workspace/core" not in command


def test_stage_uses_juicefs_and_data_uses_prisma() -> None:
    """Keep the two workstation mounts separate."""
    assert STAGING_MOUNT == Path("/mnt/juicefs")
    assert DEFAULT_STAGING_ROOT == STAGING_MOUNT / "job-staging/NexuML-predictive"
    assert DEFAULT_DATA_ROOT == Path("/mnt/juicefs_prisma/data")
    assert _stage_subpath(DEFAULT_STAGING_ROOT, "sha-job") == (
        "job-staging/NexuML-predictive/sha-job"
    )
    with pytest.raises(ValueError, match="Staging root"):
        _stage_subpath(
            Path("/mnt/juicefs_prisma/job-staging/NexuML-predictive"),
            "sha-job",
        )


def test_recovery_rejects_external_output_paths_and_unsafe_subpaths() -> None:
    """Restrict the staging and experiment output to the intended PVC."""
    with pytest.raises(ValueError, match="Experiment root"):
        build_manifest(
            job="nexuml-rul-fd003-recovery-gh200",
            subpath="job-staging/NexuML-predictive/example",
            experiment_root=Path("/tmp/experiments"),
        )
    with pytest.raises(ValueError, match="Data root"):
        build_manifest(
            job="nexuml-rul-fd003-recovery-gh200",
            subpath="job-staging/NexuML-predictive/example",
            experiment_root=Path("/mnt/juicefs_prisma/models/nexuml/.experiments"),
            data_root=Path("/tmp/data"),
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
            subpath="job-staging/NexuML-predictive/example",
            experiment_root=Path("/mnt/juicefs_prisma/models/nexuml/.experiments"),
            prisma_pvc="juicefs_prisma",
        )
