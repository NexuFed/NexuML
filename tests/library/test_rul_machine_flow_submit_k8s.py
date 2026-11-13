"""Contract checks for the GH200 FD003 recovery submission manifest."""

from pathlib import Path

import pytest

from nexuml_library.evaluation.maintenance.rul_machine_flow_submit_k8s import (
    _runtime_command,
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
    assert pod["volumes"][0]["persistentVolumeClaim"]["claimName"] == "juicefs"


def test_runtime_runs_exact_recovery_command_with_main_and_feature() -> None:
    """Recover in the target environment using both staged project revisions."""
    command = _runtime_command(Path("/mnt/juicefs_prisma/models/nexuml/.experiments"))
    assert "-e /workspace/core -e /workspace/feature/library" in command
    assert "--rebuild --retrain-missing" in command
    assert "--experiment-root /mnt/juicefs_prisma/models/nexuml/.experiments" in command
    assert "assert torch.cuda.is_available()" in command


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
