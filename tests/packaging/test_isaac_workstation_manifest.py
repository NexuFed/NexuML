"""The workstation template retains GPU, persistence and private-access requirements."""

from pathlib import Path
import hashlib
import os
import re
import runpy
import shutil
import subprocess
import tomllib
import zipfile

import pytest
from ruamel.yaml import YAML


ROOT = Path(__file__).resolve().parents[2]
STAGER = ROOT / "containers/isaac/stage.py"


def test_isaac_workstation_manifest():
    pod = YAML(typ="safe").load((ROOT / ".k8s/isaac/pod.yaml").read_text())
    assert pod["apiVersion"] == "v1" and pod["kind"] == "Pod"
    assert pod["metadata"]["namespace"] == "rglitza"
    spec = pod["spec"]
    assert spec["runtimeClassName"] == "nvidia"
    assert not spec.get("hostNetwork")
    assert "kubernetes.io/hostname" not in spec["nodeSelector"]
    assert pod["metadata"]["annotations"]["netbird.io/setup-key"] == "isaac-workstation"
    assert spec["imagePullSecrets"] == [{"name": "regcred"}]
    assert spec["restartPolicy"] == "Never"
    assert spec["automountServiceAccountToken"] is False
    assert "runAsUser" not in spec["securityContext"]
    assert "fsGroup" not in spec["securityContext"]
    assert "initContainers" not in spec

    (container,) = spec["containers"]
    assert container["name"] == "isaac"
    assert re.fullmatch(
        r"harbor\.ika\.rub\.de/library/nexuml-isaac@sha256:[a-f0-9]{64}", container["image"]
    )
    assert container["imagePullPolicy"] == "IfNotPresent"
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert container["securityContext"]["runAsNonRoot"] is True
    assert container["securityContext"]["runAsUser"] == 1234
    assert container["securityContext"]["allowPrivilegeEscalation"] is False
    assert container["securityContext"]["capabilities"] == {"drop": ["ALL"]}
    assert not container["securityContext"].get("privileged", False)
    env = {entry["name"]: entry for entry in container["env"]}
    assert env["NVIDIA_DRIVER_CAPABILITIES"]["value"] == "compute,utility,graphics,video"
    assert "ISAACSIM_HOST" not in env
    assert "NVIDIA_VISIBLE_DEVICES" not in env
    assert "PRIVACY_CONSENT" not in env
    assert env["OMNI_CRASHREPORTER_ENABLED"]["value"] == "0"
    assert env["OMNI_CRASHREPORTER_SKIPOLDDUMPUPLOAD"]["value"] == "1"
    assert all("hostPort" not in p for p in container["ports"])
    assert {(p["containerPort"], p["protocol"]) for p in container["ports"]} == {
        (49100, "TCP"),
        (47998, "UDP"),
    }

    volumes = {volume["name"]: volume for volume in spec["volumes"]}
    assert volumes["workspace"]["persistentVolumeClaim"]["claimName"] == "seaweedfs"
    assert volumes["shared-memory"]["emptyDir"] == {"medium": "Memory", "sizeLimit": "2Gi"}
    assert {m["name"]: m["mountPath"] for m in container["volumeMounts"]} == {
        "workspace": "/mnt/persistent",
        "shared-memory": "/dev/shm",
    }
    expressions = spec["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"][0]["matchExpressions"]
    assert expressions == [
        {
            "key": "nvidia.com/gpu.product",
            "operator": "In",
            "values": ["NVIDIA-RTX-4000-Ada-Generation"],
        },
        {"key": "nvidia.com/cuda.driver.major", "operator": "Gt", "values": ["594"]},
    ]
    assert "command" not in container
    shell, flag, script = container["args"]
    assert (shell, flag) == ("/bin/bash", "-c")
    assert 'mkdir -p "$NEXUML_WORKSPACE_ROOT"/{assets,tasks,runs}' in script
    assert "exec /isaac-sim/runheadless.sh --enable isaacsim.ros2.bridge" in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


@pytest.mark.parametrize("nic_present", [True, False])
def test_editor_waits_for_and_advertises_the_pod_netbird_ip(tmp_path, nic_present):
    pod = YAML(typ="safe").load((ROOT / ".k8s/isaac/pod.yaml").read_text())
    script = pod["spec"]["containers"][0]["args"][-1]
    ip = tmp_path / "ip"
    ip.write_text(
        "#!/bin/sh\n"
        '[ "$*" = "-o -4 addr show dev wt0" ] || exit 2\n'
        + ("echo '21: wt0 inet 100.100.0.7/16'\n" if nic_present else "exit 1\n")
    )
    ip.chmod(0o755)
    sleep = tmp_path / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n")
    sleep.chmod(0o755)
    workspace = tmp_path / "workspace"
    result = subprocess.run(
        [
            "bash",
            "-c",
            script.replace(
                "exec /isaac-sim/runheadless.sh --enable isaacsim.ros2.bridge",
                'printf "%s\\n" "$ISAACSIM_HOST"',
            ),
        ],
        env={
            "PATH": f"{tmp_path}:{os.defpath}",
            "ISAACSIM_HOST": "192.0.2.7",
            "NEXUML_WORKSPACE_ROOT": str(workspace),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert (result.returncode == 0) == nic_present
    assert (workspace / "assets").exists() == nic_present
    if nic_present:
        assert result.stdout == "100.100.0.7\n"
    else:
        assert "NetBird sidecar did not assign" in result.stderr


def test_netbird_enrollment_references_a_secret_and_isolates_editor_ingress():
    load = YAML(typ="safe")
    pod = load.load((ROOT / ".k8s/isaac/pod.yaml").read_text())
    key, policy = load.load_all((ROOT / ".k8s/isaac/netbird.yaml").read_text())
    assert key["kind"] == "NBSetupKey"
    assert key["metadata"]["name"] == pod["metadata"]["annotations"]["netbird.io/setup-key"]
    assert key["spec"]["secretKeyRef"] == {"name": "isaac-netbird-enrollment", "key": "setup-key"}
    assert policy["kind"] == "NetworkPolicy"
    assert policy["spec"]["podSelector"]["matchLabels"] == pod["metadata"]["labels"]
    assert policy["spec"]["policyTypes"] == ["Ingress"]
    assert policy["spec"]["ingress"] == []


def test_training_job_reuses_the_editor_image_and_pvc_without_streaming():
    load = YAML(typ="safe").load
    job = load((ROOT / ".k8s/isaac/job.yaml").read_text())
    editor = load((ROOT / ".k8s/isaac/pod.yaml").read_text())["spec"]
    assert job["kind"] == "Job"
    assert job["spec"]["backoffLimit"] == 0
    spec = job["spec"]["template"]["spec"]
    assert spec["volumes"] == editor["volumes"]
    assert spec["securityContext"]["runAsUser"] == 1234
    assert spec["securityContext"]["seccompProfile"] == editor["securityContext"]["seccompProfile"]
    assert spec["imagePullSecrets"] == editor["imagePullSecrets"]
    assert spec["automountServiceAccountToken"] is False
    assert not spec.get("hostNetwork")
    (container,) = spec["containers"]
    assert container["image"] == editor["containers"][0]["image"]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert "ports" not in container and "command" not in container
    _, _, script = container["args"]
    assert 'cd "$NEXUML_LOGS_ROOT"' in script
    assert "exec nexuml train --scenario-file" in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_workstation_stager_rejects_changed_artifacts(tmp_path):
    stage = runpy.run_path(str(STAGER))
    artifact = tmp_path / "dependency.whl"
    artifact.write_bytes(b"approved binary artifact")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    stage["check_hash"](artifact, digest)
    artifact.write_bytes(b"changed binary artifact")
    with pytest.raises(ValueError, match="Artifact hash mismatch"):
        stage["check_hash"](artifact, digest)


def test_workstation_wheel_metadata_identifies_the_built_version(tmp_path):
    stage = runpy.run_path(str(STAGER))
    wheel = tmp_path / "fixture.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "fixture.dist-info/METADATA",
            "Name: fixture\nVersion: 1.0\nRequires-Python: >=3.12\n"
            "Requires-Dist: zed>=2\nRequires-Dist: alpha<4\nProvides-Extra: dev\n",
        )
    assert stage["wheel_metadata"](wheel) == ("fixture", "1.0")


@pytest.mark.parametrize("manifest", ["pyproject.toml", "library/pyproject.toml"])
def test_isaac_lock_tracks_both_manifests_but_not_release_stamps(tmp_path, manifest):
    stage = runpy.run_path(str(STAGER))
    for directory in ("library", "containers/isaac"):
        (tmp_path / directory).mkdir(parents=True)
    for relative in (
        "pyproject.toml",
        "library/pyproject.toml",
        *(
            f"containers/isaac/{path.name}"
            for path in STAGER.parent.iterdir()
            if path.suffix in (".in", ".txt", ".json")
        ),
    ):
        shutil.copyfile(ROOT / relative, tmp_path / relative)
    lock = stage["LOCK"].read_text()
    path = tmp_path / manifest
    text = path.read_text()
    version = tomllib.loads(text)["project"]["version"]
    git_release_stamp = text.replace(f'version = "{version}"', 'version = "0.2.99"', 1)
    assert git_release_stamp != text
    path.write_text(git_release_stamp)
    stage["check_lock"](lock, stage["fingerprint"](tmp_path))
    path.write_text(git_release_stamp.replace("dependencies = [", 'dependencies = ["idna<4", ', 1))
    with pytest.raises(ValueError, match="lock is stale"):
        stage["check_lock"](lock, stage["fingerprint"](tmp_path))


def test_lock_rejects_unhashed_or_duplicate_simulator_requirements():
    stage = runpy.run_path(str(STAGER))
    header = stage["FINGERPRINT"] + "fingerprint\n"
    for requirement in ("idna==3.12", "isaacsim==6.2.0 --hash=sha256:" + "a" * 64):
        with pytest.raises(ValueError, match="Unexpected Isaac lock requirement"):
            stage["check_lock"](header + requirement, "fingerprint")
