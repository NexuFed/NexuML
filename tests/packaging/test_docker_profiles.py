"""Locked CUDA runtime profiles and their Docker publishing contract."""

import os
from pathlib import Path
import shutil
import subprocess
import tomllib

import pytest
from ruamel.yaml import YAML


ROOT = Path(__file__).resolve().parents[2]
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
WORKFLOW = YAML(typ="safe").load((ROOT / ".github/workflows/docker.yml").read_text())
PROFILES = [
    ("cu128", "12.8.1", "12.8", "nvidia-dali-cuda120", "nvidia-npp-cu12"),
    ("cu130", "13.3.1", "13.0", "nvidia-dali-cuda130", "nvidia-npp"),
]


def export(*options):
    if not shutil.which("uv"):
        pytest.skip("uv is required to verify the locked CUDA profiles")
    return subprocess.run(
        ["uv", "export", "--frozen", "--all-packages", "--all-extras", "--no-hashes", *options],
        cwd=ROOT,
        env={**os.environ, "UV_OFFLINE": "1"},
        text=True,
        capture_output=True,
    )


def test_docker_matrix_isolated_profiles_and_versions():
    job = WORKFLOW["jobs"]["build"]
    assert WORKFLOW["env"]["PYTHON_VERSION"] == "3.12"
    assert WORKFLOW["env"]["UBUNTU_VERSION"] == "24.04"
    assert job["strategy"]["fail-fast"] is False
    assert job["strategy"]["max-parallel"] == 1
    assert job["strategy"]["matrix"]["include"] == [
        {"backend": backend, "cuda": cuda, "torch-cuda": torch_cuda}
        for backend, cuda, torch_cuda, _, _ in PROFILES
    ]
    assert job["env"]["CUDA_VERSION"] == "${{ matrix.cuda }}"
    assert job["env"]["LOCAL_IMAGE"] == (
        "nexuml:ci-${{ github.run_id }}-${{ github.run_attempt }}-${{ matrix.backend }}"
    )
    build = next(
        step
        for step in job["steps"]
        if step.get("uses", "").startswith("docker/build-push-action@")
    )
    assert "TORCH_BACKEND=${{ matrix.backend }}" in build["with"]["build-args"]
    for name in ("cache-from", "cache-to"):
        assert "scope=nexuml-${{ matrix.backend }}" in build["with"][name]
    verify = next(step for step in job["steps"] if step.get("name") == "Verify local image")
    assert verify["env"]["EXPECTED_TORCH_CUDA"] == "${{ matrix.torch-cuda }}"
    assert '["nvcc", "--version"]' in verify["run"]
    assert '"$LOCAL_IMAGE" -c' in verify["run"]
    assert 'torch.version.cuda == os.environ["EXPECTED_TORCH_CUDA"]' in verify["run"]
    assert 'torch.__version__.endswith("+" + os.environ["EXPECTED_BACKEND"])' in verify["run"]
    source = verify["run"].split('python - <<"PY"\n', 1)[1].split("\nPY", 1)[0]
    compile(source, "docker-image-verification", "exec")
    if shutil.which("bash"):
        subprocess.run(["bash", "-n", "-c", verify["run"]], check=True, capture_output=True)
    metadata = next(step for step in job["steps"] if step.get("name") == "Generate image metadata")
    assert metadata["with"]["flavor"] == "latest=false"
    assert "suffix=-cuda${{ env.CUDA_VERSION }}" in metadata["with"]["tags"]


def test_docker_installs_locked_features_and_one_cuda_group():
    dockerfile = (ROOT / "Dockerfile").read_text()
    commands = [
        line.strip() for line in dockerfile.splitlines() if line.strip().startswith("uv sync")
    ]
    assert len(commands) == 2
    assert commands[0].startswith("uv sync --frozen --no-install-workspace")
    assert commands[1].startswith("uv sync --locked")
    for command in commands:
        assert '--all-packages --all-extras --no-extra dali --group "${TORCH_BACKEND}"' in command
        assert "--upgrade" not in command
    assert "ARG PYTHON_VERSION=3.12" in dockerfile
    assert "ARG TORCH_BACKEND=cu128" in dockerfile
    assert "libegl1" in dockerfile and "libvulkan1" in dockerfile
    assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video,display" in dockerfile
    assert "site-packages/nvidia/npp/lib" in dockerfile
    assert "site-packages/nvidia/cu13/lib" in dockerfile


def test_cuda_groups_do_not_change_public_feature_extras():
    uv = PROJECT["tool"]["uv"]
    assert [{"group": "cu128"}, {"group": "cu130"}] in uv["conflicts"]
    assert [{"extra": "dali"}, {"group": "cu130"}] in uv["conflicts"]
    assert PROJECT["project"]["optional-dependencies"]["dali"] == [
        "nvidia-dali-cuda120; sys_platform == 'linux'"
    ]
    indexes = {index["name"]: index for index in uv["index"]}
    for package in ("torch", "torchvision", "torchaudio", "torchcodec"):
        assert uv["sources"][package] == [
            {"index": "pytorch-cu128", "group": "cu128"},
            {"index": "pytorch-cu130", "group": "cu130"},
        ]
    for backend, _, _, _, _ in PROFILES:
        assert backend not in PROJECT["project"]["optional-dependencies"]
        index = indexes[f"pytorch-{backend}"]
        assert index["explicit"] is True
        assert index["url"] == f"https://download.pytorch.org/whl/{backend}"


@pytest.mark.parametrize("backend,cuda,torch_cuda,dali,npp", PROFILES)
def test_locked_profile_exports_matching_binary_family(backend, cuda, torch_cuda, dali, npp):
    result = export("--no-extra", "dali", "--group", backend)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    for package in ("torch", "torchvision", "torchaudio", "torchcodec"):
        selected = [line for line in lines if line.startswith(package + "==")]
        assert len(selected) == 1 and selected[0].endswith("+" + backend), selected
    selected_dali = [line for line in lines if line.startswith("nvidia-dali-cuda")]
    assert len(selected_dali) == 1 and selected_dali[0].startswith(dali + "==")
    selected_npp = [line for line in lines if line.startswith("nvidia-npp")]
    assert len(selected_npp) == 1 and selected_npp[0].startswith(npp + "==")


@pytest.mark.parametrize(
    "options", [("--group", "cu128", "--group", "cu130"), ("--group", "cu130")]
)
def test_conflicting_cuda_or_dali_families_are_rejected(options):
    result = export(*options)
    assert result.returncode != 0
    assert "conflict" in result.stderr.lower()


def test_normal_all_extras_keep_default_pypi_resolution():
    result = export()
    assert result.returncode == 0, result.stderr
    torch = next(line for line in result.stdout.splitlines() if line.startswith("torch=="))
    assert "+cu" not in torch
    assert "nvidia-dali-cuda120==" in result.stdout
    assert "nvidia-dali-cuda130==" not in result.stdout
