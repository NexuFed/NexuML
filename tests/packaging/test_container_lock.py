"""CUDA container dependency policy checks; no network, wheels, or environment changes."""

import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("container_lock", ROOT / "scripts/container_lock.py")
assert SPEC is not None and SPEC.loader is not None
container_lock = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(container_lock)


@pytest.fixture
def checkout(tmp_path):
    for relative in ("pyproject.toml", "library/pyproject.toml", "containers/cuda128.in"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    (tmp_path / container_lock.LOCK).write_text(
        container_lock.FINGERPRINT_PREFIX + container_lock.fingerprint(tmp_path) + "\n"
        "example==1.0 --hash=sha256:" + "a" * 64 + "\n"
    )
    return tmp_path


def test_current_lock_passes_without_uv_or_network():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/container_lock.py"), "--check"],
        env={**os.environ, "PATH": "", "UV_OFFLINE": "1"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_release_stamping_does_not_stale_lock(checkout):
    before = container_lock.fingerprint(checkout)
    for relative in ("pyproject.toml", "library/pyproject.toml"):
        manifest = checkout / relative
        manifest.write_text(
            re.sub(r'version = "[^"]+"', 'version = "0.2.3rc99"', manifest.read_text())
        )
    assert container_lock.fingerprint(checkout) == before
    container_lock.check(checkout)


@pytest.mark.parametrize("relative", ["pyproject.toml", "library/pyproject.toml"])
@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('requires-python = ">=3.12"', 'requires-python = ">=3.13"'),
        ("dependencies = [", 'dependencies = ["example>=2",'),
        ('"soundfile>=0.12"', '"soundfile>=0.13"'),
    ],
)
def test_dependency_changes_stale_lock(checkout, relative, old, new):
    manifest = checkout / relative
    assert old in manifest.read_text()
    manifest.write_text(manifest.read_text().replace(old, new, 1))
    with pytest.raises(ValueError, match="stale"):
        container_lock.check(checkout)


@pytest.mark.parametrize("relative", ["pyproject.toml", "library/pyproject.toml"])
@pytest.mark.parametrize(
    "configuration",
    [
        '\n[[tool.uv.index]]\nname = "other"\n'
        'url = "https://example.org/simple"\nexplicit = true\n',
        '\n[tool.uv.sources.example]\nindex = "nvidia-pypi"\n',
        '\n[tool.uv]\nconstraint-dependencies = ["example<2"]\n',
    ],
)
def test_index_source_and_constraint_changes_stale_lock(checkout, relative, configuration):
    manifest = checkout / relative
    manifest.write_text(manifest.read_text() + configuration)
    with pytest.raises(ValueError, match="stale"):
        container_lock.check(checkout)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("https://pypi.nvidia.com", "https://example.org/simple"),
        ("explicit = true", "explicit = false"),
        ('index = "nvidia-pypi"', 'index = "other"'),
    ],
)
def test_existing_index_configuration_changes_stale_lock(checkout, old, new):
    manifest = checkout / "pyproject.toml"
    manifest.write_text(manifest.read_text().replace(old, new))
    with pytest.raises(ValueError, match="stale"):
        container_lock.check(checkout)


def test_backend_constraints_stale_lock(checkout):
    constraints = checkout / container_lock.INPUT
    constraints.write_text(constraints.read_text().replace("torch==2.11.0", "torch==2.10.0"))
    with pytest.raises(ValueError, match="stale"):
        container_lock.check(checkout)


def test_missing_lock_fails(checkout):
    (checkout / container_lock.LOCK).unlink()
    with pytest.raises(ValueError, match="stale"):
        container_lock.check(checkout)


def test_generator_is_binary_only_scoped_and_sanitized(checkout, monkeypatch):
    monkeypatch.setenv("UV_TORCH_BACKEND", "cu130")
    monkeypatch.setenv("UV_INDEX", "https://credentials@example.org/simple")
    monkeypatch.setenv("PIP_INDEX_URL", "https://credentials@example.org/simple")
    monkeypatch.setattr(
        container_lock.subprocess,
        "check_output",
        lambda *args, **kwargs: "uv 0.13.5 (test)\n",
    )

    def compile_lock(command, *, cwd, env, check):
        assert cwd == checkout and check
        assert "VIRTUAL_ENV" not in env
        assert "UV_TORCH_BACKEND" not in env and "UV_INDEX" not in env
        assert "PIP_INDEX_URL" not in env
        assert env["TMPDIR"].startswith("/tmp/opencode/")
        assert env["UV_CACHE_DIR"].startswith(env["TMPDIR"])
        assert command[:3] == ["uv", "pip", "compile"]
        assert command[3:6] == ["pyproject.toml", "library/pyproject.toml", "containers/cuda128.in"]
        for flag in ("--all-extras", "--generate-hashes", "--only-binary=:all:", "--no-annotate"):
            assert flag in command
        assert "--emit-index-annotation" in command
        for flag, value in (
            ("--torch-backend", "cu128"),
            ("--python-version", "3.13"),
            ("--python-platform", "x86_64-manylinux_2_39"),
        ):
            assert command[command.index(flag) + 1] == value
        assert command.count("--no-emit-package") == 2
        assert "nexuml" in command and "nexuml-library" in command
        assert "--index" not in command and "--extra-index-url" not in command
        output = Path(command[command.index("--output-file") + 1])
        output.write_text("example==1.0 --hash=sha256:" + "b" * 64 + "\n")

    monkeypatch.setattr(container_lock.subprocess, "run", compile_lock)
    container_lock.generate(checkout)
    container_lock.check(checkout)
    assert "credentials" not in (checkout / container_lock.LOCK).read_text()


def test_generator_rejects_other_uv_versions(checkout, monkeypatch):
    monkeypatch.setattr(
        container_lock.subprocess,
        "check_output",
        lambda *args, **kwargs: "uv 0.12.9 (test)\n",
    )
    with pytest.raises(ValueError, match="requires uv 0.13.5"):
        container_lock.generate(checkout)


def test_inputs_changing_during_compile_do_not_replace_lock(checkout, monkeypatch):
    original = (checkout / container_lock.LOCK).read_text()
    monkeypatch.setattr(
        container_lock.subprocess,
        "check_output",
        lambda *args, **kwargs: "uv 0.13.5 (test)\n",
    )

    def compile_with_changed_input(*args, **kwargs):
        constraints = checkout / container_lock.INPUT
        constraints.write_text(constraints.read_text() + "example==2\n")

    monkeypatch.setattr(container_lock.subprocess, "run", compile_with_changed_input)
    with pytest.raises(ValueError, match="changed during"):
        container_lock.generate(checkout)
    assert (checkout / container_lock.LOCK).read_text() == original


def test_external_lock_has_hashes_and_exact_cu128_pins():
    lock = (ROOT / container_lock.LOCK).read_text()
    assert "pyproject.toml" not in lock
    assert "nexuml==" not in lock and "nexuml-library==" not in lock
    assert not re.search(r"cu13\d*|cuda13|cuda-13", lock)
    assert "--only-binary :all:" in lock or "--only-binary=:all:" in lock
    requirements = [
        line
        for line in lock.replace("\\\n", "").splitlines()
        if line and not line.lstrip().startswith(("#", "--"))
    ]
    assert requirements
    for requirement in requirements:
        assert re.fullmatch(
            r"[a-z0-9_.-]+==[^\s]+(?:\s+--hash=sha256:[a-f0-9]{64})+", requirement
        ), requirement
    for name, version in {
        "torch": "2.11.0+cu128",
        "torchaudio": "2.11.0+cu128",
        "torchvision": "0.26.0+cu128",
        "torchcodec": "0.11.1+cu128",
    }.items():
        assert any(line.startswith(f"{name}=={version} ") for line in requirements)
    assert any(line.startswith("nvidia-dali-cuda120==") for line in requirements)
    assert re.search(
        r"nvidia-dali-cuda120==[^\n]+(?:\n    --hash=sha256:[^\n]+)+"
        r"\n    # from https://pypi.nvidia.com/\n",
        lock,
    )


def test_docker_checks_before_hash_sync_and_only_installs_local_projects():
    docker = (ROOT / "Dockerfile").read_text()
    assert f"ghcr.io/astral-sh/uv:{container_lock.UV_VERSION}" in docker
    source_copy = docker.index("COPY --chown=${UID}:${GID} . .")
    check = docker.index("python scripts/container_lock.py --check")
    sync = docker.index("uv pip sync")
    assert check < sync < source_copy
    for path in (
        "pyproject.toml",
        "library/pyproject.toml",
        str(container_lock.INPUT),
        str(container_lock.LOCK),
        "scripts/container_lock.py",
    ):
        assert path in docker[:check]
    assert "uv.lock" not in docker and "uv sync" not in docker
    assert "--require-hashes --only-binary :all:" in docker[sync:source_copy]
    assert "--torch-backend cu128" in docker[sync:source_copy]
    assert "--python /env/bin/python" in docker[sync:source_copy]
    assert (
        "uv pip install --python /env/bin/python --no-deps --editable . --editable ./library"
        in docker[source_copy:]
    )
    assert "uv pip check --python /env/bin/python" in docker[source_copy:]


def test_docker_context_excludes_unrelated_local_resources():
    ignored = set((ROOT / ".dockerignore").read_text().splitlines())
    assert {
        "**/node_modules/",
        "**/.venv*/",
        "**/__pycache__/",
        "docs/python/.venv/",
        "/external/",
        "/graphify-out/",
        "/reports/",
        "/skills/*-workspace/",
        ".kube/",
    } <= ignored
    assert not {"/src/", "/library/", "/configs/", "/containers/", "/scripts/"} & ignored
