"""Contract tests for the branch-only FD003 Kubernetes source-staging launcher."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


LAUNCHER = (
    Path(__file__).resolve().parents[2]
    / ".k8s/scenarios/submit_rul_enhanced_cfc_fd003.py"
)


def load_launcher():
    """Load the optional experiment launcher without running its CLI."""
    spec = importlib.util.spec_from_file_location("nex233_k8s_submit", LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_small_source(root: Path, module) -> None:
    """Create the minimal runtime source and required workspace metadata."""
    for directory in module.SOURCE_DIRS:
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "src/nexuml/cli").mkdir(parents=True)
    (root / "src/nexuml/cli/main.py").write_text("VALUE = 1\n")
    (root / "library/src/nexuml_library").mkdir(parents=True, exist_ok=True)
    (root / "library/src/nexuml_library/component.py").write_text("VALUE = 1\n")
    (root / "scripts/example.sh").write_text("#!/bin/bash\nexit 0\n")
    (root / "scripts/dataset.pt").write_bytes(b"must never be staged")
    for name in module.ROOT_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"metadata for {name}\n")


def test_source_files_exclude_weights_and_include_lock(tmp_path: Path) -> None:
    """Stage executable code and dependency metadata, never data or weights."""
    module = load_launcher()
    prepare_small_source(tmp_path, module)
    names = {p.relative_to(tmp_path).as_posix() for p in module.source_files(tmp_path)}
    assert "uv.lock" in names
    assert "pyproject.toml" in names
    assert "library/pyproject.toml" in names
    assert "src/nexuml/cli/main.py" in names
    assert "scripts/example.sh" in names
    assert "scripts/dataset.pt" not in names


def test_digest_changes_when_actual_source_changes(tmp_path: Path) -> None:
    """A content digest must detect changed bytes without Git metadata."""
    module = load_launcher()
    prepare_small_source(tmp_path, module)
    original = module.digest_source(tmp_path)
    (tmp_path / "src/nexuml/cli/main.py").write_text("VALUE = 2\n")
    assert original != module.digest_source(tmp_path)


def test_source_symlinks_fail_closed(tmp_path: Path) -> None:
    """Do not silently stage out-of-tree symlinked source files."""
    module = load_launcher()
    prepare_small_source(tmp_path, module)
    (tmp_path / "src/nexuml/cli/alias.py").symlink_to(
        tmp_path / "src/nexuml/cli/main.py"
    )
    with pytest.raises(ValueError, match="symlinks"):
        module.source_files(tmp_path)


def test_render_uses_unique_native_seed_scenarios() -> None:
    """Every generated Job must resolve and train its own six-factor seed."""
    module = load_launcher()
    for seed in (42, 43, 44):
        digest = "a" * 64
        source = f"job-staging/nexuml-nex233/{digest}"
        manifest = module.render_manifest(seed, digest, source)
        expected = "maintenance-rul-enhanced-cfc-fd003-compare-six"
        if seed != 42:
            expected += f"-s{seed}"
        assert expected in manifest
        assert f"compare-six-s{seed}.yaml" in manifest
        assert "harbor.ika.rub.de/library/nexuml:latest" in manifest
        assert "uv venv" in manifest and "uv sync --locked" in manifest
        assert "kueue.x-k8s.io/queue-name: jobs" in manifest
        assert "nvidia.com/gpu: \"1\"" in manifest
        assert "namespace: lbecker" in manifest
        assert f"subPath: {source}" in manifest
        assert "__SOURCE_DIGEST__" not in manifest
        assert "nvidia.com/gpu.memory" not in manifest
        assert "nvidia.com/gpu.product" not in manifest
        assert "mountPath: /tmp/nexuml-job-venv" in manifest
    with pytest.raises(ValueError, match="unexpected"):
        module.render_manifest(45, "a" * 64, "job-staging/example")
