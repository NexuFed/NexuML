"""Keep workers in this worktree and away from the user's library configuration."""

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def selected_test_runtime(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join((str(root / "src"), str(root / "library/src")))
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(
        "nexuml.core.discovery.DEFAULT_CONFIG_PATH", tmp_path / ".config/nexuml/libraries.json"
    )
