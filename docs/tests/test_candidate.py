"""Provenance checks must fail before a trusted publisher can use downloaded files."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "verify_candidate", ROOT / "docs/scripts/verify_candidate.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_candidate_identity_and_fixture_rejection(tmp_path):
    revision = "1" * 40
    for target in ("versioned", "changelog"):
        root = tmp_path / target
        root.mkdir()
        data = {
            "schemaVersion": 1,
            "target": target,
            "version": "dev",
            "revision": revision,
            "site": "https://docs.example.com",
            "mount": "/",
        }
        (root / "build.json").write_text(json.dumps(data))
    (tmp_path / "versioned/index.html").write_text("home")
    (tmp_path / "changelog/changelog").mkdir()
    (tmp_path / "changelog/changelog/index.html").write_text("updates")
    module.verify(tmp_path, revision, "dev")
    with pytest.raises(ValueError, match="identity"):
        module.verify(tmp_path, "2" * 40, "dev")
    (tmp_path / "versioned/zz-api-proof.html").write_text("not publishable")
    with pytest.raises(ValueError, match="fixture"):
        module.verify(tmp_path, revision, "dev")
