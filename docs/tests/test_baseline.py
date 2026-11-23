"""Verify the migration inventory accounts for the original source and all archives."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "capture_baseline", ROOT / "docs/scripts/capture_baseline.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_baseline_is_reproducible():
    baseline = json.loads((ROOT / "docs/migration/baseline.json").read_text())
    routes = json.loads((ROOT / "docs/migration/route-map.json").read_text())
    assert routes["sourceRef"] == baseline["sourceCommit"]
    assert len(baseline["downloads"]) == 3
    assert set(baseline["pages"]) == set(routes["pages"])
    assert all(page["target"] == routes["pages"][name] for name, page in baseline["pages"].items())
    assert {entry["version"] for entry in baseline["versions"]} <= set(baseline["trees"])


def test_real_archive_reproduction():
    import pytest
    import subprocess

    baseline = json.loads((ROOT / "docs/migration/baseline.json").read_text())
    routes = json.loads((ROOT / "docs/migration/route-map.json").read_text())
    for ref in [baseline["sourceCommit"], baseline["archiveCommit"]]:
        if subprocess.run(["git", "cat-file", "-e", ref], cwd=ROOT, capture_output=True).returncode:
            pytest.skip(
                "Recorded archive/source objects unavailable; fetch baseline refs to reproduce"
            )
    assert module.capture(baseline["sourceCommit"], baseline["archiveCommit"], routes) == baseline


def test_missing_page_is_rejected(monkeypatch):
    import pytest

    baseline = json.loads((ROOT / "docs/migration/baseline.json").read_text())
    routes = json.loads((ROOT / "docs/migration/route-map.json").read_text())
    routes["pages"].pop("studio.md")

    def source_inventory(*args):
        if args[0] == "rev-parse":
            return args[1].encode()
        assert args == ("ls-tree", "-r", "--name-only", baseline["sourceCommit"], "docs/")
        return "\n".join(f"docs/{page}" for page in baseline["pages"]).encode()

    monkeypatch.setattr(module, "git", source_inventory)
    with pytest.raises(ValueError, match="studio.md"):
        module.capture(baseline["sourceCommit"], baseline["archiveCommit"], routes)
