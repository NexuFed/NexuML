"""Editorial validation must not impose handwritten metadata on generated API modules."""

import importlib.util
import json
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "check_content", ROOT / "docs/scripts/check_content.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_install_container_tags_use_runtime_placeholders():
    page = (ROOT / "docs/src/content/docs/start/install.mdx").read_text()
    container = page.split("## Container image\n", 1)[1].split("\n## ", 1)[0]
    assert re.search(r"(?:CUDA\s+|cuda)[0-9]", container) is None
    for tag in (
        "<nexuml-version>-cuda<cuda-version>",
        "latest-cuda<cuda-version>",
        "edge-cuda<cuda-version>",
    ):
        assert f"`{tag}`" in container


def test_editorial_frontmatter_and_missing_routes_fail(tmp_path):
    content = tmp_path / "src/content/docs"
    content.mkdir(parents=True)
    (tmp_path / "migration").mkdir()
    mapping = tmp_path / "migration/route-map.json"
    mapping.write_text(json.dumps({"pages": {"index.md": ""}}))
    page = content / "index.mdx"
    page.write_text("---\ntitle: Valid\ndescription: Useful description\n---\n\nContent\n")
    module.check_content(tmp_path)
    page.write_text("---\ntitle: Valid\n---\n\nMissing description\n")
    with pytest.raises(ValueError, match="description"):
        module.check_content(tmp_path)
    page.write_text("No frontmatter\n")
    with pytest.raises(ValueError, match="frontmatter"):
        module.check_content(tmp_path)
