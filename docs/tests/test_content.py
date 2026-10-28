"""Editorial validation must not impose handwritten metadata on generated API modules."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "check_content", ROOT / "docs/scripts/check_content.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


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


def test_physical_ai_guides_are_in_the_starlight_site():
    module.check_content(ROOT / "docs")
    sidebar = (ROOT / "docs/astro.config.mjs").read_text()
    for route in (
        "how-to/reinforcement-learning",
        "how-to/isaac-cartpole",
        "how-to/isaac-native-configuration",
        "how-to/isaac-read-only-viser",
        "how-to/isaac-sensors-and-policy-state",
        "how-to/isaac-multi-agent-learning",
        "how-to/lerobot-data",
        "how-to/so101-local-fit",
        "explanation/physical-contracts",
        "explanation/physical-ai-architecture",
        "explanation/isaac-spectator-boundary",
    ):
        assert (ROOT / f"docs/src/content/docs/{route}.mdx").is_file()
        assert f"slug: '{route}'" in sidebar
        assert not (ROOT / f"docs/{route}.md").exists()
    assert not (ROOT / "mkdocs.yml").exists()
