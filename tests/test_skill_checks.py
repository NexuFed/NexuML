"""Regression tests for the repository-only skill validator."""

import json
import re
import runpy
from pathlib import Path

import pytest


METADATA = """metadata:
  author: NexuFed AI
  version: "0.1.0"
  last-reviewed: "2026-10-08"
"""


@pytest.fixture
def skill_kit(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    evals = root / "tests/evals"
    evals.mkdir(parents=True)
    inputs = root / "tests/fixtures/input.txt"
    inputs.parent.mkdir()
    inputs.write_text("Evaluation input")
    for name in ("nexuml", "nexuml-library", "nexuml-reproduce-paper", "nexuml-autoresearch"):
        directory = root / name
        (directory / "references").mkdir(parents=True)
        (directory / "references/contract.md").write_text("# Contract\n")
        (directory / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: >-\n  A valid skill.\nlicense: Apache-2.0\n"
            f"{METADATA}---\n# Skill\n[Contract](references/contract.md)\n"
        )
        (evals / f"{name}.json").write_text(
            json.dumps(
                {
                    "skill_name": name,
                    "evals": [
                        {
                            "id": i,
                            "prompt": "Task",
                            "expected_output": "Result",
                            "files": ["fixtures/input.txt"],
                        }
                        for i in range(2)
                    ],
                }
            )
        )
    script = Path(__file__).resolve().parents[1] / "skills/tests/check_skills.py"
    check = runpy.run_path(str(script))["main"]
    monkeypatch.setitem(check.__globals__, "__file__", str(root / "tests/check_skills.py"))
    return root, check


@pytest.mark.parametrize(
    "description",
    ["A valid skill.", "'A valid skill.'", ">-\n  A valid\n  skill.", "|-\n  A valid\n  skill."],
)
def test_description_accepts_yaml_scalar_layouts(skill_kit, description):
    root, check = skill_kit
    markdown = root / "nexuml/SKILL.md"
    markdown.write_text(
        markdown.read_text().replace(
            "description: >-\n  A valid skill.", f"description: {description}"
        )
    )
    check()


def test_frontmatter_accepts_equivalent_yaml_values(skill_kit):
    root, check = skill_kit
    markdown = root / "nexuml/SKILL.md"
    markdown.write_text(
        markdown.read_text()
        .replace("name: nexuml", 'name: "nexuml"')
        .replace("license: Apache-2.0", "license: 'Apache-2.0'")
        .replace(
            METADATA,
            'metadata: {author: "NexuFed AI", version: 0.1.0, last-reviewed: 2026-10-08}\n',
        )
    )
    check()


@pytest.mark.parametrize("field", ["metadata", "author", "version", "last-reviewed"])
def test_frontmatter_requires_metadata(skill_kit, field):
    root, check = skill_kit
    markdown = root / "nexuml/SKILL.md"
    text = markdown.read_text()
    text = (
        text.replace(METADATA, "")
        if field == "metadata"
        else re.sub(rf"^  {field}:.*\n", "", text, flags=re.M)
    )
    markdown.write_text(text)
    with pytest.raises(AssertionError):
        check()


@pytest.mark.parametrize(
    "old, new",
    [
        ("name: nexuml", "name: another-skill"),
        ("description: >-\n  A valid skill.", "description: ''"),
        ("description: >-\n  A valid skill.", "description: '   '"),
        ("description: >-\n  A valid skill.", f"description: {'x' * 1025}"),
        ("description: >-\n  A valid skill.", "description: [not, text]"),
        ("license: Apache-2.0", "license: MIT"),
        ("author: NexuFed AI", "author: Someone else"),
        ('version: "0.1.0"', 'version: "0.1"'),
        ('last-reviewed: "2026-10-08"', 'last-reviewed: "October 2026"'),
    ],
)
def test_frontmatter_rejects_invalid_values(skill_kit, old, new):
    root, check = skill_kit
    markdown = root / "nexuml/SKILL.md"
    markdown.write_text(markdown.read_text().replace(old, new))
    with pytest.raises(AssertionError):
        check()


def test_skill_payload_excludes_evals(skill_kit):
    root, check = skill_kit
    (root / "nexuml/evals.json").write_text("{}")
    with pytest.raises(AssertionError, match="install payload"):
        check()


def test_eval_inputs_must_exist(skill_kit):
    root, check = skill_kit
    (root / "tests/fixtures/input.txt").unlink()
    with pytest.raises(AssertionError, match="missing eval input"):
        check()


def test_local_links_must_resolve(skill_kit):
    root, check = skill_kit
    (root / "nexuml/references/contract.md").unlink()
    with pytest.raises(AssertionError, match="broken link"):
        check()
