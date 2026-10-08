"""Offline structural checks: python skills/tests/check_skills.py."""

import json
import re
from pathlib import Path


def main():
    """Check skill identities, references and evaluation definitions."""
    root = Path(__file__).resolve().parents[1]
    assert not (root.parent / ".agents/skills/autoresearch/SKILL.md").exists(), (
        "Retired autoresearch skill is still present."
    )
    names = ("nexuml", "nexuml-library", "nexuml-reproduce-paper", "nexuml-autoresearch")
    total_cases = 0
    for name in names:
        directory = root / name
        text = (directory / "SKILL.md").read_text()
        assert text.startswith("---\n"), f"{name}: missing frontmatter"
        header, body = text[4:].split("\n---\n", 1)
        assert f"name: {name}\n" in header + "\n", f"{name}: identity mismatch"
        assert "description:" in header, f"{name}: missing description"
        assert len(body.splitlines()) < 500, f"{name}: body needs progressive disclosure"
        assert "/workspaces/" not in text and "/home/" not in text, f"{name}: machine-specific path"
        cases = json.loads((directory / "evals/evals.json").read_text())
        assert cases["skill_name"] == name
        assert len(cases["evals"]) >= 2
        total_cases += len(cases["evals"])
        assert len({case["id"] for case in cases["evals"]}) == len(cases["evals"])
        for case in cases["evals"]:
            assert case["prompt"] and case["expected_output"]
            for file in case["files"]:
                assert (directory / file).is_file(), f"{name}: missing eval input {file}"
    for markdown in root.rglob("*.md"):
        if any(part.endswith("-workspace") for part in markdown.parts):
            continue
        for link in re.findall(r"\]\(([^)]+)\)", markdown.read_text()):
            if "://" in link or link.startswith("#"):
                continue
            assert (markdown.parent / link.split("#")[0]).exists(), (
                f"{markdown}: broken link {link}"
            )
    print(f"PASS: four skills, frontmatter, {total_cases} eval cases and local reference links")


if __name__ == "__main__":
    main()
