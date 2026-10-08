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
    for name in names:
        directory = root / name
        text = (directory / "SKILL.md").read_text()
        assert text.startswith("---\n"), f"{name}: missing frontmatter"
        header, body = text[4:].split("\n---\n", 1)
        assert f"name: {name}\n" in header + "\n", f"{name}: identity mismatch"
        description = re.search(r"(?m)^description: >-\n((?:  [^\n]+\n?)+)", header + "\n")
        assert description, f"{name}: expected an explicit multiline description"
        summary = " ".join(line.strip() for line in description.group(1).splitlines())
        assert 1 <= len(summary) <= 1024, f"{name}: description length outside spec"
        assert re.search(r"(?m)^license: Apache-2\\.0$", header), (
            f"{name}: missing Apache-2.0 license"
        )
        assert re.search(r"(?m)^metadata:$", header), f"{name}: missing metadata"
        assert re.search(r"(?m)^  author: NexuFed AI$", header), (
            f"{name}: missing source attribution"
        )
        assert re.search(r'(?m)^  version: "\\d+\\.\\d+\\.\\d+"
        assert "/workspaces/" not in text and "/home/" not in text, f"{name}: machine-specific path"
        assert {entry.name for entry in directory.iterdir()} <= {
            "SKILL.md",
            "references",
            "scripts",
            "assets",
        }, f"{name}: evaluation/test apparatus leaked into install payload"
        assert json.loads((root / "tests/evals" / f"{name}.json").read_text())["skill_name"] == name
    total_cases = 0
    for case_file in (root / "tests/evals").glob("*.json"):
        cases = json.loads(case_file.read_text())
        assert cases["skill_name"] in (*names, "nexuml-agent-kit")
        assert len(cases["evals"]) >= 2
        total_cases += len(cases["evals"])
        assert len({case["id"] for case in cases["evals"]}) == len(cases["evals"])
        for case in cases["evals"]:
            assert case["prompt"] and case["expected_output"]
            for file in case["files"]:
                assert (root / "tests" / file).is_file(), f"{case_file}: missing eval input {file}"
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
, header), (
            f"{name}: expected independent semantic skill version"
        )
        assert re.search(r'(?m)^  last-reviewed: "\\d{4}-\\d{2}-\\d{2}"
        assert "/workspaces/" not in text and "/home/" not in text, f"{name}: machine-specific path"
        assert {entry.name for entry in directory.iterdir()} <= {
            "SKILL.md",
            "references",
            "scripts",
            "assets",
        }, f"{name}: evaluation/test apparatus leaked into install payload"
        assert json.loads((root / "tests/evals" / f"{name}.json").read_text())["skill_name"] == name
    total_cases = 0
    for case_file in (root / "tests/evals").glob("*.json"):
        cases = json.loads(case_file.read_text())
        assert cases["skill_name"] in (*names, "nexuml-agent-kit")
        assert len(cases["evals"]) >= 2
        total_cases += len(cases["evals"])
        assert len({case["id"] for case in cases["evals"]}) == len(cases["evals"])
        for case in cases["evals"]:
            assert case["prompt"] and case["expected_output"]
            for file in case["files"]:
                assert (root / "tests" / file).is_file(), f"{case_file}: missing eval input {file}"
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
, header), (
            f"{name}: missing review date"
        )
        assert len(body.splitlines()) < 500, f"{name}: body needs progressive disclosure"
        assert "/workspaces/" not in text and "/home/" not in text, f"{name}: machine-specific path"
        assert {entry.name for entry in directory.iterdir()} <= {
            "SKILL.md",
            "references",
            "scripts",
            "assets",
        }, f"{name}: evaluation/test apparatus leaked into install payload"
        assert json.loads((root / "tests/evals" / f"{name}.json").read_text())["skill_name"] == name
    total_cases = 0
    for case_file in (root / "tests/evals").glob("*.json"):
        cases = json.loads(case_file.read_text())
        assert cases["skill_name"] in (*names, "nexuml-agent-kit")
        assert len(cases["evals"]) >= 2
        total_cases += len(cases["evals"])
        assert len({case["id"] for case in cases["evals"]}) == len(cases["evals"])
        for case in cases["evals"]:
            assert case["prompt"] and case["expected_output"]
            for file in case["files"]:
                assert (root / "tests" / file).is_file(), f"{case_file}: missing eval input {file}"
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
