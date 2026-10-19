"""Source-level aggregate-extra coverage without installing optional packages."""

from pathlib import Path
import tomllib

from packaging.requirements import Requirement


def test_all_includes_every_root_extra_except_dev_and_robots():
    project_file = Path(__file__).resolve().parents[2] / "pyproject.toml"
    project = tomllib.loads(project_file.read_text())["project"]
    extras = project["optional-dependencies"]
    assert len(extras["all"]) == 1
    aggregate = Requirement(extras["all"][0])
    assert aggregate.name == project["name"]
    assert aggregate.extras == set(extras) - {"dev", "all", "robots"}
