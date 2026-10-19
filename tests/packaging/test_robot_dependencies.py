"""Robotics opt-in metadata stays separate from consumer environment selection."""

from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.version import Version


def test_robots_is_library_owned_forwarded_and_excluded_from_all():
    core = tomllib.loads(Path("pyproject.toml").read_text())
    library = tomllib.loads(Path("library/pyproject.toml").read_text())
    forwarding = list(map(Requirement, core["project"]["optional-dependencies"]["robots"]))
    assert len(forwarding) == 1
    assert forwarding[0].name == "nexuml-library" and forwarding[0].extras == {"robots"}
    integrations = list(map(Requirement, library["project"]["optional-dependencies"]["robots"]))
    assert {req.name for req in integrations} == {"nexuml-library", "isaaclab", "lerobot"}
    for manifest in (core, library):
        project = manifest["project"]
        assert project["requires-python"] == ">=3.12"
        assert all(
            "robots" not in Requirement(value).extras
            for value in project["optional-dependencies"]["all"]
        )
        assert all(
            Requirement(value).name not in {"isaaclab", "lerobot"}
            for value in project["dependencies"]
        )
    assert all(
        bound.operator == ">="
        for req in integrations
        if req.name in {"isaaclab", "lerobot"}
        for bound in req.specifier
    )


def test_native_source_and_backend_choices_are_only_root_tool_configuration():
    core = tomllib.loads(Path("pyproject.toml").read_text())
    library = tomllib.loads(Path("library/pyproject.toml").read_text())
    assert "uv" not in library.get("tool", {})
    assert core["tool"]["uv"]["sources"]["isaaclab"]["rev"]
    assert core["tool"]["uv"]["sources"]["torch"] == [{"index": "pytorch-cu130"}]
    assert "packaging==25.0" in core["tool"]["uv"]["override-dependencies"]
    assert core["tool"]["uv"]["index-strategy"] == "first-index"
    for manifest in (core, library):
        for requirements in [
            manifest["project"]["dependencies"],
            *manifest["project"]["optional-dependencies"].values(),
        ]:
            for value in requirements:
                req = Requirement(value)
                assert req.url is None
                assert not any("cu130" in bound.version for bound in req.specifier)


def test_native_validation_lock_has_one_opencv_provider():
    lock = tomllib.loads(Path("uv.lock").read_text())
    names = {package["name"] for package in lock["package"]}
    assert "opencv-python-headless-noffmpeg" in names
    assert not names.intersection(
        {"opencv-python", "opencv-python-headless", "opencv-contrib-python"}
    )
    requests = next(package for package in lock["package"] if package["name"] == "requests")
    assert Version(requests["version"]) >= Version("2.32.4")


def test_download_dependency_does_not_leak_netrc_to_a_different_host(tmp_path, monkeypatch):
    import base64
    import requests

    credentials = tmp_path / "netrc"
    credentials.write_text("machine trusted.example login fixture-user password fixture-secret\n")
    monkeypatch.setenv("NETRC", str(credentials))
    with requests.Session() as session:
        prepared = session.prepare_request(
            requests.Request("GET", "http://trusted.example:@attacker.example/")
        )
    leaked = "Basic " + base64.b64encode(b"fixture-user:fixture-secret").decode()
    assert prepared.headers.get("Authorization") != leaked
