"""The official SDK bundle is qualified without inventing pip metadata."""

from hashlib import sha256
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest

from nexuml_library.environments import _isaaclab_viser as viewer


def test_pip_simulator_keeps_its_existing_version_check(monkeypatch):
    monkeypatch.setattr(viewer, "distribution", lambda name: SimpleNamespace(version="6.2.0.0"))
    monkeypatch.setattr(viewer, "find_spec", lambda name: pytest.fail("Do not fall back from pip"))
    viewer._qualify_isaacsim()
    monkeypatch.setattr(viewer, "distribution", lambda name: SimpleNamespace(version="6.1.0.0"))
    with pytest.raises(RuntimeError, match="Unqualified spectator distribution"):
        viewer._qualify_isaacsim()


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    def missing_distribution(name):
        raise PackageNotFoundError(name)

    root = tmp_path / "isaac-sim"
    origin = root / "python_packages/isaacsim/__init__.py"
    origin.parent.mkdir(parents=True)
    origin.write_text("# Static bundle identity fixture.\n")
    (root / "VERSION").write_text(viewer._SIM_BUNDLE_VERSION + "\n")
    monkeypatch.setattr(viewer, "distribution", missing_distribution)
    monkeypatch.setattr(viewer, "find_spec", lambda name: SimpleNamespace(origin=str(origin)))
    monkeypatch.setattr(viewer, "_SIM_BUNDLE_INIT_SHA256", sha256(origin.read_bytes()).hexdigest())
    return root, origin


def test_exact_vendor_bundle_is_accepted(bundle):
    viewer._qualify_isaacsim()


@pytest.mark.parametrize("mutation", ["version", "payload", "missing-version", "missing-payload"])
def test_changed_or_incomplete_vendor_bundle_is_rejected(bundle, mutation):
    root, origin = bundle
    if mutation == "version":
        (root / "VERSION").write_text("6.2.0-unqualified-build")
    elif mutation == "payload":
        origin.write_text("# Changed vendor entrypoint.\n")
    elif mutation == "missing-version":
        (root / "VERSION").unlink()
    else:
        origin.unlink()
    with pytest.raises(RuntimeError, match="Isaac Sim SDK bundle"):
        viewer._qualify_isaacsim()


@pytest.mark.parametrize("origin", [None, "/unrelated/isaacsim/__init__.py"])
def test_missing_or_shadowed_bundle_origin_is_rejected(bundle, monkeypatch, origin):
    monkeypatch.setattr(viewer, "find_spec", lambda name: SimpleNamespace(origin=origin))
    with pytest.raises(RuntimeError, match="Isaac Sim SDK bundle"):
        viewer._qualify_isaacsim()


def test_missing_bundle_module_is_rejected(bundle, monkeypatch):
    monkeypatch.setattr(viewer, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="Isaac Sim SDK bundle"):
        viewer._qualify_isaacsim()
