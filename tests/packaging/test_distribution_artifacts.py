import os
import tarfile
import tomllib
import zipfile
import subprocess
import sys
from email.parser import BytesParser
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def _project_version(path: Path) -> str:
    return tomllib.loads(path.read_text())["project"]["version"]


CORE_VERSION = _project_version(Path("pyproject.toml"))
LIBRARY_VERSION = _project_version(Path("library/pyproject.toml"))

pytestmark = pytest.mark.skipif(
    "NEXUML_DIST_ROOT" not in os.environ,
    reason="distribution artifacts were not requested",
)


def _only(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    assert len(matches) == 1, f"Expected one {pattern} in {directory}, found {len(matches)}"
    return matches[0]


def _only_name(names: set[str], suffix: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    assert len(matches) == 1, f"Expected one *{suffix}, found {len(matches)}"
    return matches[0]


def _wheel_files(path: Path) -> tuple[set[str], bytes, bytes]:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        metadata_name = _only_name(names, ".dist-info/METADATA")
        license_name = _only_name(names, ".dist-info/licenses/LICENSE")
        return names, archive.read(metadata_name), archive.read(license_name)


def _sdist_files(path: Path) -> tuple[set[str], bytes]:
    with tarfile.open(path, "r:gz") as archive:
        names = set(archive.getnames())
        license_name = _only_name(names, "/LICENSE")
        member = archive.extractfile(license_name)
        assert member is not None, f"Could not read {license_name} from {path}"
        return names, member.read()


def _require_names(names: set[str], suffixes: tuple[str, ...], artifact: Path) -> None:
    for suffix in suffixes:
        assert any(name.endswith(suffix) for name in names), f"{artifact} is missing *{suffix}"


def _metadata(path: Path, expected_name: str, expected_version: str, raw: bytes):
    message = BytesParser().parsebytes(raw)
    expected = {
        "Name": expected_name,
        "Version": expected_version,
        "License-Expression": "Apache-2.0",
        "License-File": "LICENSE",
        "Maintainer": "NexuFed AI",
        "Requires-Python": ">=3.12",
        "Description-Content-Type": "text/markdown",
    }
    for key, value in expected.items():
        assert message[key] == value, f"{path}: expected {key}: {value!r}, got {message[key]!r}"
    payload = message.get_payload()
    assert isinstance(payload, str) and payload.strip(), f"{path}: README payload is empty"
    assert len(message.get_all("Project-URL", [])) >= 5
    return message


def test_distribution_artifacts() -> None:
    root = Path(os.environ["NEXUML_DIST_ROOT"])
    license_text = Path("LICENSE").read_bytes()
    core_dir = root / "nexuml"
    library_dir = root / "nexuml-library"

    core_wheel = _only(core_dir, f"nexuml-{CORE_VERSION}-*.whl")
    core_sdist = _only(core_dir, f"nexuml-{CORE_VERSION}.tar.gz")
    library_wheel = _only(library_dir, f"nexuml_library-{LIBRARY_VERSION}-*.whl")
    library_sdist = _only(library_dir, f"nexuml_library-{LIBRARY_VERSION}.tar.gz")

    core_wheel_names, core_raw_metadata, core_license = _wheel_files(core_wheel)
    library_wheel_names, library_raw_metadata, library_license = _wheel_files(library_wheel)
    core_sdist_names, core_sdist_license = _sdist_files(core_sdist)
    library_sdist_names, library_sdist_license = _sdist_files(library_sdist)

    _require_names(core_wheel_names, ("nexuml/__init__.py", "nexuml/api/app.py"), core_wheel)
    _require_names(
        library_wheel_names,
        ("nexuml_library/__init__.py", "nexuml_library/data/dcaset2/dcase_zenodo.yaml"),
        library_wheel,
    )
    _require_names(core_sdist_names, ("README.md", "src/nexuml/__init__.py"), core_sdist)
    assert not any(
        "studio/" in name or "node_modules/" in name for name in core_wheel_names | core_sdist_names
    )
    for dependency in ("fastapi", "uvicorn", "websockets", "psutil"):
        assert any(
            requirement.startswith(dependency) and "extra == 'api'" in requirement
            for requirement in BytesParser()
            .parsebytes(core_raw_metadata)
            .get_all("Requires-Dist", [])
        )
    _require_names(
        library_sdist_names,
        ("README.md", "src/nexuml_library/__init__.py", "dcase_zenodo.yaml"),
        library_sdist,
    )
    for artifact, packaged_license in (
        (core_wheel, core_license),
        (core_sdist, core_sdist_license),
        (library_wheel, library_license),
        (library_sdist, library_sdist_license),
    ):
        assert packaged_license == license_text, (
            f"{artifact}: packaged license differs from LICENSE"
        )

    core = _metadata(core_wheel, "nexuml", CORE_VERSION, core_raw_metadata)
    library = _metadata(library_wheel, "nexuml-library", LIBRARY_VERSION, library_raw_metadata)
    core_requires = core.get_all("Requires-Dist", [])
    library_requires = library.get_all("Requires-Dist", [])

    assert not any(
        requirement.startswith("nexuml-library") and "extra ==" not in requirement
        for requirement in core_requires
    )
    assert any(
        requirement.startswith("nexuml-library")
        and ">=0.2" in requirement
        and "<0.3" in requirement
        and "extra == 'library'" in requirement
        for requirement in core_requires
    )
    assert any(
        requirement.startswith("nexuml")
        and ">=0.2" in requirement
        and "<0.3" in requirement
        and "extra ==" not in requirement
        for requirement in library_requires
    )

    forbidden = ("rapids", "numba", "huggingface-hub", "ffmpeg", "einops", "omegaconf")
    assert not any(
        requirement.lower().startswith(forbidden)
        for requirement in core_requires + library_requires
    )
    assert not any(
        requirement.startswith(("fastapi", "uvicorn", "websockets", "psutil"))
        and "extra ==" not in requirement
        for requirement in core_requires
    )

    all_requires = [
        Requirement(requirement) for requirement in core_requires if "extra == 'all'" in requirement
    ]
    core_extras = tomllib.loads(Path("pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    expected_all = [
        Requirement(requirement)
        for extra, requirements in core_extras.items()
        if extra not in {"dev", "all", "robots", "isaac", "lerobot", "mujoco"}
        for requirement in requirements
        if Requirement(requirement).name != "nexuml"
    ]
    # Hatch expands self-referencing extras into their concrete published requirements.
    assert {
        (canonicalize_name(req.name), frozenset(req.extras), req.specifier) for req in all_requires
    } == {
        (canonicalize_name(req.name), frozenset(req.extras), req.specifier) for req in expected_all
    }
    dali = next(req for req in all_requires if req.name == "nvidia-dali-cuda120")
    assert dali.marker is not None
    assert dali.marker.evaluate({"extra": "all", "sys_platform": "linux"})
    assert not dali.marker.evaluate({"extra": "all", "sys_platform": "win32"})
    for requirements in (core_requires, library_requires):
        parsed = [Requirement(value) for value in requirements]
        native = [req for req in parsed if req.name in {"isaaclab", "lerobot"}]
        assert not any(
            req.marker is None or req.marker.evaluate({"extra": "all"}) for req in native
        )
    assert any(
        req.name == "nexuml-library"
        and req.extras == {"robots"}
        and req.marker is not None
        and req.marker.evaluate({"extra": "robots"})
        for req in map(Requirement, core_requires)
    )
    for name in ("isaaclab", "lerobot"):
        assert any(
            req.name == name and req.marker is not None and req.marker.evaluate({"extra": "robots"})
            for req in map(Requirement, library_requires)
        )
    assert "ray[default,train]<2.59,>=2.57; extra == 'ray'" in core_requires


def test_wheel_discovery_does_not_import_robotics(tmp_path):
    root = Path(os.environ["NEXUML_DIST_ROOT"])
    for directory in ("nexuml", "nexuml-library"):
        with zipfile.ZipFile(_only(root / directory, "*.whl")) as archive:
            archive.extractall(tmp_path)
    code = """
import importlib.abc, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
blocked = {'lerobot', 'newton', 'mujoco', 'jax', 'rclpy', 'omni', 'pxr', 'torchrl', 'gymnasium'}
class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        name = fullname.split('.')[0]
        if name in blocked or name.startswith(('isaaclab', 'isaacsim')):
            raise AssertionError('unexpected native import: ' + fullname)
sys.meta_path.insert(0, BlockNative())
import nexuml, nexuml_library
assert Path(nexuml.__file__).is_relative_to(root)
assert Path(nexuml_library.__file__).is_relative_to(root)
from nexuml.core.scenario_registry import get_scenario_registry
assert get_scenario_registry().get('isaac-cartpole-ppo')().interaction is not None
assert not blocked.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(
    os.environ.get("NEXUML_RUN_RESOLVER_TESTS") != "1",
    reason="explicit opt-in required for network-backed consumer resolution",
)
@pytest.mark.parametrize("source_kind", ["wheel", "editable"])
def test_external_consumer_owns_backend_versions_and_overrides(tmp_path, source_kind):
    root = Path(os.environ["NEXUML_DIST_ROOT"]).resolve()
    core = _only(root / "nexuml", f"nexuml-{CORE_VERSION}-*.whl")
    library = _only(root / "nexuml-library", f"nexuml_library-{LIBRARY_VERSION}-*.whl")
    core_source, library_source = core, library
    editable = ""
    if source_kind == "editable":
        core_source = Path(__file__).resolve().parents[2]
        library_source = core_source / "library"
        editable = ", editable=true"
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(
        '[project]\nname="external-consumer"\nversion="0.0.0"\n'
        'requires-python=">=3.12,<3.13"\n'
        f'dependencies=["nexuml[library]=={CORE_VERSION}", "torch==2.10.0+cpu"]\n'
        '[tool.uv]\noverride-dependencies=["numpy==2.4.6", "packaging==26.0"]\n'
        "[tool.uv.sources]\n"
        f"nexuml={{path={str(core_source)!r}{editable}}}\n"
        f"nexuml-library={{path={str(library_source)!r}{editable}}}\n"
        'torch={index="cpu"}\ntorchaudio={index="cpu"}\ntorchvision={index="cpu"}\n'
        '[[tool.uv.index]]\nname="cpu"\nurl="https://download.pytorch.org/whl/cpu"\n'
        "explicit=true\n"
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith("UV_")}
    env["UV_PYTHON_DOWNLOADS"] = "never"
    command = ["uv", "lock", "--project", str(tmp_path), "--python", sys.executable]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    locked = tomllib.loads((tmp_path / "uv.lock").read_text())
    packages = {package["name"]: package for package in locked["package"]}
    assert packages["torch"]["version"] == "2.10.0+cpu"
    assert packages["torch"]["source"]["registry"] == "https://download.pytorch.org/whl/cpu"
    assert packages["numpy"]["version"] == "2.4.6"
    assert packages["packaging"]["version"] == "26.0"
    assert not set(packages).intersection({"isaaclab", "isaacsim", "lerobot", "newton"})

    manifest.write_text(manifest.read_text().replace("torch==2.10.0+cpu", "torch==1.13.1+cpu"))
    rejected = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
    assert rejected.returncode != 0
    assert "torch>=2.0" in rejected.stderr and "unsatisfiable" in rejected.stderr

    requirements = tmp_path / "requirements.in"
    requirements.write_text(str(core) + "\n")
    rejected = subprocess.run(
        [
            "uv",
            "--no-config",
            "pip",
            "compile",
            str(requirements),
            "--python",
            sys.executable,
            "--python-version",
            "3.11",
            "--no-build",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert rejected.returncode != 0
    assert "requested Python version (>=3.11)" in rejected.stderr
    assert f"nexuml=={CORE_VERSION} only supports >=3.12" in rejected.stderr
