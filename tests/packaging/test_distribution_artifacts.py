import os
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

import pytest

VERSION = "0.2.0"

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


def _metadata(path: Path, expected_name: str, raw: bytes):
    message = BytesParser().parsebytes(raw)
    expected = {
        "Name": expected_name,
        "Version": VERSION,
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

    core_wheel = _only(core_dir, f"nexuml-{VERSION}-*.whl")
    core_sdist = _only(core_dir, f"nexuml-{VERSION}.tar.gz")
    library_wheel = _only(library_dir, f"nexuml_library-{VERSION}-*.whl")
    library_sdist = _only(library_dir, f"nexuml_library-{VERSION}.tar.gz")

    core_wheel_names, core_raw_metadata, core_license = _wheel_files(core_wheel)
    library_wheel_names, library_raw_metadata, library_license = _wheel_files(library_wheel)
    core_sdist_names, core_sdist_license = _sdist_files(core_sdist)
    library_sdist_names, library_sdist_license = _sdist_files(library_sdist)

    _require_names(core_wheel_names, ("nexuml/__init__.py",), core_wheel)
    _require_names(
        library_wheel_names,
        ("nexuml_library/__init__.py", "nexuml_library/data/dcaset2/dcase_zenodo.yaml"),
        library_wheel,
    )
    _require_names(core_sdist_names, ("README.md", "src/nexuml/__init__.py"), core_sdist)
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

    core = _metadata(core_wheel, "nexuml", core_raw_metadata)
    library = _metadata(library_wheel, "nexuml-library", library_raw_metadata)
    core_requires = core.get_all("Requires-Dist", [])
    library_requires = library.get_all("Requires-Dist", [])

    assert not any(
        requirement.startswith("nexuml-library") and "extra ==" not in requirement
        for requirement in core_requires
    )
    assert "nexuml-library; extra == 'library'" in core_requires
    assert any(requirement.startswith("nexuml>=0.2") for requirement in library_requires)

    forbidden = ("rapids", "numba", "psutil", "huggingface-hub", "ffmpeg", "einops", "omegaconf")
    assert not any(
        requirement.lower().startswith(forbidden)
        for requirement in core_requires + library_requires
    )

    all_requires = [
        requirement.lower() for requirement in core_requires if "extra == 'all'" in requirement
    ]
    assert not any(
        requirement.startswith(("pytest", "ruff", "ty;", "nvidia-dali"))
        for requirement in all_requires
    )
    assert "ray[default,train]<2.59,>=2.57; extra == 'ray'" in core_requires
