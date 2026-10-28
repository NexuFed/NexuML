"""Fixed installed-wheel CPU controls, called only by the guarded Ray worker.

Import is inert and stdlib-only. The caller owns admission, budgets and processes;
this profile never installs into its interpreter or claims the complete D gate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import tomllib


_PROBE = """
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

def verify_wheel(distribution, wheel_path, prefix):
    verified = 0
    with Path(wheel_path).open('rb') as wheel:
        digest = hashlib.file_digest(wheel, 'sha256').hexdigest()
        wheel.seek(0)
        with zipfile.ZipFile(wheel) as archive:
            for entry in archive.infolist():
                relative = PurePosixPath(entry.filename)
                parts = entry.filename.rstrip('/').split('/')
                if (relative.is_absolute() or '\\\\' in entry.filename
                        or any(part in {'', '.', '..'} or part.endswith('.data')
                               for part in parts)):
                    raise AssertionError('unsafe or unsupported wheel entry')
                if entry.is_dir():
                    continue
                if (len(parts) == 2 and parts[0].endswith('.dist-info')
                        and parts[1] == 'RECORD'):
                    continue  # Installers rewrite RECORD; all other wheel bytes must match.
                installed = Path(distribution.locate_file(entry.filename)).resolve(strict=True)
                if not installed.is_relative_to(prefix) or not installed.is_file():
                    raise AssertionError('wheel file is outside the owned environment')
                if installed.read_bytes() != archive.read(entry):
                    raise AssertionError('installed file differs from the exact built wheel')
                verified += 1
    if verified <= 0:
        raise AssertionError('wheel verification matched no installed files')
    return digest, verified

result = {'prefix': sys.prefix, 'executable': sys.executable,
          'sys_path': sys.path, 'modules': {}}
for (module, name), wheel_path in zip(
        (('nexuml', 'nexuml'), ('nexuml_library', 'nexuml-library')), sys.argv[2:], strict=True):
    imported = importlib.import_module(module)
    distribution = importlib.metadata.distribution(name)
    wheel_hash, verified = verify_wheel(
        distribution, wheel_path, Path(sys.prefix).resolve(strict=True))
    direct = json.loads(distribution.read_text('direct_url.json') or '{}')
    archive = direct.get('archive_info', {})
    direct_hash = archive.get('hashes', {}).get('sha256')
    if direct_hash is None and archive.get('hash', '').startswith('sha256='):
        direct_hash = archive['hash'].removeprefix('sha256=')
    result['modules'][module] = {
        'name': distribution.metadata['Name'], 'version': distribution.version,
        'origin': str(Path(imported.__file__).resolve(strict=True)),
        'distribution_root': str(Path(distribution.locate_file('')).resolve(strict=True)),
        'distribution_origin': str(Path(distribution.locate_file(
            module + '/__init__.py')).resolve(strict=True)),
        'editable': direct.get('dir_info', {}).get('editable', False),
        'wheel_sha256': wheel_hash,
        'verified_file_count': verified,
        'direct_url_sha256': direct_hash,
    }
with Path(sys.argv[1]).open('x') as output:
    json.dump(result, output, indent=2, allow_nan=False)
"""


def _admit_installed(provenance, venv, source, artifacts):
    """Reject source, foreign, editable or wrong-wheel evidence before runtime tests."""
    owned = venv.resolve(strict=True)
    if Path(provenance["prefix"]).resolve(strict=True) != owned:
        raise AssertionError("installed probe did not use the private environment")
    if Path(provenance["executable"]).absolute() != venv / "bin/python":
        raise AssertionError("installed probe interpreter differs from the owned interpreter")
    excluded = (source, source / "src", source / "library/src")
    if any(
        Path(entry).resolve().is_relative_to(directory)
        for entry in provenance["sys_path"]
        for directory in excluded
    ):
        raise AssertionError("uploaded source roots remain on the installed import path")
    if set(provenance["modules"]) != {"nexuml", "nexuml_library"}:
        raise AssertionError("installed probe must identify both application distributions")
    wheels = {item["package"]: item for item in artifacts if item["kind"] == "wheel"}
    for module, name in (("nexuml", "nexuml"), ("nexuml_library", "nexuml-library")):
        identity = provenance["modules"][module]
        wheel = wheels[name]
        if (
            identity["name"] != name
            or identity["version"] != wheel["version"]
            or identity["editable"] is not False
            or identity["wheel_sha256"] != wheel["sha256"]
            or type(identity.get("verified_file_count")) is not int
            or identity["verified_file_count"] <= 0
        ):
            raise AssertionError("installed distribution does not identify the exact built wheel")
        for key in ("origin", "distribution_root", "distribution_origin"):
            path = Path(identity[key]).resolve(strict=True)
            if not path.is_relative_to(owned) or path == owned:
                raise AssertionError("application import/distribution origin is outside owned venv")
        if identity["origin"] != identity["distribution_origin"]:
            raise AssertionError("application import is not owned by the installed distribution")


def run_package(record: dict, root: Path, env: dict, stage):
    """Build current distributions, then test exact wheels in a fresh locked environment."""
    source = Path.cwd().resolve(strict=True)
    scratch = Path(tempfile.mkdtemp(prefix="nexuml-distribution-", dir="/tmp/ray"))
    venv = scratch / "venv"
    python = str(venv / "bin/python")
    private = {
        key: value
        for key, value in env.items()
        if key
        not in {
            "PYTHONPATH",
            "PYTEST_ADDOPTS",
            "PYTEST_PLUGINS",
        }
    }
    private.update(
        UV_PROJECT_ENVIRONMENT=str(venv),
        VIRTUAL_ENV=str(venv),
        CUDA_VISIBLE_DEVICES="",
        TMPDIR=str(scratch),
        NEXUML_LOGS_ROOT=str(root / "logs"),
        NEXUML_DIST_ROOT=str(root / "dist"),
        NEXUML_INSTALLED_DIST_TEST="1",
        NEXUML_RUN_RESOLVER_TESTS="0",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    for tier in ("SIMULATOR", "SYSTEM", "HARDWARE", "DATA"):
        private[f"NEXUML_RUN_{tier}_TESTS"] = "0"
    record["gates"].update({key: "UNVERIFIED" for key in "DRVL"})
    record["gate_scope"]["D"] = (
        "installed wheels, synthetic supervised evaluation and actor controls only; "
        "approved SO-101 data/local-fit and full distribution gate UNVERIFIED"
    )
    record["package"] = {
        "scratch_path": str(scratch),
        "venv_path": str(venv),
        "artifacts": [],
        "provenance": None,
        "so101_approved_data": "UNVERIFIED",
    }
    checks = record["checks"]
    checks.update(
        dict.fromkeys(
            (
                "distribution_artifacts",
                "installed_import_origins",
                "installed_library_distribution",
                "installed_supervised_evaluation",
                "installed_actor_inference",
            ),
            "UNVERIFIED",
        )
    )
    for package in ("nexuml", "nexuml-library"):
        stage(
            record,
            "build-" + package,
            [
                "uv",
                "build",
                "--package",
                package,
                "--out-dir",
                str(root / "dist" / package),
            ],
            private,
        )

    pytest = [
        "-I",
        "-m",
        "pytest",
        "-c",
        "/dev/null",
        "--rootdir=tests/packaging",
        "--confcutdir=tests/packaging",
        "--import-mode=importlib",
        "-o",
        "pythonpath=",
        "-p",
        "no:cacheprovider",
        "-q",
        "--tb=short",
    ]
    stage(
        record,
        "distribution-artifacts",
        [
            sys.executable,
            *pytest,
            "tests/packaging/test_distribution_artifacts.py",
            "--basetemp",
            str(scratch / "artifacts-test"),
        ],
        private,
    )
    artifacts = record["package"]["artifacts"]
    for package, project in (
        ("nexuml", "pyproject.toml"),
        ("nexuml-library", "library/pyproject.toml"),
    ):
        version = tomllib.loads(Path(project).read_text())["project"]["version"]
        for kind, pattern in (("wheel", "*.whl"), ("sdist", "*.tar.gz")):
            matches = list((root / "dist" / package).glob(pattern))
            if len(matches) != 1 or matches[0].is_symlink() or not matches[0].is_file():
                raise AssertionError("expected exactly one owned current wheel and sdist")
            path = matches[0]
            with path.open("rb") as output:
                digest = hashlib.file_digest(output, "sha256").hexdigest()
            artifacts.append(
                {
                    "package": package,
                    "version": version,
                    "kind": kind,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                }
            )
    checks["distribution_artifacts"] = "PASS"
    stage(
        record,
        "installed-locked-sync",
        [
            "uv",
            "sync",
            "--locked",
            "--no-editable",
            "--python",
            sys.executable,
            "--extra",
            "library",
            "--extra",
            "ray",
            "--extra",
            "dev",
            "--extra",
            "api",
            "--extra",
            "reinforcement",
        ],
        private,
    )
    stage(
        record,
        "installed-exact-wheels",
        [
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "--no-deps",
            "--reinstall",
            *(item["path"] for item in artifacts if item["kind"] == "wheel"),
        ],
        private,
    )
    provenance_path = root / "installed-provenance.json"
    stage(
        record,
        "installed-import-origins",
        [
            python,
            "-I",
            "-c",
            _PROBE,
            str(provenance_path),
            *(item["path"] for item in artifacts if item["kind"] == "wheel"),
        ],
        private,
    )
    provenance = json.loads(provenance_path.read_text())
    record["package"]["provenance"] = provenance
    _admit_installed(provenance, venv, source, artifacts)
    checks["installed_import_origins"] = "PASS"
    stage(
        record,
        "installed-library",
        [
            python,
            *pytest,
            "tests/packaging/test_installed_distributions.py",
            "-k",
            "library",
            "--basetemp",
            str(scratch / "library-test"),
        ],
        private,
    )
    checks["installed_library_distribution"] = "PASS"
    stage(
        record,
        "installed-supervised",
        [
            python,
            *pytest,
            "tests/core/test_model_evaluation_cli.py",
            "-k",
            "not policy and not explicit_cuda",
            "--basetemp",
            str(scratch / "supervised-test"),
        ],
        private,
    )
    checks["installed_supervised_evaluation"] = "PASS"
    stage(
        record,
        "installed-actor",
        [
            python,
            *pytest,
            "tests/core/test_multiagent_learning.py::"
            "test_registered_marl_actor_packages_run_cleanly_without_learner_or_native_imports",
            "tests/core/test_policy.py::"
            "test_preserved_batch_layout_keeps_temporal_computation_and_package_parity",
            "tests/core/test_policy.py::test_complete_policy_checks_contract_and_preserves_observations",
            "--basetemp",
            str(scratch / "actor-test"),
        ],
        private,
    )
    checks["installed_actor_inference"] = "PASS"
