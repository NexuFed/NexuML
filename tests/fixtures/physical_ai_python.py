"""Fixed ordinary Python support profile, invoked only inside the outer worker guard.

Import is inert and stdlib-only. The caller owns admission, budgets and processes;
this source/runtime gate is neither native nor installed-distribution acceptance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import tomllib
from urllib.parse import urlsplit


_PROBE = """
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

result = {'python_version': '.'.join(map(str, sys.version_info[:3])),
          'prefix': sys.prefix, 'executable': sys.executable,
          'sys_path': sys.path, 'modules': {}}
for module, name in (('nexuml', 'nexuml'), ('nexuml_library', 'nexuml-library'),
                     ('torch', 'torch'), ('numpy', 'numpy'), ('tensordict', 'tensordict')):
    imported = importlib.import_module(module)
    distribution = importlib.metadata.distribution(name)
    direct = json.loads(distribution.read_text('direct_url.json') or '{}')
    result['modules'][module] = {
        'name': distribution.metadata['Name'], 'version': distribution.version,
        'origin': str(Path(imported.__file__).resolve(strict=True)),
        'distribution_root': str(Path(distribution.locate_file('')).resolve(strict=True)),
        'distribution_origin': str(Path(distribution.locate_file(
            module + '/__init__.py')).resolve(strict=True)),
        'editable': direct.get('dir_info', {}).get('editable', False),
        'wheel_url': direct.get('url'),
    }
import torch
result['precision'] = {'default_dtype': str(torch.get_default_dtype()),
                       'float32_matmul_precision': torch.get_float32_matmul_precision()}
result['cuda'] = {'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                  'available': torch.cuda.is_available(),
                  'device_count': torch.cuda.device_count(),
                  'torch_cuda_version': torch.version.cuda, 'ones': None}
if result['cuda']['available']:
    result['cuda']['ones'] = torch.ones(1, device='cuda:0').item()
    torch.cuda.synchronize()
with Path(sys.argv[1]).open('x') as output:
    json.dump(result, output, indent=2, allow_nan=False)
"""


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _admit_probe(probe, venv, source, wheels, python_version, visible_devices):
    """Reject a foreign interpreter, source imports, wrong wheels or unusable CUDA."""
    owned = venv.resolve(strict=True)
    if (
        Path(probe["prefix"]).resolve(strict=True) != owned
        or Path(probe["executable"]).absolute() != venv / "bin/python"
        or probe["python_version"].rsplit(".", 1)[0] != python_version
    ):
        raise AssertionError("probe must use the owned requested Python interpreter")
    if any(Path(entry).resolve().is_relative_to(source) for entry in probe["sys_path"]):
        raise AssertionError("uploaded source remains on the isolated probe import path")
    modules = {"nexuml", "nexuml_library", "torch", "numpy", "tensordict"}
    if set(probe["modules"]) != modules:
        raise AssertionError("probe must identify both wheels and the ordinary runtime stack")
    artifacts = {item["package"]: item for item in wheels}
    for module, identity in probe["modules"].items():
        name = module.replace("_", "-")
        if identity["name"] != name or identity["editable"] is not False:
            raise AssertionError("probe distribution identity is foreign or editable")
        for key in ("origin", "distribution_origin", "distribution_root"):
            path = Path(identity[key]).resolve(strict=True)
            if path == owned or not path.is_relative_to(owned):
                raise AssertionError("probe import/distribution origin is outside the owned venv")
        if identity["origin"] != identity["distribution_origin"]:
            raise AssertionError("probe import is not owned by its installed distribution")
        if name in artifacts:
            wheel = artifacts[name]
            if (
                identity["version"] != wheel["version"]
                or identity["wheel_url"] != Path(wheel["path"]).as_uri()
            ):
                raise AssertionError("probe application identity differs from the built wheel")
    if (
        probe["modules"]["torch"]["version"].split("+", 1)[0] != "2.14.1"
        or probe["modules"]["numpy"]["version"]
        != {"3.13": "2.3.1", "3.14": "2.3.3"}[python_version]
    ):
        raise AssertionError("probe runtime versions differ from the fixed CI stack")
    cuda = probe["cuda"]
    if (
        cuda["visible_devices"] != visible_devices
        or cuda["available"] is not True
        or cuda["device_count"] != 1
        or cuda["ones"] != 1
    ):
        raise AssertionError("probe did not validate the parent's one job-visible CUDA device")


def run_matrix(record: dict, root: Path, env: dict, stage, python_version: str):
    """Build generic wheels and run the fixed ordinary suite in a new private environment."""
    if python_version not in ("3.13", "3.14"):
        raise ValueError("Python matrix accepts only 3.13 or 3.14")
    source = Path.cwd().resolve(strict=True)
    scratch = Path(tempfile.mkdtemp(prefix=f"nexuml-python-{python_version}-", dir="/tmp/ray"))
    venv = scratch / ".venv"
    python = str(venv / "bin/python")
    private = {
        key: value
        for key, value in env.items()
        if key not in {"PYTHONPATH", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"}
    }
    private.update(
        UV_PROJECT_ENVIRONMENT=str(venv),
        VIRTUAL_ENV=str(venv),
        UV_PYTHON_INSTALL_DIR=str(scratch / "pythons"),
        TMPDIR=str(scratch),
        COVERAGE_FILE=str(scratch / ".coverage"),
    )
    for tier in ("SIMULATOR", "SYSTEM", "HARDWARE", "DATA", "RESOLVER"):
        private[f"NEXUML_RUN_{tier}_TESTS"] = "0"
    record["gates"].update(dict.fromkeys("CVLRD", "UNVERIFIED"))
    matrix: dict
    matrix = record["python_matrix"] = {
        "python_version": python_version,
        "actual_python_version": None,
        "scratch_path": str(scratch),
        "venv_path": str(venv),
        "wheels": [],
        "lock_sha256": None,
        "pyproject_sha256": None,
        "probe": None,
        "probe_gate": "UNVERIFIED",
        "ordinary_gate": "UNVERIFIED",
        "native_proof": False,
    }
    for package, project in (
        ("nexuml", source / "pyproject.toml"),
        ("nexuml-library", source / "library/pyproject.toml"),
    ):
        output = root / "matrix-wheels" / package
        stage(
            record,
            "python-build-" + package,
            ["uv", "build", "--package", package, "--wheel", "--out-dir", str(output)],
            private,
        )
        matches = list(output.glob("*.whl"))
        if len(matches) != 1 or matches[0].is_symlink() or not matches[0].is_file():
            raise AssertionError("expected exactly one current generic wheel per package")
        wheel = matches[0]
        if not wheel.name.endswith("-py3-none-any.whl"):
            raise AssertionError("Python matrix requires generic wheels")
        matrix["wheels"].append(
            {
                "package": package,
                "version": tomllib.loads(project.read_text())["project"]["version"],
                "path": str(wheel),
                "sha256": _sha256(wheel),
                "bytes": wheel.stat().st_size,
            }
        )

    dependencies = [
        f"{wheel['package']}[{extras}] @ {Path(wheel['path']).as_uri()}"
        for wheel, extras in zip(matrix["wheels"], ("dev,all,dali", "all"), strict=True)
    ]
    dependencies += [
        "torch==2.14.1",
        "numpy==" + {"3.13": "2.3.1", "3.14": "2.3.3"}[python_version],
    ]
    uv = tomllib.loads((source / "pyproject.toml").read_text())["tool"]["uv"]
    overrides = [item for item in uv.get("override-dependencies", []) if isinstance(item, str)]
    lines = [
        "[project]",
        'name = "qualification"',
        'version = "0"',
        "requires-python = " + json.dumps("==" + python_version + ".*"),
        "dependencies = " + json.dumps(dependencies, ensure_ascii=False),
        "[tool.uv]",
        "override-dependencies = " + json.dumps(overrides, ensure_ascii=False),
    ]
    for index in uv.get("index", []):
        url = urlsplit(index["url"])
        if url.scheme != "https" or not url.hostname or url.username or url.password:
            raise ValueError("Python matrix copies only public HTTPS index definitions")
        lines.append("[[tool.uv.index]]")
        for key, value in index.items():
            if not isinstance(value, (str, bool)):
                raise ValueError("unsupported public index field; review the fixed profile")
            lines.append(json.dumps(key) + " = " + json.dumps(value, ensure_ascii=False))
    project_path = scratch / "pyproject.toml"
    with project_path.open("x") as output:
        output.write("\n".join(lines) + "\n")
    retained_project = root / "python-matrix-pyproject.toml"
    with retained_project.open("xb") as output:
        output.write(project_path.read_bytes())
    matrix["pyproject_path"] = str(retained_project)
    matrix["pyproject_sha256"] = _sha256(retained_project)
    stage(
        record,
        "python-lock",
        ["uv", "lock", "--project", str(scratch), "--python", python_version],
        private,
    )
    retained_lock = root / "python-matrix-uv.lock"
    with retained_lock.open("xb") as output:
        output.write((scratch / "uv.lock").read_bytes())
    matrix["lock_path"] = str(retained_lock)
    matrix["lock_sha256"] = _sha256(retained_lock)
    stage(
        record,
        "python-sync",
        ["uv", "sync", "--locked", "--project", str(scratch), "--python", python_version],
        private,
    )
    probe_path = root / "python-matrix-probe.json"
    stage(record, "python-probe", [python, "-I", "-c", _PROBE, str(probe_path)], private)
    probe = matrix["probe"] = json.loads(probe_path.read_text())
    matrix["actual_python_version"] = probe["python_version"]
    _admit_probe(
        probe, venv, source, matrix["wheels"], python_version, env.get("CUDA_VISIBLE_DEVICES")
    )
    matrix["probe_gate"] = "PASS"
    stage(
        record,
        "python-pytest",
        [
            python,
            "-m",
            "pytest",
            "tests",
            "--cov=nexuml",
            "--cov=nexuml_library",
            "--cov-report=term-missing",
            "--cov-report=xml:" + str(root / "python-matrix-coverage.xml"),
            "-q",
            "--tb=short",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(scratch / "pytest"),
        ],
        private,
    )
    matrix["ordinary_gate"] = "PASS"
