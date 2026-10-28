"""Hermetic Python-profile contracts: fake stages only, no builds, installs or GPU imports."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import tomllib
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "fixtures/physical_ai_python.py"


class PhysicalAIPythonTest(unittest.TestCase):
    def setUp(self):
        namespace = runpy.run_path(str(SCRIPT))
        self.run_matrix = namespace["run_matrix"]
        self.admit = namespace["_admit_probe"]
        self.probe_script = namespace["_PROBE"]
        temporary = tempfile.TemporaryDirectory(prefix="physical-ai-python-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.source = self.base / "uploaded source"
        self.source.mkdir()
        (self.source / "library").mkdir()
        self.uv: dict = {
            "override-dependencies": [
                "numpy==2.3.1; python_version < '3.14'",
                "numpy==2.3.3; python_version == '3.14'",
                {"package": {"name": "native"}, "dependencies": ["foreign"]},
            ],
            "index": [
                {"name": "nvidia-pypi", "url": "https://pypi.nvidia.com", "explicit": False},
                {"name": "pypi", "url": "https://pypi.org/simple", "default": True},
            ],
        }
        self.write_source_project()
        (self.source / "library/pyproject.toml").write_text('[project]\nversion="0.2.1"\n')
        self.version = "3.13"
        self.failed_stage = None
        self.probe_change = None
        self.wheel_mode = None
        self.new_attempt()
        self.env = {
            "PYTHONPATH": "/foreign/source",
            "PYTEST_ADDOPTS": "-k hidden-selector",
            "PYTEST_PLUGINS": "foreign_plugin",
            "UV_PROJECT_ENVIRONMENT": "/shared/environment",
            "VIRTUAL_ENV": "/worker/environment",
            "UV_PYTHON_INSTALL_DIR": "/shared/pythons",
            "UV_CACHE_DIR": "/shared/cache",
            "UV_LINK_MODE": "hardlink",
            "UV_HTTP_TIMEOUT": "600",
            "UV_CONCURRENT_DOWNLOADS": "4",
            "CUDA_VISIBLE_DEVICES": "GPU-job-owned",
            "TMPDIR": "/foreign/tmp",
            "COVERAGE_FILE": "/foreign/.coverage",
            "HTTP_PROXY": "http://parent-proxy",
            "PARENT_TOKEN": "not-in-receipt",
            **{
                f"NEXUML_RUN_{tier}_TESTS": "1"
                for tier in ("SIMULATOR", "SYSTEM", "HARDWARE", "DATA", "RESOLVER")
            },
        }

    def write_source_project(self):
        text = '[project]\nversion="0.2.1"\n[tool.uv]\n'
        text += 'default-groups = ["native-sources"]\n'
        text += "environments = [\"python_version == '3.12'\"]\n"
        text += "override-dependencies = [\n"
        for item in self.uv["override-dependencies"]:
            if isinstance(item, str):
                text += json.dumps(item) + ",\n"
            else:
                text += '{package = {name = "native"}, dependencies = ["foreign"]},\n'
        text += "]\n"
        for index in self.uv["index"]:
            text += "[[tool.uv.index]]\n"
            text += "\n".join(key + " = " + json.dumps(value) for key, value in index.items())
            text += "\n"
        text += "[tool.uv.sources]\nnexuml-library = {workspace = true}\n"
        text += '[tool.uv.workspace]\nmembers = ["library"]\n'
        (self.source / "pyproject.toml").write_text(text)

    def new_attempt(self):
        self.root = Path(tempfile.mkdtemp(prefix="attempt with spaces; $()", dir=self.base))
        self.scratch = Path(tempfile.mkdtemp(prefix="scratch ", dir=self.base))
        self.venv = self.scratch / ".venv"
        self.site = self.venv / "lib/site-packages"
        self.site.mkdir(parents=True)
        for module in ("nexuml", "nexuml_library", "torch", "numpy", "tensordict"):
            (self.site / module).mkdir()
            (self.site / module / "__init__.py").write_text("# inert fixture\n")
        self.calls = []
        self.record: dict = {"gates": dict.fromkeys("CVLRD", "PASS")}

    def provenance(self):
        wheels = {item["package"]: item for item in self.record["python_matrix"]["wheels"]}
        versions = {
            "torch": "2.14.1+cu130",
            "numpy": {"3.13": "2.3.1", "3.14": "2.3.3"}[self.version],
            "tensordict": "0.14.0",
        }
        return {
            "python_version": self.version + ".9",
            "prefix": str(self.venv),
            "executable": str(self.venv / "bin/python"),
            "sys_path": ["/usr/lib/python" + self.version, str(self.site)],
            "modules": {
                module: {
                    "name": module.replace("_", "-"),
                    "version": versions.get(module, "0.2.1"),
                    "origin": str(self.site / module / "__init__.py"),
                    "distribution_origin": str(self.site / module / "__init__.py"),
                    "distribution_root": str(self.site),
                    "editable": False,
                    "wheel_url": Path(wheels[module.replace("_", "-")]["path"]).as_uri()
                    if module.replace("_", "-") in wheels
                    else None,
                }
                for module in ("nexuml", "nexuml_library", "torch", "numpy", "tensordict")
            },
            "precision": {"default_dtype": "torch.float32", "float32_matmul_precision": "highest"},
            "cuda": {
                "visible_devices": self.env["CUDA_VISIBLE_DEVICES"],
                "available": True,
                "device_count": 1,
                "torch_cuda_version": "13.0",
                "ones": 1.0,
            },
        }

    def stage(self, record, name, argv, env):
        self.assertIs(record, self.record)
        self.calls.append((name, list(argv), dict(env)))
        if name == self.failed_stage:
            raise RuntimeError("original stage failure")
        if name.startswith("python-build-"):
            output = Path(argv[-1])
            output.mkdir(parents=True)
            if self.wheel_mode != "missing":
                wheel = output / (argv[3].replace("-", "_") + "-0.2.1-py3-none-any.whl")
                wheel.write_bytes(argv[3].encode())
                if self.wheel_mode == "duplicate":
                    (output / "stale-0-py3-none-any.whl").write_bytes(b"preserve stale evidence")
                if self.wheel_mode == "native":
                    wheel.rename(output / "native-0-cp313-cp313-linux_x86_64.whl")
                if self.wheel_mode == "symlink":
                    wheel.unlink()
                    wheel.symlink_to(self.source / "pyproject.toml")
        if name == "python-lock":
            (self.scratch / "uv.lock").write_text("version = 1\nrevision = 3\n")
        if name == "python-probe":
            probe = self.provenance()
            if self.probe_change:
                self.probe_change(probe)
            Path(argv[4]).write_text(json.dumps(probe))

    def run_profile(self):
        with (
            contextlib.chdir(self.source),
            mock.patch("tempfile.mkdtemp", return_value=str(self.scratch)) as scratch,
            mock.patch("subprocess.Popen", side_effect=AssertionError("no local launches")),
        ):
            self.run_matrix(self.record, self.root, self.env, self.stage, self.version)
        scratch.assert_called_once_with(prefix=f"nexuml-python-{self.version}-", dir="/tmp/ray")

    def test_import_is_inert_stdlib_only(self):
        compile(self.probe_script, "<python-matrix-probe>", "exec")
        original = __import__
        imports = []

        def capture(name, *args, **kwargs):
            imports.append(name)
            return original(name, *args, **kwargs)

        with (
            mock.patch("builtins.__import__", side_effect=capture),
            mock.patch("tempfile.mkdtemp", side_effect=AssertionError("inert import")),
            mock.patch("subprocess.Popen", side_effect=AssertionError("inert import")),
        ):
            runpy.run_path(str(SCRIPT))
        self.assertTrue(all(name.split(".")[0] in sys.stdlib_module_names for name in imports))

    def test_both_versions_fixed_commands_private_environment_and_retained_identity(self):
        before = self.env.copy()
        parent = os.environ.copy()
        for version in ("3.13", "3.14"):
            with self.subTest(version=version):
                self.version = version
                self.new_attempt()
                self.run_profile()
                matrix = self.record["python_matrix"]
                self.assertEqual(self.env, before)
                self.assertEqual(dict(os.environ), parent)
                self.assertEqual(
                    [call[0] for call in self.calls],
                    [
                        "python-build-nexuml",
                        "python-build-nexuml-library",
                        "python-lock",
                        "python-sync",
                        "python-probe",
                        "python-pytest",
                    ],
                )
                commands = {name: argv for name, argv, _ in self.calls}
                for package in ("nexuml", "nexuml-library"):
                    self.assertEqual(
                        commands["python-build-" + package],
                        [
                            "uv",
                            "build",
                            "--package",
                            package,
                            "--wheel",
                            "--out-dir",
                            str(self.root / "matrix-wheels" / package),
                        ],
                    )
                self.assertEqual(
                    commands["python-lock"],
                    ["uv", "lock", "--project", str(self.scratch), "--python", version],
                )
                self.assertEqual(
                    commands["python-sync"],
                    ["uv", "sync", "--locked", "--project", str(self.scratch), "--python", version],
                )
                python = str(self.venv / "bin/python")
                self.assertEqual(
                    commands["python-probe"],
                    [
                        python,
                        "-I",
                        "-c",
                        self.probe_script,
                        str(self.root / "python-matrix-probe.json"),
                    ],
                )
                self.assertEqual(
                    commands["python-pytest"],
                    [
                        python,
                        "-m",
                        "pytest",
                        "tests",
                        "--cov=nexuml",
                        "--cov=nexuml_library",
                        "--cov-report=term-missing",
                        "--cov-report=xml:" + str(self.root / "python-matrix-coverage.xml"),
                        "-q",
                        "--tb=short",
                        "-p",
                        "no:cacheprovider",
                        "--basetemp",
                        str(self.scratch / "pytest"),
                    ],
                )
                for _, _, private in self.calls:
                    self.assertEqual(
                        private,
                        {
                            **{
                                key: value
                                for key, value in before.items()
                                if key not in {"PYTHONPATH", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"}
                            },
                            "UV_PROJECT_ENVIRONMENT": str(self.venv),
                            "VIRTUAL_ENV": str(self.venv),
                            "UV_PYTHON_INSTALL_DIR": str(self.scratch / "pythons"),
                            "TMPDIR": str(self.scratch),
                            "COVERAGE_FILE": str(self.scratch / ".coverage"),
                            **{
                                f"NEXUML_RUN_{tier}_TESTS": "0"
                                for tier in ("SIMULATOR", "SYSTEM", "HARDWARE", "DATA", "RESOLVER")
                            },
                        },
                    )
                project = tomllib.loads((self.scratch / "pyproject.toml").read_text())
                self.assertEqual(
                    project["project"],
                    {
                        "name": "qualification",
                        "version": "0",
                        "requires-python": "==" + version + ".*",
                        "dependencies": [
                            "nexuml[dev,all,dali] @ " + Path(matrix["wheels"][0]["path"]).as_uri(),
                            "nexuml-library[all] @ " + Path(matrix["wheels"][1]["path"]).as_uri(),
                            "torch==2.14.1",
                            "numpy==" + {"3.13": "2.3.1", "3.14": "2.3.3"}[version],
                        ],
                    },
                )
                self.assertEqual(
                    project["tool"]["uv"],
                    {
                        "override-dependencies": self.uv["override-dependencies"][:2],
                        "index": self.uv["index"],
                    },
                )
                for key, scratch_name in (("lock", "uv.lock"), ("pyproject", "pyproject.toml")):
                    retained = Path(matrix[key + "_path"])
                    self.assertEqual(
                        retained.read_bytes(), (self.scratch / scratch_name).read_bytes()
                    )
                    self.assertEqual(
                        matrix[key + "_sha256"], hashlib.sha256(retained.read_bytes()).hexdigest()
                    )
                for wheel in matrix["wheels"]:
                    self.assertEqual(
                        wheel["sha256"],
                        hashlib.sha256(Path(wheel["path"]).read_bytes()).hexdigest(),
                    )
                self.assertEqual(matrix["actual_python_version"], version + ".9")
                self.assertEqual(matrix["probe_gate"], "PASS")
                self.assertEqual(matrix["ordinary_gate"], "PASS")
                self.assertIs(matrix["native_proof"], False)
                self.assertEqual(set(self.record["gates"].values()), {"UNVERIFIED"})
                self.assertNotIn("not-in-receipt", json.dumps(self.record))

    def test_unsupported_versions_rejected_before_any_work(self):
        for version in ("3.12", "3.13.1", "3.14t", "3.15", "--help", "3.13; touch bad", None):
            with (
                self.subTest(version=version),
                mock.patch("tempfile.mkdtemp", side_effect=AssertionError("no scratch")),
                self.assertRaisesRegex(ValueError, "only 3.13 or 3.14"),
            ):
                self.run_matrix(self.record, self.root, self.env, self.stage, version)
        self.assertEqual(self.calls, [])
        self.assertNotIn("python_matrix", self.record)

    def test_every_failed_stage_preserves_failure_and_stops(self):
        names = [
            "python-build-nexuml",
            "python-build-nexuml-library",
            "python-lock",
            "python-sync",
            "python-probe",
            "python-pytest",
        ]
        for index, name in enumerate(names):
            with self.subTest(stage=name):
                self.new_attempt()
                self.failed_stage = name
                with self.assertRaisesRegex(RuntimeError, "original stage failure"):
                    self.run_profile()
                self.assertEqual([call[0] for call in self.calls], names[: index + 1])
                self.assertEqual(self.record["python_matrix"]["ordinary_gate"], "UNVERIFIED")
                self.assertEqual(set(self.record["gates"].values()), {"UNVERIFIED"})

    def test_missing_duplicate_native_or_symlinked_wheels_stop_without_deletion(self):
        for mode in ("missing", "duplicate", "native", "symlink"):
            with self.subTest(mode=mode):
                self.new_attempt()
                self.wheel_mode = mode
                with self.assertRaisesRegex(AssertionError, "wheel"):
                    self.run_profile()
                self.assertEqual(len(self.calls), 1)
                if mode == "duplicate":
                    stale = self.root / "matrix-wheels/nexuml/stale-0-py3-none-any.whl"
                    self.assertEqual(stale.read_bytes(), b"preserve stale evidence")

    def test_probe_rejects_wrong_interpreter_origins_identity_versions_and_cuda(self):
        self.run_profile()
        valid = self.provenance()
        wheels = self.record["python_matrix"]["wheels"]

        def admit(probe):
            self.admit(
                probe,
                self.venv,
                self.source,
                wheels,
                self.version,
                self.env["CUDA_VISIBLE_DEVICES"],
            )

        admit(valid)
        for field, value in (
            ("prefix", str(self.base)),
            ("executable", sys.executable),
            ("python_version", "3.12.99"),
            ("sys_path", [str(self.source / "src")]),
        ):
            with self.subTest(field=field):
                probe = copy.deepcopy(valid)
                probe[field] = value
                with self.assertRaises(AssertionError):
                    admit(probe)
        for module in valid["modules"]:
            for field, value in (
                ("name", "foreign"),
                ("editable", True),
                ("origin", str(self.source / "pyproject.toml")),
                ("distribution_origin", str(self.source / "pyproject.toml")),
                ("distribution_root", str(self.source)),
            ):
                with self.subTest(module=module, field=field):
                    probe = copy.deepcopy(valid)
                    probe["modules"][module][field] = value
                    with self.assertRaises(AssertionError):
                        admit(probe)
        for module, field, value in (
            ("nexuml", "wheel_url", "file:///foreign.whl"),
            ("nexuml_library", "version", "0.0.0"),
            ("torch", "version", "2.11.0"),
            ("numpy", "version", "2.2.0"),
        ):
            with self.subTest(module=module, field=field):
                probe = copy.deepcopy(valid)
                probe["modules"][module][field] = value
                with self.assertRaises(AssertionError):
                    admit(probe)
        for field, value in (
            ("available", False),
            ("visible_devices", "foreign"),
            ("device_count", 2),
            ("ones", None),
            ("ones", 0),
        ):
            with self.subTest(cuda=field):
                probe = copy.deepcopy(valid)
                probe["cuda"][field] = value
                with self.assertRaises(AssertionError):
                    admit(probe)
        escaped = self.site / "escape.py"
        escaped.symlink_to(self.source / "pyproject.toml")
        probe = copy.deepcopy(valid)
        probe["modules"]["nexuml"]["origin"] = str(escaped)
        with self.assertRaises(AssertionError):
            admit(probe)

    def test_bad_probe_prevents_ordinary_gate(self):
        self.probe_change = lambda value: value["modules"]["nexuml"].update(editable=True)
        with self.assertRaisesRegex(AssertionError, "editable"):
            self.run_profile()
        self.assertEqual(self.calls[-1][0], "python-probe")
        matrix = self.record["python_matrix"]
        self.assertEqual(matrix["probe_gate"], "UNVERIFIED")
        self.assertEqual(matrix["ordinary_gate"], "UNVERIFIED")
        self.assertTrue(matrix["probe"]["modules"]["nexuml"]["editable"])


if __name__ == "__main__":
    unittest.main()
