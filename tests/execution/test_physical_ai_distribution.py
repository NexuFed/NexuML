"""Hermetic package-profile contracts; never build, install or run application code."""

from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import sys
import sysconfig
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "fixtures/physical_ai_distribution.py"


class PhysicalAIDistributionTest(unittest.TestCase):
    def setUp(self):
        namespace = runpy.run_path(str(SCRIPT))
        self.run_package = namespace["run_package"]
        self.admit = namespace["_admit_installed"]
        self.probe = namespace["_PROBE"]
        temporary = tempfile.TemporaryDirectory(prefix="physical-ai-distribution-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.source = self.base / "uploaded source"
        self.source.mkdir()
        (self.source / "library").mkdir()
        for name in ("pyproject.toml", "library/pyproject.toml"):
            (self.source / name).write_text('[project]\nversion="0.2.1"\n')
        self.root = self.base / "attempt with spaces; $(not-a-command)"
        self.root.mkdir()
        self.scratch = self.base / "owned scratch"
        self.scratch.mkdir()
        self.venv = self.scratch / "venv"
        self.site = self.venv / "lib/python3.12/site-packages"
        self.site.mkdir(parents=True)
        for module in ("nexuml", "nexuml_library"):
            (self.site / module).mkdir()
            (self.site / module / "__init__.py").write_text("# inert fake distribution\n")
        self.record: dict = {
            "gates": dict.fromkeys("CVLRD", "UNVERIFIED"),
            "gate_scope": {},
            "checks": {},
        }
        self.env = {
            "PYTHONPATH": "/foreign/source",
            "VIRTUAL_ENV": "/worker/environment",
            "UV_PROJECT_ENVIRONMENT": "/shared/environment",
            "CUDA_VISIBLE_DEVICES": "7",
            "UV_CACHE_DIR": "/shared/cache",
            "HTTP_PROXY": "http://private-proxy",
            "PYTEST_ADDOPTS": "--override-ini=pythonpath=/foreign/source",
            "PYTEST_PLUGINS": "foreign_plugin",
            "NEXUML_RUN_DATA_TESTS": "1",
            "NEXUML_RUN_RESOLVER_TESTS": "1",
            "PARENT_TOKEN": "never-in-receipt",
        }
        self.calls = []
        self.failed_stage = None
        self.provenance_change = None

    def provenance(self):
        return {
            "prefix": str(self.venv),
            "executable": str(self.venv / "bin/python"),
            "sys_path": ["/usr/lib/python3.12", str(self.site)],
            "modules": {
                module: {
                    "name": name,
                    "version": "0.2.1",
                    "editable": False,
                    "origin": str(self.site / module / "__init__.py"),
                    "distribution_origin": str(self.site / module / "__init__.py"),
                    "distribution_root": str(self.site),
                    "wheel_sha256": next(
                        item["sha256"]
                        for item in self.record["package"]["artifacts"]
                        if item["package"] == name and item["kind"] == "wheel"
                    ),
                    "verified_file_count": 1,
                    "direct_url_sha256": None,
                }
                for module, name in (("nexuml", "nexuml"), ("nexuml_library", "nexuml-library"))
            },
        }

    def stage(self, record, name, argv, env):
        self.assertIs(record, self.record)
        self.calls.append((name, list(argv), dict(env)))
        if name == self.failed_stage:
            raise AssertionError("fake stage failure")
        if name.startswith("build-"):
            package = argv[3]
            output = Path(argv[5])
            output.mkdir(parents=True)
            for suffix in ("-py3-none-any.whl", ".tar.gz"):
                (output / (package.replace("-", "_") + "-0.2.1" + suffix)).write_bytes(
                    (package + suffix).encode()
                )
        if name == "installed-import-origins":
            provenance = self.provenance()
            if self.provenance_change:
                self.provenance_change(provenance)
            Path(argv[4]).write_text(json.dumps(provenance))

    def run_profile(self):
        with (
            contextlib.chdir(self.source),
            mock.patch("tempfile.mkdtemp", return_value=str(self.scratch)) as scratch,
            mock.patch("subprocess.Popen", side_effect=AssertionError("local launch forbidden")),
        ):
            self.run_package(self.record, self.root, self.env, self.stage)
        scratch.assert_called_once_with(prefix="nexuml-distribution-", dir="/tmp/ray")

    def test_import_is_inert_stdlib_only(self):
        original = __import__
        imports = []

        def capture(name, *args, **kwargs):
            imports.append(name)
            return original(name, *args, **kwargs)

        with (
            mock.patch("builtins.__import__", side_effect=capture),
            mock.patch("subprocess.Popen", side_effect=AssertionError("inert import")),
            mock.patch("tempfile.mkdtemp", side_effect=AssertionError("inert import")),
        ):
            runpy.run_path(str(SCRIPT))
        self.assertTrue(all(name.split(".")[0] in sys.stdlib_module_names for name in imports))

    def test_fixed_commands_private_environment_and_truthful_scope(self):
        before = self.env.copy()
        parent = os.environ.copy()
        self.run_profile()
        self.assertEqual(self.env, before)
        self.assertEqual(dict(os.environ), parent)
        self.assertEqual(
            [name for name, _, _ in self.calls],
            [
                "build-nexuml",
                "build-nexuml-library",
                "distribution-artifacts",
                "installed-locked-sync",
                "installed-exact-wheels",
                "installed-import-origins",
                "installed-library",
                "installed-supervised",
                "installed-actor",
            ],
        )
        commands = {name: argv for name, argv, _ in self.calls}
        for package in ("nexuml", "nexuml-library"):
            self.assertEqual(
                commands["build-" + package],
                [
                    "uv",
                    "build",
                    "--package",
                    package,
                    "--out-dir",
                    str(self.root / "dist" / package),
                ],
            )
        self.assertEqual(
            commands["installed-locked-sync"],
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
        )
        wheels = [
            item["path"] for item in self.record["package"]["artifacts"] if item["kind"] == "wheel"
        ]
        python = str(self.venv / "bin/python")
        self.assertEqual(
            commands["installed-exact-wheels"],
            [
                "uv",
                "pip",
                "install",
                "--python",
                python,
                "--no-deps",
                "--reinstall",
                *wheels,
            ],
        )
        self.assertEqual(
            commands["installed-import-origins"][4:],
            [
                str(self.root / "installed-provenance.json"),
                *wheels,
            ],
        )
        for name, argv, private in self.calls:
            self.assertNotIn("PYTHONPATH", private)
            self.assertNotIn("PYTEST_ADDOPTS", private)
            self.assertNotIn("PYTEST_PLUGINS", private)
            self.assertEqual(private["UV_PROJECT_ENVIRONMENT"], str(self.venv))
            self.assertEqual(private["VIRTUAL_ENV"], str(self.venv))
            self.assertEqual(private["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(private["TMPDIR"], str(self.scratch))
            self.assertEqual(private["NEXUML_DIST_ROOT"], str(self.root / "dist"))
            self.assertEqual(private["PYTEST_DISABLE_PLUGIN_AUTOLOAD"], "1")
            self.assertEqual(private["NEXUML_RUN_RESOLVER_TESTS"], "0")
            for tier in ("DATA", "SIMULATOR", "SYSTEM", "HARDWARE"):
                self.assertEqual(private[f"NEXUML_RUN_{tier}_TESTS"], "0")
            for key in ("UV_CACHE_DIR", "HTTP_PROXY", "PARENT_TOKEN"):
                self.assertEqual(private[key], before[key])
            if "pytest" in argv:
                self.assertEqual(
                    argv[0], sys.executable if name == "distribution-artifacts" else python
                )
                self.assertEqual(argv[1:6], ["-I", "-m", "pytest", "-c", "/dev/null"])
                self.assertIn("--confcutdir=tests/packaging", argv)
                self.assertIn("--import-mode=importlib", argv)
                self.assertEqual(argv[argv.index("-o") + 1], "pythonpath=")
        self.assertEqual(
            commands["installed-library"][commands["installed-library"].index("-k") + 1], "library"
        )
        actor = " ".join(commands["installed-actor"])
        self.assertIn("test_registered_marl_actor_packages_run_cleanly", actor)
        self.assertIn("test_preserved_batch_layout", actor)
        self.assertIn("test_model_evaluation_cli.py", " ".join(commands["installed-supervised"]))
        for artifact in self.record["package"]["artifacts"]:
            path = Path(artifact["path"])
            self.assertEqual(artifact["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(len(self.record["package"]["artifacts"]), 4)
        self.assertEqual(set(self.record["checks"].values()), {"PASS"})
        self.assertEqual(set(self.record["gates"].values()), {"UNVERIFIED"})
        self.assertEqual(self.record["package"]["so101_approved_data"], "UNVERIFIED")
        self.assertIn("full distribution gate UNVERIFIED", self.record["gate_scope"]["D"])
        self.assertNotIn("never-in-receipt", json.dumps(self.record))

    def test_each_stage_failure_stops_without_later_passes(self):
        self.run_profile()
        names = [name for name, _, _ in self.calls]
        for index, name in enumerate(names):
            with self.subTest(stage=name):
                # Reuse fake build files only; no helper performs real builds or installs.
                with tempfile.TemporaryDirectory(dir=self.base) as attempt:
                    self.root = Path(attempt)
                    self.calls = []
                    self.failed_stage = name
                    self.record = {"gates": {}, "gate_scope": {}, "checks": {}}
                    with self.assertRaisesRegex(AssertionError, "fake stage failure"):
                        self.run_profile()
                    self.assertEqual([call[0] for call in self.calls], names[: index + 1])
                    self.assertNotEqual(
                        self.record["checks"].get("installed_actor_inference"), "PASS"
                    )
                    self.assertEqual(set(self.record["gates"].values()), {"UNVERIFIED"})

    def test_identity_admission_rejects_foreign_source_editable_and_wrong_wheel(self):
        self.run_profile()
        valid = self.provenance()
        artifacts = self.record["package"]["artifacts"]
        self.admit(valid, self.venv, self.source, artifacts)
        for module in valid["modules"]:
            for field, value in (
                ("name", "foreign"),
                ("version", "0.0.0"),
                ("editable", True),
                ("wheel_sha256", "0" * 64),
                ("wheel_sha256", None),
                ("verified_file_count", 0),
                ("verified_file_count", -1),
                ("verified_file_count", None),
                ("verified_file_count", True),
                ("verified_file_count", "1"),
                ("origin", str(self.source / "pyproject.toml")),
                ("distribution_root", str(self.source)),
                ("distribution_origin", str(self.source / "pyproject.toml")),
            ):
                with self.subTest(module=module, field=field):
                    provenance = copy.deepcopy(valid)
                    provenance["modules"][module][field] = value
                    with self.assertRaises(AssertionError):
                        self.admit(provenance, self.venv, self.source, artifacts)
        for field, value in (
            ("prefix", str(self.base)),
            ("executable", sys.executable),
            ("sys_path", [str(self.source / "src"), str(self.site)]),
        ):
            with self.subTest(field=field):
                provenance = copy.deepcopy(valid)
                provenance[field] = value
                with self.assertRaises(AssertionError):
                    self.admit(provenance, self.venv, self.source, artifacts)
        escaped = self.site / "escaped.py"
        escaped.symlink_to(self.source / "pyproject.toml")
        provenance = copy.deepcopy(valid)
        provenance["modules"]["nexuml"]["origin"] = str(escaped)
        with self.assertRaises(AssertionError):
            self.admit(provenance, self.venv, self.source, artifacts)
        provenance = copy.deepcopy(valid)
        core, library = provenance["modules"].values()
        core["wheel_sha256"], library["wheel_sha256"] = (
            library["wheel_sha256"],
            core["wheel_sha256"],
        )
        with self.assertRaises(AssertionError):
            self.admit(provenance, self.venv, self.source, artifacts)

    def test_bad_probe_blocks_runtime_tests(self):
        self.provenance_change = lambda value: value["modules"]["nexuml"].update(editable=True)
        with self.assertRaisesRegex(AssertionError, "exact built wheel"):
            self.run_profile()
        self.assertEqual(self.calls[-1][0], "installed-import-origins")
        self.assertEqual(self.record["checks"]["installed_import_origins"], "UNVERIFIED")

    def execute_probe(self, direct_hash=None):
        expected = self.provenance()
        output = self.root / f"sanitized-{len(list(self.root.glob('sanitized-*')))}.json"
        real_import_module = importlib.import_module

        def import_module(name, package=None):
            if name in {"nexuml", "nexuml_library"}:
                return SimpleNamespace(__file__=str(self.site / name / "__init__.py"))
            return real_import_module(name, package)

        def distribution(name):
            direct = json.dumps(
                {
                    "url": "https://secret:credential@private.invalid/wheel?token=hidden",
                    "archive_info": {"hashes": {"sha256": direct_hash}},
                }
            )
            return SimpleNamespace(
                metadata={"Name": name},
                version="0.2.1",
                read_text=lambda _: direct,
                locate_file=lambda path: self.site / path,
            )

        def wheel_archive(wheel):
            module = Path(wheel.name).name.split("-")[0]
            return self.archive(
                {
                    module + "/__init__.py": (self.site / module / "__init__.py").read_bytes(),
                    module + ".dist-info/RECORD": b"installer-rewritten RECORD must be excluded",
                }
            )

        wheels = [
            item["path"] for item in self.record["package"]["artifacts"] if item["kind"] == "wheel"
        ]
        with (
            mock.patch.object(
                importlib,
                "import_module",
                side_effect=import_module,
            ),
            mock.patch.object(importlib.metadata, "distribution", side_effect=distribution),
            mock.patch.object(sys, "prefix", str(self.venv)),
            mock.patch.object(sys, "executable", str(self.venv / "bin/python")),
            mock.patch.object(sys, "path", expected["sys_path"]),
            mock.patch.object(sys, "argv", ["-c", str(output), *wheels]),
            mock.patch.object(zipfile, "ZipFile", side_effect=wheel_archive),
        ):
            self.assertIs(importlib.import_module("sysconfig"), sysconfig)
            self.assertIs(importlib.import_module(".decoder", package="json"), json.decoder)
            self.assertIn("purelib", sysconfig.get_paths())
            exec(compile(self.probe, "<fixed-distribution-probe>", "exec"), {})
        actual = json.loads(output.read_text())
        for identity in expected["modules"].values():
            identity["direct_url_sha256"] = direct_hash
        self.assertEqual(actual, expected)
        self.assertNotIn("secret", output.read_text())
        self.assertNotIn("token", output.read_text())
        self.admit(actual, self.venv, self.source, self.record["package"]["artifacts"])

    def archive(self, entries):
        archive = mock.MagicMock()
        archive.__enter__.return_value = archive
        archive.infolist.return_value = [zipfile.ZipInfo(name) for name in entries]
        archive.read.side_effect = lambda entry: entries[entry.filename]
        return archive

    def test_probe_retains_sanitized_provenance_with_optional_direct_url_hash(self):
        self.run_profile()
        self.execute_probe()
        self.execute_probe(direct_hash="metadata hash is not wheel identity")

    def test_probe_verifies_all_installed_bytes_and_rejects_unsafe_wheel_entries(self):
        # Execute only the probe's stdlib imports and pure verifier, never application imports.
        tree = ast.parse(self.probe)
        definitions: list[ast.stmt] = [
            node
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
        ]
        namespace = {}
        exec(
            compile(ast.Module(body=definitions, type_ignores=[]), "<wheel-verifier>", "exec"),
            namespace,
        )
        verify = namespace["verify_wheel"]
        wheel = self.root / "inert-wheel-fixture"
        wheel.write_bytes(b"fixed wheel identity")
        distribution = SimpleNamespace(locate_file=lambda name: self.site / name)
        prefix = self.venv.resolve()
        (self.site / "nexuml/resource.txt").write_bytes(b"installed resource")
        entries = {
            "nexuml/__init__.py": (self.site / "nexuml/__init__.py").read_bytes(),
            "nexuml/resource.txt": b"installed resource",
            "nexuml.dist-info/RECORD": b"ignored changed RECORD",
        }
        with mock.patch.object(zipfile, "ZipFile", return_value=self.archive(entries)):
            digest, count = verify(distribution, wheel, prefix)
        self.assertEqual(digest, hashlib.sha256(wheel.read_bytes()).hexdigest())
        self.assertEqual(count, 2)
        mismatched = {**entries, "nexuml/resource.txt": b"different wheel resource"}
        with (
            mock.patch.object(zipfile, "ZipFile", return_value=self.archive(mismatched)),
            self.assertRaisesRegex(AssertionError, "installed file differs"),
        ):
            verify(distribution, wheel, prefix)
        for name in (
            "../escape.py",
            "/absolute.py",
            "nexuml/../escape.py",
            "./relative.py",
            "nexuml//empty.py",
            "nexuml\\backslash.py",
            "nexuml.data/purelib/file.py",
            "nexuml.data/",
        ):
            with (
                self.subTest(entry=name),
                mock.patch.object(zipfile, "ZipFile", return_value=self.archive({name: b"bad"})),
                self.assertRaisesRegex(AssertionError, "unsafe or unsupported"),
            ):
                verify(distribution, wheel, prefix)
        with (
            mock.patch.object(
                zipfile,
                "ZipFile",
                return_value=self.archive(
                    {
                        "nexuml.dist-info/RECORD": b"not proof",
                    }
                ),
            ),
            self.assertRaisesRegex(AssertionError, "no installed files"),
        ):
            verify(distribution, wheel, prefix)
        escaped = self.site / "nexuml/escaped.py"
        escaped.symlink_to(self.source / "pyproject.toml")
        with (
            mock.patch.object(
                zipfile,
                "ZipFile",
                return_value=self.archive(
                    {
                        "nexuml/escaped.py": escaped.read_bytes(),
                    }
                ),
            ),
            self.assertRaisesRegex(AssertionError, "outside the owned environment"),
        ):
            verify(distribution, wheel, prefix)


if __name__ == "__main__":
    unittest.main()
