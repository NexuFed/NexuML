"""Mocked worker-fixture contracts: no training, inference, Ray, CUDA or native imports."""

from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import runpy
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "fixtures" / "physical_ai_acceptance.py"


class PhysicalAICasesTest(unittest.TestCase):
    def setUp(self):
        namespace = runpy.run_path(str(SCRIPT))
        self.case = namespace["run_case"]
        self.g = self.case.__globals__
        self.source_admission = self.g["_require_current_source"]
        self.module_inventory = self.g["_module_inventory"]
        self.source_identity = self.g["_source_identity"]
        self.graphics_inventory = self.g["_graphics_inventory"]
        scratch = tempfile.TemporaryDirectory(prefix="physical-ai-cases-")
        self.addCleanup(scratch.cleanup)
        self.base = Path(scratch.name)
        self.root = self.base / "attempt with spaces; $(not-a-command)"
        self.config = self.base / "local config.yaml"
        self.config.write_text("trusted resolved config")
        self.patch_globals(
            ATTEMPT_BASE=self.base,
            _source_identity=lambda: "tree-sha",
            _module_inventory=lambda: {},
            _graphics_inventory=lambda: {"scope": "inert test inventory"},
            _object_store_observations=lambda: [
                {
                    "pid": 79,
                    "bytes": 1024**3,
                    "plasma_directory": "/dev/shm",
                    "filesystem": "tmpfs",
                    "backing": "shared_memory",
                }
            ],
            _require_current_source=mock.Mock(
                side_effect=lambda record: record.update(imports=self.g["_module_inventory"]())
            ),
        )
        environment = mock.patch.dict(
            os.environ,
            {
                "CUDA_VISIBLE_DEVICES": "",
                "TMPDIR": str(self.base),
                "RAY_JOB_ID": "mock-worker",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)

    def patch_globals(self, **values):
        patch = mock.patch.dict(self.g, values)
        patch.start()
        self.addCleanup(patch.stop)

    def guarded_root(self):
        self.root.mkdir()
        (self.root / "guard.json.lock").touch()

    def run_case(self, case, options=None):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = self.case(case, self.root, self.config, options or {})
        receipt = json.loads((self.root / "case.json").read_text())
        markers = [
            line.removeprefix("NEXUML_CASE_RESULT=")
            for line in output.getvalue().splitlines()
            if line.startswith("NEXUML_CASE_RESULT=")
        ]
        self.assertEqual(len(markers), 1)
        self.assertEqual(json.loads(markers[0]), receipt)
        self.assertFalse((self.root / "case.json.lock").exists())
        return code, receipt

    def test_runpy_import_is_stdlib_only(self):
        imported = []
        original = __import__

        def capture(name, *args, **kwargs):
            imported.append(name)
            return original(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=capture):
            runpy.run_path(str(SCRIPT))
        self.assertFalse(
            any(
                name.split(".")[0]
                in {
                    "torch",
                    "ray",
                    "nexuml",
                    "nexuml_library",
                    "isaacsim",
                    "isaaclab",
                }
                for name in imported
            )
        )

    def test_inventory_matches_all_matrix_names_and_registered_decorators(self):
        matrix = (
            SCRIPT.parents[2]
            / "openspec/changes/qualify-physical-ai-workflows-via-ray-jobs/verification-matrix.md"
        )
        names = set()
        for path in (SCRIPT.parents[2] / "library/src/nexuml_library/scenarios/reinforcement").glob(
            "*.py"
        ):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if (
                        node.func.id == "scenario"
                        and node.args
                        and isinstance(node.args[0], ast.Constant)
                    ):
                        names.add(node.args[0].value)
        self.assertEqual(len(self.g["RECIPES"]), 21)
        self.assertEqual(self.g["RECIPES"], names)
        self.assertTrue(all(f"`{name}`" in matrix.read_text() for name in names))

    def test_source_identity_covers_owned_trees_and_metadata_not_unrelated_files(self):
        # Obtain the real helper, not setUp's receipt-only stub.
        identity = runpy.run_path(str(SCRIPT))["_source_identity"]
        source = self.base / "source"
        for directory in ("src", "library/src", "tests", "scripts"):
            (source / directory).mkdir(parents=True, exist_ok=True)
            (source / directory / "module.py").write_text(directory)
        for filename in ("pyproject.toml", "library/pyproject.toml", "uv.lock"):
            (source / filename).write_text(filename)
        (source / "src/module").mkdir()
        (source / "src/module/nested.py").write_text("sort by relative string, not Path components")
        for name in ("tests/browser.cjs", "scripts/browser.js"):
            (source / name).write_text("fixed browser source")
        expected = hashlib.sha256()
        names = [
            f"{directory}/module.py" for directory in ("src", "library/src", "tests", "scripts")
        ]
        names += [
            "src/module/nested.py",
            "tests/browser.cjs",
            "scripts/browser.js",
            "pyproject.toml",
            "library/pyproject.toml",
            "uv.lock",
        ]
        for name in sorted(names):
            expected.update(name.encode() + b"\0" + (source / name).read_bytes() + b"\0")
        with contextlib.chdir(source):
            initial = identity()
            self.assertEqual(initial, expected.hexdigest())
            driver_identity = runpy.run_path(
                str(SCRIPT.parents[2] / "scripts/physical_ai_acceptance.py")
            )["source_identity"]
            self.assertEqual(driver_identity(source), initial)
            (source / "README.md").write_text("not executable source")
            self.assertEqual(initial, identity())
            (source / "tests/module.py").write_text("changed fixture")
            self.assertNotEqual(initial, identity())
            for name in ("tests/browser.cjs", "scripts/browser.js"):
                previous = identity()
                (source / name).write_text("changed executed browser fixture")
                self.assertNotEqual(previous, identity())
                self.assertEqual(driver_identity(source), identity())
        self.assertEqual(len(initial), 64)

    def test_unknown_case_and_arbitrary_options_block_without_subprocess(self):
        for case, options in (
            ("shell", {}),
            ("regression", {"cmd": "echo secret"}),
            ("recipe", {"module": "arbitrary.module"}),
        ):
            with self.subTest(case=case):
                if self.root.exists():
                    self.root = self.base / f"next-{case}"
                with mock.patch.object(self.g["subprocess"], "Popen") as launch:
                    code, receipt = self.run_case(case, options)
                launch.assert_not_called()
                self.assertEqual((code, receipt["status"]), (2, "BLOCKED"))
                self.assertEqual(receipt["gates"]["C"], "UNVERIFIED")

    def source_fixture(self):
        source = self.base / "uploaded"
        origins = {}
        for module, directory in (
            ("nexuml", "src/nexuml"),
            ("nexuml_library", "library/src/nexuml_library"),
        ):
            package = source / directory
            package.mkdir(parents=True)
            origin = package / "__init__.py"
            origin.write_text("raise AssertionError('application import forbidden')")
            origins[module] = str(origin)
        for name in ("pyproject.toml", "library/pyproject.toml", "uv.lock"):
            (source / name).write_text("uploaded identity")
        self.patch_globals(
            _require_current_source=self.source_admission,
            _module_inventory=self.module_inventory,
            _source_identity=self.source_identity,
        )
        original_import = __import__

        def no_application_import(name, *args, **kwargs):
            if name.split(".")[0] in {
                "nexuml",
                "nexuml_library",
                "torch",
                "ray",
                "isaacsim",
                "isaaclab",
            }:
                raise AssertionError("application/native import forbidden")
            return original_import(name, *args, **kwargs)

        for patch in (
            mock.patch("builtins.__import__", side_effect=no_application_import),
            mock.patch.object(
                self.g["importlib"].util,
                "find_spec",
                side_effect=lambda name: (
                    SimpleNamespace(origin=origins[name]) if name in origins else None
                ),
            ),
            mock.patch.object(
                self.g["importlib"].metadata,
                "version",
                side_effect=self.g["importlib"].metadata.PackageNotFoundError,
            ),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        return source, origins

    def test_foreign_origins_block_otherwise_passing_cases_despite_matching_source_hash(self):
        source, origins = self.source_fixture()
        valid = origins.copy()
        foreign = self.base / "prepared-site-packages/__init__.py"
        foreign.parent.mkdir()
        foreign.write_text("older prepared application")
        stages = {name: mock.Mock() for name in ("_preflight", "_regression", "_recipe", "_stage")}
        self.patch_globals(**stages, _gpu_inventory=mock.Mock())
        with contextlib.chdir(source):
            expected_hash = self.source_identity()
            for case in ("preflight", "regression", "recipe", "learning", "logging"):
                for module in valid:
                    with self.subTest(case=case, module=module):
                        origins.update(valid)
                        origins[module] = str(foreign)
                        self.root = self.base / f"{case}-{module}"
                        self.guarded_root()
                        code, receipt = self.run_case(case)
                        self.assertEqual((code, receipt["status"]), (2, "BLOCKED"))
                        self.assertEqual(receipt["source_tree_sha256"], expected_hash)
                        self.assertFalse(receipt["source_origin_match"])
                        self.assertEqual(receipt["imports"][module]["origin"], str(foreign))
                        self.assertFalse(receipt["imports"][module]["source_origin_match"])
                        other = next(name for name in valid if name != module)
                        self.assertTrue(receipt["imports"][other]["source_origin_match"])
                        self.assertEqual(receipt["stages"], [])
                        self.assertTrue(all(v == "UNVERIFIED" for v in receipt["gates"].values()))
        for stage in stages.values():
            stage.assert_not_called()
        self.g["_gpu_inventory"].assert_not_called()

    def test_current_source_origins_admit_all_source_cases_without_changing_pythonpath(self):
        source, origins = self.source_fixture()
        stages = {
            name: mock.Mock() for name in ("_preflight", "_regression", "_recipe", "_package")
        }
        self.patch_globals(**stages)
        with contextlib.chdir(source), mock.patch.dict(os.environ, PYTHONPATH="/prepared/profile"):
            for case, stage in zip(
                ("preflight", "regression", "recipe", "package"), stages.values()
            ):
                with self.subTest(case=case):
                    self.root = self.base / case
                    self.guarded_root()
                    code, receipt = self.run_case(case)
                    self.assertEqual((code, receipt["status"]), (0, "PASS"))
                    self.assertTrue(receipt["source_origin_match"])
                    for module, origin in origins.items():
                        self.assertEqual(receipt["imports"][module]["resolved_origin"], origin)
                        self.assertTrue(receipt["imports"][module]["source_origin_match"])
                    stage.assert_called_once()
                    self.assertEqual(stage.call_args.args[-1]["PYTHONPATH"], "/prepared/profile")

    def test_source_admission_rejects_foreign_symlink_files_and_package_roots(self):
        source, origins = self.source_fixture()
        stage = mock.Mock()
        self.patch_globals(_preflight=stage, _gpu_inventory=mock.Mock())
        for module in origins:
            origin = Path(origins[module])
            foreign = self.base / f"foreign-{module}"
            foreign.mkdir()
            (foreign / "__init__.py").write_text("foreign application")
            origin.unlink()
            origin.symlink_to(foreign / "__init__.py")
            with contextlib.chdir(source):
                for kind in ("file", "package"):
                    with self.subTest(module=module, kind=kind):
                        if kind == "package":
                            origin.unlink()
                            origin.parent.rmdir()
                            origin.parent.symlink_to(foreign, target_is_directory=True)
                        self.root = self.base / f"{module}-{kind}"
                        self.guarded_root()
                        code, receipt = self.run_case("preflight")
                        self.assertEqual((code, receipt["status"]), (2, "BLOCKED"))
                        self.assertFalse(receipt["source_origin_match"])
                        self.assertEqual(receipt["stages"], [])
            origin.parent.unlink()
            origin.parent.mkdir()
            origin.write_text("restored uploaded source")
        stage.assert_not_called()
        self.g["_gpu_inventory"].assert_not_called()

    def test_source_admission_rejects_missing_and_unresolvable_origins_for_both_packages(self):
        source, origins = self.source_fixture()
        valid = origins.copy()
        stage = mock.Mock()
        self.patch_globals(_preflight=stage, _gpu_inventory=mock.Mock())
        find_spec = self.g["importlib"].util.find_spec
        for module in valid:
            for kind in ("missing-spec", "missing-origin", "missing-file", "lookup-error", "loop"):
                with self.subTest(module=module, kind=kind), contextlib.chdir(source):
                    origins.update(valid)
                    if kind == "missing-spec":
                        del origins[module]
                    elif kind == "missing-origin":
                        origins[module] = None
                    elif kind == "missing-file":
                        origins[module] = str(source / "absent.py")
                    elif kind == "loop":
                        loop = source / f"{module}-loop"
                        loop.symlink_to(loop)
                        origins[module] = str(loop)
                    original_lookup = find_spec.side_effect
                    if kind == "lookup-error":

                        def lookup(name):
                            if name == module:
                                raise ValueError("prepared module has no resolvable spec")
                            return original_lookup(name)

                        find_spec.side_effect = lookup
                    self.root = self.base / f"{module}-{kind}"
                    self.guarded_root()
                    code, receipt = self.run_case("preflight")
                    find_spec.side_effect = original_lookup
                    self.assertEqual((code, receipt["status"]), (2, "BLOCKED"))
                    self.assertFalse(receipt["source_origin_match"])
                    self.assertFalse(receipt["imports"][module]["source_origin_match"])
                    self.assertEqual(receipt["stages"], [])
        stage.assert_not_called()
        self.g["_gpu_inventory"].assert_not_called()

    def test_unprepared_profiles_never_pass(self):
        for case, gate in (("so101", "L"), ("browser", "V")):
            self.root = self.base / case
            code, receipt = self.run_case(case)
            self.assertEqual(
                (code, receipt["status"], receipt["gates"][gate]), (2, "BLOCKED", "BLOCKED")
            )
            self.assertEqual(receipt["stages"], [])
            self.assertIsNone(receipt.get("training_frames"))

    def test_prepared_browser_dispatch_keeps_visual_gate_unverified(self):
        self.guarded_root()
        venv = self.base / "private-venv"
        setup = venv / "browser-runtime/setup.json"
        setup.parent.mkdir(parents=True)
        setup.write_text('{"status":"success"}')
        producer = {
            "source_attempt": "/retained/producer",
            "source_case_sha256": "a" * 64,
            "source_guard_sha256": "b" * 64,
            "logger_mode": "both",
        }
        profile = mock.Mock(
            side_effect=lambda record, *args: record["checks"].update(
                logger_ui_reconnaissance="PASS"
            )
        )
        with (
            mock.patch.object(sys, "prefix", str(venv)),
            mock.patch("runpy.run_path", return_value={"run_browser": profile}),
        ):
            code, receipt = self.run_case("browser", producer)
        self.assertEqual((code, receipt["status"]), (0, "PASS"))
        self.assertTrue(all(value == "UNVERIFIED" for value in receipt["gates"].values()))
        self.assertEqual(profile.call_args.args[2], producer)
        self.assertEqual(
            profile.call_args.args[4:],
            tuple(self.g[name] for name in ("_stage", "_recovery", "_group_members")),
        )

    def test_prepared_browser_missing_identity_never_loads_probe(self):
        self.guarded_root()
        venv = self.base / "private-venv"
        setup = venv / "browser-runtime/setup.json"
        setup.parent.mkdir(parents=True)
        setup.write_text('{"status":"success"}')
        with (
            mock.patch.object(sys, "prefix", str(venv)),
            mock.patch("runpy.run_path") as load,
        ):
            code, receipt = self.run_case("browser")
        self.assertEqual((code, receipt["status"]), (2, "BLOCKED"))
        load.assert_not_called()
        self.assertIn("producer", receipt["reason"])

    def test_logging_failure_requires_qualified_source_and_prepared_browser(self):
        self.guarded_root()
        with mock.patch("runpy.run_path") as load:
            code, receipt = self.run_case("logging-failure")
        self.assertEqual((code, receipt["status"]), (2, "BLOCKED"))
        load.assert_not_called()
        self.assertEqual(receipt["stages"], [])

    def test_logging_failure_observer_checks_cannot_promote_failed_job(self):
        venv = self.base / "private-venv"
        setup = venv / "browser-runtime/setup.json"
        setup.parent.mkdir(parents=True)
        setup.write_text('{"status":"success"}')
        producer = {
            "source_attempt": "/retained/producer",
            "source_case_sha256": "a" * 64,
            "source_guard_sha256": "b" * 64,
        }
        original = RuntimeError("original partial delivery CLI failed")

        def observe(*args):
            record, root, options = args[:3]
            self.assertEqual(options, producer)
            self.assertEqual(len(args), 8)
            (root / "original-cli.log").write_text("MLFlowLogger failed after TensorBoard delivery")
            record["stages"].append(
                {"name": "logging-partial-delivery-evaluate", "status": "FAIL", "exit_code": 1}
            )
            record["checks"]["partial_delivery_native_failed_ui"] = "PASS"
            raise original

        self.guarded_root()
        with (
            mock.patch.object(sys, "prefix", str(venv)),
            mock.patch("runpy.run_path", return_value={"run_logging_failure": observe}),
        ):
            code, receipt = self.run_case("logging-failure", producer)
        self.assertEqual((code, receipt["status"]), (1, "FAIL"))
        self.assertEqual(receipt["stages"][0]["exit_code"], 1)
        self.assertEqual(receipt["checks"]["partial_delivery_native_failed_ui"], "PASS")
        self.assertEqual(set(receipt["gates"].values()), {"UNVERIFIED"})
        self.assertIn(
            "original partial delivery CLI failed",
            (self.root / "logging-failure-controller.log").read_text(),
        )

    def test_recorded_recipe_rejects_other_backends_before_any_cli_stage(self):
        for recipe, selection, renderer in (
            ("cartpole-ppo", "both", "egl"),
            ("mujoco-reacher-ppo", "both", "osmesa"),
            ("mujoco-humanoid-ppo", "gif", "egl"),
            ("mujoco-reacher-ppo", "both", None),
        ):
            with self.subTest(recipe=recipe, renderer=renderer):
                prepare = mock.Mock(return_value=SimpleNamespace(name=recipe))
                stage = mock.Mock(side_effect=AssertionError("must not launch"))
                with mock.patch.dict(self.g, _prepare_recipe=prepare, _stage=stage):
                    with self.assertRaisesRegex(self.g["Blocked"], "explicitly selected"):
                        self.g["_recipe"](
                            {},
                            self.root,
                            self.config,
                            {"recording": selection},
                            {"MUJOCO_GL": renderer},
                        )
                stage.assert_not_called()

    def test_browser_controller_failure_has_private_traceback_and_preserves_primary(self):
        self.root.mkdir()
        original = RuntimeError("original private server readiness failure")
        options = {
            key: "fixed" for key in ("source_attempt", "source_case_sha256", "source_guard_sha256")
        }
        profile = mock.Mock(side_effect=original)
        with mock.patch("runpy.run_path", return_value={"run_browser": profile}):
            with self.assertRaises(RuntimeError) as caught:
                self.g["_browser"]({}, self.root, options, {})
            self.assertIs(caught.exception, original)
            log = self.root / "browser-controller-failure.log"
            self.assertIn(
                "RuntimeError: original private server readiness failure", log.read_text()
            )
            log.unlink()
            with mock.patch.object(Path, "open", side_effect=OSError("diagnostic storage failed")):
                with self.assertRaises(RuntimeError) as caught:
                    self.g["_browser"]({}, self.root, options, {})
            self.assertIs(caught.exception, original)
            self.assertTrue(any("diagnostic also failed" in note for note in original.__notes__))

    def test_root_absolute_contained_not_symlink_escape_and_never_reused(self):
        for bad in (Path("relative"), self.base, self.base.parent / "outside"):
            with self.assertRaises(ValueError):
                self.case("package", bad, self.config, {})
        self.root.symlink_to(self.base.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.case("package", self.root / "escape", self.config, {})
        self.root.unlink()
        self.guarded_root()
        (self.root / "policy").mkdir()
        with self.assertRaises(FileExistsError):
            self.case("package", self.root, self.config, {})
        (self.root / "policy").rmdir()
        self.run_case("browser")
        original = (self.root / "case.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.case("package", self.root, self.config, {})
        self.assertEqual(original, (self.root / "case.json").read_bytes())

    def test_requires_remote_identity_and_outer_guard_before_gpu_or_workloads(self):
        self.patch_globals(_gpu_inventory=mock.Mock(side_effect=AssertionError("must not run")))
        code, receipt = self.run_case("preflight")
        self.assertEqual(code, 2)
        self.assertIn("guard", receipt["reason"])
        self.g["_gpu_inventory"].assert_not_called()
        self.root = self.base / "not-remote"
        self.guarded_root()
        with mock.patch.dict(os.environ, clear=True):
            code, receipt = self.run_case("preflight")
        self.assertEqual(code, 2)
        self.assertIn("local acceptance execution is forbidden", receipt["reason"])
        self.g["_require_current_source"].assert_not_called()

    def fake_launch(self, returncode=0, log_text="private test output"):
        calls = []

        def launch(argv, **kwargs):
            calls.append((list(argv), kwargs))
            kwargs["stdout"].write(log_text.encode())
            process = mock.Mock(pid=991, returncode=returncode)
            process.poll.return_value = returncode
            process.wait.return_value = returncode
            return process

        self.patch_globals(_group_members=lambda: {os.getpid()})
        patch = mock.patch.object(self.g["subprocess"], "Popen", side_effect=launch)
        patch.start()
        self.addCleanup(patch.stop)
        return calls

    def test_regression_fixed_full_argv_literal_paths_and_cpu_gating(self):
        self.guarded_root()
        calls = self.fake_launch(log_text="secret-not-for-stdout: 7 passed, 3 skipped")
        with mock.patch.object(
            self.g["tempfile"], "mkdtemp", return_value=str(self.base / "scratch")
        ) as scratch:
            code, receipt = self.run_case("regression")
        self.assertEqual(code, 0)
        argv, kwargs = calls[0]
        self.assertEqual(
            argv,
            [
                sys.executable,
                "-m",
                "pytest",
                "tests",
                "-q",
                "--tb=short",
                "-p",
                "no:cacheprovider",
                "--basetemp",
                str(self.base / "scratch/pytest"),
            ],
        )
        scratch.assert_called_once_with(prefix="nexuml-regression-", dir="/tmp/ray")
        self.assertNotIn("start_new_session", kwargs)
        self.assertNotIn("shell", kwargs)
        self.assertNotIn("process_group", kwargs)
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(kwargs["env"]["TMPDIR"], "/tmp/ray")
        for tier in ("SIMULATOR", "SYSTEM", "HARDWARE", "DATA"):
            self.assertEqual(kwargs["env"][f"NEXUML_RUN_{tier}_TESTS"], "0")
        self.assertEqual(receipt["regression"]["full_suite_gate"], "PASS")
        self.assertFalse(receipt["regression"]["native_proof"])
        self.assertNotIn("secret-not-for-stdout", json.dumps(receipt))
        self.assertTrue(all(value == "UNVERIFIED" for value in receipt["gates"].values()))

    def test_focused_suite_is_not_full_gate_and_failure_retains_exit_and_log(self):
        self.guarded_root()
        calls = self.fake_launch(returncode=9)
        with mock.patch.object(
            self.g["tempfile"], "mkdtemp", return_value=str(self.base / "scratch")
        ):
            code, receipt = self.run_case("regression", {"suite": "focused"})
        self.assertEqual((code, receipt["status"]), (1, "FAIL"))
        selected = self.g["FOCUSED_TESTS"]
        self.assertEqual(calls[0][0][3 : 3 + len(selected)], list(selected))
        self.assertEqual(receipt["stages"][0]["exit_code"], 9)
        self.assertEqual(receipt["regression"]["full_suite_gate"], "UNVERIFIED")
        self.assertEqual(
            receipt["stages"][0]["log_sha256"], self.g["_sha256"](self.root / "pytest.log")
        )

    def test_full_suite_failure_is_not_pass(self):
        self.guarded_root()
        self.fake_launch(returncode=2)
        with mock.patch.object(
            self.g["tempfile"], "mkdtemp", return_value=str(self.base / "scratch")
        ):
            code, receipt = self.run_case("regression")
        self.assertEqual((code, receipt["regression"]["full_suite_gate"]), (1, "FAIL"))

    def test_static_suite_is_fixed_and_does_not_claim_application_acceptance(self):
        self.guarded_root()
        calls = self.fake_launch()
        with mock.patch.object(
            self.g["tempfile"], "mkdtemp", return_value=str(self.base / "scratch")
        ):
            code, receipt = self.run_case("regression", {"suite": "static"})
        self.assertEqual(code, 0)
        self.assertEqual(
            [stage["name"] for stage in receipt["stages"]],
            ["lint", "format", "types", "docs", "lock"],
        )
        self.assertEqual(calls[0][0], [sys.executable, "-m", "ruff", "check", "."])
        self.assertEqual(calls[-1][0], ["uv", "lock", "--check"])
        self.assertEqual(
            calls[2][0], [sys.executable, "-m", "ty", "check", "--python", sys.executable]
        )
        self.assertEqual(receipt["checks"]["static"], "PASS")
        self.assertEqual(receipt["regression"]["full_suite_gate"], "UNVERIFIED")
        self.assertTrue(all(value == "UNVERIFIED" for value in receipt["gates"].values()))

    def test_native_state_suite_keeps_gpu_and_requires_each_real_oracle(self):
        self.guarded_root()

        def stage(record, name, argv, env):
            self.assertEqual(name, "native-state")
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(env["NEXUML_RUN_SIMULATOR_TESTS"], "1")
            self.assertEqual(env["NEXUML_RUN_SYSTEM_TESTS"], "0")
            self.assertEqual(argv[3:9], list(self.g["NATIVE_STATE_TESTS"]))
            names = [selector.rsplit("::", 1)[1] for selector in self.g["NATIVE_STATE_TESTS"]]
            (self.root / "native-state-junit.xml").write_text(
                "<testsuites><testsuite>"
                + "".join(f'<testcase name="{name}"/>' for name in names)
                + "</testsuite></testsuites>"
            )

        self.patch_globals(_stage=stage)
        with mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES="0"):
            code, receipt = self.run_case("regression", {"suite": "native-state"})
        self.assertEqual(code, 0)
        self.assertTrue(receipt["regression"]["native_proof"])
        self.assertEqual(receipt["checks"]["native_state_oracles"], "PASS")
        self.assertEqual(receipt["regression"]["full_suite_gate"], "UNVERIFIED")
        self.assertTrue(all(value == "UNVERIFIED" for value in receipt["gates"].values()))

    def test_native_state_suite_never_counts_skipped_or_missing_tests(self):
        for skipped in (False, True):
            with self.subTest(skipped=skipped):
                self.root = self.base / f"native-state-{skipped}"
                self.guarded_root()

                def stage(*args):
                    names = [s.rsplit("::", 1)[1] for s in self.g["NATIVE_STATE_TESTS"]]
                    bodies = (
                        [f'<testcase name="{name}"><skipped/></testcase>' for name in names]
                        if skipped
                        else []
                    )
                    (self.root / "native-state-junit.xml").write_text(
                        "<testsuites><testsuite>" + "".join(bodies) + "</testsuite></testsuites>"
                    )

                self.patch_globals(_stage=stage)
                code, receipt = self.run_case("regression", {"suite": "native-state"})
                self.assertEqual((code, receipt["status"]), (1, "FAIL"))
                self.assertFalse(receipt["regression"]["native_proof"])
                self.assertNotIn("native_state_oracles", receipt["checks"])

    def test_native_sensor_suite_requires_exact_executed_owner_and_scene_oracles(self):
        self.guarded_root()

        def stage(record, name, argv, env):
            self.assertEqual(name, "native-sensors")
            self.assertEqual(env["NEXUML_RUN_SIMULATOR_TESTS"], "1")
            selected = self.g["NATIVE_SENSOR_TESTS"]
            self.assertEqual(argv[3 : 3 + len(selected)], list(selected))
            self.assertEqual(len(selected), 13)
            self.assertIn("--maxfail=1", argv)
            names = [selector.rsplit("::", 1)[1] for selector in self.g["NATIVE_SENSOR_TESTS"]]
            (self.root / "native-sensors-junit.xml").write_text(
                "<testsuites><testsuite>"
                + "".join(f'<testcase name="{name}"/>' for name in names)
                + "</testsuite></testsuites>"
            )

        self.patch_globals(_stage=stage)
        code, receipt = self.run_case("regression", {"suite": "native-sensors"})
        self.assertEqual(code, 0)
        self.assertTrue(receipt["regression"]["native_proof"])
        self.assertEqual(receipt["checks"]["native_sensor_oracles"], "PASS")
        self.assertNotIn("native_state_oracles", receipt["checks"])
        self.assertTrue(all(value == "UNVERIFIED" for value in receipt["gates"].values()))

    def test_declared_python_suites_reuse_guarded_profile_and_preserve_gpu_visibility(self):
        for suite, version in (("python313", "3.13"), ("python314", "3.14")):
            with self.subTest(suite=suite):
                self.root = self.base / suite
                self.guarded_root()
                profile = mock.Mock()
                with (
                    mock.patch("runpy.run_path", return_value={"run_matrix": profile}),
                    mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}),
                ):
                    code, receipt = self.run_case("regression", {"suite": suite})
                self.assertEqual(code, 0)
                args = profile.call_args.args
                self.assertEqual(args[1], self.root)
                self.assertEqual(args[4], version)
                self.assertEqual(args[2]["CUDA_VISIBLE_DEVICES"], "0")
                for tier in ("SIMULATOR", "SYSTEM", "HARDWARE", "DATA"):
                    self.assertEqual(args[2][f"NEXUML_RUN_{tier}_TESTS"], "0")
                self.assertEqual(receipt["regression"]["full_suite_gate"], "PASS")
                self.assertFalse(receipt["regression"]["native_proof"])
                self.assertTrue(all(value == "UNVERIFIED" for value in receipt["gates"].values()))

    def test_stage_refuses_resume_when_descendants_survive(self):
        self.guarded_root()
        self.fake_launch()
        self.patch_globals(_group_members=mock.Mock(side_effect=[{1}, {1, 992}]))
        receipt = {"attempt_root": str(self.root), "stages": []}
        with self.assertRaisesRegex(AssertionError, "live descendants"):
            self.g["_stage"](receipt, "train", ["fixed-command"], {})
        self.assertEqual(receipt["stages"][0]["status"], "FAIL")

    def test_partial_interruption_must_be_real_nonfinal_and_nonzero(self):
        self.guarded_root()
        self.fake_launch()
        receipt = {"attempt_root": str(self.root), "stages": []}
        with self.assertRaisesRegex(AssertionError, "proven partial interruption"):
            self.g["_stage"](receipt, "train", ["fixed-command"], {}, interrupt_total=64)
        self.assertIsNone(receipt["stages"][0]["partial"])

    def test_partial_train_signals_only_child_then_waits_and_preserves_copy_hash(self):
        self.guarded_root()
        partial = self.root / "partial.ckpt"
        partial.write_bytes(b"readable mocked checkpoint")
        process = mock.Mock(pid=991, returncode=-signal.SIGTERM)
        process.poll.return_value = None
        process.wait.return_value = -signal.SIGTERM
        self.patch_globals(
            _copy_partial=lambda root, total: {
                "path": str(partial),
                "frames": 32,
                "global_step": 2,
                "sha256": self.g["_sha256"](partial),
            },
            _group_members=lambda: {1},
        )
        receipt = {"attempt_root": str(self.root), "stages": []}
        with mock.patch.object(self.g["subprocess"], "Popen", return_value=process):
            result = self.g["_stage"](receipt, "train", ["fixed-command"], {}, interrupt_total=64)
        process.send_signal.assert_called_once_with(signal.SIGTERM)
        process.wait.assert_called_once_with(timeout=30)
        self.assertEqual(result["status"], "EXPECTED_INTERRUPTION")
        self.assertTrue(result["cleanup"])

    def test_recipe_stage_order_literal_argv_and_gate_scope(self):
        self.guarded_root()
        for name in ("case.yaml", "eval-case.yaml", "partial.ckpt"):
            (self.root / name).write_text("mocked")
        (self.root / "checkpoints").mkdir()
        (self.root / "checkpoints/final.ckpt").write_bytes(b"mocked")
        stages = []
        partial = {
            "path": str(self.root / "partial.ckpt"),
            "frames": 32,
            "global_step": 2,
            "sha256": self.g["_sha256"](self.root / "partial.ckpt"),
        }

        def stage(record, name, argv, env, **kwargs):
            stages.append((name, argv, kwargs))
            if name == "fresh-evaluate":
                (self.root / "fresh-evaluate.log").write_text(
                    "Policy evaluation (frame step 64): {'eval/return_mean': 1.25}"
                )
            return {"partial": partial}

        config = SimpleNamespace(
            name="cartpole-ppo", training=SimpleNamespace(total_frames=64, accelerator="cpu")
        )
        self.patch_globals(
            _prepare_recipe=lambda *args: config,
            _stage=stage,
            _checkpoint=lambda path: ({}, 64, 4),
            _module_inventory=lambda: {},
            _validate_policy=lambda *args: {"training_frames": 64, "global_step": 4},
        )
        record = {
            "attempt_root": str(self.root),
            "gates": {key: "UNVERIFIED" for key in "CVLRD"},
            "checks": {},
            "gate_scope": {},
        }
        self.g["_recipe"](record, self.root, self.config, {}, {})
        self.assertEqual(
            [stage[0] for stage in stages],
            [
                "untrained-export",
                "partial-train",
                "resume",
                "checkpoint-export",
                "fresh-evaluate",
                "exhausted-resume",
            ],
        )
        self.assertEqual(stages[1][2], {"interrupt_total": 64})
        self.assertEqual(
            stages[2][1][-2:], ["--trainer-checkpoint", str(self.root / "partial.ckpt")]
        )
        self.assertEqual(stages[4][1][-2:], ["--device", "cpu"])
        self.assertIn(str(self.root / "checkpoint-policy"), stages[4][1])
        self.assertEqual(record["gates"]["C"], "PASS")
        self.assertEqual(record["gates"]["R"], "UNVERIFIED")
        self.assertEqual(record["checks"]["partial_resume"], "PASS")

    def test_recipe_failure_stops_dependent_stages_and_fails_c_gate(self):
        self.guarded_root()
        config = SimpleNamespace(name="cartpole-ppo", training=SimpleNamespace(total_frames=64))

        def prepare(*args):
            (self.root / "case.yaml").write_text("mocked")
            return config

        stage = mock.Mock(side_effect=AssertionError("controlled failure"))
        self.patch_globals(_prepare_recipe=prepare, _stage=stage, _module_inventory=lambda: {})
        code, receipt = self.run_case("recipe")
        self.assertEqual((code, receipt["gates"]["C"]), (1, "FAIL"))
        stage.assert_called_once()

    def test_learning_reuses_fixed_profile_after_worker_admission_and_preserves_failures(self):
        self.guarded_root()
        profile = mock.Mock(side_effect=AssertionError("controlled learning failure"))
        with mock.patch("runpy.run_path", return_value={"run_learning": profile}):
            code, receipt = self.run_case("learning")
        self.assertEqual((code, receipt["status"], receipt["gates"]["L"]), (1, "FAIL", "FAIL"))
        self.assertEqual(receipt["gates"]["C"], "UNVERIFIED")
        self.g["_require_current_source"].assert_called_once()
        args = profile.call_args.args
        self.assertEqual(args[1], self.root)
        self.assertEqual(args[2]["TMPDIR"], "/tmp/ray")
        self.assertEqual(
            args[3:],
            tuple(
                self.g[name]
                for name in (
                    "_prepare_recipe",
                    "_stage",
                    "_checkpoint",
                    "_validate_policy",
                    "_evaluation_metrics",
                )
            ),
        )

    def test_learning_rejects_arbitrary_options_before_loading_fixed_profile(self):
        with mock.patch("runpy.run_path") as load:
            code, receipt = self.run_case("learning", {"frames": "128"})
        self.assertEqual((code, receipt["status"]), (2, "BLOCKED"))
        load.assert_not_called()

    def test_private_dependency_setup_receipt_is_retained_and_hashed(self):
        self.guarded_root()
        venv = self.base / "private-venv"
        setup = venv / "native-runtime/setup.json"
        setup.parent.mkdir(parents=True)
        setup.write_text('{"status": "success", "packages": []}')
        browser = venv / "browser-runtime/setup.json"
        browser.parent.mkdir()
        browser.write_text('{"status": "success", "assets": []}')
        self.patch_globals(_preflight=mock.Mock())
        with mock.patch.object(sys, "prefix", str(venv)):
            code, receipt = self.run_case("preflight")
        self.assertEqual(code, 0)
        retained = self.root / "native-setup.json"
        self.assertEqual(retained.read_bytes(), setup.read_bytes())
        self.assertEqual(receipt["native_setup_sha256"], self.g["_sha256"](retained))
        self.assertIn(str(retained), [artifact["path"] for artifact in receipt["artifacts"]])
        browser_retained = self.root / "browser-setup.json"
        self.assertEqual(browser_retained.read_bytes(), browser.read_bytes())
        self.assertEqual(receipt["browser_setup_sha256"], self.g["_sha256"](browser_retained))

    def test_logging_fixed_profile_failure_preserves_unqualified_gates(self):
        self.guarded_root()
        profile = mock.Mock(side_effect=RuntimeError("controlled logger failure"))
        with mock.patch("runpy.run_path", return_value={"run_logging": profile}):
            code, receipt = self.run_case("logging")
        self.assertEqual((code, receipt["status"]), (1, "FAIL"))
        self.assertTrue(all(value == "UNVERIFIED" for value in receipt["gates"].values()))
        self.assertNotIn("logger_payloads", receipt["checks"])
        args = profile.call_args.args
        self.assertEqual(args[1], self.root)
        self.assertEqual(
            args[3:],
            tuple(
                self.g[name]
                for name in (
                    "_prepare_recipe",
                    "_stage",
                    "_checkpoint",
                    "_validate_policy",
                    "_evaluation_metrics",
                )
            ),
        )

    def test_logging_rejects_options_before_loading_profile(self):
        with mock.patch("runpy.run_path") as load:
            code, receipt = self.run_case("logging", {"command": "arbitrary"})
        self.assertEqual((code, receipt["status"]), (2, "BLOCKED"))
        load.assert_not_called()

    def test_metrics_require_observed_finite_values(self):
        log = self.base / "evaluation.log"
        for text in ("Evaluation complete!", "Policy evaluation (frame step 64): {}"):
            log.write_text(text)
            with self.assertRaises(AssertionError):
                self.g["_evaluation_metrics"](log)
        log.write_text("Policy evaluation (frame step 64): {'return': 2.0,\n 'length': 3}")
        self.assertEqual(self.g["_evaluation_metrics"](log), {"return": 2.0, "length": 3})
        with self.assertRaisesRegex(AssertionError, "frame label"):
            self.g["_evaluation_metrics"](log, expected_frames=128)

    def test_graphics_inventory_reads_only_bounded_metadata_without_loading_driver(self):
        icds = self.base / "usr/share/vulkan/icd.d"
        icds.mkdir(parents=True)
        manifest = {
            "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.0"},
            "ignored": "not-retained",
        }
        (icds / "nvidia_icd.json").write_text(json.dumps(manifest))
        libraries = self.base / "usr/lib/x86_64-linux-gnu"
        libraries.mkdir(parents=True)
        (libraries / "libGLX_nvidia.so.0").write_bytes(b"inert metadata fixture, never loaded")
        with (
            mock.patch.dict(os.environ, {"NVIDIA_DRIVER_CAPABILITIES": "compute,utility"}),
            mock.patch.object(self.g["subprocess"], "run") as launch,
        ):
            result = self.graphics_inventory(self.base)
        self.assertIn("UNVERIFIED", result["scope"])
        self.assertEqual(result["driver_capabilities"], "compute,utility")
        self.assertEqual(result["manifests"][0]["library_path"], "libGLX_nvidia.so.0")
        self.assertEqual(len(result["libraries"]), 1)
        self.assertNotIn("not-retained", str(result))
        launch.assert_not_called()

    def test_graphics_inventory_retains_manifest_errors_and_caps_entries(self):
        icds = self.base / "etc/vulkan/icd.d"
        icds.mkdir(parents=True)
        for index in range(35):
            (icds / f"bad-{index}.json").write_text("{" * 65537 if index == 0 else "[]")
        result = self.graphics_inventory(self.base)
        self.assertTrue(result["truncated"])
        self.assertEqual(len(result["manifests"]), 32)
        self.assertTrue(all(entry["observation"] == "ValueError" for entry in result["manifests"]))
        self.assertFalse(result["libraries"])

    def test_preflight_base_pass_keeps_native_and_private_data_blocked(self):
        self.guarded_root()
        self.fake_launch(log_text="GPU-mocked-uuid, mock-driver\n")
        imports = {
            name: {"available": True, "origin": "mocked", "version": "mocked"}
            for name in ("nexuml", "nexuml_library", "torch")
        }
        self.patch_globals(
            _module_inventory=lambda: imports,
            _gpu_inventory=lambda: {
                "visible_count": 1,
                "device": "cuda:0",
                "allocation_uuid": "GPU-mocked-uuid",
            },
        )
        shm = SimpleNamespace(f_blocks=100, f_bavail=90, f_bfree=90, f_frsize=4096)
        with (
            mock.patch.object(self.g["shutil"], "which", side_effect=lambda name: f"/mock/{name}"),
            mock.patch.object(self.g["os"], "statvfs", return_value=shm),
        ):
            code, receipt = self.run_case("preflight")
        self.assertEqual((code, receipt["status"]), (0, "PASS"))
        self.assertIsNone(receipt["family_prerequisites"]["so101"]["data_root"])
        self.assertFalse(receipt["family_prerequisites"]["so101"]["data_present"])
        self.assertTrue(
            all(row["state"] == "BLOCKED" for row in receipt["family_prerequisites"].values())
        )
        self.assertEqual(receipt["shared_memory"]["available_bytes"], 90 * 4096)
        self.assertEqual(receipt["shared_memory"]["campaign_cap_bytes"], 8 * 1024**3)
        self.assertEqual(receipt["shared_memory"]["object_store_limit_bytes"], 1024**3)
        self.assertEqual(receipt["shared_memory"]["shared_object_store_limit_bytes"], 1024**3)
        self.assertTrue(receipt["durable_output"]["writable"])
        self.assertEqual(receipt["gpu"]["nvidia_smi_rows"], ["GPU-mocked-uuid, mock-driver"])
        self.assertTrue(all(state == "UNVERIFIED" for state in receipt["gates"].values()))

    def test_preflight_public_s3_reader_is_fixed_offline_and_does_not_pass_frozen_so101(self):
        self.guarded_root()
        invocations = []

        def stage(record, name, argv, env):
            invocations.append((name, argv, env))
            if name == "gpu-driver":
                (self.root / "gpu-driver.log").write_text("GPU-mocked-uuid, mock-driver\n")
            elif name == "public-so101-s3-reader":
                Path(argv[-1]).write_text(json.dumps({"files_sha256_verified": 15}))
            else:
                self.fail("unexpected preflight stage")

        self.patch_globals(
            _module_inventory=lambda: {
                name: {"available": True} for name in ("nexuml", "nexuml_library", "torch")
            },
            _gpu_inventory=lambda: {"visible_count": 1},
            _stage=stage,
        )
        probe = runpy.run_path(str(SCRIPT.with_name("physical_ai_public_data.py")))
        with (
            mock.patch.object(self.g["shutil"], "which", return_value="/mock/nvidia-smi"),
            mock.patch.object(
                self.g["os"],
                "statvfs",
                return_value=SimpleNamespace(f_blocks=100, f_bavail=90, f_bfree=90, f_frsize=4096),
            ),
        ):
            code, receipt = self.run_case("preflight", {"data_root": probe["SOURCE_URI"]})
        self.assertEqual((code, receipt["status"]), (0, "PASS"))
        self.assertEqual(receipt["external_data"]["files_sha256_verified"], 15)
        self.assertEqual(receipt["family_prerequisites"]["so101"]["state"], "BLOCKED")
        self.assertTrue(all(state == "UNVERIFIED" for state in receipt["gates"].values()))
        self.assertEqual(invocations[-1][0], "public-so101-s3-reader")
        self.assertEqual(invocations[-1][2]["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(invocations[-1][2]["HF_HUB_OFFLINE"], "1")

    def test_preflight_gpu_block_does_not_drop_family_rows_or_probe_private_data_implicitly(self):
        self.guarded_root()
        imports = {name: {"available": True} for name in ("nexuml", "nexuml_library", "torch")}
        self.patch_globals(
            _module_inventory=lambda: imports,
            _gpu_inventory=mock.Mock(side_effect=self.g["Blocked"]("GPU absent")),
        )
        with (
            mock.patch.dict(os.environ, NEXUML_DATA_ROOT=str(self.base)),
            mock.patch.object(
                self.g["os"],
                "statvfs",
                return_value=SimpleNamespace(f_blocks=100, f_bavail=90, f_bfree=90, f_frsize=4096),
            ),
        ):
            code, receipt = self.run_case("preflight")
        self.assertEqual((code, receipt["status"]), (2, "BLOCKED"))
        self.assertIsNone(receipt["family_prerequisites"]["so101"]["data_root"])
        self.assertIsNone(receipt["family_prerequisites"]["ros_gazebo"]["controllers"])

    def test_preflight_shm_cap_blocks_without_mutation_or_gpu_probe(self):
        self.guarded_root()
        self.patch_globals(
            _module_inventory=lambda: {
                name: {"available": True} for name in ("nexuml", "nexuml_library", "torch")
            },
            _gpu_inventory=mock.Mock(),
        )
        with mock.patch.object(
            self.g["os"],
            "statvfs",
            return_value=SimpleNamespace(f_blocks=10, f_bavail=1, f_bfree=1, f_frsize=1024**3),
        ):
            code, receipt = self.run_case("preflight")
        self.assertEqual((code, receipt["status"]), (2, "BLOCKED"))
        self.assertIn("8 GiB", receipt["reason"])
        self.g["_gpu_inventory"].assert_not_called()

    def test_object_store_observations_identify_actual_raylet_backing(self):
        inventory = runpy.run_path(str(SCRIPT))["_object_store_observations"]
        proc = self.base / "proc"
        raylet = proc / "79"
        raylet.mkdir(parents=True)
        plasma = self.base / "plasma with spaces"
        plasma.mkdir()
        escaped = str(plasma).replace(" ", r"\040")
        (raylet / "cmdline").write_bytes(
            b"/usr/local/bin/raylet\0--object_store_memory=10000000000\0"
            + f"--plasma_directory={plasma}".encode()
            + b"\0"
        )
        # The controller's namespace says RAM; the worker's namespace is authoritative.
        (proc / "self").mkdir()
        for process in (raylet, proc / "self"):
            (process / "ns").mkdir()
            (process / "ns/mnt").symlink_to("mnt:[123]")
        (proc / "self" / "mountinfo").write_text(f"1 0 0:1 / {escaped} rw - tmpfs shm rw\n")
        for filesystem, backing in (
            ("ext4", "disk"),
            ("overlay", "unknown"),
            ("tmpfs", "shared_memory"),
            ("hugetlbfs", "shared_memory"),
            ("fuse.juicefs", "unknown"),
        ):
            with self.subTest(filesystem=filesystem):
                (raylet / "mountinfo").write_text(
                    "1 0 0:1 / / rw - tmpfs root rw\n"
                    f"2 1 259:2 / {escaped} rw - {filesystem} plasma rw\n"
                )
                self.assertEqual(
                    inventory(proc),
                    [
                        {
                            "pid": 79,
                            "bytes": 10_000_000_000,
                            "plasma_directory": str(plasma),
                            "filesystem": filesystem,
                            "backing": backing,
                        }
                    ],
                )
        (raylet / "ns/mnt").unlink()
        (raylet / "ns/mnt").symlink_to("mnt:[456]")
        (raylet / "mountinfo").write_text(f"1 0 259:2 / {escaped} rw - ext4 plasma rw\n")
        self.assertEqual(inventory(proc)[0]["backing"], "unknown")
        (raylet / "ns/mnt").unlink()
        self.assertEqual(inventory(proc)[0]["backing"], "unknown")
        (raylet / "ns/mnt").symlink_to("mnt:[123]")
        (raylet / "mountinfo").write_text(
            f"1 0 259:2 / {escaped} rw - ext4 plasma rw\n2 0 0:1 / {escaped} rw - tmpfs shm rw\n"
        )
        self.assertEqual(inventory(proc)[0]["backing"], "unknown")
        (raylet / "mountinfo").unlink()
        self.assertEqual(inventory(proc)[0]["backing"], "unknown")
        (raylet / "cmdline").write_bytes(b"raylet\0--object_store_memory=invalid\0")
        observation = inventory(proc)[0]
        self.assertIsNone(observation["bytes"])
        self.assertEqual(observation["backing"], "unknown")

    def test_disk_object_store_cap_is_not_a_shared_memory_reservation(self):
        imports = {name: {"available": True} for name in ("nexuml", "nexuml_library", "torch")}
        self.fake_launch(log_text="GPU-mocked-uuid, mock-driver\n")
        self.patch_globals(
            _module_inventory=lambda: imports,
            _gpu_inventory=mock.Mock(return_value={"visible_count": 1}),
        )
        for index, (backing, capacity, expected) in enumerate(
            (
                ("disk", 10_000_000_000, 0),
                ("shared_memory", 10_000_000_000, 2),
                ("unknown", 10_000_000_000, 2),
                ("disk", None, 2),
            )
        ):
            with self.subTest(backing=backing, capacity=capacity):
                self.root = self.base / f"attempt-{index}"
                self.guarded_root()
                self.g["_gpu_inventory"].reset_mock()
                observed = [{"pid": 79, "bytes": capacity, "backing": backing}]
                with (
                    mock.patch.dict(self.g, _object_store_observations=lambda: observed),
                    mock.patch.object(self.g["shutil"], "which", return_value="/mock/nvidia-smi"),
                    mock.patch.object(
                        self.g["os"],
                        "statvfs",
                        return_value=SimpleNamespace(
                            f_blocks=8,
                            f_bavail=8,
                            f_bfree=8,
                            f_frsize=1024**3,
                        ),
                    ),
                ):
                    code, receipt = self.run_case("preflight")
                self.assertEqual(code, expected)
                if code == 0:
                    self.assertEqual(receipt["shared_memory"]["used_bytes"], 0)
                    self.assertEqual(receipt["shared_memory"]["shared_object_store_limit_bytes"], 0)
                    self.assertEqual(
                        receipt["shared_memory"]["disk_object_store_limit_bytes"], 10_000_000_000
                    )
                    self.g["_gpu_inventory"].assert_called_once()
                else:
                    self.g["_gpu_inventory"].assert_not_called()

    def test_missing_object_store_observation_blocks_before_gpu_probe(self):
        self.guarded_root()
        self.patch_globals(
            _module_inventory=lambda: {
                name: {"available": True} for name in ("nexuml", "nexuml_library", "torch")
            },
            _gpu_inventory=mock.Mock(),
            _object_store_observations=lambda: [],
        )
        code, receipt = self.run_case("preflight")
        self.assertEqual(code, 2)
        self.assertIn("unobservable", receipt["reason"])
        self.g["_gpu_inventory"].assert_not_called()

    def test_gpu_inventory_requires_exactly_one_visible_gpu_before_tiny_allocation(self):
        cuda = mock.Mock()
        cuda.is_available.return_value = True
        cuda.device_count.return_value = 2
        cuda.mem_get_info.return_value = (100, 200)
        cuda.get_device_name.return_value = "mocked GPU"
        cuda.get_device_properties.return_value = SimpleNamespace(uuid="GPU-mock")
        torch = SimpleNamespace(
            cuda=cuda, empty=mock.Mock(), version=SimpleNamespace(cuda="mocked")
        )
        with mock.patch.dict(sys.modules, torch=torch):
            with self.assertRaisesRegex(self.g["Blocked"], "exactly one"):
                self.g["_gpu_inventory"]()
            torch.empty.assert_not_called()
            cuda.device_count.return_value = 1
            result = self.g["_gpu_inventory"]()
        torch.empty.assert_called_once_with(1, device="cuda:0")
        cuda.synchronize.assert_called_once_with()
        self.assertEqual((result["free_bytes"], result["allocation_uuid"]), (100, "GPU-mock"))

    def test_policy_validation_rejects_bad_identity_counters_weights_and_training_sidecars(self):
        self.root.mkdir()
        policy = self.root / "checkpoint-policy"
        policy.mkdir()
        checkpoint = self.root / "final.ckpt"
        checkpoint.write_bytes(b"mock checkpoint")
        for name in ("state_dict.pt", "action_adapter_state.pt"):
            (policy / name).write_bytes(f"mock {name}".encode())
        metadata = {
            "artifact_kind": "policy",
            "training_frames": 64,
            "checkpoint": {"sha256": self.g["_sha256"](checkpoint)},
            "policy_files_sha256": {
                name: self.g["_sha256"](policy / name)
                for name in ("state_dict.pt", "action_adapter_state.pt")
            },
        }
        original = copy.deepcopy(metadata)
        saved = {
            "state_dict": {
                "policy.pipeline.actor.weight": 2.0,
                "policy.action_adapter.log_std": 0.1,
            }
        }
        values = {
            "state_dict.pt": {"actor.weight": 2.0},
            "action_adapter_state.pt": {"log_std": 0.1},
            "untrained": {"actor.weight": 1.0},
        }
        torch = SimpleNamespace(
            load=lambda path, **kwargs: values[
                "untrained" if path.parent.name == "untrained" else path.name
            ],
            equal=lambda first, second: first == second,
            isfinite=lambda value: SimpleNamespace(all=lambda: math.isfinite(value)),
        )
        counters = [64, 4]
        self.patch_globals(_checkpoint=lambda path: (saved, *counters))
        with mock.patch.dict(sys.modules, torch=torch):
            (policy / "metadata.json").write_text(json.dumps(metadata))
            result = self.g["_validate_policy"](self.root, checkpoint, 64)
            self.assertEqual((result["training_frames"], result["global_step"]), (64, 4))
            for field, value in (
                ("training_frames", None),
                ("artifact_kind", "pipeline"),
                ("checkpoint", {"sha256": "wrong"}),
                ("policy_files_sha256", {}),
            ):
                with self.subTest(field=field):
                    (policy / "metadata.json").write_text(json.dumps({**original, field: value}))
                    with self.assertRaises(AssertionError):
                        self.g["_validate_policy"](self.root, checkpoint, 64)
            (policy / "metadata.json").write_text(json.dumps(original))
            counters[0] = 32
            with self.assertRaisesRegex(AssertionError, "exact physical budget"):
                self.g["_validate_policy"](self.root, checkpoint, 64)
            counters[0] = 64
            values["untrained"]["actor.weight"] = 2.0
            with self.assertRaisesRegex(AssertionError, "did not change"):
                self.g["_validate_policy"](self.root, checkpoint, 64)
            values["untrained"]["actor.weight"] = 1.0
            values["state_dict.pt"]["actor.weight"] = float("nan")
            with self.assertRaisesRegex(AssertionError, "nonfinite"):
                self.g["_validate_policy"](self.root, checkpoint, 64)
            values["state_dict.pt"]["actor.weight"] = 2.0
            values["action_adapter_state.pt"] = {}
            with self.assertRaisesRegex(AssertionError, "complete checkpoint actor/adapter"):
                self.g["_validate_policy"](self.root, checkpoint, 64)
            values["action_adapter_state.pt"] = {"log_std": 0.1}
            (policy / "training_state.pt").touch()
            with self.assertRaisesRegex(AssertionError, "training-only"):
                self.g["_validate_policy"](self.root, checkpoint, 64)

    def fake_config_imports(self, config):
        class LocalExecution:
            pass

        config.execution = LocalExecution()
        config.model_copy = lambda **kwargs: copy.deepcopy(config)
        config.save = lambda path: path.write_text("mocked yaml")

        class ReinforcementLearning(SimpleNamespace):
            pass

        training = ReinforcementLearning()
        training.__dict__.update(vars(config.training))
        config.training = training

        class PolicyEvaluation:
            @classmethod
            def model_validate(cls, values):
                return cls(**values)

            def __init__(self, **values):
                self.__dict__.update(values)

            def model_dump(self):
                return dict(self.__dict__)

        config.training.evaluation = PolicyEvaluation(
            episodes=0, seed=0, at_start=False, every_n_frames=None
        )
        modules = {
            "nexuml.core.config": SimpleNamespace(
                ResolvedConfig=SimpleNamespace(load=lambda path: config)
            ),
            "nexuml.core.types": SimpleNamespace(
                CallbackSpec=lambda **kw: SimpleNamespace(**kw),
                ExportSpec=lambda **kw: SimpleNamespace(**kw),
                PolicyEvaluationSpec=PolicyEvaluation,
                ReinforcementLearningSpec=ReinforcementLearning,
            ),
            "nexuml.execution.definitions": SimpleNamespace(LocalExecution=LocalExecution),
        }
        patch = mock.patch.dict(sys.modules, modules)
        patch.start()
        self.addCleanup(patch.stop)
        return config

    def test_recipe_config_owns_checkpoint_export_logs_preserves_training_semantics(self):
        self.root.mkdir()
        logging = SimpleNamespace(
            tensorboard=SimpleNamespace(log_dir="shared"),
            dvclive=None,
            diagram=SimpleNamespace(output_dir="shared"),
            temp_artifact_dir="shared",
            mlflow=SimpleNamespace(
                tracking_uri="sqlite:///shared.db",
                run_id="old",
                parent_run_id="old",
                artifact_location=None,
            ),
        )
        config = self.fake_config_imports(
            SimpleNamespace(
                name="cartpole-ppo",
                checkpoint=None,
                tuning=None,
                callbacks=[],
                exports=[],
                logging=logging,
                policy_evaluation=None,
                training=SimpleNamespace(total_frames=64, frames_per_batch=16, seed=4),
            )
        )
        prepared = self.g["_prepare_recipe"](self.config, self.root, {})
        self.assertEqual(
            prepared.callbacks[0].kwargs,
            {
                "dirpath": str(self.root / "checkpoints"),
                "every_n_train_steps": 1,
                "save_top_k": -1,
                "save_last": False,
                "filename": "step-{step}",
            },
        )
        self.assertEqual(prepared.exports[0].output, str(self.root / "policy"))
        self.assertEqual(prepared.logging.mlflow.tracking_uri, (self.root / "logs/mlflow").as_uri())
        self.assertIsNone(prepared.logging.mlflow.run_id)
        self.assertEqual(
            (config.training.total_frames, config.training.frames_per_batch, config.training.seed),
            (64, 16, 4),
        )
        self.assertEqual(config.training.evaluation.episodes, 0)
        config.name = "unapproved-factory"
        with self.assertRaisesRegex(self.g["Blocked"], "21 matrix"):
            self.g["_prepare_recipe"](self.config, self.root, {})
        config.name = "cartpole-ppo"
        config.training = SimpleNamespace()
        with self.assertRaisesRegex(self.g["Blocked"], "reinforcement learning"):
            self.g["_prepare_recipe"](self.config, self.root, {})

    def test_partial_copy_is_readable_nonfinal_immutable_and_has_real_counters(self):
        self.root.mkdir()
        checkpoints = self.root / "checkpoints"
        checkpoints.mkdir()
        source = checkpoints / "step.ckpt"
        source.write_bytes(b"mock checkpoint")
        self.patch_globals(_checkpoint=lambda path: ({}, 32, 7))
        result = self.g["_copy_partial"](self.root, 64)
        self.assertEqual((result["frames"], result["global_step"]), (32, 7))
        self.assertEqual(result["sha256"], self.g["_sha256"](source))
        self.assertEqual((self.root / "partial.ckpt").stat().st_mode & 0o777, 0o400)
        (self.root / "partial.ckpt").unlink()
        self.patch_globals(_checkpoint=lambda path: ({}, 64, 8))
        self.assertIsNone(self.g["_copy_partial"](self.root, 64))


if __name__ == "__main__":
    unittest.main()
