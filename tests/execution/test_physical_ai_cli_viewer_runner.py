"""Hermetic fixed viewer-runner checks; never start CLI, browser or native code."""

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock
import xml.etree.ElementTree as ET

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "fixtures/physical_ai_cli_viewer.py"
PATH_KINDS = {
    "node": False,
    "playwrightModulePath": True,
    "chromeBinary": False,
    "fontsConf": False,
    "fontCache": True,
    "homePath": True,
}
NAMES = (
    "test_native_spectator_public_cli_browser_train_replay_and_lifecycle",
    "test_native_spectator_uses_isolated_storage_and_preserves_sensor_trajectory[False]",
    "test_native_spectator_uses_isolated_storage_and_preserves_sensor_trajectory[True]",
    "test_native_spectator_partial_startup_releases_context_listener_and_threads[port-conflict]",
    "test_native_spectator_partial_startup_releases_context_listener_and_threads[startup-failure]",
    "test_safe_spectator_native_constructor_and_owned_cleanup[dispatch]",
    "test_safe_spectator_native_constructor_and_owned_cleanup[display]",
    "test_safe_spectator_native_constructor_and_owned_cleanup[restart]",
)


def test_import_is_stdlib_only_and_inert():
    tree = ast.parse(SCRIPT.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] in sys.stdlib_module_names for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module and node.module.split(".")[0] in sys.stdlib_module_names
    with (
        mock.patch.object(subprocess, "run", side_effect=AssertionError("unexpected execution")),
        mock.patch.object(subprocess, "Popen", side_effect=AssertionError("unexpected execution")),
        mock.patch.object(os, "system", side_effect=AssertionError("unexpected execution")),
        mock.patch.object(
            Path, "read_bytes", side_effect=AssertionError("unexpected receipt read")
        ),
    ):
        helper = runpy.run_path(str(SCRIPT))
    assert helper["VIEWER_TESTS"] == tuple(
        "tests/core/test_isaaclab_integration.py::" + name for name in NAMES[:5]
    ) + tuple("tests/core/test_isaaclab_viser.py::" + name for name in NAMES[5:])
    assert issubclass(helper["MissingViewerRuntime"], RuntimeError)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    helper = runpy.run_path(str(SCRIPT))
    run = helper["run_viewer"]
    base = tmp_path / "envs ; $(literal)"
    venv = base / ("a" * 32)
    owned = venv / "browser-runtime"
    (venv / "bin").mkdir(parents=True)
    interpreter = tmp_path / "managed-python"
    interpreter.touch()
    (venv / "bin/python").symlink_to(interpreter)
    (venv / "pyvenv.cfg").touch()
    owned.mkdir(mode=0o700)
    paths = {}
    for key, directory in PATH_KINDS.items():
        path = owned / key / ("bin/node" if key == "node" else "asset")
        path.parent.mkdir(parents=True, exist_ok=True)
        if directory:
            path.mkdir()
        else:
            path.touch()
        paths[key] = str(path)
    libraries = [
        owned / "clients/usr/lib/x86_64-linux-gnu",
        owned / "clients/lib/x86_64-linux-gnu",
        venv / "native-runtime/usr/lib/x86_64-linux-gnu",
    ]
    for path in libraries:
        path.mkdir(parents=True)
    paths["libraryPrefix"] = ":".join(map(str, libraries))
    icd = Path(paths["chromeBinary"]).parent / "vk_swiftshader_icd.json"
    icd.touch()
    receipt = owned / "setup.json"
    setup = {"status": "success", "paths": paths, "private_credential": "never-echo-this"}
    receipt.write_text(json.dumps(setup))
    monkeypatch.setitem(run.__globals__, "ENV_BASE", base)
    monkeypatch.setitem(
        run.__globals__, "sys", SimpleNamespace(prefix=str(venv), executable=sys.executable)
    )
    root = tmp_path / "attempt ; $(literal)"
    root.mkdir()
    env = {
        "UV_PROJECT_ENVIRONMENT": str(venv),
        "PATH": "/native/bin:/ordinary/bin",
        "HOME": "/ordinary home",
        "LD_LIBRARY_PATH": "/vendor/nvidia:/normal/native",
        "FONTCONFIG_FILE": "/ordinary/fonts.conf",
        "XDG_CACHE_HOME": "/ordinary/cache",
        "CUDA_VISIBLE_DEVICES": "owned-device",
        "NEXUML_RUN_SIMULATOR_TESTS": "0",
        "NEXUML_RUN_SYSTEM_TESTS": "0",
        "NEXUML_RUN_HARDWARE_TESTS": "0",
        "NEXUML_RUN_DATA_TESTS": "0",
        "PRIVATE_CREDENTIAL": "never-echo-this",
    }
    record = {"checks": {"other": "PASS"}, "gates": {"V": "UNVERIFIED", "L": "UNVERIFIED"}}
    return SimpleNamespace(
        helper=helper,
        run=run,
        venv=venv,
        owned=owned,
        receipt=receipt,
        setup=setup,
        paths=paths,
        icd=icd,
        root=root,
        env=env,
        record=record,
    )


def write_junit(root, names=NAMES):
    results = ET.Element("testsuites")
    suite = ET.SubElement(
        results, "testsuite", tests=str(len(names)), skipped="0", failures="0", errors="0"
    )
    for name in names:
        ET.SubElement(suite, "testcase", name=name)
    ET.ElementTree(results).write(root / "native-viewer-junit.xml")
    return results


def assert_unverified(prepared):
    assert prepared.record["regression"]["native_proof"] is False
    assert prepared.record["regression"]["full_suite_gate"] == "UNVERIFIED"
    assert prepared.record["checks"] == {"other": "PASS", "native_cli_viewer": "UNVERIFIED"}
    assert prepared.record["gates"] == {"V": "UNVERIFIED", "L": "UNVERIFIED"}


@pytest.mark.parametrize("path_value", ["/native/bin:/ordinary/bin", "", None])
def test_fixed_stage_preserves_native_environment_and_limits_claims(prepared, path_value):
    p = prepared
    if path_value is None:
        del p.env["PATH"]
    else:
        p.env["PATH"] = path_value
    before = p.env.copy()
    global_before = os.environ.copy()

    def stage(record, name, argv, env):
        assert record is p.record and name == "native-viewer"
        assert_unverified(p)
        assert argv == [
            sys.executable,
            "-m",
            "pytest",
            *("tests/core/test_isaaclab_integration.py::" + name for name in NAMES[:5]),
            *("tests/core/test_isaaclab_viser.py::" + name for name in NAMES[5:]),
            "-q",
            "--maxfail=1",
            "--tb=short",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(p.root / "native-viewer-tests"),
            "--junitxml",
            str(p.root / "native-viewer-junit.xml"),
        ]
        expected = {
            **before,
            "NEXUML_RUN_SIMULATOR_TESTS": "1",
            "NEXUML_RUN_SYSTEM_TESTS": "1",
            "NEXUML_PLAYWRIGHT_MODULE": p.paths["playwrightModulePath"],
            "NEXUML_CHROMIUM_EXECUTABLE": p.paths["chromeBinary"],
            "NEXUML_BROWSER_SWIFTSHADER_ICD": str(p.icd),
            "PATH": str(Path(p.paths["node"]).parent) + (":" + path_value if path_value else ""),
        }
        assert env == expected and env is not p.env
        assert "NEXUML_SPECTATOR_PYTHON" not in env
        write_junit(p.root)

    launch = mock.Mock(side_effect=stage)
    p.run(p.record, p.root, p.env, launch)
    launch.assert_called_once()
    assert p.env == before and dict(os.environ) == global_before
    assert p.record["regression"] == {
        "suite": "native-viewer",
        "full_suite_gate": "UNVERIFIED",
        "native_proof": True,
        "scope": (
            "fixed public-CLI spectator/storage/startup proof; "
            "full V gate requires retained visual/private-route review"
        ),
        "browser_setup_sha256": hashlib.sha256(p.receipt.read_bytes()).hexdigest(),
    }
    assert p.record["checks"] == {"other": "PASS", "native_cli_viewer": "PASS"}
    assert p.record["gates"] == {"V": "UNVERIFIED", "L": "UNVERIFIED"}
    assert "never-echo-this" not in json.dumps(p.record)


@pytest.mark.parametrize("key", [*PATH_KINDS, "libraryPrefix"])
@pytest.mark.parametrize(
    "fault", ["missing-key", "invalid-value", "missing-path", "wrong-kind", "foreign-symlink"]
)
def test_invalid_receipt_paths_block_before_stage(prepared, key, fault):
    p = prepared
    if fault == "missing-key":
        del p.setup["paths"][key]
    elif fault == "invalid-value" or key == "libraryPrefix":
        p.setup["paths"][key] = "https://user:never-echo-this@example.invalid/path"
    else:
        path = Path(p.paths[key])
        path.rmdir() if PATH_KINDS[key] else path.unlink()
        if fault == "wrong-kind":
            path.touch() if PATH_KINDS[key] else path.mkdir()
        elif fault == "foreign-symlink":
            target = p.root / "never-echo-this"
            target.mkdir() if PATH_KINDS[key] else target.touch()
            path.symlink_to(target, target_is_directory=PATH_KINDS[key])
    p.receipt.write_text(json.dumps(p.setup))
    assert_blocked(p)


def assert_blocked(p):
    stage = mock.Mock()
    before = copy.deepcopy(p.env)
    with pytest.raises(p.helper["MissingViewerRuntime"]) as caught:
        p.run(p.record, p.root, p.env, stage)
    assert str(caught.value) == p.helper["MISSING_RUNTIME"]
    assert caught.value.__suppress_context__
    assert "never-echo-this" not in str(caught.value)
    assert "never-echo-this" not in json.dumps(p.record)
    assert p.env == before
    stage.assert_not_called()
    assert_unverified(p)


@pytest.mark.parametrize(
    "fault",
    [
        "missing-receipt",
        "malformed-json",
        "non-object",
        "failed-status",
        "missing-status",
        "missing-paths",
        "invalid-paths",
        "receipt-symlink",
        "receipt-directory",
        "missing-icd",
        "foreign-icd",
        "icd-directory",
        "foreign-runtime",
        "foreign-venv",
        "missing-venv-config",
        "mismatched-prefix",
        "missing-prefix",
        "unowned",
        "missing-library",
    ],
)
def test_missing_or_unsafe_runtime_blocks_before_stage(prepared, monkeypatch, fault):
    p = prepared
    if fault in {"missing-receipt", "receipt-symlink", "receipt-directory"}:
        p.receipt.unlink()
        if fault == "receipt-symlink":
            target = p.root / "setup.json"
            target.write_text(json.dumps(p.setup))
            p.receipt.symlink_to(target)
        elif fault == "receipt-directory":
            p.receipt.mkdir()
    elif fault in {"malformed-json", "non-object"}:
        p.receipt.write_text("never-echo-this" if fault == "malformed-json" else "[]")
    elif fault in {"failed-status", "missing-status", "missing-paths", "invalid-paths"}:
        if fault == "failed-status":
            p.setup["status"] = "failed: never-echo-this"
        elif fault == "invalid-paths":
            p.setup["paths"] = ["never-echo-this"]
        else:
            del p.setup["status" if fault == "missing-status" else "paths"]
        p.receipt.write_text(json.dumps(p.setup))
    elif fault in {"missing-icd", "foreign-icd", "icd-directory"}:
        p.icd.unlink()
        if fault == "foreign-icd":
            target = p.root / "never-echo-this"
            target.touch()
            p.icd.symlink_to(target)
        elif fault == "icd-directory":
            p.icd.mkdir()
    elif fault in {"foreign-runtime", "foreign-venv"}:
        path = p.owned if fault == "foreign-runtime" else p.venv
        target = p.root / "foreign"
        path.rename(target)
        path.symlink_to(target, target_is_directory=True)
    elif fault == "missing-venv-config":
        (p.venv / "pyvenv.cfg").unlink()
    elif fault == "mismatched-prefix":
        p.env["UV_PROJECT_ENVIRONMENT"] = "/never-echo-this"
    elif fault == "missing-prefix":
        del p.env["UV_PROJECT_ENVIRONMENT"]
    elif fault == "unowned":
        monkeypatch.setitem(p.run.__globals__, "os", SimpleNamespace(getuid=lambda: -1))
    elif fault == "missing-library":
        Path(p.paths["libraryPrefix"].split(":")[0]).rmdir()
    assert_blocked(p)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "malformed",
        "empty",
        "missing-case",
        "extra-case",
        "wrong-name",
        "duplicate",
        "skipped",
        "failure",
        "error",
        "suite-error",
        "suite-skipped-count",
        "suite-failure-count",
    ],
)
def test_junit_errors_are_failures_not_missing_prerequisites(prepared, fault):
    p = prepared

    def stage(*_):
        if fault == "missing":
            return
        report = p.root / "native-viewer-junit.xml"
        if fault == "malformed":
            report.write_text("not XML")
            return
        names = {
            "empty": (),
            "missing-case": NAMES[:-1],
            "extra-case": (*NAMES, "extra"),
            "wrong-name": (*NAMES[:-1], "wrong"),
            "duplicate": (*NAMES[:-1], NAMES[0]),
        }.get(fault, NAMES)
        results = write_junit(p.root, names)
        if fault in {"skipped", "failure", "error"}:
            ET.SubElement(results.findall(".//testcase")[0], fault)
        elif fault == "suite-error":
            ET.SubElement(results[0], "error")
        elif fault in {"suite-skipped-count", "suite-failure-count"}:
            results[0].set("skipped" if fault == "suite-skipped-count" else "failures", "1")
        ET.ElementTree(results).write(report)

    launch = mock.Mock(side_effect=stage)
    with pytest.raises(AssertionError):
        p.run(p.record, p.root, p.env, launch)
    launch.assert_called_once()
    assert_unverified(p)


def test_stage_failure_preserves_original_exception_without_retry(prepared):
    p = prepared
    original = RuntimeError("original stage failure")
    launch = mock.Mock(side_effect=original)
    with pytest.raises(RuntimeError) as caught:
        p.run(p.record, p.root, p.env, launch)
    assert caught.value is original
    launch.assert_called_once()
    assert not (p.root / "native-viewer-junit.xml").exists()
    assert_unverified(p)
