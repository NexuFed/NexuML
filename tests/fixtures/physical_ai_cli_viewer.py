"""Fixed native CLI spectator proof; importing this helper starts nothing."""

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from xml.etree import ElementTree

ENV_BASE = Path("/tmp/ray/nexuml-envs")
VIEWER_TESTS = tuple(
    "tests/core/test_isaaclab_integration.py::" + name
    for name in (
        "test_native_spectator_public_cli_browser_train_replay_and_lifecycle",
        "test_native_spectator_uses_isolated_storage_and_preserves_sensor_trajectory[False]",
        "test_native_spectator_uses_isolated_storage_and_preserves_sensor_trajectory[True]",
        "test_native_spectator_partial_startup_releases_context_listener_and_threads[port-conflict]",
        "test_native_spectator_partial_startup_releases_context_listener_and_threads[startup-failure]",
    )
) + tuple(
    "tests/core/test_isaaclab_viser.py::"
    "test_safe_spectator_native_constructor_and_owned_cleanup[" + mode + "]"
    for mode in ("dispatch", "display", "restart")
)
MISSING_RUNTIME = (
    "native-viewer requires a successful owned private browser-runtime/setup.json "
    "with prepared Node, Playwright, Chrome, SwiftShader and browser library/font/home paths"
)


class MissingViewerRuntime(RuntimeError):
    """The already-prepared private browser runtime is absent or unsafe."""


def _owned_path(path: Path, boundary: Path, *, directory: bool) -> None:
    resolved = path.resolve(strict=True)
    mode = resolved.stat()
    if (
        not path.is_absolute()
        or not path.is_relative_to(boundary)
        or not resolved.is_relative_to(boundary)
        or mode.st_uid != os.getuid()
        or not (stat.S_ISDIR(mode.st_mode) if directory else stat.S_ISREG(mode.st_mode))
    ):
        raise ValueError


def _runtime(env: dict) -> tuple[dict, Path, str]:
    try:
        venv = Path(sys.prefix)
        if (
            not venv.is_absolute()
            or venv.parent != ENV_BASE
            or re.fullmatch(r"[0-9a-f]{32}", venv.name) is None
            or env.get("UV_PROJECT_ENVIRONMENT") != str(venv)
            or venv.resolve(strict=True) != venv
            or not (venv / "bin/python").is_file()
        ):
            raise ValueError
        _owned_path(venv, venv, directory=True)
        _owned_path(venv / "pyvenv.cfg", venv, directory=False)
        owned = venv / "browser-runtime"
        if owned.resolve(strict=True) != owned:
            raise ValueError
        _owned_path(owned, venv, directory=True)
        receipt = owned / "setup.json"
        if receipt.is_symlink():
            raise ValueError
        _owned_path(receipt, owned, directory=False)
        content = receipt.read_bytes()
        setup = json.loads(content)
        if not isinstance(setup, dict) or setup.get("status") != "success":
            raise ValueError
        paths = setup["paths"]
        if not isinstance(paths, dict):
            raise ValueError
        for key in (
            "node",
            "playwrightModulePath",
            "chromeBinary",
            "fontsConf",
            "fontCache",
            "homePath",
        ):
            if not isinstance(paths[key], str):
                raise ValueError
            _owned_path(
                Path(paths[key]),
                owned,
                directory=key in {"playwrightModulePath", "fontCache", "homePath"},
            )
        libraries = (
            owned / "clients/usr/lib/x86_64-linux-gnu",
            owned / "clients/lib/x86_64-linux-gnu",
            venv / "native-runtime/usr/lib/x86_64-linux-gnu",
        )
        if paths["libraryPrefix"] != ":".join(map(str, libraries)):
            raise ValueError
        for library in libraries:
            _owned_path(library, venv, directory=True)
        icd = Path(paths["chromeBinary"]).parent / "vk_swiftshader_icd.json"
        _owned_path(icd, owned, directory=False)
        return paths, icd, hashlib.sha256(content).hexdigest()
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        raise MissingViewerRuntime(MISSING_RUNTIME) from None


def run_viewer(record: dict, root: Path, env: dict, stage) -> None:
    record["regression"] = {
        "suite": "native-viewer",
        "full_suite_gate": "UNVERIFIED",
        "native_proof": False,
        "scope": (
            "fixed public-CLI spectator/storage/startup proof; "
            "full V gate requires retained visual/private-route review"
        ),
    }
    record["checks"]["native_cli_viewer"] = "UNVERIFIED"
    paths, icd, digest = _runtime(env)
    record["regression"]["browser_setup_sha256"] = digest
    private = {
        **env,
        "NEXUML_RUN_SIMULATOR_TESTS": "1",
        "NEXUML_RUN_SYSTEM_TESTS": "1",
        "NEXUML_PLAYWRIGHT_MODULE": paths["playwrightModulePath"],
        "NEXUML_CHROMIUM_EXECUTABLE": paths["chromeBinary"],
        "NEXUML_BROWSER_SWIFTSHADER_ICD": str(icd),
        "PATH": str(Path(paths["node"]).parent) + (":" + env["PATH"] if env.get("PATH") else ""),
    }
    report = root / "native-viewer-junit.xml"
    stage(
        record,
        "native-viewer",
        [
            sys.executable,
            "-m",
            "pytest",
            *VIEWER_TESTS,
            "-q",
            "--maxfail=1",
            "--tb=short",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(root / "native-viewer-tests"),
            "--junitxml",
            str(report),
        ],
        private,
    )
    try:
        results = ElementTree.parse(report)
    except (OSError, ElementTree.ParseError):
        raise AssertionError("native-viewer requires its exact passing JUnit results") from None
    cases = results.findall(".//testcase")
    expected = sorted(selector.rsplit("::", 1)[1] for selector in VIEWER_TESTS)
    if (
        sorted(case.get("name", "") for case in cases) != expected
        or any(results.find(".//" + tag) is not None for tag in ("skipped", "failure", "error"))
        or any(
            element.get(key, "0") != "0"
            for element in results.iter()
            if element.tag in {"testsuites", "testsuite"}
            for key in ("skipped", "failures", "errors")
        )
    ):
        raise AssertionError("every fixed native-viewer test must execute and pass; no skips")
    record["regression"]["native_proof"] = True
    record["checks"]["native_cli_viewer"] = "PASS"
