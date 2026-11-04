"""Remote Chromium launch/forced-parent-death proof, never a browser UI/V pass."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def main(root, manifest):
    """Run the fixed browser under a dedicated nested guard and verify owned cleanup.

    Raises:
        AssertionError: Real browser launch or the expected failure/cleanup proof is absent.
    """
    root.mkdir(mode=0o700)
    output = root / "browser"
    output.mkdir(mode=0o700)
    setup = json.loads(
        (Path(os.environ["UV_PROJECT_ENVIRONMENT"]) / "browser-runtime/setup.json").read_text()
    )
    if setup["status"] != "success":
        raise AssertionError("successful private browser setup required")
    receipt = root / "guard.json"
    log = root / "guard.log"
    fixtures = Path(__file__).parent
    with log.open("xb") as stream:
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                str(fixtures / "physical_ai_guard.py"),
                "--receipt",
                str(receipt),
                "--deadline-seconds",
                "90",
                "--disk-floor-bytes",
                str(4 * 2**30),
                "--disk-path",
                "/tmp/ray",
                "--",
                setup["paths"]["node"],
                str(fixtures / "physical_ai_logger_browser.cjs"),
                str(output),
                str(manifest),
                "--guard-forced-exit",
            ],
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=105,
        )
    guard = json.loads(receipt.read_text())
    launched = [
        line.removeprefix("NEXUML_BROWSER_LAUNCHED=")
        for line in log.read_text().splitlines()
        if line.startswith("NEXUML_BROWSER_LAUNCHED=")
    ]
    if len(launched) != 1:
        raise AssertionError("real sandboxed Chromium launch marker absent; inspect guard.log")
    identity = json.loads(launched[0])
    if (
        identity["version"] != "153.0.8010.12"
        or identity["pid"] != guard["pid"]
        or result.returncode != 137
        or guard["exit_code"] != 137
        or guard["child_returncode"] != -9
        or guard["process_ownership"] != "dedicated-linux-child-subreaper-pidfds"
        or guard["errors"] != []
        or guard["reason"] != "signal"
    ):
        raise AssertionError("browser identity, forced failure or owned guard cleanup differs")
    proof = {
        "scope": "real sandboxed Chromium launch and forced-parent-death cleanup only",
        "browser_version": identity["version"],
        "expected_exit_code": 137,
        "absence_proof": "dedicated guard returned only after kernel ECHILD",
        "V_gate": "UNVERIFIED",
        "cleanup": "PASS",
    }
    with (root / "proof.json").open("x") as stream:
        json.dump(proof, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("NEXUML_BROWSER_FORCED_PARENT_DEATH_CLEANUP_VERIFIED", flush=True)


if __name__ == "__main__":
    assert len(sys.argv) == 3
    main(Path(sys.argv[1]), Path(sys.argv[2]))
