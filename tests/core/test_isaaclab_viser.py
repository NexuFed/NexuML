"""Bounded private spectator qualification and the retained bare-native deadlock diagnostic."""

import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

import pytest


@pytest.mark.requires_system
def test_pinned_viser_startup_failure_never_signals_constructor(tmp_path):
    """Confirm the upstream failure without leaving its stuck constructor alive."""
    root = Path(__file__).resolve().parents[2]
    assert shutil.disk_usage(root).free > 4 * 2**30
    process = subprocess.Popen(
        [sys.executable, "-u", str(root / "tests/fixtures/isaaclab_viser_startup_probe.py")],
        cwd=root,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = ""
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            output, _ = process.communicate(timeout=8)
    finally:
        if process.poll() is None:
            process.kill()
        retained, _ = process.communicate(timeout=5)
        output = retained or output
        (tmp_path / "viser-startup-blocker.log").write_text(output)
    assert "OWNED_LOOPBACK_65535_RESERVED" in output, output
    assert "OverflowError: bind(): port must be 0-65535" in output, output
    assert "CONSTRUCTOR_RETURNED" not in output, output
    assert process.returncode is not None
    with socket.socket() as released:
        released.bind(("127.0.0.1", 65535))
    assert shutil.disk_usage(root).free > 4 * 2**30


@pytest.mark.requires_system
@pytest.mark.parametrize(
    "mode",
    [
        "static",
        "prechecks",
        "entry-sigterm",
        "entry-sigint",
        "stop-late-ready",
        "stop-readiness-deadline",
        "dispatch",
        "display",
        "restart",
        "scene-error",
        "scene-interrupt",
        "scene-sigterm",
        "scene-cleanup-error",
        "readiness-timeout",
        "early-failure",
        "closed-loop",
        "port-mismatch",
        "uncooperative-peer",
    ],
)
def test_safe_spectator_native_constructor_and_owned_cleanup(mode, tmp_path):
    root = Path(__file__).resolve().parents[2]
    assert shutil.disk_usage(root).free > 4 * 2**30
    log_path = tmp_path / f"viser-safe-{mode}.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", str(root / "tests/fixtures/isaaclab_viser_safe_probe.py"), mode],
            cwd=root,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            process.wait(timeout=65)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
    output = log_path.read_text()
    assert process.returncode == 0, output
    assert (
        "STATIC_IMPORT_AND_FAIL_CLOSED_QUALIFICATION_VERIFIED"
        if mode == "static"
        else "OWNED_CLEANUP_SIGNAL_RESTORE_AND_VENDOR_IDENTITY_VERIFIED"
    ) in output, output
    assert shutil.disk_usage(root).free > 4 * 2**30
