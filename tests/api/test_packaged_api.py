"""Opt-in proof of the built Python distribution without altering an installation."""

import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile

import pytest


def test_built_wheel_startup(tmp_path):
    wheel = os.environ.get("NEXUML_API_WHEEL")
    if not wheel:
        pytest.skip("Set NEXUML_API_WHEEL to a wheel built from this worktree")
    target = tmp_path / "installed distribution"
    # Extract the pure-Python wheel only in this private test location. Runtime deps
    # come from this test's interpreter; no uv/pip install or user environment mutation.
    with zipfile.ZipFile(wheel) as archive:
        assert all(not name.startswith("studio/") for name in archive.namelist())
        archive.extractall(target)
    working = tmp_path / "working directory"
    working.mkdir()
    env = {**os.environ, "PYTHONPATH": str(target), "NEXUML_API_TOKEN": secrets.token_urlsafe(32)}
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with (tmp_path / "server-output.txt").open("w+") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "nexuml.cli.main",
                "serve",
                "--directory",
                str(working),
                "--port",
                str(port),
                "--origin",
                "http://127.0.0.1:3000",
            ],
            cwd=working,
            env=env,
            stdout=output,
            stderr=output,
        )
        try:
            deadline = time.monotonic() + 30
            while True:
                if process.poll() is not None:
                    output.seek(0)
                    pytest.fail(output.read())
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/api/v1/runtime",
                    headers={"Authorization": f"Bearer {env['NEXUML_API_TOKEN']}"},
                )
                try:
                    with urllib.request.urlopen(request, timeout=1) as response:
                        identity = json.load(response)
                    break
                except urllib.error.URLError:
                    assert time.monotonic() < deadline, "API startup exceeded deadline"
                    time.sleep(0.1)
            assert identity["python_executable"] == sys.executable
            assert identity["working_directory"] == str(working.resolve())
            assert identity["interface_version"] == 2
            check = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import builtins; original=builtins.__import__; "
                    "builtins.__import__=lambda name,*a,**kw: "
                    "(_ for _ in ()).throw(ImportError('blocked API')) "
                    "if name in {'fastapi','uvicorn','websockets'} else original(name,*a,**kw); "
                    "from nexuml.cli.main import app; "
                    "from typer.testing import CliRunner; "
                    "assert CliRunner().invoke(app,['--help']).exit_code == 0",
                ],
                cwd=working,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert check.returncode == 0, check.stderr
        finally:
            process.terminate()
            process.wait(timeout=15)
        output.seek(0)
        assert env["NEXUML_API_TOKEN"] not in output.read()
    assert not list(working.glob("*.env"))
    assert not (working / "pyproject.toml").exists()
