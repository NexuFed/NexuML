"""Explicit release test setup, never used by Studio's installed launcher."""

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def main() -> None:
    """Install built distributions into isolated tool/project smoke environments.

    Raises:
        RuntimeError: If the selected installed API cannot start or reach readiness.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--library-wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("tool", "project"), required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    home = root / "isolated user home"
    home.mkdir(exist_ok=True)
    env = {
        **os.environ,
        "HOME": str(home),
        "USERPROFILE": str(home),
        "UV_TOOL_DIR": str(root / "tools"),
        "UV_TOOL_BIN_DIR": str(root / "tool bin"),
        "UV_PYTHON_DOWNLOADS": "never",
    }
    # Verify packaging, not an editable checkout or its configured GPU index.
    env.pop("PYTHONPATH", None)
    index = (
        []
        if os.name == "posix" and os.uname().sysname == "Darwin"
        else [
            "--index",
            "https://download.pytorch.org/whl/cpu",
        ]
    )
    wheel = str(args.wheel.resolve()) + "[api]"
    library = str(args.library_wheel.resolve())
    if args.mode == "tool":
        subprocess.run(
            [
                "uv",
                "tool",
                "install",
                "--no-config",
                "--python",
                args.python,
                *index,
                "--with",
                library,
                wheel,
            ],
            env=env,
            check=True,
        )
        command = [str(root / "tool bin" / ("nexuml.exe" if os.name == "nt" else "nexuml"))]
    else:
        venv = root / "project environment"
        subprocess.run(
            ["uv", "venv", "--no-config", "--python", args.python, str(venv)], env=env, check=True
        )
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--no-config",
                "--python",
                str(python),
                *index,
                wheel,
                library,
            ],
            env=env,
            check=True,
        )
        command = [str(python), "-m", "nexuml.cli.main"]
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    token = secrets.token_urlsafe(32)
    process = subprocess.Popen(
        [
            *command,
            "serve",
            "--directory",
            str(root),
            "--port",
            str(port),
            "--origin",
            "http://127.0.0.1:3000",
        ],
        env={**env, "NEXUML_API_TOKEN": token},
    )
    try:
        deadline = time.monotonic() + 60
        while True:
            if process.poll() is not None:
                raise RuntimeError("Selected built installation exited before runtime handshake")
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/v1/runtime",
                headers={"Authorization": f"Bearer {token}"},
            )
            try:
                with urllib.request.urlopen(request, timeout=1) as response:
                    identity = json.load(response)
                break
            except urllib.error.URLError:
                if time.monotonic() > deadline:
                    raise RuntimeError("Built installation startup timed out") from None
                time.sleep(0.1)
        assert identity["interface_version"] == 1
        assert identity["working_directory"] == str(root)
        executable = Path(identity["python_executable"])
        script = executable.parent / ("nexuml.exe" if os.name == "nt" else "nexuml")
        selected = {"STUDIO_TEST_PYTHON": str(executable), "STUDIO_TEST_NEXUML": str(script)}
        (root / "runtime.json").write_text(json.dumps(selected), encoding="utf-8")
        if os.environ.get("GITHUB_ENV"):
            with Path(os.environ["GITHUB_ENV"]).open("a", encoding="utf-8") as output:
                for key, value in selected.items():
                    output.write(f"{key}={value}\n")
        print(f"Built {args.mode} runtime verified: {executable}")
    finally:
        process.terminate()
        process.wait(timeout=15)


if __name__ == "__main__":
    main()
