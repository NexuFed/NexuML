"""Private Jammy client libraries; dependency setup only, never a native/GPU probe.

Run inside the registered SDK job's existing disk/wall guard after uv creates its
private venv. The caller owns interpreter/library-path configuration afterward.
Pins come from authenticated Ubuntu indexes retained by the qualification owner.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import TypedDict
import urllib.request


class _Package(TypedDict):
    name: str
    version: str
    filename: str
    sha256: str
    size: int


ENV_BASE = Path("/tmp/ray/nexuml-envs")
ARCHIVE = "https://archive.ubuntu.com/ubuntu/"
PACKAGES: tuple[_Package, ...] = (
    {
        "name": "libgl1",
        "version": "1.4.0-1",
        "filename": "pool/main/libg/libglvnd/libgl1_1.4.0-1_amd64.deb",
        "sha256": "4fb2bbff94bb1a59bbdb40041e7a8ab58db2a003a151f4055357af03f0b78666",
        "size": 110500,
    },
    {
        "name": "libglx0",
        "version": "1.4.0-1",
        "filename": "pool/main/libg/libglvnd/libglx0_1.4.0-1_amd64.deb",
        "sha256": "950068e8a3ad3905058d21a8c870c37989e48c0a537a7c57231c277af19f5395",
        "size": 41024,
    },
    {
        "name": "libglvnd0",
        "version": "1.4.0-1",
        "filename": "pool/main/libg/libglvnd/libglvnd0_1.4.0-1_amd64.deb",
        "sha256": "6956d8be93b160b439a043141a061adfd54561d877423a956ace84d2fb3bfb4e",
        "size": 73582,
    },
    {
        "name": "libegl1",
        "version": "1.4.0-1",
        "filename": "pool/main/libg/libglvnd/libegl1_1.4.0-1_amd64.deb",
        "sha256": "4a35c0e925e15a076e7ce11d7c76f8ecb16615fc347c2c09afa4c205a7ca8ec4",
        "size": 28564,
    },
    {
        "name": "libxt6",
        "version": "1:1.2.1-1",
        "filename": "pool/main/libx/libxt/libxt6_1.2.1-1_amd64.deb",
        "sha256": "d2da572bc7200facc5683ccf349c8105fd5bd8d2c3755a299ee8569c71f6bdc1",
        "size": 176536,
    },
    {
        "name": "libice6",
        "version": "2:1.0.10-1build2",
        "filename": "pool/main/libi/libice/libice6_1.0.10-1build2_amd64.deb",
        "sha256": "eea6d52d12ad610d70b0f70c825ed3c560dafb09f4c272e3c6b0b493d984d13f",
        "size": 42606,
    },
    {
        "name": "libsm6",
        "version": "2:1.2.3-1build2",
        "filename": "pool/main/libs/libsm/libsm6_1.2.3-1build2_amd64.deb",
        "sha256": "e652e286a79d9e8e15189d79b290bdce20aca83651f3dfed1b9b7dc0bbf0702f",
        "size": 16736,
    },
    {
        "name": "libxrandr2",
        "version": "2:1.5.2-1build1",
        "filename": "pool/main/libx/libxrandr/libxrandr2_1.5.2-1build1_amd64.deb",
        "sha256": "5fbce1f1ec150c715fd83eeef5626363580218db09cedc8ac888b172476b0c82",
        "size": 20394,
    },
    {
        "name": "libxrender1",
        "version": "1:0.9.10-1build4",
        "filename": "pool/main/libx/libxrender/libxrender1_0.9.10-1build4_amd64.deb",
        "sha256": "a68f6db4320b5963dd499b83b6cc8c7b5cd39b05e84862d5945da9244e4c96d6",
        "size": 19678,
    },
    {
        "name": "libxfixes3",
        "version": "1:6.0.0-1",
        "filename": "pool/main/libx/libxfixes/libxfixes3_6.0.0-1_amd64.deb",
        "sha256": "f627555440060edf19801d77d369c99547cf47123fe4e30c5bdb8b664f0b4467",
        "size": 11656,
    },
    {
        "name": "libbsd0",
        "version": "0.11.5-1",
        "filename": "pool/main/libb/libbsd/libbsd0_0.11.5-1_amd64.deb",
        "sha256": "09367acdf59f28ffb71ba7bc36c516cff03b2765930cf9be7bb7ea7273e4c312",
        "size": 44760,
    },
    {
        "name": "libmd0",
        "version": "1.0.4-1build1",
        "filename": "pool/main/libm/libmd/libmd0_1.0.4-1build1_amd64.deb",
        "sha256": "bcdad5e400aeb660f0337a1c5c1a77e3effa91d0803e64ce2d000e9006f67cdb",
        "size": 22964,
    },
    {
        "name": "libuuid1",
        "version": "2.37.2-4ubuntu3.6",
        "filename": "pool/main/u/util-linux/libuuid1_2.37.2-4ubuntu3.6_amd64.deb",
        "sha256": "97eaa1090c2e7ddaad55d58f9d0d7907d64b70f89228405d68a7ce7d56a89d67",
        "size": 23978,
    },
)
# libglx0/libegl1 declare Mesa in APT metadata; extract only GLVND client loaders for the
# existing NVIDIA vendor, never install Mesa or claim this is an APT closure.


def main():
    if len(sys.argv) != 1:
        raise ValueError("native setup accepts no arguments")
    release = platform.freedesktop_os_release()
    if (
        sys.platform != "linux"
        or release.get("ID") != "ubuntu"
        or release.get("VERSION_ID") != "22.04"
        or platform.machine() != "x86_64"
    ):
        raise RuntimeError("native setup requires Ubuntu 22.04 amd64")
    if (
        len(PACKAGES) != 13
        or len({package["name"] for package in PACKAGES}) != 13
        or len({Path(package["filename"]).name for package in PACKAGES}) != 13
    ):
        raise ValueError("native package lock must contain 13 unique packages/files")
    venv = Path(os.environ["UV_PROJECT_ENVIRONMENT"])
    if (
        not venv.is_absolute()
        or venv.parent != ENV_BASE
        or re.fullmatch(r"[0-9a-f]{32}", venv.name) is None
        or venv.resolve(strict=True) != venv
        or not venv.is_dir()
        or venv.stat().st_uid != os.getuid()
        or not (venv / "pyvenv.cfg").is_file()
        or not (venv / "bin/python").is_file()
    ):
        raise ValueError("UV_PROJECT_ENVIRONMENT must be an owned real private SDK venv")
    if shutil.which("dpkg-deb") is None or shutil.which("ldd") is None:
        raise RuntimeError("dpkg-deb and ldd must already be installed")
    destination = venv / "native-runtime"
    destination.mkdir(mode=0o700)
    record = {
        "directory": str(destination),
        "packages": PACKAGES,
        "libraries": {},
        "ldd": [],
        "status": "failed",
    }
    try:
        cache = destination / "cache"
        cache.mkdir(mode=0o700)
        for package in PACKAGES:
            url = ARCHIVE + package["filename"]
            with urllib.request.urlopen(url, timeout=30) as response:
                if response.geturl() != url:
                    raise ValueError("archive redirects are not permitted")
                content = response.read(package["size"] + 1)
            if len(content) != package["size"]:
                raise ValueError(f"size mismatch: {package['name']}")
            if hashlib.sha256(content).hexdigest() != package["sha256"]:
                raise ValueError(f"SHA256 mismatch: {package['name']}")
            with (cache / Path(package["filename"]).name).open("xb") as output:
                output.write(content)
        for package in PACKAGES:
            subprocess.run(
                [
                    "dpkg-deb",
                    "--extract",
                    str(cache / Path(package["filename"]).name),
                    str(destination),
                ],
                check=True,
                timeout=60,
            )
        libraries = destination / "usr/lib/x86_64-linux-gnu"
        for path in sorted(libraries.glob("*.so*")):
            if path.is_file() and not path.is_symlink():
                record["libraries"][str(path.relative_to(destination))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
        environment = os.environ.copy()
        previous = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = str(libraries) + (":" + previous if previous else "")
        for name in ("libGL.so.1", "libEGL.so.1", "libXt.so.6", "libXrandr.so.2"):
            path = libraries / name
            if not path.resolve(strict=True).is_relative_to(destination):
                raise ValueError(f"library escapes private runtime: {name}")
            result = subprocess.run(
                ["ldd", str(path)],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=30,
            )
            record["ldd"].append(
                {"library": name, "stdout": result.stdout, "exit": result.returncode}
            )
            if result.returncode != 0 or "not found" in result.stdout:
                raise RuntimeError(f"ldd failed ({result.returncode}): {result.stdout}")
        record["status"] = "success"
    finally:
        primary = sys.exception()
        try:
            with (destination / "setup.json").open("x") as output:
                json.dump(record, output, indent=2)
                output.write("\n")
            print("NEXUML_NATIVE_RUNTIME_SETUP=" + json.dumps(record), flush=True)
        except Exception as error:
            if primary is None:
                raise
            primary.add_note(f"Native setup receipt also failed: {error}")


if __name__ == "__main__":
    main()
