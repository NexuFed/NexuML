"""Stage the frozen binary inputs and fresh application wheels for the workstation image."""

import argparse
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOCK = HERE / "external-requirements.txt"
ARTIFACTS = HERE / "artifacts.json"
FINGERPRINT = "# dependency-input-sha256: "
EXTRAS = {"nexuml": "all,dev", "nexuml-library": "all,mujoco,video", "isaaclab": "ov,viser"}
RESOLVER = (
    "--only-binary",
    ":all:",
    "--generate-hashes",
    "--no-header",
    "--no-annotate",
    "--python-version",
    "3.12",
    "--python-platform",
    "x86_64-manylinux_2_39",
    "--torch-backend",
    "cu130",
    "--default-index",
    "https://pypi.org/simple",
    "--prerelease-package",
    "opencv-python-headless-noffmpeg=allow",
    "--prerelease-package",
    "tinyobjloader=allow",
)


def fingerprint(root=ROOT):
    """Hash inputs for the container resolution, excluding tag-stamped versions.

    Returns:
        SHA-256 of the dependency inputs.
    """
    projects = []
    for relative in ("pyproject.toml", "library/pyproject.toml"):
        data = tomllib.loads((root / relative).read_text())
        projects.append(
            {
                key: data["project"].get(key)
                for key in ("name", "requires-python", "dependencies", "optional-dependencies")
            }
            | {"build-system": data["build-system"]}
        )
    inputs = {
        name: (root / "containers/isaac" / name).read_text()
        for name in ("requirements.in", "constraints.in", "overrides.txt", "artifacts.json")
    }
    return hashlib.sha256(
        json.dumps([projects, inputs, EXTRAS, RESOLVER], sort_keys=True).encode()
    ).hexdigest()


def check_lock(lock, expected):
    """Reject stale, unhashed or out-of-profile requirements.

    Raises:
        ValueError: Dependency inputs changed or a requirement is not safe for this image.
    """
    if not lock.startswith(FINGERPRINT + expected + "\n"):
        raise ValueError("Isaac lock is stale; rerun stage.py with --refresh-lock")
    requirements = [
        line
        for line in lock.replace("\\\n", "").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not requirements:
        raise ValueError("Isaac dependency lock is empty")
    for line in requirements:
        match = re.fullmatch(
            r"([a-z0-9-]+)(?:==[^ \t;]+| @ https://[^ \t;]+)"
            r"(?: +--hash=sha256:[a-f0-9]{64})+",
            line,
        )
        if (
            not match
            or match[1].startswith(("lerobot", "isaacsim"))
            or match[1]
            in {
                "isaaclab",
                "antlr4-python3-runtime",
                "nexuml",
                "nexuml-library",
            }
        ):
            raise ValueError(f"Unexpected Isaac lock requirement: {line.split()[0]}")


def check_hash(path, expected):
    """Reject changed binary inputs before they enter the build context.

    Raises:
        ValueError: The artifact is missing or does not match its recorded identity.
    """
    if not path.is_file():
        raise ValueError(f"Missing approved artifact: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Artifact hash mismatch: {path.name}")


def wheel_metadata(path):
    """Return the identity of a freshly built application wheel."""
    with zipfile.ZipFile(path) as archive:
        (entry,) = (name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = BytesParser().parsebytes(archive.read(entry))
    return metadata["Name"], metadata["Version"]


def refresh_lock(uv, output, wheels, lab, antlr, env, expected, upgrades):
    """Resolve against the current workspace wheel metadata, then replace the lock."""
    inputs = output / "workspace.in"
    candidate = output / LOCK.name
    inputs.write_text(
        "".join(f"{name}[{EXTRAS[name]}] @ {path.as_uri()}\n" for name, path in wheels.items())
        + f"isaaclab[{EXTRAS['isaaclab']}] @ {lab.as_uri()}\n"
        + f"antlr4-python3-runtime @ {antlr.as_uri()}\n"
    )
    shutil.copy2(LOCK, candidate)
    subprocess.run(
        [
            str(uv),
            "--no-config",
            "pip",
            "compile",
            str(inputs),
            str(HERE / "requirements.in"),
            "--constraints",
            str(HERE / "constraints.in"),
            "--overrides",
            str(HERE / "overrides.txt"),
            *RESOLVER,
            *upgrades,
            "--quiet",
            "--output-file",
            str(candidate),
            *(
                arg
                for name in (*EXTRAS, "antlr4-python3-runtime")
                for arg in ("--no-emit-package", name)
            ),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    lock = FINGERPRINT + expected + "\n" + candidate.read_text()
    check_lock(lock, fingerprint())
    with tempfile.NamedTemporaryFile(mode="w", dir=HERE, delete=False) as staged:
        staged.write(lock)
    Path(staged.name).chmod(0o644)
    Path(staged.name).replace(LOCK)
    inputs.unlink()


def main():
    """Stage a local build or explicitly refresh the Isaac-only lock.

    Raises:
        ValueError: A required artifact, source identity or dependency input is invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, help="New build-context directory")
    parser.add_argument("--artifacts", type=Path, help="Approved local wheel and VCS cache inputs")
    parser.add_argument(
        "--check", action="store_true", help="Check the lock without Docker or artifacts"
    )
    parser.add_argument(
        "--refresh-lock", action="store_true", help="Resolve from current manifests"
    )
    upgrade = parser.add_mutually_exclusive_group()
    upgrade.add_argument("--upgrade", action="store_true", help="Refresh all unpinned dependencies")
    upgrade.add_argument("--upgrade-package", action="append", default=[], metavar="PACKAGE")
    args = parser.parse_args()
    if (args.upgrade or args.upgrade_package) and not args.refresh_lock:
        parser.error("Package upgrades require --refresh-lock")
    policy = json.loads(ARTIFACTS.read_text())
    if (HERE / "Dockerfile").read_text().splitlines()[1] != f"FROM {policy['base']}":
        raise ValueError("Dockerfile base differs from the approved artifact manifest")
    expected = fingerprint()
    if not args.refresh_lock:
        check_lock(LOCK.read_text(), expected)
    if args.check:
        if args.output or args.artifacts or args.refresh_lock:
            parser.error("--check does not stage or refresh an image")
        return
    if not args.output or not args.artifacts:
        parser.error("Staging requires a new output directory and --artifacts")
    output = args.output.resolve()
    artifacts = args.artifacts.resolve(strict=True)
    uv = Path(shutil.which(os.environ.get("CONTAINER_UV", "uv")) or "").resolve()
    if not uv.is_file():
        raise ValueError("Set CONTAINER_UV to the recorded uv binary")
    check_hash(uv, policy["uv_sha256"])
    cache = artifacts / "uv-lab-cache"
    lab = cache / policy["lab_cache"]
    check_hash(lab / policy["lab_wheel"], policy["lab_sha256"])
    archive = (lab / policy["lab_wheel"].removesuffix(".whl")).resolve()
    if not archive.is_relative_to(cache / "archive-v0"):
        raise ValueError("Lab cached wheel archive is missing or outside the uv cache")
    with zipfile.ZipFile(lab / policy["lab_wheel"]) as wheel:
        for name in wheel.namelist():
            if not name.endswith("/"):
                if (archive / name).read_bytes() != wheel.read(name):
                    raise ValueError(f"Lab cached payload differs from its wheel: {name}")
    checkout = cache / policy["lab_checkout"]
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True)
    if revision.strip() != policy["lab_revision"]:
        raise ValueError("Cached Lab checkout revision does not match the wheel")
    antlr = artifacts / "workspace-wheels" / policy["antlr_wheel"]
    check_hash(antlr, policy["antlr_sha256"])
    shim = checkout / "docker/scripts/install_carb_env_shim.sh"
    check_hash(shim, policy["shim_sha256"])
    output.mkdir(parents=True, exist_ok=False)
    wheels = output / "workspace-wheels"
    wheels.mkdir()
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("UV_", "PIP_")) and k != "VIRTUAL_ENV"
    }
    env.update(
        TMPDIR=str(output.parent),
        UV_PROJECT_ENVIRONMENT=str(output / "build-env"),
        UV_PYTHON_DOWNLOADS="never",
    )
    for package in ("nexuml", "nexuml-library"):
        subprocess.run(
            [
                str(uv),
                "--no-config",
                "build",
                "--wheel",
                "--no-build",
                "--package",
                package,
                "--python",
                sys.executable,
                "--out-dir",
                str(wheels),
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
    workspace = {}
    for wheel in wheels.glob("*.whl"):
        name, version = wheel_metadata(wheel)
        manifest = ROOT / ("pyproject.toml" if name == "nexuml" else "library/pyproject.toml")
        project = tomllib.loads(manifest.read_text())["project"]
        if name != project["name"] or version != project["version"] or name in workspace:
            raise ValueError(f"Unexpected workspace wheel: {wheel.name}")
        workspace[name] = wheel
    if workspace.keys() != {"nexuml", "nexuml-library"}:
        raise ValueError("Both workspace wheels are required")
    if args.refresh_lock:
        upgrades = (
            ["--upgrade"]
            if args.upgrade
            else [arg for package in args.upgrade_package for arg in ("--upgrade-package", package)]
        )
        refresh_lock(
            uv, output, workspace, lab / policy["lab_wheel"], antlr, env, expected, upgrades
        )
    check_lock(LOCK.read_text(), fingerprint())
    shutil.copy2(antlr, wheels / antlr.name)
    for name in ("Dockerfile", "environment.sh", "external-requirements.txt", "overrides.txt"):
        shutil.copy2(HERE / name, output / name)
    shutil.copy2(uv, output / "uv")
    (output / "upstream").mkdir()
    shutil.copy2(shim, output / "upstream" / shim.name)
    for relative in (
        policy["lab_cache"],
        policy["lab_checkout"],
        policy["lab_db"],
        str(archive.relative_to(cache)),
    ):
        shutil.copytree(cache / relative, output / "uv-lab-cache" / relative, symlinks=True)
    entries = {
        wheel.name: "/opt/nexuml-build/wheels/"
        + wheel.name
        + " --hash=sha256:"
        + hashlib.sha256(wheel.read_bytes()).hexdigest()
        + "\n"
        for wheel in wheels.glob("*.whl")
    }
    (output / "antlr-requirement.txt").write_text(entries.pop(antlr.name))
    (output / "workspace-requirements.txt").write_text("".join(entries.values()))
    print(f"Prepared binary-only workstation build context: {output}")


if __name__ == "__main__":
    main()
