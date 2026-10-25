"""Fixed worker cases, not a runtime qualification or an arbitrary command runner.

The outer physical_ai_guard owns setup/workload budgets and the process group.
Importing this file is stdlib-only; native runtimes are never imported here.
"""

from __future__ import annotations

import ast
from contextlib import ExitStack, contextmanager
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time

ATTEMPT_BASE = Path("/mnt/data/nexuml-qualification")
CASES = frozenset({"preflight", "regression", "package", "recipe", "so101", "browser", "recovery"})
RECIPES = frozenset(
    {
        "cartpole-ppo",
        "pendulum-ppo",
        "visual-cartpole-ppo",
        "mujoco-humanoid-ppo",
        "mujoco-reacher-ppo",
        "isaac-cartpole-ppo",
        "isaac-cartpole-learning-ppo",
        "isaac-franka-reach-ppo",
        "isaac-so101-joint-reach-ppo",
        "isaac-custom-cartpole-ppo",
        "isaac-rgb-cartpole-ppo",
        "isaac-depth-cartpole-ppo",
        "isaac-segmentation-cartpole-ppo",
        "isaac-mixed-cartpole-ppo",
        "isaac-categorical-cartpole-ppo",
        "isaac-pendulum-ippo",
        "isaac-pendulum-mappo",
        "isaac-handover-ippo",
        "isaac-handover-mappo",
        "ros2-gazebo-joint-reach-ppo",
        "carla-lane-following-ppo",
    }
)
FOCUSED_TESTS = (
    "tests/execution/test_definitions.py",
    "tests/execution/test_service.py",
    "tests/execution/test_physical_ai_cases.py",
)
_CLI = (sys.executable, "-m", "nexuml.cli.main")


class Blocked(Exception):
    """An input/profile prerequisite is missing; no passing gate is inferred."""


def _sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _source_identity():
    paths = {
        path
        for directory in ("src", "library/src", "tests", "scripts")
        for path in Path(directory).rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    }
    paths.update(Path(name) for name in ("pyproject.toml", "library/pyproject.toml", "uv.lock"))
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(path.as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_receipt(path, record):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            json.dump(record, output, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)  # Atomic publication, never replace another attempt's receipt.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _claim_root(root):
    root = Path(root)
    if not root.is_absolute() or not root.resolve().is_relative_to(ATTEMPT_BASE.resolve()):
        raise ValueError("attempt_root must be absolute beneath the approved durable root")
    if root.resolve() == ATTEMPT_BASE.resolve():
        raise ValueError("attempt_root must be a private child of the durable root")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    # The enclosing guard creates this root/lock before setup. Nothing else is reusable.
    if any(path.name != "guard.json.lock" for path in root.iterdir()):
        raise FileExistsError("attempt root already contains evidence or stage outputs")
    lock = root / "case.json.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    root.chmod(0o700)
    return root, lock


def _allocation():
    # Read only Ray-provided identity, never ray.init() or a submitter-side CUDA probe.
    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "pgid": os.getpgrp(),
        "ray_job_id": os.environ.get("RAY_JOB_ID"),
        "ray_submission_id": os.environ.get("RAY_JOB_SUBMISSION_ID"),
        "ray_job_config_present": bool(os.environ.get("RAY_JOB_CONFIG_JSON_ENV_VAR")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "sys_executable": sys.executable,
        "python_version": sys.version.split()[0],
    }


def _require_worker(record):
    identity = record["allocation"]
    if not any(
        identity[key]
        for key in (
            "ray_job_id",
            "ray_submission_id",
            "ray_job_config_present",
        )
    ):
        raise Blocked("Ray worker identity absent; local acceptance execution is forbidden")
    if not (Path(record["attempt_root"]) / "guard.json.lock").is_file():
        raise Blocked("combined setup/workload guard ownership lock absent")


def _group_members():
    members = set()
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if fields[0] != "Z" and int(fields[2]) == os.getpgrp():
                members.add(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            continue  # Process exited during enumeration.
    return members


def _checkpoint(path):
    import torch

    raw = torch.load(path, map_location="cpu", weights_only=False)
    frames = raw["state_dict"]["frames"]
    frames = frames.item() if isinstance(frames, torch.Tensor) else frames
    step = raw.get("global_step")
    if type(frames) is not int or frames < 0 or type(step) is not int or step < 0:
        raise ValueError("checkpoint lacks physical frames/global_step counters")
    return raw, frames, step


def _copy_partial(root, total):
    for source in sorted((root / "checkpoints").glob("*.ckpt")):
        destination = root / "partial.ckpt"
        try:
            before = source.stat()
            _, frames, step = _checkpoint(source)
            if not 0 < frames < total or step <= 0:
                continue
            shutil.copyfile(source, destination)
            after = source.stat()
            _, copied_frames, copied_step = _checkpoint(destination)
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                destination.unlink()
                continue
            if (frames, step) != (copied_frames, copied_step):
                destination.unlink()
                continue
            destination.chmod(0o400)
            return {
                "path": str(destination),
                "source": str(source),
                "frames": frames,
                "global_step": step,
                "sha256": _sha256(destination),
            }
        except (OSError, EOFError, RuntimeError, ValueError, KeyError):
            destination.unlink(missing_ok=True)  # A checkpoint may still be being written.
    return None


def _stage(record, name, argv, env, *, interrupt_total=None):
    root = Path(record["attempt_root"])
    log = root / f"{name}.log"
    stage = {
        "name": name,
        "argv": list(argv),
        "cwd": str(Path.cwd()),
        "exit_code": None,
        "pid": None,
        "status": "RUNNING",
        "log_path": str(log),
        "log_sha256": None,
        "partial": None,
        "cleanup": None,
    }
    record["stages"].append(stage)
    print("NEXUML_CASE_PHASE=" + json.dumps({"stage": name, "state": "started"}), flush=True)
    baseline = _group_members()
    process = None
    try:
        with log.open("xb") as output:
            # Inherit the guard's group: never start_new_session/process_group here.
            process = subprocess.Popen(argv, stdout=output, stderr=subprocess.STDOUT, env=env)
            stage["pid"] = process.pid
            if interrupt_total is not None:
                while process.poll() is None:
                    partial = _copy_partial(root, interrupt_total)
                    if partial is not None:
                        stage["partial"] = partial
                        print(
                            "NEXUML_CASE_PHASE="
                            + json.dumps(
                                {
                                    "stage": name,
                                    "state": "checkpoint_saved",
                                    "physical_frames": partial["frames"],
                                    "optimizer_steps": partial["global_step"],
                                }
                            ),
                            flush=True,
                        )
                        process.send_signal(signal.SIGTERM)  # Only this training child.
                        break
                    time.sleep(0.05)
            returned = process.wait(timeout=30) if stage["partial"] else process.wait()
            stage["exit_code"] = returned
        stage["log_sha256"] = _sha256(log)
        stage["cleanup"] = not (_group_members() - baseline)
        if not stage["cleanup"]:
            raise AssertionError("stage left live descendants; refuse overlapping native resume")
        if interrupt_total is not None:
            if stage["partial"] is None or returned == 0:
                raise AssertionError(
                    "training completed/failed before a proven partial interruption"
                )
            for checkpoint in (root / "checkpoints").glob("*.ckpt"):
                if _checkpoint(checkpoint)[1] >= interrupt_total:
                    raise AssertionError("training reached full budget before interruption")
            if (root / "policy").exists():
                raise AssertionError("training exported a completed policy before interruption")
            stage["status"] = "EXPECTED_INTERRUPTION"
        else:
            if returned != 0:
                raise AssertionError("public CLI/test stage exited nonzero")
            stage["status"] = "PASS"
        print(
            "NEXUML_CASE_PHASE="
            + json.dumps(
                {
                    "stage": name,
                    "state": stage["status"],
                    "exit_code": stage["exit_code"],
                }
            ),
            flush=True,
        )
        return stage
    except BaseException:
        stage["status"] = "FAIL"
        if process is not None and process.poll() is None:
            process.kill()  # Outer guard retains ownership of any surviving descendants.
            process.wait(timeout=5)
        if process is not None:
            stage["exit_code"] = process.returncode
        if log.exists():
            stage["log_sha256"] = _sha256(log)
        raise


def _module_inventory():
    result = {}
    for module, distribution in (
        ("nexuml", "nexuml"),
        ("nexuml_library", "nexuml-library"),
        ("torch", "torch"),
        ("lightning", "lightning"),
        ("torchrl", "torchrl"),
        ("ray", "ray"),
        ("gymnasium", "gymnasium"),
        ("mujoco", "mujoco"),
        ("isaacsim", "isaacsim"),
        ("isaaclab", "isaaclab"),
        ("rclpy", "rclpy"),
        ("playwright", "playwright"),
        ("tensorboard", "tensorboard"),
        ("mlflow", "mlflow"),
        ("imageio", "imageio"),
    ):
        try:
            spec = importlib.util.find_spec(module)  # Top-level only; no Isaac/ROS import.
        except (ImportError, OSError, ValueError):
            spec = None
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
        result[module] = {
            "available": spec is not None,
            "origin": spec.origin if spec else None,
            "version": version,
        }
    return result


def _require_current_source(record):
    record["imports"] = _module_inventory()
    record["source_origin_match"] = False
    matched = []
    for module, directory in (
        ("nexuml", "src/nexuml"),
        ("nexuml_library", "library/src/nexuml_library"),
    ):
        inventory = record["imports"][module]
        expected = Path.cwd() / directory
        inventory.update(
            expected_source_root=str(expected), resolved_origin=None, source_origin_match=False
        )
        try:
            origin = inventory["origin"]
            if origin is not None:
                resolved = Path(origin).resolve(strict=True)
                inventory["resolved_origin"] = str(resolved)
                # Resolving the expected root must not admit a foreign symlinked source tree.
                inventory["source_origin_match"] = (
                    expected.resolve(strict=True) == expected
                    and resolved.is_relative_to(expected)
                    and resolved.is_file()
                )
        except (OSError, RuntimeError, ValueError):
            pass
        matched.append(inventory["source_origin_match"])
    record["source_origin_match"] = all(matched)
    if not record["source_origin_match"]:
        raise Blocked(
            "source-only case requires both application import origins "
            "in the current uploaded source tree"
        )


def _gpu_inventory():
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise Blocked("preflight requires exactly one job-visible CUDA device")
    allocation = torch.empty(1, device="cuda:0")
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info(0)
    allocation_uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
    result = {
        "visible_count": torch.cuda.device_count(),
        "name": torch.cuda.get_device_name(0),
        "device": "cuda:0",
        "allocation_uuid": str(allocation_uuid) if allocation_uuid is not None else None,
        "free_bytes": free,
        "total_bytes": total,
        "torch_cuda_version": torch.version.cuda,
        "tiny_allocation": "PASS",
    }
    del allocation
    return result


def _object_store_observations(proc=Path("/proc")):
    observations = []
    disk_filesystems = {
        "ext2",
        "ext3",
        "ext4",
        "xfs",
        "btrfs",
        "zfs",
        "f2fs",
        "bcachefs",
        "jfs",
        "reiserfs",
        "ufs",
    }
    for path in proc.glob("[0-9]*/cmdline"):
        try:
            arguments = path.read_bytes().split(b"\0")
        except OSError:
            continue
        if not arguments or Path(os.fsdecode(arguments[0])).name != "raylet":
            continue
        settings = dict(
            os.fsdecode(argument).split("=", 1)
            for argument in arguments
            if argument.startswith((b"--object_store_memory=", b"--plasma_directory="))
        )
        value = settings.get("--object_store_memory", "")
        observation = {
            "pid": int(path.parent.name),
            "bytes": int(value) if value.isdigit() and int(value) > 0 else None,
            "plasma_directory": settings.get("--plasma_directory"),
            "filesystem": None,
            "backing": "unknown",
        }
        try:
            directory = observation["plasma_directory"]
            if directory is not None and Path(directory).is_absolute():
                if os.readlink(proc / "self/ns/mnt") != os.readlink(path.parent / "ns/mnt"):
                    raise ValueError("raylet mount namespace differs; plasma backing unknown")
                directory = Path(directory).resolve(strict=True)
                mounts = []
                # Use the raylet's mount namespace, not a path-name assumption about /tmp.
                for line in (path.parent / "mountinfo").read_text().splitlines():
                    left, right = line.split(" - ", 1)
                    mount = Path(
                        re.sub(
                            r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), left.split()[4]
                        )
                    )
                    if directory.is_relative_to(mount):
                        mounts.append((len(mount.parts), right.split()[0]))
                if mounts:
                    depth = max(depth for depth, _ in mounts)
                    filesystems = {kind for level, kind in mounts if level == depth}
                    filesystem = next(iter(filesystems)) if len(filesystems) == 1 else None
                    observation["filesystem"] = filesystem
                    if filesystem in disk_filesystems:
                        observation["backing"] = "disk"
                    elif filesystem in {"tmpfs", "ramfs", "hugetlbfs"}:
                        observation["backing"] = "shared_memory"
        except (OSError, RuntimeError, ValueError, IndexError):
            pass  # Unknown reservations/backing must block, never count as free headroom.
        observations.append(observation)
    return observations


def _preflight(record, root, options, env):
    record["licenses"] = {
        name: os.environ.get(name) in {"Y", "YES", "1"}
        for name in ("ACCEPT_EULA", "OMNI_KIT_ACCEPT_EULA")
    }
    record["binaries"] = {
        name: shutil.which(name)
        for name in (
            "nvidia-smi",
            "ros2",
            "gz",
            "gazebo",
            "Xvfb",
            "ffmpeg",
            "chromium",
            "chromium-browser",
            "google-chrome",
        )
    }
    shm = os.statvfs("/dev/shm")
    record["shared_memory"] = {
        "total_bytes": shm.f_blocks * shm.f_frsize,
        "available_bytes": shm.f_bavail * shm.f_frsize,
        "object_store_limit_bytes": None,
        "campaign_cap_bytes": 8 * 1024**3,
        "existing_object_store_observations": _object_store_observations(),
    }
    data = options.get("data_root")
    present = data is not None and Path(data).is_dir()
    record["family_prerequisites"] = {
        "so101": {
            "state": "BLOCKED",
            "data_root": data,
            "data_present": present,
            "reason": "current private-data/runtime qualification not executed"
            if present
            else "explicit private data_root unset or absent",
        },
        "ros_gazebo": {
            "state": "BLOCKED",
            "ros_binary": record["binaries"]["ros2"],
            "gazebo_binary": record["binaries"]["gz"] or record["binaries"]["gazebo"],
            "world": None,
            "controllers": None,
            "reset_bridge": None,
            "clock": None,
            "reason": "prepared live world/controller/reset/clock profile unqualified",
        },
        "display_browser_codec": {
            "state": "BLOCKED",
            "display_configured": bool(os.environ.get("DISPLAY")),
            "browser_binary_available": any(
                record["binaries"][key] for key in ("chromium", "chromium-browser", "google-chrome")
            ),
            "ffmpeg_available": record["binaries"]["ffmpeg"] is not None,
            "reason": "actual private display/browser/codec journey unqualified",
        },
        "native": {"state": "BLOCKED", "reason": "metadata inventory is not native startup proof"},
    }
    if not all(
        record["imports"][key]["available"] for key in ("nexuml", "nexuml_library", "torch")
    ):
        raise Blocked("base application imports absent")
    used = (shm.f_blocks - shm.f_bfree) * shm.f_frsize
    record["shared_memory"]["used_bytes"] = used
    observed = record["shared_memory"]["existing_object_store_observations"]
    if not observed or any(
        item["bytes"] is None or item["backing"] == "unknown" for item in observed
    ):
        raise Blocked(
            "Ray object-store reservation/backing unobservable; shared-memory budget unknown"
        )
    shared = sum(item["bytes"] for item in observed if item["backing"] == "shared_memory")
    record["shared_memory"].update(
        object_store_limit_bytes=sum(item["bytes"] for item in observed),
        shared_object_store_limit_bytes=shared,
        disk_object_store_limit_bytes=sum(
            item["bytes"] for item in observed if item["backing"] == "disk"
        ),
    )
    if max(used, shared) > 8 * 1024**3:
        raise Blocked("observed shared-memory use/reservation exceeds the 8 GiB campaign cap")
    probe = root / "durable-write-check"
    payload = os.urandom(32)
    with probe.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    if probe.read_bytes() != payload or stat.S_IMODE(root.stat().st_mode) != 0o700:
        raise AssertionError("durable output write/privacy check failed")
    record["durable_output"] = {
        "writable": True,
        "private_mode": "0700",
        "retrieval_after_worker_cleanup": "UNVERIFIED",
    }
    record["gpu"] = _gpu_inventory()
    if record["binaries"]["nvidia-smi"] is None:
        raise Blocked("nvidia-smi unavailable; GPU UUID/driver identity unverified")
    _stage(
        record,
        "gpu-driver",
        ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
        env,
    )
    # nvidia-smi can enumerate host GPUs: retain its rows without claiming allocation mapping.
    record["gpu"]["nvidia_smi_rows"] = (root / "gpu-driver.log").read_text().splitlines()


def _package(record, root, options, env):
    import runpy

    profile = runpy.run_path(str(Path(__file__).with_name("physical_ai_distribution.py")))
    profile["run_package"](record, root, env, _stage)


@contextmanager
def _open_recovery_file(source_fd, relative):
    if relative.is_absolute() or not relative.parts or any(part == ".." for part in relative.parts):
        raise AssertionError("recovery file must stay under its source attempt")
    with ExitStack() as owned:
        directory = source_fd
        for part in relative.parts[:-1]:
            directory = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
            )
            owned.callback(os.close, directory)
        descriptor = os.open(
            relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
        owned.callback(os.close, descriptor)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise AssertionError("recovery accepts regular artifact files only")
        stream = owned.enter_context(os.fdopen(descriptor, "rb", closefd=False))
        yield stream


def _recovery(record, root, options, env):
    source = Path(options.get("source_attempt", ""))
    if (
        not source.is_absolute()
        or source.parent.parent != ATTEMPT_BASE
        or not re.fullmatch(r"[0-9a-f]{32}", source.parent.name)
        or not re.fullmatch(r"(?:recipe|package|preflight|regression)-[0-9a-f]{32}", source.name)
        or source == root
    ):
        raise Blocked("recovery requires a distinct previous qualification attempt")
    for name in ("source_case_sha256", "source_guard_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", options.get(name, "")):
            raise Blocked("recovery requires previously retained case and guard hashes")
    if source.resolve(strict=True) != source:
        raise Blocked("recovery source must not contain symlinks")
    with ExitStack() as owned:
        source_fd = os.open(ATTEMPT_BASE, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        owned.callback(os.close, source_fd)
        for part in source.relative_to(ATTEMPT_BASE).parts:
            source_fd = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=source_fd
            )
            owned.callback(os.close, source_fd)
        receipts = []
        for name, key in (
            ("case.json", "source_case_sha256"),
            ("guard.json", "source_guard_sha256"),
        ):
            with _open_recovery_file(source_fd, Path(name)) as stream:
                data = stream.read()
            if hashlib.sha256(data).hexdigest() != options[key]:
                raise AssertionError("retained source receipt identity differs")
            receipts.append(json.loads(data))
        previous, guard = receipts
        if (
            previous.get("attempt_root") != str(source)
            or previous.get("status") != "PASS"
            or previous.get("exit_code") != 0
            or guard.get("exit_code") != 0
            or guard.get("child_returncode") != 0
            or guard.get("errors") != []
        ):
            raise AssertionError("source attempt did not finish successfully with owned cleanup")
        artifacts = previous.get("artifacts", [])
        if not artifacts:
            raise AssertionError("source attempt retained no artifact inventory")
        for artifact in artifacts:
            path = Path(artifact["path"])
            if not path.is_relative_to(source) or path.resolve(strict=True) != path:
                raise AssertionError("retained artifact path escaped its source attempt")
            with _open_recovery_file(source_fd, path.relative_to(source)) as stream:
                if (
                    os.fstat(stream.fileno()).st_size != artifact["bytes"]
                    or hashlib.file_digest(stream, "sha256").hexdigest() != artifact["sha256"]
                ):
                    raise AssertionError(
                        "retained artifact identity differs from the producing attempt"
                    )
    record["checks"]["durable_retrieval"] = "PASS"
    record["recovery"] = {
        "source_attempt": str(source),
        "source_case_sha256": options["source_case_sha256"],
        "source_guard_sha256": options["source_guard_sha256"],
        "source_tree_sha256": previous.get("source_tree_sha256"),
        "source_worker": previous["allocation"]["hostname"],
        "reader_worker": record["allocation"]["hostname"],
        "verified_artifacts": len(artifacts),
    }
    record["gate_scope"]["R"] = (
        "durable receipts/artifacts verified; worker removal needs control-plane evidence; "
        "other recovery paths UNVERIFIED"
    )


def _regression(record, root, options, env):
    suite = options.get("suite", "full")
    if suite not in {"full", "focused", "static", "python313", "python314"}:
        raise Blocked("suite must be a fixed full, focused, static or declared-Python inventory")
    for tier in ("SIMULATOR", "SYSTEM", "HARDWARE", "DATA"):
        env[f"NEXUML_RUN_{tier}_TESTS"] = "0"
    if suite in {"python313", "python314"}:
        import runpy

        profile = runpy.run_path(str(Path(__file__).with_name("physical_ai_python.py")))
        record["regression"] = {
            "suite": suite,
            "full_suite_gate": "UNVERIFIED",
            "native_proof": False,
            "gpu_tests": "job-visible GPU; ordinary CUDA controls only",
        }
        profile["run_matrix"](record, root, env, _stage, "3.13" if suite == "python313" else "3.14")
        record["regression"]["full_suite_gate"] = "PASS"
        return
    env.update(CUDA_VISIBLE_DEVICES="", TMPDIR="/tmp/ray")
    scratch = Path(tempfile.mkdtemp(prefix="nexuml-regression-", dir="/tmp/ray"))
    argv = [
        sys.executable,
        "-m",
        "pytest",
        *(FOCUSED_TESTS if suite == "focused" else ("tests",)),
        "-q",
        "--tb=short",
        "-p",
        "no:cacheprovider",
        "--basetemp",
        str(scratch / "pytest"),
    ]
    record["regression"] = {
        "suite": suite,
        "full_suite_gate": "UNVERIFIED",
        "native_proof": False,
        "gpu_tests": "CUDA hidden; skips are not proof",
        "scratch_path": str(scratch),
        "environment": {
            key: value
            for key, value in env.items()
            if key
            in {"CUDA_VISIBLE_DEVICES", "TMPDIR"}
            | {f"NEXUML_RUN_{tier}_TESTS" for tier in ("SIMULATOR", "SYSTEM", "HARDWARE", "DATA")}
        },
    }
    if suite == "static":
        for name, command in (
            ("lint", [sys.executable, "-m", "ruff", "check", "."]),
            ("format", [sys.executable, "-m", "ruff", "format", "--check", "."]),
            ("types", [sys.executable, "-m", "ty", "check", "--python", sys.executable]),
            (
                "docs",
                [
                    sys.executable,
                    "-m",
                    "mkdocs",
                    "build",
                    "--strict",
                    "--site-dir",
                    str(root / "site"),
                ],
            ),
            ("lock", ["uv", "lock", "--check"]),
        ):
            _stage(record, name, command, env)
        record["checks"]["static"] = "PASS"
        return
    _stage(record, "pytest", argv, env)
    record["regression"]["full_suite_gate"] = "PASS" if suite == "full" else "UNVERIFIED"
    # Raw pytest failure output stays in the private log, never in control-plane stdout.


def _prepare_recipe(config_path, root, options):
    from nexuml.core.config import ResolvedConfig
    from nexuml.core.types import (
        CallbackSpec,
        ExportSpec,
        PolicyEvaluationSpec,
        ReinforcementLearningSpec,
    )
    from nexuml.execution.definitions import LocalExecution

    if not config_path.is_file():
        raise Blocked("supplied local resolved scenario YAML is absent on the worker")
    config = ResolvedConfig.load(config_path)
    if config.name not in RECIPES or options.get("recipe", config.name) != config.name:
        raise Blocked("recipe must match one of the fixed 21 matrix recipe names")
    if not isinstance(config.training, ReinforcementLearningSpec):
        raise Blocked("recipe requires reinforcement learning training settings")
    if not isinstance(config.execution, LocalExecution):
        raise Blocked("worker configuration must use registered LocalExecution")
    if config.checkpoint is not None or config.tuning is not None:
        raise Blocked("fresh baseline requires no selective checkpoint or tuning")
    total = getattr(config.training, "total_frames", None)
    batch = getattr(config.training, "frames_per_batch", None)
    if type(total) is not int or type(batch) is not int or not 0 < batch < total:
        raise Blocked("budget must admit a genuine nonfinal saved rollout checkpoint")
    if type(config.training.seed) is not int:
        raise Blocked("a fixed training seed is required for a matched untrained baseline")
    if any(not callback.factory.endswith(":ModelCheckpoint") for callback in config.callbacks):
        raise Blocked("unqualified custom callbacks cannot write outside attempt ownership")
    config.callbacks = [
        CallbackSpec(
            factory="lightning.pytorch.callbacks:ModelCheckpoint",
            kwargs={
                "dirpath": str(root / "checkpoints"),
                "every_n_train_steps": 1,
                "save_top_k": -1,
                "save_last": True,
                "filename": "step-{step}",
            },
        )
    ]
    if config.exports and any(export.kind != "train_package" for export in config.exports):
        raise Blocked("recipe fixture requires actor train_package exports only")
    config.exports = [ExportSpec(output=str(root / "policy"))]
    logging = config.logging
    if logging is not None:
        logging.temp_artifact_dir = str(root / "logs" / "artifacts")
        for name, field, directory in (
            ("tensorboard", "log_dir", "tensorboard"),
            ("dvclive", "dir", "dvclive"),
            ("diagram", "output_dir", "diagrams"),
        ):
            writer = getattr(logging, name)
            if writer is not None:
                setattr(writer, field, str(root / "logs" / directory))
        if logging.mlflow is not None:
            logging.mlflow.tracking_uri = (root / "logs" / "mlflow").as_uri()
            logging.mlflow.artifact_location = (root / "logs" / "mlflow-artifacts").as_uri()
            logging.mlflow.run_id = logging.mlflow.parent_run_id = None
    config.save(root / "case.yaml")
    evaluation = config.model_copy(deep=True)
    existing = config.policy_evaluation or config.training.evaluation
    evaluation.policy_evaluation = PolicyEvaluationSpec.model_validate(
        {
            **existing.model_dump(),
            "episodes": 2,
            "seed": 10000,
            "at_start": False,
            "every_n_frames": None,
        }
    )
    evaluation.save(root / "eval-case.yaml")
    # Native training validation remains authoritative; never enable in-fit Isaac evaluation.
    return config


def _validate_policy(root, checkpoint, total):
    import torch

    raw, frames, step = _checkpoint(checkpoint)
    if frames != total or step <= 0:
        raise AssertionError(
            "final checkpoint must contain the exact physical budget and real steps"
        )
    policy = root / "checkpoint-policy"
    metadata = json.loads((policy / "metadata.json").read_text())
    if metadata.get("artifact_kind") != "policy" or metadata.get("training_frames") != total:
        raise AssertionError("export metadata lacks correct actor identity/physical frames")
    if metadata.get("checkpoint", {}).get("sha256") != _sha256(checkpoint):
        raise AssertionError("export checkpoint SHA does not match final checkpoint")
    if any(path.name in {"training_state.pt", "lightning.ckpt"} for path in policy.rglob("*")):
        raise AssertionError("inference policy contains training-only state")
    states = {}
    for name, prefix in (
        ("state_dict.pt", "policy.pipeline."),
        ("action_adapter_state.pt", "policy.action_adapter."),
    ):
        value = torch.load(policy / name, map_location="cpu", weights_only=True)
        expected = {
            key[len(prefix) :]: tensor
            for key, tensor in raw["state_dict"].items()
            if key.startswith(prefix)
        }
        if set(value) != set(expected) or (name == "state_dict.pt" and not value):
            raise AssertionError("export is not the complete checkpoint actor/adapter state")
        if any(
            not torch.isfinite(tensor).all() or not torch.equal(tensor, expected[key])
            for key, tensor in value.items()
        ):
            raise AssertionError(
                "exported actor/adapter state is nonfinite or differs from checkpoint"
            )
        if metadata.get("policy_files_sha256", {}).get(name) != _sha256(policy / name):
            raise AssertionError("policy actor/adapter file hash mismatch")
        states[name] = value
    untrained = torch.load(
        root / "untrained" / "state_dict.pt", map_location="cpu", weights_only=True
    )
    trained = states["state_dict.pt"]
    if set(untrained) != set(trained) or not any(
        not torch.equal(untrained[key], trained[key]) for key in trained
    ):
        raise AssertionError("complete actor weights did not change from the untrained baseline")
    return {
        "training_frames": frames,
        "global_step": step,
        "actor_changed": True,
        "finite_complete_actor": True,
        "checkpoint_sha256": _sha256(checkpoint),
        "actor_sha256": _sha256(policy / "state_dict.pt"),
        "adapter_sha256": _sha256(policy / "action_adapter_state.pt"),
    }


def _evaluation_metrics(log, expected_frames=None):
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text())
    match = re.search(r"Policy evaluation \(frame step (\d+)\):\s*(\{.*\})", text, re.S)
    if match is None:
        raise AssertionError("CLI evaluation did not report actual policy metrics")
    if expected_frames is not None and int(match[1]) != expected_frames:
        raise AssertionError("CLI evaluation source frame label differs from the checkpoint")
    metrics = ast.literal_eval(match[2])
    if not metrics or not all(
        type(value) in (int, float) and math.isfinite(value) for value in metrics.values()
    ):
        raise AssertionError("evaluation metrics absent or nonfinite")
    return metrics


def _recipe(record, root, config_path, options, env):
    config = _prepare_recipe(config_path, root, options)
    total = config.training.total_frames
    record["recipe"] = config.name
    record["config_sha256"] = _sha256(root / "case.yaml")
    if config.name == "visual-cartpole-ppo":
        _stage(
            record,
            "image-input-contract",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "tests/core/test_simulation_scenarios.py::test_visual_policy_uses_real_nonuniform_rgb_inputs",
                "--basetemp",
                str(root / "image-input-check"),
            ],
            {**env, "NEXUML_RUN_SIMULATOR_TESTS": "1"},
        )
        if "1 passed" not in (root / "image-input-contract.log").read_text():
            raise AssertionError("image input contract test did not execute")
        record["checks"]["image_policy_inputs"] = "PASS"
    argv = [
        *_CLI,
        "export",
        "--config",
        str(root / "case.yaml"),
        "--output",
        str(root / "untrained"),
    ]
    _stage(record, "untrained-export", argv, env)
    train = [*_CLI, "train", "--config", str(root / "case.yaml")]
    interrupted = _stage(record, "partial-train", train, env, interrupt_total=total)
    partial = interrupted["partial"]
    record["partial_checkpoint"] = partial
    _stage(record, "resume", [*train, "--trainer-checkpoint", partial["path"]], env)
    if _sha256(root / "partial.ckpt") != partial["sha256"]:
        raise AssertionError("resume mutated the immutable partial checkpoint")
    finals = [
        path
        for path in sorted((root / "checkpoints").glob("*.ckpt"))
        if _checkpoint(path)[1] == total
    ]
    if not finals:
        raise AssertionError("resume did not produce a full physical-budget checkpoint")
    final = finals[-1]
    _stage(
        record,
        "checkpoint-export",
        [
            *_CLI,
            "export",
            "--config",
            str(root / "case.yaml"),
            "--checkpoint",
            str(final),
            "--output",
            str(root / "checkpoint-policy"),
        ],
        env,
    )
    record["assertions"] = _validate_policy(root, final, total)
    device = "cpu" if config.training.accelerator == "cpu" else "cuda:0"
    _stage(
        record,
        "fresh-evaluate",
        [
            *_CLI,
            "evaluate",
            str(root / "checkpoint-policy"),
            "--config",
            str(root / "eval-case.yaml"),
            "--device",
            device,
        ],
        env,
    )
    record["metrics"] = _evaluation_metrics(root / "fresh-evaluate.log", expected_frames=total)
    checkpoints = {path.name: _sha256(path) for path in (root / "checkpoints").glob("*.ckpt")}
    _stage(record, "exhausted-resume", [*train, "--trainer-checkpoint", str(final)], env)
    if checkpoints != {path.name: _sha256(path) for path in (root / "checkpoints").glob("*.ckpt")}:
        raise AssertionError("exhausted resume changed checkpoints or performed more updates")
    record["checks"]["exhausted_resume"] = "PASS"
    record["gates"]["C"] = "PASS"
    record["checks"]["partial_resume"] = "PASS"
    record["gate_scope"]["R"] = (
        "partial SIGTERM continuation passed; full recovery matrix UNVERIFIED"
    )


def run_case(case: str, attempt_root: Path, config_path: Path, options: dict) -> int:
    """Run an allowlisted guarded worker case and publish an immutable JSON receipt.

    Invalid/occupied roots raise without writing into another attempt. Missing
    case inputs/profiles produce BLOCKED receipts; failed executed stages fail.
    """
    root, lock = _claim_root(attempt_root)
    record: dict = {
        "case": case,
        "attempt_root": str(root),
        "status": "BLOCKED",
        "exit_code": None,
        "source_tree_sha256": None,
        "source_origin_match": None,
        "allocation": _allocation(),
        "stages": [],
        "gates": {key: "UNVERIFIED" for key in "CVLRD"},
        "checks": {},
        "gate_scope": {
            "L": "finite-update smoke only, no learning-showcase proof",
            "V": "no actual viewer/browser interaction",
            "D": "no installed-distribution profile executed",
        },
        "artifacts": [],
        "reason": None,
    }
    code = 2
    try:
        record["source_tree_sha256"] = _source_identity()
        allowed_options = {
            "preflight": {"data_root"},
            "regression": {"suite"},
            "recipe": {"recipe"},
            "recovery": {"source_attempt", "source_case_sha256", "source_guard_sha256"},
        }
        if case not in CASES:
            raise Blocked("unsupported case; only the fixed case inventory is executable")
        if not isinstance(options, dict) or set(options) - allowed_options.get(case, set()):
            raise Blocked("unknown options; arbitrary commands/scripts/modules are forbidden")
        if any(not isinstance(value, str) or "\0" in value for value in options.values()):
            raise Blocked("case option values must be literal strings without NUL")
        if case in {"so101", "browser"}:
            gate = {"so101": "L", "browser": "V"}[case]
            record["gates"][gate] = "BLOCKED"
            raise Blocked(
                f"{case}: missing qualified fixed profile; not implemented acceptance proof"
            )
        _require_worker(record)
        _require_current_source(record)
        env = {**os.environ, "NEXUML_LOGS_ROOT": str(root / "logs"), "TMPDIR": "/tmp/ray"}
        if case == "preflight":
            _preflight(record, root, options, env)
        elif case == "regression":
            _regression(record, root, options, env)
        elif case == "package":
            _package(record, root, options, env)
        elif case == "recovery":
            _recovery(record, root, options, env)
        else:
            _recipe(record, root, Path(config_path), options, env)
        record["status"], code = "PASS", 0
    except Blocked as error:
        record["reason"] = str(
            error
        )  # Only fixed fixture diagnostics, not arbitrary config contents.
    except Exception as error:
        record["status"], code = "FAIL", 1
        record["reason"] = (
            f"{type(error).__name__}: worker assertion/stage failed; inspect private logs"
        )
        if case == "recipe":
            record["gates"]["C"] = "FAIL"
        if case == "regression" and "regression" in record:
            record["regression"]["full_suite_gate"] = (
                "FAIL" if record["regression"]["suite"] == "full" else "UNVERIFIED"
            )
    finally:
        record["exit_code"] = code
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.name.endswith(".lock"):
                record["artifacts"].append(
                    {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
                )
        try:
            _atomic_receipt(root / "case.json", record)
        finally:
            lock.unlink()
    print("NEXUML_CASE_RESULT=" + json.dumps(record, allow_nan=False), flush=True)
    return code
