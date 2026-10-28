"""Linux amd64 worker subprocess guard, usable before dependency setup."""

from __future__ import annotations

import argparse
import ctypes
import datetime
import json
from itertools import islice
import math
import os
from pathlib import Path
import re
import select
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, BinaryIO, cast

LOG_CAP_BYTES = 32 * 1024**2
ATTEMPT_BASE = Path("/mnt/data/nexuml-qualification")
_TERM_SECONDS = 1.0
_KILL_SECONDS = 2.0
_WAIT_ALL = 0x40000000  # Linux __WALL also waits for non-SIGCHLD clone children.
_SUPERVISE_DESCENDANTS = False
_QUOTA_FILESYSTEMS = {"fuse.juicefs", "nfs", "nfs4", "ceph", "fuse.ceph", "fuse.ceph-fuse"}
_DISK_FILESYSTEMS = {
    "ext2",
    "ext3",
    "ext4",
    "xfs",
    "btrfs",
    "zfs",
    "overlay",
    "f2fs",
    "bcachefs",
    "jfs",
    "reiserfs",
    "ufs",
}


def _timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _memory_observations(root=Path("/sys/fs/cgroup")):
    """Retain visible cgroup counters, never reset them or claim app-only peaks."""
    files = {}
    for name in ("memory.peak", "memory.max", "memory.current", "memory.events"):
        try:
            files[name] = (root / name).read_text(encoding="ascii").strip()
        except (OSError, UnicodeError):
            files[name] = None
    return {
        "scope": "visible cgroup v2; includes Ray/setup/workload, not application-only",
        "files": files,
    }


def _unescape_mount(value):
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def _disk_sample(path):
    """Report visible available bytes; distributed quota is not backing-disk proof."""
    path = path.resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"disk path is not a directory: {path}")
    mounts = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        fields, filesystem = left.split(), right.split()
        mount = Path(_unescape_mount(fields[4]))
        if path.is_relative_to(mount):
            mounts.append((mount, fields, filesystem))
    if not mounts:
        raise ValueError(f"cannot identify filesystem for {path}")
    mount, fields, filesystem = max(mounts, key=lambda entry: len(entry[0].parts))
    physical = filesystem[0] in _DISK_FILESYSTEMS
    if not physical and filesystem[0] not in _QUOTA_FILESYSTEMS:
        raise ValueError(f"not a supported disk filesystem: {filesystem[0]} at {mount}")
    stats = os.statvfs(path)
    if stats.f_flag & os.ST_RDONLY or stats.f_frsize <= 0 or stats.f_blocks <= 0:
        raise ValueError(f"unusable disk filesystem at {path}")
    return stats.f_bavail * stats.f_frsize, {
        "path": str(path),
        "mountpoint": str(mount),
        "type": filesystem[0],
        "source": _unescape_mount(filesystem[1]),
        "device": fields[2],
        "physical": physical,
        "availability_kind": "filesystem_available_bytes"
        if physical
        else "distributed_quota_available_bytes",
    }


def _open_directory(parent_fd, name):
    return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)


def _claim_attempt_root(root, receipt, log_path):
    """Validate without writes, then claim exclusively using no-follow directory FDs."""
    if (
        not root.is_absolute()
        or ".." in root.parts
        or "\0" in str(root)
        or not root.is_relative_to(ATTEMPT_BASE)
        or root == ATTEMPT_BASE
    ):
        raise ValueError(
            "attempt_root must be an absolute private child of the approved root without '..'"
        )
    if receipt != root / "guard.json":
        raise ValueError("campaign receipt must be attempt_root/guard.json")
    if log_path is not None and (
        not log_path.is_absolute()
        or ".." in log_path.parts
        or "\0" in str(log_path)
        or not log_path.is_relative_to(root)
        or log_path == root
        or log_path.is_relative_to(receipt)
        or log_path.is_relative_to(root / "guard.json.lock")
    ):
        raise ValueError("campaign log must be inside attempt_root, separate from receipt/lock")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in root.parts[1:]:
            try:
                child_fd = _open_directory(fd, part)
            except FileNotFoundError:
                break
            os.close(fd)
            fd = child_fd
        else:
            raise FileExistsError(f"attempt root already exists: {root}")
    finally:
        os.close(fd)
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for index, part in enumerate(root.parts[1:], 1):
            final = index == len(root.parts) - 1
            if final:
                os.mkdir(part, 0o700, dir_fd=fd)  # Never adopt even an empty raced root.
            else:
                try:
                    os.mkdir(part, 0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            child_fd = _open_directory(fd, part)
            os.close(fd)
            fd = child_fd
        os.fchmod(fd, 0o700)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _open_campaign_log(root_fd, relative):
    fd = os.dup(root_fd)
    try:
        for part in relative.parts[:-1]:
            try:
                os.mkdir(part, 0o700, dir_fd=fd)
            except FileExistsError:
                pass
            child_fd = _open_directory(fd, part)
            os.close(fd)
            fd = child_fd
        return os.fdopen(
            os.open(
                relative.name,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                0o600,
                dir_fd=fd,
            ),
            "wb",
        )
    finally:
        os.close(fd)


def _group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def _pidfd_call(number, *arguments):
    # Standalone Python builds can omit pidfd wrappers even on capable Linux kernels.
    if sys.platform != "linux" or os.uname().machine != "x86_64":
        raise RuntimeError("dedicated guard requires Linux amd64 pidfd syscalls")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *(ctypes.c_long(value) for value in arguments))
    if result == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def _enable_descendant_supervision():
    if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
        raise RuntimeError("dedicated guard requires default SIGCHLD ownership")
    try:
        os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOHANG | os.WNOWAIT | _WAIT_ALL)
    except ChildProcessError:
        pass
    else:
        raise RuntimeError("dedicated guard must have no pre-existing children")
    children = Path(f"/proc/self/task/{os.getpid()}/children")
    if children.read_text().strip():
        raise RuntimeError("dedicated guard must have no pre-existing children")
    descriptor = _pidfd_call(434, os.getpid(), 0)  # pidfd_open
    try:
        _pidfd_call(424, descriptor, 0, 0, 0)  # pidfd_send_signal
    finally:
        os.close(descriptor)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER affects only this guard.
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _children_gone(process, observation=None):
    # Fork churn must not keep zombie draining ahead of TERM-to-KILL escalation.
    for _ in range(64):
        process.poll()  # Only Popen may reap the workload leader and set its return code.
        try:
            status = os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOHANG | os.WNOWAIT | _WAIT_ALL)
        except ChildProcessError:
            if observation is not None:
                observation["waitid"] = "ECHILD"
            return True
        if observation is not None:
            observation["waitid"] = None if status is None else {"pid": status.si_pid}
        if status is None:
            return False
        if status.si_pid != process.pid or process.returncode is not None:
            result = os.waitpid(status.si_pid, os.WNOHANG | _WAIT_ALL)
            if observation is not None:
                observation["last_waitpid"] = {"pid": status.si_pid, "result": list(result)}
    return False


def _descendant_snapshot(pid):
    info: dict[str, Any] = {"pid": pid}
    try:
        base = Path(f"/proc/{pid}")
        fields = (base / "stat").read_text().rpartition(")")[2].split()
        info.update(
            state=fields[0],
            ppid=int(fields[1]),
            start_ticks=int(fields[19]),
            exit_signal=int(fields[35]),
        )
        status = {}
        for line in (base / "status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"Threads", "TracerPid"}:
                status[key] = int(value)
        info.update(threads=status["Threads"], tracer_pid=status["TracerPid"])
        descriptor = _pidfd_call(434, pid, 0)
        try:
            info["pidfd_ready"] = bool(select.select([descriptor], [], [], 0)[0])
        finally:
            os.close(descriptor)
        if info["threads"] > 1:
            tasks = list(islice((base / "task").iterdir(), 65))
            states = []
            for task in sorted(tasks[:64], key=lambda path: int(path.name)):
                entry: dict[str, Any] = {"tid": int(task.name)}
                try:
                    entry["state"] = (task / "stat").read_text().rpartition(")")[2].split()[0]
                except (OSError, IndexError) as error:
                    entry["observation"] = type(error).__name__
                if entry.get("state") == "D":
                    try:
                        entry["wchan"] = (task / "wchan").read_text().strip()[:128]
                    except OSError as error:
                        entry["wchan_observation"] = type(error).__name__
                    try:
                        # Retain only the syscall number, never argument values or addresses.
                        entry["syscall"] = int((task / "syscall").read_text().split()[0])
                    except (OSError, ValueError, IndexError) as error:
                        entry["syscall_observation"] = type(error).__name__
                states.append(entry)
            info.update(thread_states=states, thread_states_truncated=len(tasks) > 64)
    except (OSError, ValueError, IndexError, KeyError) as error:
        info["observation"] = type(error).__name__
    return info


def _stop_descendants(process):
    children = Path(f"/proc/self/task/{os.getpid()}/children")
    observation: dict[str, Any] = {}
    for sig, grace in ((signal.SIGTERM, _TERM_SECONDS), (signal.SIGKILL, _KILL_SECONDS)):
        held = {}
        try:
            until = time.monotonic() + grace
            while True:
                process.poll()
                # A direct child cannot be reused until this dedicated parent reaps it.
                for pid in map(int, children.read_text().split()):
                    if pid in held:
                        if not select.select([held[pid]], [], [], 0)[0]:
                            continue
                        os.close(held.pop(pid))
                    try:
                        descriptor = _pidfd_call(434, pid, 0)
                        held[pid] = descriptor
                        _pidfd_call(424, descriptor, sig, 0, 0)
                    except ProcessLookupError:
                        pass
                if _children_gone(process, observation):
                    return
                if time.monotonic() >= until:
                    break
                time.sleep(0.02)
        finally:
            for descriptor in held.values():
                os.close(descriptor)
    pids = [int(pid) for pid in children.read_text().split()]
    remaining = [_descendant_snapshot(pid) for pid in pids[:64]]
    raise RuntimeError(
        "owned descendants survived bounded cleanup: "
        + json.dumps(
            {
                "direct_children": remaining,
                "term_seconds": _TERM_SECONDS,
                "kill_seconds": _KILL_SECONDS,
                "direct_children_truncated": len(pids) > 64,
                "wait_observation": observation,
            }
        )
    )


def _stop_group(process):
    """Stop only the owned group, or the dedicated CLI's adopted descendants."""
    if _SUPERVISE_DESCENDANTS:
        _stop_descendants(process)
        return
    for sig, grace in ((signal.SIGTERM, _TERM_SECONDS), (signal.SIGKILL, _KILL_SECONDS)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        until = time.monotonic() + grace
        while time.monotonic() < until:
            process.poll()
            if not _group_exists(process.pid):
                break
            time.sleep(0.02)
        if not _group_exists(process.pid):
            break
    process.wait(timeout=_KILL_SECONDS)


def _write_receipt(path, record):
    # link publishes atomically without replacing even a concurrently created receipt.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as output:
            temporary = Path(output.name)
            json.dump(record, output, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _exit_code(returncode):
    return 128 - returncode if returncode < 0 else returncode


def run_guarded(
    command: list[str],
    *,
    receipt: Path,
    deadline_seconds: float,
    disk_floor_bytes: int = 4 * 1024**3,
    disk_path: Path = Path("/tmp/ray"),
    poll_seconds: float = 0.5,
    log_path: Path | None = None,
    attempt_root: Path | None = None,
) -> int:
    """Run one argv under a combined wall/disk budget and retain its exit receipt.

    Must run on the main thread (owns SIGINT/SIGTERM handlers). The dedicated CLI
    additionally owns detached descendants; direct in-process calls own only the
    fresh process group, without adopting unrelated children of their caller.
    """
    started = time.monotonic()
    started_at = _timestamp()
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(arg, str) or "\0" in arg for arg in command)
        or not command[0]
    ):
        raise ValueError("command must be a nonempty argv list without NULs")
    for name, value in (("deadline_seconds", deadline_seconds), ("poll_seconds", poll_seconds)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be finite and positive")
        try:
            valid = math.isfinite(value) and value > 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"{name} must be finite and positive")
    if type(disk_floor_bytes) is not int or disk_floor_bytes <= 0:
        raise ValueError("disk_floor_bytes must be a positive integer")
    receipt = Path(receipt)
    log_path = Path(log_path) if log_path is not None else None
    root_fd = None
    if attempt_root is not None:
        attempt_root = Path(attempt_root)
        root_fd = _claim_attempt_root(attempt_root, receipt, log_path)
        # Anchor evidence writes to our claimed inode even if a pathname is replaced.
        receipt = Path(f"/proc/self/fd/{root_fd}") / "guard.json"
        lock = receipt.with_name("guard.json.lock")
    else:
        receipt = receipt.absolute()
        log_path = log_path.absolute() if log_path is not None else None
        if log_path is not None and receipt.resolve() == log_path.resolve():
            raise ValueError("receipt and log must have different paths")
        lock = receipt.with_name(receipt.name + ".lock")
        if log_path is not None and log_path.resolve() == lock.resolve():
            raise ValueError("log conflicts with receipt ownership lock")
    # A second guard cannot claim the same receipt while this one is running.
    try:
        if root_fd is None:
            receipt.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(receipt):
            raise FileExistsError(f"receipt already exists: {receipt}")
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except BaseException:
        if root_fd is not None:
            os.close(root_fd)
        raise
    lock_identity = os.fstat(lock_fd)
    record: dict[str, Any] = {
        "argv": list(command),
        "pid": None,
        "pgid": None,
        "started_at": started_at,
        "ended_at": None,
        "elapsed_seconds": None,
        "exit_code": None,
        "child_returncode": None,
        "reason": "guard_error",
        "min_free_bytes": None,
        "disk_filesystem": None,
        "disk_spaces": {},
        "min_available_bytes": None,
        "min_physical_free_bytes": None,
        "disk_stop": None,
        "attempt_root": str(attempt_root) if attempt_root is not None else None,
        "disk_path": str(disk_path),
        "disk_floor_bytes": disk_floor_bytes,
        "deadline_seconds": deadline_seconds,
        "log_path": str(log_path) if log_path else None,
        "log_truncated": False,
        "output_incomplete": False,
        "errors": [],
        "process_ownership": "fresh-process-group",
    }
    process = log = selector = None
    old_handlers = {}
    cancelled = None
    code = 70
    logged = 0
    output_fd = output_blocking = None
    pending = b""

    def cancel(signum, _frame):
        nonlocal cancelled
        if cancelled is None:
            cancelled = signum

    def sample():
        # Legacy min_free_bytes/disk_filesystem remain scratch-only. Output quota
        # contributes to min_available_bytes, never min_physical_free_bytes.
        minimum = None
        paths = [("scratch", Path(disk_path))]
        if attempt_root is not None:
            paths.append(("output", attempt_root))
        for role, path in paths:
            stop = {"role": role, "path": str(path)}
            try:
                if role == "output":
                    assert root_fd is not None
                    current, owned = path.lstat(), os.fstat(root_fd)
                    if path.resolve(strict=True) != path or (current.st_dev, current.st_ino) != (
                        owned.st_dev,
                        owned.st_ino,
                    ):
                        raise ValueError("claimed output root was replaced")
                free, filesystem = _disk_sample(path)
                if role == "scratch" and not filesystem["physical"]:
                    raise ValueError(
                        f"scratch requires physical disk, not {filesystem['type']} at {path}"
                    )
            except (OSError, ValueError):
                record["disk_stop"] = stop
                raise
            previous = record["disk_spaces"].get(role, {}).get("min_available_bytes")
            record["disk_spaces"][role] = {
                "filesystem": filesystem,
                "min_available_bytes": free if previous is None else min(previous, free),
            }
            for key in ("min_available_bytes", "min_physical_free_bytes"):
                if key == "min_physical_free_bytes" and not filesystem["physical"]:
                    continue
                previous = record[key]
                record[key] = free if previous is None else min(previous, free)
            if role == "scratch":
                record["disk_filesystem"] = filesystem
                record["min_free_bytes"] = record["disk_spaces"][role]["min_available_bytes"]
            if free <= disk_floor_bytes and (minimum is None or minimum > disk_floor_bytes):
                record["disk_stop"] = dict(stop, available_bytes=free, filesystem=filesystem)
            minimum = free if minimum is None else min(minimum, free)
        return minimum

    def emit(chunk):
        nonlocal logged, pending
        if log is not None:
            retained = chunk[: max(0, LOG_CAP_BYTES - logged)]
            log.write(retained)
            log.flush()
            logged += len(retained)
            if len(retained) < len(chunk):
                record["log_truncated"] = True
        if output_fd is None:
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        else:
            pending += chunk

    def flush_live():
        nonlocal pending
        if pending:
            assert output_fd is not None
            try:
                pending = pending[os.write(output_fd, pending) :]
            except BlockingIOError:
                pass

    def secondary_error(label, exc, failure_code=70):
        nonlocal code
        record["errors"].append(f"{label}: {exc}")
        if code == 0:
            record["reason"], code = f"{label}_error", failure_code

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, cancel)
        if _SUPERVISE_DESCENDANTS:
            _enable_descendant_supervision()
            record["process_ownership"] = "dedicated-linux-child-subreaper-pidfds"
        if os.path.lexists(receipt):
            raise FileExistsError(f"receipt already exists: {receipt}")
        if log_path is not None:
            if root_fd is not None:
                assert attempt_root is not None
                log = _open_campaign_log(root_fd, log_path.relative_to(attempt_root))
            else:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open("xb")
        try:
            output_fd = sys.stdout.buffer.fileno()
        except (AttributeError, OSError):
            pass  # In-memory test sinks have no descriptor and cannot block.
        if output_fd is not None:
            sys.stdout.buffer.flush()
            output_blocking = os.get_blocking(output_fd)
            os.set_blocking(output_fd, False)
        try:
            free = sample()
        except (OSError, ValueError) as exc:
            record["errors"].append(str(exc))
            record["reason"], code = "disk_unusable", 75
        else:
            if cancelled is not None:
                record["reason"], code = "cancelled", 128 + cancelled
            elif free <= disk_floor_bytes:
                record["reason"], code = "disk_floor", 75
            elif time.monotonic() - started >= deadline_seconds:
                record["reason"], code = "deadline", 124
            else:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    bufsize=0,
                )
                record["pid"] = record["pgid"] = process.pid
                assert process.stdout is not None
                os.set_blocking(process.stdout.fileno(), False)
                selector = selectors.DefaultSelector()
                selector.register(process.stdout, selectors.EVENT_READ)
                next_sample = time.monotonic()
                while True:
                    returned = process.poll()
                    if cancelled is not None:
                        record["reason"], code = "cancelled", 128 + cancelled
                        break
                    if returned is not None:
                        code = _exit_code(returned)
                        record["reason"] = (
                            "success" if returned == 0 else "signal" if returned < 0 else "nonzero"
                        )
                        break
                    now = time.monotonic()
                    if now - started >= deadline_seconds:
                        record["reason"], code = "deadline", 124
                        break
                    if now >= next_sample:
                        try:
                            free = sample()
                        except (OSError, ValueError) as exc:
                            record["errors"].append(str(exc))
                            record["reason"], code = "disk_unusable", 75
                            break
                        if free <= disk_floor_bytes:
                            record["reason"], code = "disk_floor", 75
                            break
                        next_sample = now + poll_seconds
                    wait = max(0, min(next_sample - now, deadline_seconds - (now - started)))
                    if pending:
                        assert output_fd is not None
                        # Backpressure must not stall the deadline or disk monitor.
                        select.select([], [output_fd], [], wait)
                        flush_live()
                        continue
                    for key, _ in selector.select(wait):
                        chunk = os.read(key.fd, 65536)
                        if chunk:
                            emit(chunk)
                            flush_live()
                        else:
                            selector.unregister(key.fileobj)
    except Exception as exc:
        record["errors"].append(f"guard: {exc}")
        returned = process.poll() if process is not None else None
        if returned:
            code = _exit_code(returned)
            record["reason"] = "signal" if returned < 0 else "nonzero"
    finally:
        if process is not None:
            stdout = cast(BinaryIO, process.stdout)  # Popen above always requests PIPE.
            try:
                _stop_group(process)
            except Exception as exc:
                secondary_error("cleanup", exc)
            record["child_returncode"] = process.returncode
            try:
                # Never wait for EOF from a descendant that escaped the session.
                until = time.monotonic() + 0.2
                while time.monotonic() < until:
                    if pending:
                        assert output_fd is not None
                        select.select([], [output_fd], [], max(0, until - time.monotonic()))
                        flush_live()
                        if pending:
                            record["output_incomplete"] = True
                            pending = b""
                    chunk = os.read(stdout.fileno(), 65536)
                    if not chunk:
                        break
                    emit(chunk)
                    flush_live()
            except BlockingIOError:
                pass
            except Exception as exc:
                secondary_error("output", exc, 74)
            if pending:
                record["output_incomplete"] = True
            try:
                stdout.close()
            except OSError as exc:
                secondary_error("cleanup", exc)
        if selector is not None:
            try:
                selector.close()
            except OSError as exc:
                secondary_error("cleanup", exc)
        if log is not None:
            try:
                log.close()
            except OSError as exc:
                secondary_error("log", exc, 74)
        if output_blocking is not None:
            try:
                assert output_fd is not None
                os.set_blocking(output_fd, output_blocking)
            except OSError as exc:
                secondary_error("output", exc, 74)
        if cancelled is not None:
            record["reason"], code = "cancelled", 128 + cancelled
        record.update(
            ended_at=_timestamp(),
            elapsed_seconds=time.monotonic() - started,
            exit_code=code,
            memory_observations=_memory_observations(),
        )
        try:
            _write_receipt(receipt, record)
        except Exception as exc:
            print(f"guard receipt: {exc}", file=sys.stderr)
            if code == 0:
                code = 74
        finally:
            try:
                current = lock.lstat() if os.path.lexists(lock) else None
                if current is not None:
                    if (current.st_dev, current.st_ino) != (
                        lock_identity.st_dev,
                        lock_identity.st_ino,
                    ):
                        raise OSError("ownership lock was replaced; refusing removal")
                    lock.unlink(missing_ok=True)
            except OSError as exc:
                print(f"guard ownership lock: {exc}", file=sys.stderr)
                if code == 0:
                    code = 74
            finally:
                for sig, handler in old_handlers.items():
                    signal.signal(sig, handler)
                if root_fd is not None:
                    os.close(root_fd)
                os.close(lock_fd)
    return code


def main():
    global _SUPERVISE_DESCENDANTS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--deadline-seconds", required=True, type=float)
    parser.add_argument("--disk-floor-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--disk-path", type=Path, default=Path("/tmp/ray"))
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--attempt-root", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    _SUPERVISE_DESCENDANTS = True
    try:
        return run_guarded(
            command,
            receipt=args.receipt,
            deadline_seconds=args.deadline_seconds,
            disk_floor_bytes=args.disk_floor_bytes,
            disk_path=args.disk_path,
            log_path=args.log_path,
            attempt_root=args.attempt_root,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
