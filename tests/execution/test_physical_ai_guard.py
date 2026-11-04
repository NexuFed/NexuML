"""Hermetic stdlib-only tests; no application, Ray, or native runtime imports."""

from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "fixtures" / "physical_ai_guard.py"
SPEC = importlib.util.spec_from_file_location("physical_ai_guard", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


def python(code):
    return [sys.executable, "-I", "-S", "-u", "-c", code]


def running(pid):
    try:
        # A killed orphan can remain a zombie until the container's init reaps it.
        return Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


class PhysicalAIGuardTest(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="guard-test-")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.receipt = self.root / "attempt" / "receipt.json"
        self.env = mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES="", TMPDIR=self.scratch.name)
        self.env.start()
        self.addCleanup(self.env.stop)
        _, filesystem = guard._disk_sample(self.root)
        self.assertNotIn(filesystem["type"], ("tmpfs", "ramfs"))

    def run_child(self, code, **options):
        output = io.BytesIO()
        defaults = dict(
            receipt=self.receipt,
            deadline_seconds=5,
            disk_floor_bytes=1,
            disk_path=self.root,
            poll_seconds=0.01,
        )
        defaults.update(options)
        with mock.patch.object(guard.sys, "stdout", types.SimpleNamespace(buffer=output)):
            result = guard.run_guarded(python(code), **defaults)
        return result, output.getvalue(), json.loads(self.receipt.read_text())

    def run_campaign(self, code, **options):
        base = self.root / "campaign"
        root = base / "attempt"
        self.receipt = root / "guard.json"
        with mock.patch.object(guard, "ATTEMPT_BASE", base):
            return self.run_child(code, attempt_root=root, **options)

    def test_failed_descendant_cleanup_records_only_owned_kernel_state(self):
        original_read = Path.read_text

        def read(path, *args, **kwargs):
            if str(path).startswith("/proc/self/task/") and path.name == "children":
                return "321"
            if path == Path("/proc/321/stat"):
                return "321 (chrome worker) " + " ".join(["D", "123", *(["0"] * 17), "999"])
            return original_read(path, *args, **kwargs)

        process = mock.Mock(pid=123)
        with (
            mock.patch.object(Path, "read_text", read),
            mock.patch.object(guard, "_pidfd_call", return_value=11),
            mock.patch.object(guard, "_children_gone", return_value=False),
            mock.patch.object(
                guard,
                "_descendant_snapshot",
                return_value={"pid": 321, "state": "D", "ppid": 123, "start_ticks": 999},
            ),
            mock.patch.object(guard.select, "select", return_value=([], [], [])),
            mock.patch.object(guard.os, "close"),
            mock.patch.object(guard.time, "monotonic", side_effect=[0, 1, 2, 4]),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "owned descendants survived bounded cleanup"
            ) as caught:
                guard._stop_descendants(process)
        detail = json.loads(str(caught.exception).split(": ", 1)[1])
        self.assertEqual(
            detail["direct_children"], [{"pid": 321, "state": "D", "ppid": 123, "start_ticks": 999}]
        )
        self.assertEqual(detail["term_seconds"], 1.0)
        self.assertEqual(detail["kill_seconds"], 2.0)
        self.assertNotIn("cmdline", str(caught.exception))

    @unittest.skipUnless(
        sys.platform == "linux" and os.uname().machine == "x86_64", "dedicated Linux amd64 guard"
    )
    def test_owned_nonsigchld_clone_is_reaped_before_absence_is_reported(self):
        code = f"""
import ctypes, os, runpy, signal, time, types
from pathlib import Path
guard = runpy.run_path({str(SCRIPT)!r})
libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long
child = libc.syscall(
    ctypes.c_long(56), ctypes.c_ulong(0), ctypes.c_void_p(),
    ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_ulong(0))
if child == -1:
    raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
if child == 0:
    os._exit(0)
try:
    until = time.monotonic() + 2
    while time.monotonic() < until:
        state = Path(f"/proc/{{child}}/stat").read_text().rpartition(")")[2].split()[0]
        if state == "Z": break
        time.sleep(0.005)
    assert state == "Z"
    leader = types.SimpleNamespace(pid=0, returncode=0, poll=lambda: 0)
    assert guard["_children_gone"](leader)
    assert not Path(f"/proc/{{child}}").exists(), "guard falsely declared a clone zombie absent"
    assert leader.returncode == 0
finally:
    try:
        waited, status = os.waitpid(child, 0x40000000 | os.WNOHANG)
        if waited == 0:
            os.kill(child, signal.SIGKILL)
            os.waitpid(child, 0x40000000)
    except ChildProcessError:
        pass
"""
        result = subprocess.run(python(code), capture_output=True, timeout=8)
        self.assertEqual(result.returncode, 0, (result.stdout, result.stderr))

    def test_memory_receipt_preserves_worker_scope_and_missing_counters(self):
        base = self.root / "cgroup"
        base.mkdir()
        (base / "memory.peak").write_text("123456\n")
        (base / "memory.max").write_text("max\n")
        (base / "memory.events").write_text("oom 0\noom_kill 0\n")
        observed = guard._memory_observations(base)
        self.assertIn("not application-only", observed["scope"])
        self.assertEqual(
            observed["files"],
            {
                "memory.peak": "123456",
                "memory.max": "max",
                "memory.current": None,
                "memory.events": "oom 0\noom_kill 0",
            },
        )
        self.assertEqual((base / "memory.peak").read_text(), "123456\n")

    def test_unavailable_memory_counters_do_not_hide_primary_failure(self):
        observed = guard._memory_observations(self.root / "absent")
        self.assertTrue(all(value is None for value in observed["files"].values()))
        with mock.patch.object(guard, "_memory_observations", return_value=observed):
            code, _, receipt = self.run_child("raise SystemExit(7)")
        self.assertEqual((code, receipt["child_returncode"], receipt["reason"]), (7, 7, "nonzero"))
        self.assertEqual(receipt["memory_observations"], observed)

    def assert_campaign_rejected(self, root, *, base=None, receipt=None, log_path=None):
        marker = self.root / "setup-marker"
        before = sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*"))
        with mock.patch.object(guard, "ATTEMPT_BASE", base or self.root / "campaign"):
            with mock.patch.object(guard.subprocess, "Popen") as launch:
                with self.assertRaises((OSError, ValueError)):
                    guard.run_guarded(
                        python(f"open({str(marker)!r}, 'w').write('setup ran')"),
                        receipt=receipt if receipt is not None else root / "guard.json",
                        log_path=log_path,
                        attempt_root=root,
                        deadline_seconds=1,
                        disk_path=self.root,
                        disk_floor_bytes=1,
                    )
                launch.assert_not_called()
        self.assertFalse(marker.exists())
        self.assertEqual(
            before, sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*"))
        )

    def test_campaign_invalid_lexical_roots_make_no_files_or_child(self):
        base = self.root / "campaign"
        self.assertEqual(guard.ATTEMPT_BASE, Path("/mnt/data/nexuml-qualification"))
        for root in (
            Path("relative"),
            self.root / "outside",
            base,
            base / ".." / "outside",
            base / "missing" / "bad\0root",
        ):
            with self.subTest(root=root):
                self.assert_campaign_rejected(root)

    def test_campaign_receipt_and_log_paths_validate_before_claim(self):
        root = self.root / "campaign" / "attempt"
        for options in (
            {"receipt": root / "other.json"},
            {"receipt": Path("relative.json")},
            {"log_path": self.root / "outside.log"},
            {"log_path": root / ".." / "outside.log"},
            {"log_path": root / "bad\0log"},
            {"log_path": root},
            {"log_path": root / "guard.json"},
            {"log_path": root / "guard.json.lock"},
            {"log_path": root / "guard.json.lock" / "nested.log"},
        ):
            with self.subTest(options=options):
                self.assert_campaign_rejected(root, **options)

    def test_campaign_symlink_root_and_any_ancestor_make_no_outside_writes(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "sentinel").write_text("untouched")
        base = self.root / "campaign"
        base.mkdir()
        root = base / "attempt"
        for link in (root, base / "parent", base):
            with self.subTest(link=link):
                if link == base:
                    base.rmdir()
                link.symlink_to(outside, target_is_directory=True)
                target = root if link in (root, base) else link / "attempt"
                self.assert_campaign_rejected(target)
                self.assertEqual(sorted(path.name for path in outside.iterdir()), ["sentinel"])
                link.unlink()
        # A symlink above the approved base is forbidden too.
        ancestor = self.root / "ancestor"
        ancestor.symlink_to(outside, target_is_directory=True)
        self.assert_campaign_rejected(ancestor / "base" / "attempt", base=ancestor / "base")

    def test_campaign_existing_empty_or_occupied_roots_are_never_adopted(self):
        root = self.root / "campaign" / "attempt"
        root.mkdir(parents=True, mode=0o755)
        for occupied in (False, True):
            with self.subTest(occupied=occupied):
                if occupied:
                    (root / "evidence").write_text("owned elsewhere")
                self.assert_campaign_rejected(root)
                self.assertEqual(root.stat().st_mode & 0o777, 0o755)
        self.assertEqual((root / "evidence").read_text(), "owned elsewhere")

    def test_campaign_raced_claim_is_not_adopted_or_removed(self):
        base = self.root / "campaign"
        root = base / "attempt"
        mkdir = os.mkdir

        def race(path, mode=0o777, **kwargs):
            if path == "attempt":
                mkdir(path, mode, **kwargs)
                (root / "other-owner").write_text("preserve")
            return mkdir(path, mode, **kwargs)

        with mock.patch.object(guard, "ATTEMPT_BASE", base):
            with mock.patch.object(guard.os, "mkdir", side_effect=race):
                with mock.patch.object(guard.subprocess, "Popen") as launch:
                    with self.assertRaises(FileExistsError):
                        guard.run_guarded(
                            python("pass"),
                            receipt=root / "guard.json",
                            attempt_root=root,
                            deadline_seconds=1,
                        )
                    launch.assert_not_called()
        self.assertEqual([path.name for path in root.iterdir()], ["other-owner"])

    def test_campaign_private_root_allows_later_worker_claim(self):
        root = self.root / "campaign" / "attempt"
        result, _, receipt = self.run_campaign(
            "import os; "
            f"root={str(root)!r}; "
            "assert os.stat(root).st_mode & 0o777 == 0o700; "
            "assert os.listdir(root) == ['guard.json.lock']; "
            "fd=os.open(root+'/case.json.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600); "
            "os.close(fd)"
        )
        self.assertEqual(result, 0)
        self.assertEqual(root.stat().st_mode & 0o777, 0o700)
        self.assertEqual(
            sorted(path.name for path in root.iterdir()), ["case.json.lock", "guard.json"]
        )
        self.assertEqual(set(receipt["disk_spaces"]), {"scratch", "output"})
        self.assertEqual(receipt["attempt_root"], str(root))

    def test_campaign_nested_log_stays_in_owned_root(self):
        log = self.root / "campaign" / "attempt" / "logs" / "setup.log"
        result, output, receipt = self.run_campaign("print('setup')", log_path=log)
        self.assertEqual(result, 0)
        self.assertEqual(log.read_bytes(), output)
        self.assertEqual(receipt["log_path"], str(log))

    def test_campaign_replaced_root_does_not_redirect_receipt_or_cleanup(self):
        base = self.root / "campaign"
        root, owned, other = base / "attempt", base / "owned", self.root / "other-attempt"
        other.mkdir()
        (other / "guard.json.lock").write_text("other lock")
        (other / "guard.json").write_text("other receipt")
        self.receipt = root / "guard.json"
        actual = guard._disk_sample
        scratch_samples = 0

        def sample(path):
            nonlocal scratch_samples
            if path == self.root:
                scratch_samples += 1
                if scratch_samples == 2:
                    root.rename(owned)
                    root.symlink_to(other, target_is_directory=True)
            return actual(path)

        with (
            mock.patch.object(guard, "ATTEMPT_BASE", base),
            mock.patch.object(guard, "_disk_sample", side_effect=sample),
            mock.patch.object(guard.sys, "stdout", types.SimpleNamespace(buffer=io.BytesIO())),
        ):
            result = guard.run_guarded(
                python("import time; time.sleep(30)"),
                receipt=self.receipt,
                attempt_root=root,
                deadline_seconds=5,
                disk_path=self.root,
                disk_floor_bytes=1,
                poll_seconds=0.01,
            )
        receipt = json.loads((owned / "guard.json").read_text())
        self.assertEqual(result, 75)
        self.assertEqual(receipt["reason"], "disk_unusable")
        self.assertEqual(receipt["disk_stop"]["role"], "output")
        self.assertFalse(running(receipt["pid"]))
        self.assertEqual((other / "guard.json.lock").read_text(), "other lock")
        self.assertEqual((other / "guard.json").read_text(), "other receipt")
        self.assertEqual([path.name for path in owned.iterdir()], ["guard.json"])

    def test_cli_campaign_invalid_root_never_runs_setup(self):
        root = self.root / "outside-approved-base"
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                str(SCRIPT),
                "--attempt-root",
                str(root),
                "--receipt",
                str(root / "guard.json"),
                "--deadline-seconds",
                "1",
                "--",
                *python(f"open({str(self.root / 'setup-marker')!r},'w').write('ran')"),
            ],
            capture_output=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn(b"attempt_root must be", result.stderr)
        self.assertFalse(root.exists())
        self.assertFalse((self.root / "setup-marker").exists())

    def test_campaign_output_floor_stops_only_owned_child_and_reports_distinct_mount(self):
        sentinel = subprocess.Popen(python("import time; time.sleep(30)"), start_new_session=True)
        self.addCleanup(lambda: self.stop_process(sentinel))
        actual = guard._disk_sample(self.root)[1]
        output_samples = iter((100, 100, 5))

        def sample(path):
            role = "scratch" if path == self.root else "output"
            filesystem = dict(actual, device=role, path=str(path), mountpoint=str(path))
            return (1000 if role == "scratch" else next(output_samples, 5)), filesystem

        with mock.patch.object(guard, "_disk_sample", side_effect=sample):
            result, _, receipt = self.run_campaign(
                "import time; time.sleep(30)", disk_floor_bytes=5
            )
        self.assertEqual(result, 75)
        self.assertEqual(receipt["reason"], "disk_floor")
        self.assertIsNotNone(receipt["pid"])
        self.assertFalse(running(receipt["pid"]))
        self.assertIsNone(sentinel.poll())
        self.assertEqual(receipt["disk_stop"]["role"], "output")
        self.assertEqual(receipt["disk_stop"]["path"], str(self.receipt.parent))
        self.assertEqual(receipt["disk_spaces"]["output"]["min_available_bytes"], 5)
        self.assertEqual(receipt["disk_spaces"]["scratch"]["filesystem"]["device"], "scratch")
        self.assertEqual(receipt["disk_spaces"]["output"]["filesystem"]["device"], "output")
        self.assertEqual(receipt["min_free_bytes"], 1000)
        self.assertEqual(receipt["min_physical_free_bytes"], 5)

    def test_campaign_fuse_quota_is_not_reported_as_physical_disk(self):
        actual = guard._disk_sample
        statvfs = os.statvfs
        mountinfo = (
            "1 0 0:1 / / rw - ext4 /dev/test rw\n2 1 0:2 / "
            + str(self.root / "campaign")
            + " rw - fuse.juicefs JuiceFS:test rw\n"
        )

        def sample(path):
            with mock.patch.object(Path, "read_text", return_value=mountinfo):
                stats = statvfs(path)
                with mock.patch.object(
                    guard.os,
                    "statvfs",
                    return_value=types.SimpleNamespace(
                        f_flag=stats.f_flag,
                        f_frsize=1,
                        f_blocks=10000,
                        f_bavail=1000 if path == self.root else 5,
                    ),
                ):
                    return actual(path)

        with mock.patch.object(guard, "_disk_sample", side_effect=sample):
            with mock.patch.object(guard.subprocess, "Popen") as launch:
                result, _, receipt = self.run_campaign("pass", disk_floor_bytes=5)
                launch.assert_not_called()
        self.assertEqual(result, 75)
        self.assertEqual(receipt["reason"], "disk_floor")
        self.assertEqual(receipt["disk_stop"]["role"], "output")
        output = receipt["disk_spaces"]["output"]
        self.assertEqual(output["filesystem"]["type"], "fuse.juicefs")
        self.assertFalse(output["filesystem"]["physical"])
        self.assertEqual(
            output["filesystem"]["availability_kind"], "distributed_quota_available_bytes"
        )
        self.assertEqual(receipt["min_available_bytes"], 5)
        self.assertEqual(receipt["min_physical_free_bytes"], 1000)
        self.assertEqual(receipt["min_free_bytes"], 1000)

    def test_campaign_scratch_rejects_quota_and_ram_and_output_rejects_ram_unknown_readonly(self):
        mount = self.root / "campaign"
        for role, filesystem in (
            ("scratch", "tmpfs"),
            ("scratch", "fuse.juicefs"),
            ("output", "tmpfs"),
            ("output", "unknown"),
            ("output", "nfs4"),
        ):
            with self.subTest(role=role, filesystem=filesystem):
                root = mount / f"attempt-{role}-{filesystem}"
                self.receipt = root / "guard.json"
                mountinfo = (
                    "1 0 0:1 / / rw - ext4 /dev/test rw\n2 1 0:2 / "
                    + str(self.root if role == "scratch" else mount)
                    + f" rw - {filesystem} test rw\n"
                )
                stats = types.SimpleNamespace(
                    f_flag=0,
                    f_frsize=1,
                    f_blocks=10000,
                    f_bavail=1000,
                )

                def statvfs(path):
                    return (
                        types.SimpleNamespace(**dict(vars(stats), f_flag=os.ST_RDONLY))
                        if filesystem == "nfs4" and path == root
                        else stats
                    )

                with (
                    mock.patch.object(guard, "ATTEMPT_BASE", mount),
                    mock.patch.object(Path, "read_text", return_value=mountinfo),
                    mock.patch.object(guard.os, "statvfs", side_effect=statvfs),
                    mock.patch.object(guard.subprocess, "Popen") as launch,
                ):
                    output = io.BytesIO()
                    with mock.patch.object(
                        guard.sys, "stdout", types.SimpleNamespace(buffer=output)
                    ):
                        result = guard.run_guarded(
                            python("pass"),
                            receipt=self.receipt,
                            attempt_root=root,
                            deadline_seconds=1,
                            disk_path=self.root,
                            disk_floor_bytes=1,
                        )
                    launch.assert_not_called()
                receipt = json.loads(self.receipt.read_text())
                self.assertEqual(result, 75)
                self.assertEqual(receipt["reason"], "disk_unusable")
                self.assertEqual(receipt["disk_stop"]["role"], role)

    def test_success_receipt_and_combined_live_log(self):
        log = self.root / "child.log"
        code, output, receipt = self.run_child(
            "import sys; print('stdout'); print('stderr', file=sys.stderr)", log_path=log
        )
        self.assertEqual(code, 0)
        self.assertEqual(output, b"stdout\nstderr\n")
        self.assertEqual(log.read_bytes(), output)
        self.assertEqual(receipt["argv"][1:4], ["-I", "-S", "-u"])
        self.assertEqual(receipt["pid"], receipt["pgid"])
        self.assertNotEqual(receipt["pgid"], os.getpgrp())
        self.assertEqual(receipt["reason"], "success")
        self.assertEqual(receipt["exit_code"], 0)
        self.assertEqual(receipt["child_returncode"], 0)
        self.assertIsInstance(receipt["min_free_bytes"], int)
        self.assertIn("type", receipt["disk_filesystem"])
        self.assertLessEqual(receipt["started_at"], receipt["ended_at"])
        self.assertGreaterEqual(receipt["elapsed_seconds"], 0)
        self.assertFalse(receipt["log_truncated"])
        self.assertFalse(self.receipt.with_name("receipt.json.lock").exists())

    def test_child_failure_and_signal_are_preserved(self):
        for source, expected, raw, reason in (
            ("raise SystemExit(17)", 17, 17, "nonzero"),
            ("raise SystemExit(141)", 141, 141, "nonzero"),
            ("import os, signal; os.kill(os.getpid(), signal.SIGTERM)", 143, -15, "signal"),
            (
                "import signal; signal.signal(signal.SIGPIPE, signal.SIG_DFL); "
                "signal.raise_signal(signal.SIGPIPE)",
                141,
                -13,
                "signal",
            ),
        ):
            with self.subTest(source=source):
                self.receipt.unlink(missing_ok=True)
                result, _, receipt = self.run_child(source)
                self.assertEqual(result, expected)
                self.assertEqual(receipt["child_returncode"], raw)
                self.assertEqual(receipt["reason"], reason)

    def test_quiet_setup_deadline_and_kill_escalation(self):
        started = time.monotonic()
        result, _, receipt = self.run_child(
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
            deadline_seconds=0.2,
        )
        self.assertEqual(result, 124)
        self.assertEqual(receipt["reason"], "deadline")
        self.assertEqual(receipt["child_returncode"], -signal.SIGKILL)
        self.assertLess(time.monotonic() - started, 5)
        self.assertFalse(running(receipt["pid"]))

    def test_floor_transition_during_quiet_setup(self):
        actual = guard._disk_sample
        samples = iter((100, 100, 5))

        def sample(path):
            _, filesystem = actual(path)
            return next(samples, 5), filesystem

        with mock.patch.object(guard, "_disk_sample", side_effect=sample):
            result, _, receipt = self.run_child("import time; time.sleep(30)", disk_floor_bytes=5)
        self.assertEqual(result, 75)
        self.assertEqual(receipt["reason"], "disk_floor")
        self.assertEqual(receipt["min_free_bytes"], 5)
        self.assertIsNotNone(receipt["pid"])
        self.assertFalse(running(receipt["pid"]))

    def test_preflight_floor_has_no_child(self):
        free, _ = guard._disk_sample(self.root)
        with mock.patch.object(guard.subprocess, "Popen", side_effect=AssertionError("child")):
            result, _, receipt = self.run_child("pass", disk_floor_bytes=free + 1024**3)
        self.assertEqual(result, 75)
        self.assertEqual(receipt["reason"], "disk_floor")
        self.assertIsNone(receipt["pid"])

    def test_invalid_disk_paths_and_tmpfs_have_no_child(self):
        for path in (self.root / "missing", Path("/proc"), Path("/dev/shm")):
            with self.subTest(path=path):
                self.receipt.unlink(missing_ok=True)
                with mock.patch.object(
                    guard.subprocess, "Popen", side_effect=AssertionError("child")
                ):
                    result, _, receipt = self.run_child("pass", disk_path=path)
                self.assertEqual(result, 75)
                self.assertEqual(receipt["reason"], "disk_unusable")
                self.assertIsNone(receipt["pid"])

    def test_disk_becomes_unusable_after_launch(self):
        actual = guard._disk_sample(self.root)
        with mock.patch.object(guard, "_disk_sample", side_effect=[actual, OSError("gone")]):
            result, _, receipt = self.run_child("import time; time.sleep(30)")
        self.assertEqual(result, 75)
        self.assertEqual(receipt["reason"], "disk_unusable")
        self.assertFalse(running(receipt["pid"]))

    def test_combined_deadline_includes_preflight(self):
        actual = guard._disk_sample

        def slow_sample(path):
            time.sleep(0.04)
            return actual(path)

        with mock.patch.object(guard, "_disk_sample", side_effect=slow_sample):
            with mock.patch.object(guard.subprocess, "Popen", side_effect=AssertionError("child")):
                result, _, receipt = self.run_child("pass", deadline_seconds=0.02)
        self.assertEqual(result, 124)
        self.assertEqual(receipt["reason"], "deadline")
        self.assertIsNone(receipt["pid"])

    def test_receipt_and_lock_conflicts_do_not_overwrite_or_launch(self):
        self.receipt.parent.mkdir()
        for path in (self.receipt, self.receipt.with_name("receipt.json.lock")):
            with self.subTest(path=path):
                path.write_text("owned elsewhere")
                with mock.patch.object(
                    guard.subprocess, "Popen", side_effect=AssertionError("child")
                ):
                    with self.assertRaises(FileExistsError):
                        self.run_child("pass")
                self.assertEqual(path.read_text(), "owned elsewhere")
                path.unlink()

    def test_safety_validation_precedes_child_and_file_creation(self):
        invalid = [
            {"command": []},
            {"command": "python"},
            {"command": [""]},
            {"command": ["python", "bad\0arg"]},
            {"command": [1]},
            {"deadline_seconds": 0},
            {"deadline_seconds": float("nan")},
            {"deadline_seconds": float("inf")},
            {"deadline_seconds": 10**1000},
            {"deadline_seconds": True},
            {"poll_seconds": -1},
            {"poll_seconds": float("inf")},
            {"disk_floor_bytes": -1},
            {"disk_floor_bytes": 1.5},
            {"disk_floor_bytes": True},
            {"log_path": self.receipt},
            {"log_path": self.receipt.with_name("receipt.json.lock")},
        ]
        for options in invalid:
            with self.subTest(options=options):
                args = dict(command=python("pass"), receipt=self.receipt, deadline_seconds=1)
                args.update(options)
                with mock.patch.object(
                    guard.subprocess, "Popen", side_effect=AssertionError("child")
                ):
                    with self.assertRaises(ValueError):
                        guard.run_guarded(**args)
                self.assertFalse(self.receipt.parent.exists())

    def test_log_conflict_has_no_child_and_preserves_log(self):
        log = self.root / "existing.log"
        log.write_text("original")
        with mock.patch.object(guard.subprocess, "Popen", side_effect=AssertionError("child")):
            result, _, receipt = self.run_child("pass", log_path=log)
        self.assertNotEqual(result, 0)
        self.assertIsNone(receipt["pid"])
        self.assertEqual(log.read_text(), "original")

    def test_fixed_log_cap_keeps_live_stream_and_receipt(self):
        self.assertEqual(guard.LOG_CAP_BYTES, 32 * 1024**2)
        log, live = self.root / "capped.log", self.root / "live.log"
        size = guard.LOG_CAP_BYTES + 8192
        with live.open("wb") as output:
            with mock.patch.object(guard.sys, "stdout", types.SimpleNamespace(buffer=output)):
                result = guard.run_guarded(
                    python(
                        f"import os; data=b'x'*65536; "
                        f"[os.write(1, data) for _ in range({size // 65536})]; "
                        f"os.write(1, b'y'*{size % 65536})"
                    ),
                    receipt=self.receipt,
                    deadline_seconds=10,
                    disk_floor_bytes=1,
                    disk_path=self.root,
                    poll_seconds=0.01,
                    log_path=log,
                )
        self.assertEqual(result, 0)
        self.assertEqual(log.stat().st_size, guard.LOG_CAP_BYTES)
        self.assertEqual(live.stat().st_size, size)
        self.assertTrue(json.loads(self.receipt.read_text())["log_truncated"])

    def test_cleanup_failure_does_not_replace_child_failure(self):
        stop = guard._stop_group

        def stop_then_fail(process):
            stop(process)
            raise OSError("cleanup diagnostic")

        with mock.patch.object(guard, "_stop_group", side_effect=stop_then_fail):
            result, _, receipt = self.run_child("raise SystemExit(17)")
        self.assertEqual(result, 17)
        self.assertEqual(receipt["exit_code"], 17)
        self.assertTrue(any("cleanup diagnostic" in error for error in receipt["errors"]))

    def test_receipt_failure_does_not_replace_child_failure(self):
        with mock.patch.object(guard, "_write_receipt", side_effect=OSError("receipt diagnostic")):
            with mock.patch.object(guard.sys, "stderr", io.StringIO()) as errors:
                with mock.patch.object(
                    guard.sys, "stdout", types.SimpleNamespace(buffer=io.BytesIO())
                ):
                    result = guard.run_guarded(
                        python("raise SystemExit(17)"),
                        receipt=self.receipt,
                        deadline_seconds=5,
                        disk_path=self.root,
                        disk_floor_bytes=1,
                    )
        self.assertEqual(result, 17)
        self.assertIn("receipt diagnostic", errors.getvalue())
        self.assertFalse(self.receipt.exists())

    def test_atomic_publish_refuses_concurrent_receipt(self):
        self.receipt.parent.mkdir()
        self.receipt.write_text("someone else")
        with self.assertRaises(FileExistsError):
            guard._write_receipt(self.receipt, {"exit_code": 0})
        self.assertEqual(self.receipt.read_text(), "someone else")
        self.assertEqual(list(self.receipt.parent.iterdir()), [self.receipt])

    def test_ownership_lock_cleanup_failure_preserves_primary_exit(self):
        original = Path.unlink

        def unlink(path, *args, **kwargs):
            if path == self.receipt.with_name("receipt.json.lock"):
                raise OSError("lock diagnostic")
            return original(path, *args, **kwargs)

        with mock.patch.object(Path, "unlink", unlink):
            with mock.patch.object(guard.sys, "stderr", io.StringIO()) as errors:
                result, _, receipt = self.run_child("raise SystemExit(17)")
        self.assertEqual(result, 17)
        self.assertEqual(receipt["exit_code"], 17)
        self.assertIn("lock diagnostic", errors.getvalue())

    def test_blocked_live_output_does_not_block_deadline(self):
        invocation = [
            sys.executable,
            "-I",
            "-S",
            str(SCRIPT),
            "--receipt",
            str(self.receipt),
            "--deadline-seconds",
            "0.2",
            "--disk-floor-bytes",
            "1",
            "--disk-path",
            str(self.root),
            "--",
            *python("import os; data=b'x'*65536; exec('while True: os.write(1,data)')"),
        ]
        process = subprocess.Popen(invocation, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            # Deliberately do not drain stdout until after the guard exits.
            self.assertEqual(process.wait(timeout=5), 124)
            receipt = json.loads(self.receipt.read_text())
            self.assertEqual(receipt["reason"], "deadline")
            self.assertTrue(receipt["output_incomplete"])
            self.assertFalse(running(receipt["pid"]))
        finally:
            self.stop_process(process)

    def test_cli_cancellation_kills_owned_descendant_not_unrelated_sentinel(self):
        sentinel = subprocess.Popen(python("import time; time.sleep(30)"), start_new_session=True)
        self.addCleanup(lambda: self.stop_process(sentinel))
        for sig, detached in (
            (signal.SIGTERM, False),
            (signal.SIGINT, False),
            (signal.SIGTERM, True),
            (signal.SIGINT, True),
        ):
            with self.subTest(signal=sig, detached=detached):
                self.receipt.unlink(missing_ok=True)
                ready = self.root / f"ready-{sig}-{detached}.json"
                descendant = (
                    "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                    "print('ready',flush=True); time.sleep(30)"
                )
                child = (
                    "import subprocess,sys,os,json,signal,time; "
                    "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                    f"p=subprocess.Popen({python(descendant)!r},stdout=subprocess.PIPE,"
                    f"start_new_session={detached!r}); "
                    "p.stdout.readline(); "
                    f"open({str(ready)!r},'w').write(json.dumps([os.getpid(),p.pid])); "
                    "time.sleep(30)"
                )
                invocation = [
                    sys.executable,
                    "-I",
                    "-S",
                    str(SCRIPT),
                    "--receipt",
                    str(self.receipt),
                    "--deadline-seconds",
                    "10",
                    "--disk-floor-bytes",
                    "1",
                    "--disk-path",
                    str(self.root),
                    "--",
                    *python(child),
                ]
                process = subprocess.Popen(
                    invocation, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                try:
                    until = time.monotonic() + 5
                    while not ready.exists() and time.monotonic() < until:
                        self.assertIsNone(process.poll())
                        time.sleep(0.01)
                    self.assertTrue(ready.exists(), "child and descendant did not become ready")
                    pids = json.loads(ready.read_text())
                    process.send_signal(sig)
                    stdout, stderr = process.communicate(timeout=8)
                    self.assertEqual(process.returncode, 128 + sig, (stdout, stderr))
                    receipt = json.loads(self.receipt.read_text())
                    self.assertEqual(receipt["reason"], "cancelled")
                    self.assertEqual(receipt["exit_code"], 128 + sig)
                    self.assertEqual(receipt["pid"], pids[0])
                    self.assertEqual(receipt["errors"], [])
                    self.assertTrue(all(not running(pid) for pid in pids))
                    self.assertIsNone(sentinel.poll())
                finally:
                    self.stop_process(process)
                    # Only these exact test-owned groups are eligible for cleanup.
                    if ready.exists():
                        pids = json.loads(ready.read_text())
                        for pgid in pids if detached else pids[:1]:
                            try:
                                os.killpg(pgid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass

    def test_child_absence_requires_kernel_confirmation_and_only_reaps_adopted_children(self):
        process = mock.Mock(pid=77, returncode=None)
        observation = {}
        with mock.patch.object(guard.os, "waitid", return_value=None):
            self.assertFalse(guard._children_gone(process, observation))
        self.assertIsNone(observation["waitid"])
        with (
            mock.patch.object(
                guard.os,
                "waitid",
                side_effect=[types.SimpleNamespace(si_pid=88), ChildProcessError()],
            ),
            mock.patch.object(guard.os, "waitpid", return_value=(88, 0)) as reap,
        ):
            self.assertTrue(guard._children_gone(process, observation))
        reap.assert_called_once_with(88, os.WNOHANG | guard._WAIT_ALL)
        self.assertEqual(
            observation, {"waitid": "ECHILD", "last_waitpid": {"pid": 88, "result": [88, 0]}}
        )
        self.assertIsNone(process.returncode)

    def test_zombie_snapshot_distinguishes_live_threads_without_command_lines(self):
        fields = ["0"] * 36
        fields[0], fields[1], fields[19], fields[35] = "Z", "123", "999", "17"
        texts = {
            "/proc/321/stat": "321 (private name) " + " ".join(fields),
            "/proc/321/status": "Name:\tprivate name\nThreads:\t2\nTracerPid:\t0\n",
            "/proc/321/task/321/stat": "321 (private name) Z",
            "/proc/321/task/322/stat": "322 (private name) D",
            "/proc/321/task/322/wchan": "wait_on_page_bit_common\n",
            "/proc/321/task/322/syscall": "0 0x1234 0x5678 99 0x9012\n",
        }
        with (
            mock.patch.object(Path, "read_text", lambda path, **_: texts[str(path)]),
            mock.patch.object(
                Path,
                "iterdir",
                return_value=iter([Path("/proc/321/task/321"), Path("/proc/321/task/322")]),
            ),
            mock.patch.object(guard, "_pidfd_call", return_value=11),
            mock.patch.object(guard.select, "select", return_value=([], [], [])),
            mock.patch.object(guard.os, "close") as close,
        ):
            snapshot = guard._descendant_snapshot(321)
        self.assertEqual(snapshot["state"], "Z")
        self.assertEqual(snapshot["threads"], 2)
        self.assertEqual(snapshot["tracer_pid"], 0)
        self.assertEqual(snapshot["exit_signal"], 17)
        self.assertFalse(snapshot["pidfd_ready"])
        self.assertEqual(
            snapshot["thread_states"],
            [
                {"tid": 321, "state": "Z"},
                {"tid": 322, "state": "D", "wchan": "wait_on_page_bit_common", "syscall": 0},
            ],
        )
        self.assertNotIn("private name", str(snapshot))
        self.assertNotIn("0x", str(snapshot))
        close.assert_called_once_with(11)

    def test_d_thread_diagnostics_preserve_state_on_denial_exit_and_syscall_races(self):
        fields = ["0"] * 36
        fields[0], fields[1], fields[19], fields[35] = "Z", "123", "999", "17"
        texts = {
            "/proc/321/stat": "321 (not retained) " + " ".join(fields),
            "/proc/321/status": "Threads:\t2\nTracerPid:\t0\n",
            "/proc/321/task/321/stat": "321 (not retained) Z",
            "/proc/321/task/322/stat": "322 (not retained) D",
        }
        for wchan, syscall, expected in (
            ("x" * 200, "16 0xprivate secret-arguments", {"wchan": "x" * 128, "syscall": 16}),
            (
                PermissionError("private error"),
                "-1 0xprivate",
                {"wchan_observation": "PermissionError", "syscall": -1},
            ),
            (
                "0",
                PermissionError("private error"),
                {"wchan": "0", "syscall_observation": "PermissionError"},
            ),
            (
                FileNotFoundError("private path"),
                FileNotFoundError("private path"),
                {
                    "wchan_observation": "FileNotFoundError",
                    "syscall_observation": "FileNotFoundError",
                },
            ),
            (
                "kernel_wait",
                "running",
                {"wchan": "kernel_wait", "syscall_observation": "ValueError"},
            ),
            ("kernel_wait", "", {"wchan": "kernel_wait", "syscall_observation": "IndexError"}),
        ):
            with self.subTest(expected=expected):
                values = {
                    **texts,
                    "/proc/321/task/322/wchan": wchan,
                    "/proc/321/task/322/syscall": syscall,
                }

                def read(path, **_):
                    value = values[str(path)]
                    if isinstance(value, OSError):
                        raise value
                    return value

                with (
                    mock.patch.object(Path, "read_text", read),
                    mock.patch.object(
                        Path,
                        "iterdir",
                        return_value=iter([Path("/proc/321/task/321"), Path("/proc/321/task/322")]),
                    ),
                    mock.patch.object(guard, "_pidfd_call", return_value=11),
                    mock.patch.object(guard.select, "select", return_value=([], [], [])),
                    mock.patch.object(guard.os, "close") as close,
                ):
                    snapshot = guard._descendant_snapshot(321)
                self.assertEqual(
                    snapshot["thread_states"],
                    [{"tid": 321, "state": "Z"}, {"tid": 322, "state": "D", **expected}],
                )
                self.assertFalse(snapshot["pidfd_ready"])
                self.assertNotIn("private", str(snapshot))
                self.assertNotIn("secret-arguments", str(snapshot))
                self.assertNotIn("0x", str(snapshot))
                close.assert_called_once_with(11)

    def make_sync_proc(self, syscall):
        base = self.root / "proc321"
        fields = ["0"] * 36
        fields[0], fields[1], fields[19], fields[35] = "Z", "123", "999", "17"
        for directory in ("task/321", "task/322/fd", "task/322/fdinfo"):
            (base / directory).mkdir(parents=True, exist_ok=True)
        (base / "stat").write_text("321 (private name) " + " ".join(fields))
        (base / "status").write_text("Threads:\t2\nTracerPid:\t0\n")
        (base / "task/321/stat").write_text("321 (private name) Z")
        (base / "task/322/stat").write_text("322 (private name) D")
        (base / "task/322/wchan").write_text("wait_on_page_bit_common\n")
        (base / "task/322/syscall").write_text(syscall)
        blocked = base / "task/322"
        (blocked / "fd/7").unlink(missing_ok=True)
        (blocked / "fd/7").symlink_to("/data/checkpoint")
        (blocked / "fdinfo/7").write_text(
            "pos:\t123\nflags:\t0100001\nmnt_id:\t44\nino:\t987\nsecret:\tpassword\n"
        )
        (blocked / "mountinfo").write_text(
            "43 1 8:2 / /other rw - ext4 private-source rw\n"
            r"44 1 8:3 / /data\040disk rw,password=secret shared:1"
            r" - fuse.juicefs volume\040name rw,token=secret" + "\n"
        )
        return base

    def snapshot_proc(self, base):
        with (
            mock.patch.object(guard, "Path", return_value=base) as path,
            mock.patch.object(guard, "_pidfd_call", return_value=11),
            mock.patch.object(guard.select, "select", return_value=([], [], [])),
            mock.patch.object(guard.os, "close"),
        ):
            snapshot = guard._descendant_snapshot(321)
        path.assert_called_once_with("/proc/321")
        self.assertNotIn("observation", snapshot)
        return snapshot

    def test_sync_syscalls_capture_only_owned_fd_and_allowlisted_mount_evidence(self):
        for number, argument in ((74, "0x7"), (75, "7")):
            with self.subTest(number=number):
                base = self.make_sync_proc(f"{number} {argument} 0xdeadbeef secret-args 0xcafe\n")
                with mock.patch.object(guard.os, "readlink", wraps=os.readlink) as readlink:
                    snapshot = self.snapshot_proc(base)
                readlink.assert_called_once_with(base / "task/322/fd/7")
                self.assertFalse((base / "fd").exists(), "zombie leader has no descriptor table")
                self.assertEqual(
                    snapshot["thread_states"][1],
                    {
                        "tid": 322,
                        "state": "D",
                        "wchan": "wait_on_page_bit_common",
                        "syscall": number,
                        "fd": 7,
                        "fd_target": "/data/checkpoint",
                        "fdinfo": {"pos": "123", "flags": "0100001", "mnt_id": "44", "ino": "987"},
                        "fd_mount": {
                            "id": "44",
                            "device": "8:3",
                            "mountpoint": "/data disk",
                            "filesystem": "fuse.juicefs",
                            "source": "volume name",
                        },
                    },
                )
                for excluded in ("0x", "secret", "private", "password", "token", "shared"):
                    self.assertNotIn(excluded, json.dumps(snapshot))

    def test_nonsync_syscalls_never_inspect_descriptors(self):
        for number in (0, 16, 73, 76, -1):
            with self.subTest(number=number):
                base = self.make_sync_proc(f"{number} 7 0xprivate secret-args\n")
                with (
                    mock.patch.object(guard.os, "readlink") as readlink,
                    mock.patch.object(guard, "_bounded_proc_text") as read,
                ):
                    entry = self.snapshot_proc(base)["thread_states"][1]
                self.assertEqual(set(entry), {"tid", "state", "wchan", "syscall"})
                readlink.assert_not_called()
                read.assert_not_called()

    def test_sync_fd_validation_is_bounded_without_losing_syscall(self):
        for argument in ("", "-1", "1048577", "0x100001", "0xdeadbeef", "7.0", "7_0", "9" * 5000):
            with self.subTest(argument=argument[:20]):
                base = self.make_sync_proc(f"75 {argument}")
                with mock.patch.object(guard.os, "readlink") as readlink:
                    entry = self.snapshot_proc(base)["thread_states"][1]
                self.assertEqual(entry["syscall"], 75)
                self.assertEqual(entry["fd_observation"], "ValueError")
                self.assertNotIn("fd", entry)
                readlink.assert_not_called()
        for argument, expected in (
            ("0", 0),
            ("1048576", 1048576),
            ("0x100000", 1048576),
            ("0000007", 7),
        ):
            with self.subTest(argument=argument):
                base = self.make_sync_proc(f"74 {argument}")
                entry = self.snapshot_proc(base)["thread_states"][1]
                self.assertEqual(entry["fd"], expected)

    def test_sync_fd_disappearance_and_denial_remain_optional_observations(self):
        original_open, original_readlink = Path.open, os.readlink
        for error_type in (FileNotFoundError, PermissionError):
            for stage in ("fd_target", "fdinfo", "fd_mount"):
                with self.subTest(error=error_type, stage=stage):
                    base = self.make_sync_proc("75 7 0xprivate secret-args")
                    failed = base / "task/322" / ("fdinfo/7" if stage == "fdinfo" else "mountinfo")

                    def open_file(path, *args, **kwargs):
                        if stage != "fd_target" and path == failed:
                            raise error_type("private error with credentials")
                        return original_open(path, *args, **kwargs)

                    def readlink(path):
                        if stage == "fd_target":
                            raise error_type("private error with credentials")
                        return original_readlink(path)

                    with (
                        mock.patch.object(Path, "open", open_file),
                        mock.patch.object(guard.os, "readlink", readlink),
                    ):
                        snapshot = self.snapshot_proc(base)
                    entry = snapshot["thread_states"][1]
                    self.assertEqual((entry["state"], entry["syscall"], entry["fd"]), ("D", 75, 7))
                    self.assertEqual(entry[f"{stage}_observation"], error_type.__name__)
                    self.assertNotIn("private", json.dumps(snapshot))
                    if stage == "fd_target":
                        self.assertIn("fd_mount", entry)

    def test_sync_fd_malformed_metadata_does_not_escape_diagnostics(self):
        for fdinfo, mountinfo, observation in (
            (b"pos: secret\nflags: 089\nmnt_id: -1\nino: " + b"9" * 21 + b"\n", None, "fdinfo"),
            (b"\xff", None, "fdinfo"),
            (None, b"44 1\n", "fd_mount"),
            (None, b"44 1 private-device / /data rw - ext4 source rw\n", "fd_mount"),
            (None, b"\xff", "fd_mount"),
        ):
            with self.subTest(fdinfo=fdinfo, mountinfo=mountinfo):
                base = self.make_sync_proc("75 7 secret-args")
                if fdinfo is not None:
                    (base / "task/322/fdinfo/7").write_bytes(fdinfo)
                if mountinfo is not None:
                    (base / "task/322/mountinfo").write_bytes(mountinfo)
                entry = self.snapshot_proc(base)["thread_states"][1]
                self.assertEqual((entry["syscall"], entry["fd_target"]), (75, "/data/checkpoint"))
                self.assertIn(f"{observation}_observation", entry)
                self.assertNotIn("secret", json.dumps(entry))
                self.assertNotIn("fd_mount", entry)

    def test_sync_fd_targets_mount_fields_and_numeric_values_are_bounded(self):
        base = self.make_sync_proc("75 7")
        target = "/" + "target/" * 100
        (base / "task/322/fd/7").unlink()
        (base / "task/322/fd/7").symlink_to(target)
        (base / "task/322/fdinfo/7").write_text(
            "pos: " + "9" * 20 + "\nflags: " + "7" * 24 + "\nmnt_id: 44\nino: " + "9" * 21 + "\n"
        )
        (base / "task/322/mountinfo").write_text(
            "44 1 8:3 / /" + "m" * 600 + " rw - ext4 " + "s" * 600 + " rw\n"
        )
        entry = self.snapshot_proc(base)["thread_states"][1]
        self.assertEqual(entry["fd_target"], target[:512])
        self.assertEqual(entry["fd_mount"]["mountpoint"], "/" + "m" * 511)
        self.assertEqual(entry["fd_mount"]["source"], "s" * 512)
        self.assertEqual(entry["fdinfo"], {"pos": "9" * 20, "flags": "7" * 24, "mnt_id": "44"})

    def test_sync_proc_reads_are_capped_and_partial_final_records_are_not_used(self):
        original_open = Path.open
        for stage, cap in (("fdinfo", 4096), ("fd_mount", 65536)):
            with self.subTest(stage=stage):
                base = self.make_sync_proc("75 7")
                path = base / "task/322" / ("fdinfo/7" if stage == "fdinfo" else "mountinfo")
                # A valid-looking record crossing the cap must not be used as complete evidence.
                prefix = b"pos: 123\n" if stage == "fdinfo" else b""
                crossing = (
                    b"mnt_id: 44" if stage == "fdinfo" else b"44 1 8:3 / /data rw - ext4 source "
                )
                payload = prefix + b"\n" * (cap - len(prefix) - len(crossing)) + crossing
                payload += b"999\nsecret: not-retained\n"
                source = mock.MagicMock(wraps=io.BytesIO(payload))
                source.__enter__.return_value = source

                def open_file(opened, *args, **kwargs):
                    if opened == path:
                        self.assertEqual(args, ("rb",))
                        return source
                    return original_open(opened, *args, **kwargs)

                with mock.patch.object(Path, "open", open_file):
                    entry = self.snapshot_proc(base)["thread_states"][1]
                source.read.assert_called_once_with(cap)
                if stage == "fdinfo":
                    self.assertEqual(entry["fdinfo"], {"pos": "123"})
                    self.assertNotIn("fd_mount_observation", entry)
                else:
                    self.assertEqual(entry["fd_mount_observation"], "not_found")
                self.assertNotIn("fd_mount", entry)
                self.assertNotIn("secret", json.dumps(entry))

    def test_thread_failure_snapshot_is_capped_and_process_races_are_retained(self):
        fields = ["0"] * 36
        fields[0], fields[1], fields[19], fields[35] = "Z", "123", "999", "17"
        scanned = []

        def tasks(_):
            for tid in range(100, 200):
                scanned.append(tid)
                yield Path(f"/proc/321/task/{tid}")

        def read(path, **_):
            if path.name == "status":
                return "Threads:\t100\nTracerPid:\t0\n"
            return "321 (not retained) " + " ".join(fields)

        with (
            mock.patch.object(Path, "read_text", read),
            mock.patch.object(Path, "iterdir", tasks),
            mock.patch.object(guard, "_pidfd_call", return_value=11),
            mock.patch.object(guard.select, "select", return_value=([11], [], [])),
            mock.patch.object(guard.os, "close"),
        ):
            snapshot = guard._descendant_snapshot(321)
        self.assertEqual(len(scanned), 65)
        self.assertEqual(len(snapshot["thread_states"]), 64)
        self.assertTrue(snapshot["thread_states_truncated"])
        self.assertTrue(snapshot["pidfd_ready"])
        with mock.patch.object(Path, "read_text", side_effect=FileNotFoundError("exited")):
            self.assertEqual(
                guard._descendant_snapshot(321), {"pid": 321, "observation": "FileNotFoundError"}
            )

    def test_unsupported_descendant_supervision_fails_before_workload(self):
        with (
            mock.patch.object(guard, "_SUPERVISE_DESCENDANTS", True),
            mock.patch.object(
                guard, "_enable_descendant_supervision", side_effect=OSError("pidfd unavailable")
            ),
            mock.patch.object(guard.subprocess, "Popen") as launch,
        ):
            code, _, receipt = self.run_child("raise AssertionError('must not execute')")
        self.assertEqual(code, 70)
        self.assertIsNone(receipt["pid"])
        self.assertIn("pidfd unavailable", receipt["errors"][0])
        launch.assert_not_called()

    def test_dedicated_supervisor_refuses_preexisting_children(self):
        with mock.patch.object(guard.os, "waitid", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "no pre-existing children"):
                guard._enable_descendant_supervision()

    def test_continuously_ready_zombies_cannot_defer_cleanup_escalation(self):
        process = mock.Mock(pid=77, returncode=None)
        ready = types.SimpleNamespace(si_pid=88)
        with (
            mock.patch.object(
                guard.os, "waitid", side_effect=[ready] * 64 + [AssertionError("unbounded reaping")]
            ) as wait,
            mock.patch.object(guard.os, "waitpid") as reap,
        ):
            self.assertFalse(guard._children_gone(process))
        self.assertEqual(wait.call_count, 64)
        self.assertEqual(reap.call_count, 64)

    @staticmethod
    def stop_process(process):
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)

    def test_cli_forced_parent_death_cleans_detached_descendant_without_touching_sentinel(self):
        sentinel = subprocess.Popen(python("import time; time.sleep(30)"), start_new_session=True)
        self.addCleanup(lambda: self.stop_process(sentinel))
        ready = self.root / "detached-ready.json"
        descendant = (
            "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            "print('ready',flush=True); time.sleep(30)"
        )
        child = (
            "import subprocess,os,json,signal; "
            f"p=subprocess.Popen({python(descendant)!r},stdout=subprocess.PIPE,"
            "stderr=subprocess.DEVNULL,start_new_session=True); "
            "p.stdout.readline(); "
            f"open({str(ready)!r},'w').write(json.dumps([os.getpid(),p.pid])); "
            "os.kill(os.getpid(),signal.SIGKILL)"
        )
        invocation = [
            sys.executable,
            "-I",
            "-S",
            str(SCRIPT),
            "--receipt",
            str(self.receipt),
            "--deadline-seconds",
            "10",
            "--disk-floor-bytes",
            "1",
            "--disk-path",
            str(self.root),
            "--",
            *python(child),
        ]
        process = subprocess.Popen(invocation, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            stdout, stderr = process.communicate(timeout=8)
            self.assertEqual(process.returncode, 137, (stdout, stderr))
            self.assertTrue(ready.is_file())
            pids = json.loads(ready.read_text())
            receipt = json.loads(self.receipt.read_text())
            self.assertEqual((receipt["exit_code"], receipt["child_returncode"]), (137, -9))
            self.assertEqual(receipt["errors"], [])
            self.assertTrue(all(not running(pid) for pid in pids), "detached owned child survived")
            self.assertIsNone(sentinel.poll())
        finally:
            self.stop_process(process)
            if ready.is_file():
                for pid in json.loads(ready.read_text()):
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main()
