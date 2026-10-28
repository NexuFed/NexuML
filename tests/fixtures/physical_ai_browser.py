"""Fixed remote loopback logger UI reconnaissance, not visual acceptance.

The guarded worker supplies source admission, subprocess stages and budgets.
Producer files stay read-only; MLflow serves a private copy of its SQLite store.
"""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import select
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


PORTS = {"tensorboard": 6007, "mlflow": 5001}
PROC = Path("/proc")


def _pidfd_call(number, *arguments):
    if sys.platform != "linux" or os.uname().machine != "x86_64":
        raise RuntimeError("owned listener checks require Linux amd64 pidfds")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *(ctypes.c_long(value) for value in arguments))
    if result == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def _signal_members(members, sig):
    for pid in members:
        try:
            descriptor = _pidfd_call(434, pid, 0)
        except ProcessLookupError:
            continue
        try:
            fields = (PROC / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
            if select.select([descriptor], [], [], 0)[0]:
                continue
            if int(fields[2]) != os.getpgrp():
                raise RuntimeError("UI cleanup refuses a process outside its owned group")
            _pidfd_call(424, descriptor, sig, 0, 0)
        except (FileNotFoundError, ProcessLookupError):
            pass  # Process exited; the pidfd never targets a reused numeric PID.
        finally:
            os.close(descriptor)


def _listener_inodes(port):
    inodes = set()
    for table in ("tcp", "tcp6"):
        for line in (PROC / "net" / table).read_text().splitlines()[1:]:
            fields = line.split()
            address, encoded_port = fields[1].split(":")
            if fields[3] != "0A" or int(encoded_port, 16) != port:
                continue
            if table != "tcp" or address != "0100007F":
                raise RuntimeError("UI listener is not exclusively bound to IPv4 loopback")
            inodes.add(fields[9])
    return inodes


def _has_socket(pid, inode):
    for descriptor in (PROC / str(pid) / "fd").iterdir():
        try:
            if os.readlink(descriptor) == f"socket:[{inode}]":
                return True
        except FileNotFoundError:
            pass  # A descriptor closed during enumeration.
    return False


@contextmanager
def _owned_listener(process, port):
    inodes = _listener_inodes(port)
    if not inodes:
        yield None
        return
    if len(inodes) != 1:
        raise RuntimeError("multiple listeners occupy the fixed UI port")
    (inode,) = inodes
    with ExitStack() as owned:
        pending = [(process.pid, os.getpid(), [])]
        seen = set()
        while pending:
            pid, parent, ancestors = pending.pop()
            if pid in seen:
                continue
            seen.add(pid)
            if len(seen) > 256:
                raise RuntimeError("native UI process tree exceeds the bounded ownership scan")
            try:
                descriptor = _pidfd_call(434, pid, 0)
                owned.callback(os.close, descriptor)
                fields = (PROC / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
                lineage = [*ancestors, descriptor]
                if int(fields[1]) != parent or select.select(lineage, [], [], 0)[0]:
                    continue
                found = _has_socket(pid, inode)
                if not found:
                    for task in (PROC / str(pid) / "task").iterdir():
                        pending.extend(
                            (int(child), pid, lineage)
                            for child in (task / "children").read_text().split()
                        )
            except (FileNotFoundError, ProcessLookupError):
                continue  # A starting or stopping server child left the tree.
            if found:
                identity = {"pid": pid, "start_ticks": int(fields[19]), "socket_inode": inode}
                if process.poll() is not None:
                    raise RuntimeError("native UI server exited before readiness")
                yield identity
                try:
                    valid = (
                        process.poll() is None
                        and not select.select(lineage, [], [], 0)[0]
                        and _listener_inodes(port) == inodes
                        and _has_socket(pid, inode)
                    )
                except FileNotFoundError:
                    valid = False
                if not valid:
                    raise RuntimeError("owned listener changed during HTTP readiness")
                return
    raise RuntimeError("foreign listener occupies the fixed UI port")


class _ReadinessRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, port):
        self.authority = f"127.0.0.1:{port}"

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urllib.parse.urlsplit(newurl)
        if target.scheme != "http" or target.netloc != self.authority:
            raise urllib.error.URLError("readiness redirect left the owned loopback endpoint")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _wait_server(process, port, *, seconds=90):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _ReadinessRedirect(port))
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"native UI server exited before readiness: {process.returncode}")
        with _owned_listener(process, port) as listener:
            if listener is not None:
                try:
                    with opener.open(f"http://127.0.0.1:{port}/", timeout=1) as response:
                        if response.status == 200:
                            return listener
                except (OSError, urllib.error.URLError):
                    pass
        time.sleep(0.25)
    raise TimeoutError("native UI server readiness deadline exceeded")


def run_browser(record, root, options, env, stage, recover, group_members):
    """Open actual native UIs with current tools; keep all visual gates unverified.

    Raises:
        AssertionError: Producer identity or the bounded probe claim differs.
        ValueError: Qualified producer or private browser setup is absent.
        RuntimeError: Owned UI processes do not stop cleanly.
    """
    mode = options.get("logger_mode", "separate")
    if mode not in ("separate", "both"):
        raise ValueError("browser logger mode must be separate or both")
    tb_mode = "both" if mode == "both" else "tb-only"
    ml_mode = "both" if mode == "both" else "mlflow-only"
    recover(record, root, options, env)
    source = Path(options["source_attempt"])
    data = (source / "case.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != options["source_case_sha256"]:
        raise AssertionError("producer receipt changed after input admission")
    previous = json.loads(data)
    if previous["case"] != "logging" or previous["checks"].get("logger_payloads") != "PASS":
        raise ValueError("browser reconnaissance requires the qualified four-mode logger producer")
    manifest = source / "logging-manifest.json"
    data = manifest.read_bytes()
    if hashlib.sha256(data).hexdigest() != previous["logging"]["manifest_sha256"]:
        raise AssertionError("producer logger manifest changed after input admission")
    payloads = json.loads(data)
    owned_manifest = root / "source-logging-manifest.json"
    owned_manifest.write_bytes(data)
    setup = json.loads((root / "browser-setup.json").read_text())
    if setup["status"] != "success":
        raise ValueError("successful private browser setup is required")
    paths = setup["paths"]
    tb = source / f"{tb_mode}/logs/tb"
    original_db = source / f"{ml_mode}/logs/mlflow.db"
    expected_uri = f"sqlite:///{original_db}"
    if (
        payloads["modes"][tb_mode]["ui_inputs"]["tensorboard_log_root"] != str(tb)
        or payloads["modes"][ml_mode]["ui_inputs"]["mlflow_tracking_uri"] != expected_uri
    ):
        raise AssertionError("producer UI inputs differ from fixed owned tracking locations")
    stage(
        record,
        "sandboxed-browser-cancellation",
        [
            sys.executable,
            str(Path(__file__).with_name("physical_ai_browser_cancel.py")),
            str(root / "browser-cancellation"),
            str(owned_manifest),
        ],
        {**env, "CUDA_VISIBLE_DEVICES": ""},
    )
    proof = json.loads((root / "browser-cancellation/proof.json").read_text())
    if proof["cleanup"] != "PASS" or proof["V_gate"] != "UNVERIFIED":
        raise AssertionError("browser cancellation proof absent or falsely promotes viewing")
    record["checks"]["sandboxed_browser_cancellation"] = "PASS"
    database = root / "mlflow-ui.db"
    shutil.copyfile(original_db, database)
    if (
        hashlib.sha256(database.read_bytes()).hexdigest()
        != hashlib.sha256(original_db.read_bytes()).hexdigest()
    ):
        raise AssertionError("private UI database copy differs from producer store")
    for port in PORTS.values():
        with socket.socket() as reservation:
            reservation.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            reservation.bind(("127.0.0.1", port))
    ui = root / "ui"
    ui.mkdir(mode=0o700)
    commands = {
        "tensorboard": [
            sys.executable,
            "-m",
            "tensorboard.main",
            "--logdir",
            str(tb),
            "--host",
            "127.0.0.1",
            "--port",
            str(PORTS["tensorboard"]),
        ],
        "mlflow": [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--backend-store-uri",
            f"sqlite:///{database}",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORTS["mlflow"]),
            "--workers",
            "1",
        ],
    }
    state: dict = {
        "scope": (
            f"Native {tb_mode} GIF/CSV history and {ml_mode} selected Overview rows/media; "
            "other logger modes and full V gate UNVERIFIED"
        ),
        "logger_mode": mode,
        "producer_tree_sha256": previous["source_tree_sha256"],
        "source_manifest_sha256": previous["logging"]["manifest_sha256"],
        "servers": {},
        "cleanup": "UNVERIFIED",
        "cleanup_scope": "server group/listeners only; outer guard verifies detached descendants",
    }
    record["browser"] = state
    server_env = {**env, "CUDA_VISIBLE_DEVICES": "", "MLFLOW_DISABLE_TELEMETRY": "true"}
    # Browser/UI clients never alter the ordinary worker/native interpreter environment.
    server_env["LD_LIBRARY_PATH"] = paths["libraryPrefix"] + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    baseline = group_members()
    processes, outputs = [], []
    primary = None
    try:
        for name, argv in commands.items():
            output = (root / f"{name}-ui.log").open("xb")
            outputs.append(output)
            process = subprocess.Popen(
                argv,
                env=server_env,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            processes.append(process)
            state["servers"][name] = {"argv": argv, "pid": process.pid, "port": PORTS[name]}
            state["servers"][name]["listener"] = _wait_server(process, PORTS[name])
        stage(
            record,
            "logger-ui-reconnaissance",
            [
                paths["node"],
                str(Path(__file__).with_name("physical_ai_logger_browser.cjs")),
                str(ui),
                str(owned_manifest),
                *(["--both-loggers"] if mode == "both" else []),
            ],
            server_env,
        )
        for name, process in zip(commands, processes):
            with _owned_listener(process, PORTS[name]) as listener:
                if listener != state["servers"][name]["listener"]:
                    raise RuntimeError("native UI listener changed during browser reconnaissance")
        probe = json.loads((ui / "probe.json").read_text())
        if (
            probe["status"] != "success"
            or probe["claim"] != "UI_OPENED_ONLY"
            or probe["V_gate"] != "UNVERIFIED"
            or probe.get("logger_mode") != mode
        ):
            raise AssertionError("reconnaissance status/claim differs from its bounded scope")
        overview = probe.get("mlflow_overview", {})
        ml_source = payloads["modes"][ml_mode]
        main = ml_source["native"]["mlflow"]["main"]
        expected_rows = [
            ("train/rl/frames", "64"),
            (
                "eval/rl/return_mean",
                str(main["scalar_history"]["eval/rl/return_mean"][-1]["value"]),
            ),
            ("scenario/training/total_frames", "64"),
            ("scenario/interaction/environment/params/env_name", "CartPole-v1"),
            ("scenario/logging/experiment_name", ml_source["logging"]["experiment_name"]),
        ]
        if (
            probe.get("overview_inspection") != "PASS"
            or overview.get("mode") != ml_mode
            or overview.get("run_id") != main["run_id"]
            or overview.get("experiment_id") != main["experiment_id"]
            or overview.get("source_frames") != 64
            or overview.get("source_config") != ml_source["configs"]["case.yaml"]
            or overview.get("status", {}).get("actual") != "Finished"
            or [(row.get("key"), row.get("actual")) for row in overview.get("rows", [])]
            != expected_rows
        ):
            raise AssertionError("native MLflow Overview status/metrics/config proof is incomplete")
        record["checks"]["mlflow_native_overview"] = "PASS"
        previews = probe.get("tensorboard_previews", {})
        if (
            probe.get("preview_selection") != "PASS"
            or previews.get("mode") != tb_mode
            or [stage.get("step") for stage in previews.get("stages", [])] != [0, 32, 64]
            or previews.get("reopened", {}).get("step") != 0
            or previews.get("reopened", {}).get("reopened") is not True
        ):
            raise AssertionError(
                "native TensorBoard preview selection/reopening proof is incomplete"
            )
        record["checks"]["tensorboard_native_previews"] = "PASS"
        if (
            probe.get("scalar_inspection") != "PASS"
            or [item.get("tag") for item in probe.get("tensorboard_scalars", {}).get("metrics", [])]
            != ["train/rl/frames", "eval/rl/return_mean"]
            or probe.get("tensorboard_scalars", {}).get("mode") != tb_mode
        ):
            raise AssertionError("native TensorBoard CSV/history proof is incomplete")
        record["checks"]["tensorboard_scalar_history"] = "PASS"
        artifacts = probe.get("mlflow_artifacts", {})
        if (
            artifacts.get("mode") != ml_mode
            or artifacts.get("source_frames") != 64
            or [item.get("filename") for item in artifacts.get("artifacts", [])]
            != [f"frame_000000000064.{ext}" for ext in ("mp4", "gif", "json")]
            or artifacts.get("mp4", {}).get("ended") is not True
            or len(artifacts.get("gif", {}).get("samples", [])) < 2
        ):
            raise AssertionError("native MLflow artifact retrieval/playback proof is incomplete")
        record["checks"]["mlflow_native_artifacts"] = "PASS"
        record["checks"]["logger_ui_reconnaissance"] = "PASS"
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            owned = group_members() - baseline
            _signal_members(owned, signal.SIGTERM)
            deadline = time.monotonic() + 15
            while group_members() - baseline and time.monotonic() < deadline:
                for process in processes:
                    process.poll()
                time.sleep(0.1)
            survivors = group_members() - baseline
            _signal_members(survivors, signal.SIGKILL)
            for process in processes:
                process.wait(timeout=5)
            if survivors:
                raise RuntimeError("native UI processes needed forced cleanup; inspect logs")
            for port in PORTS.values():
                with socket.socket() as reservation:
                    reservation.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    reservation.bind(("127.0.0.1", port))
            state["cleanup"] = "PASS"
        except BaseException as error:
            state["cleanup"] = "FAIL"
            if primary is None:
                raise
            primary.add_note(f"UI cleanup also failed: {error}")
        finally:
            for output in outputs:
                output.close()
