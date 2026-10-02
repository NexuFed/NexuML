"""Own and observe API-started calls, not a second ML execution engine."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psutil

from nexuml.api.errors import ApiError
from nexuml.api.process import call_worker, worker_environment
from nexuml.api.security import Settings
from nexuml.core.log_paths import resolve_logs_root

TERMINAL = {"succeeded", "failed", "cancelled", "timeout", "ownership_unknown"}
MAX_EVENT = 65500


def write_json(path: Path, data: dict) -> None:
    """Atomically publish a status snapshot for reconnect/restart readers."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def stop_process_tree(process: subprocess.Popen) -> None:
    """Stop only an owned process group and wait for local execution to exit.

    Raises:
        ApiError: If the OS cannot confirm stopping the owned process tree.
    """
    try:
        owner = psutil.Process(process.pid)
        owned = owner.children(recursive=True)
    except psutil.NoSuchProcess:
        owned = []
    # Process handles cover descendants that start their own sessions and Windows
    # trees. POSIX groups additionally cover children reparented after leader exit.
    if os.name != "nt":
        for child in psutil.process_iter():
            try:
                if os.getpgid(child.pid) == process.pid and child.pid != process.pid:
                    owned.append(child)
            except (ProcessLookupError, PermissionError):
                pass
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        pass
    for child in owned:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    time.sleep(0.5)
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()
    for child in owned:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    process.wait(timeout=10)
    deadline = time.monotonic() + 10
    while True:
        alive = []
        for child in owned:
            try:
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    alive.append(child)
            except psutil.NoSuchProcess:
                pass
        if not alive:
            return
        if time.monotonic() >= deadline:
            raise ApiError(500, "stop_failed", "Owned descendants have not confirmed exit.")
        time.sleep(0.05)


class Operations:
    """One local invocation owner with disk-backed, sequenced observation."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = (
            settings.directory / resolve_logs_root(".experiments/.studio/operations")
        ).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.active: tuple[str, subprocess.Popen] | None = None
        self.threads: list[threading.Thread] = []
        self.stopping: dict[str, str] = {}
        for path in self.root.glob("*/status.json"):
            status = json.loads(path.read_text(encoding="utf-8"))
            if status["status"] not in TERMINAL:
                status["status"] = "ownership_unknown"
                status["message"] = "API restarted; prior execution ownership cannot be verified."
                write_json(path, status)

    def folder(self, identity: str) -> Path:
        """Locate an observed invocation safely.

        Returns:
            Existing observation directory.

        Raises:
            ApiError: For an invalid or missing invocation identity.
        """
        if not re.fullmatch(r"[a-f0-9]{32}", identity):
            raise ApiError(404, "not_found", "Unknown operation.")
        path = self.root / identity
        if not (path / "status.json").is_file() or path.is_symlink():
            raise ApiError(404, "not_found", "Unknown operation.")
        return path

    def status(self, identity: str) -> dict:
        """Read actual persisted status.

        Returns:
            Invocation snapshot, including its frozen source revision and capabilities.
        """
        return json.loads((self.folder(identity) / "status.json").read_text(encoding="utf-8"))

    def list(self) -> list[dict]:
        """List API-observed invocations, not unrelated CLI processes.

        Returns:
            Persisted snapshots, newest first.
        """
        return sorted(
            [
                self.status(p.name)
                for p in self.root.iterdir()
                if p.is_dir() and not p.is_symlink() and (p / "status.json").exists()
            ],
            key=lambda item: item["created_at"],
            reverse=True,
        )

    def append(self, folder: Path, kind: str, payload: dict) -> None:
        """Record bounded observations before clients can replay them."""
        with self.lock:
            status = self.status(folder.name)
            sequence = status["sequence"] + 1
            # The worker never receives the token; redact defensively at the recording boundary.
            payload = json.loads(
                json.dumps(payload, allow_nan=False).replace(
                    self.settings.token,
                    "[redacted]",
                )
            )
            event = {
                "operation_id": folder.name,
                "sequence": sequence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "kind": kind,
                "payload": payload,
            }
            encoded = json.dumps(event, allow_nan=False)
            if len(encoded.encode()) > MAX_EVENT:
                event["kind"] = "replay_gap"
                event["payload"] = {
                    "message": "Event exceeds transport limit; raw source retained on disk."
                }
                encoded = json.dumps(event)
            with (folder / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
            status["sequence"] = sequence
            write_json(folder / "status.json", status)

    def start(self, kind: str, payload: dict, *, timeout: float | None = None) -> dict:
        """Freeze a validated config and call the existing framework in an owned process.

        Returns:
            Running invocation snapshot.

        Raises:
            ApiError: If another resource-consuming call is active or startup fails.
        """
        with self.lock:
            if self.active:
                raise ApiError(
                    409, "busy", "A build, training or export operation is already active."
                )
        checked = call_worker(
            "prepare_train" if kind == "train" else "validate", payload, self.settings.directory
        )
        with self.lock:
            # ponytail: one resource-consuming invocation; add allocation policy before concurrency.
            if self.active:
                raise ApiError(409, "busy", "Another operation started while validating.")
            identity = uuid.uuid4().hex
            folder = self.root / identity
            folder.mkdir(mode=0o700)
            (folder / "config.yaml").write_text(checked["yaml"], encoding="utf-8")
            request = {
                **payload,
                "yaml": checked["yaml"],
                "data": None,
                "observations": str(folder / "observations.jsonl"),
            }
            write_json(folder / "request.json", request)
            remote = checked["data"]["execution"]["kind"] != "local" and kind == "train"
            status = {
                "id": identity,
                "kind": kind,
                "status": "running",
                "sequence": 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "semantic_revision": checked["semantic_revision"],
                "stage_order": checked["stage_order"],
                "cancellation": not remote,
                "telemetry": "driver logs and final results" if remote else "local callbacks",
                "artifacts": [],
            }
            write_json(folder / "status.json", status)
            (folder / "events.jsonl").touch()
            log = (folder / "logs.txt").open("wb")
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "nexuml.api.worker",
                        kind,
                        str(folder / "request.json"),
                        str(folder / "response.json"),
                    ],
                    cwd=self.settings.directory,
                    env=worker_environment(),
                    stdout=log,
                    stderr=log,
                    start_new_session=os.name != "nt",
                )
            except OSError as exc:
                status["status"] = "failed"
                write_json(folder / "status.json", status)
                raise ApiError(
                    500, "runtime_error", "Unable to start the selected runtime."
                ) from exc
            finally:
                log.close()
            self.active = (identity, process)
            thread = threading.Thread(
                target=self.observe, args=(folder, process, timeout), daemon=True
            )
            self.threads = [t for t in self.threads if t.is_alive()]
            self.threads.append(thread)
            thread.start()
            return status

    def observe(self, folder: Path, process: subprocess.Popen, timeout: float | None) -> None:
        """Drain actual logs/callbacks and publish the real terminal result."""
        offsets = {"observations.jsonl": 0, "logs.txt": 0}
        discarding = False
        started = time.monotonic()
        try:
            while True:
                for name, offset in offsets.items():
                    path = folder / name
                    if not path.exists():
                        continue
                    with path.open("rb") as stream:
                        stream.seek(offset)
                        while chunk := stream.readline(16384):
                            if name == "observations.jsonl":
                                if discarding:
                                    offsets[name] = stream.tell()
                                    discarding = not chunk.endswith(b"\n")
                                    continue
                                if not chunk.endswith(b"\n"):
                                    if len(chunk) < 16384 and process.poll() is None:
                                        # Partial publication: retry on the next tick.
                                        break
                                    self.append(
                                        folder,
                                        "replay_gap",
                                        {
                                            "message": (
                                                "Callback exceeds limit; "
                                                "raw source retained on disk."
                                            ),
                                        },
                                    )
                                    offsets[name] = stream.tell()
                                    discarding = True
                                    continue
                            offsets[name] = stream.tell()
                            if name == "logs.txt":
                                self.append(
                                    folder, "log", {"text": chunk.decode("utf-8", errors="replace")}
                                )
                            else:
                                event = json.loads(chunk)
                                self.append(folder, event["kind"], event["payload"])
                if process.poll() is not None:
                    # Re-read once after exit to include buffered final logs/callbacks.
                    if all(
                        not (folder / name).exists() or (folder / name).stat().st_size == offset
                        for name, offset in offsets.items()
                    ):
                        break
                if timeout and time.monotonic() - started > timeout and process.poll() is None:
                    with self.lock:
                        self.stopping[folder.name] = "timeout"
                    stop_process_tree(process)
                time.sleep(0.1)
            with self.lock:
                status = self.status(folder.name)
                result_file = folder / "response.json"
                result = (
                    json.loads(result_file.read_text(encoding="utf-8"))
                    if result_file.exists()
                    else {}
                )
                status["status"] = self.stopping.pop(folder.name, None) or (
                    "succeeded"
                    if process.returncode == 0 and result and "error" not in result
                    else "failed"
                )
                status["result"] = result
                status["artifacts"] = [
                    *result.get("artifacts", []),
                    {"path": str(folder / "logs.txt"), "kind": "log"},
                ]
                self.append(
                    folder,
                    "terminal",
                    {"status": status["status"], "result_available": bool(result)},
                )
                status["sequence"] = self.status(folder.name)["sequence"]
                write_json(folder / "status.json", status)
        except Exception:
            with self.lock:
                status = self.status(folder.name)
                status["status"] = "ownership_unknown"
                status["message"] = "Observation failed; execution state cannot be confirmed."
                write_json(folder / "status.json", status)
        finally:
            if process.poll() is None:
                stop_process_tree(process)
            with self.lock:
                if self.active and self.active[0] == folder.name:
                    self.active = None

    def cancel(self, identity: str) -> dict:
        """Stop confirmed owned local work only.

        Returns:
            Current snapshot; terminal cancellation is published after confirmed exit.

        Raises:
            ApiError: For unsupported remote stop or unknown ownership.
        """
        with self.lock:
            status = self.status(identity)
            if status["status"] in TERMINAL:
                return status
            if not status["cancellation"]:
                raise ApiError(
                    409,
                    "unsupported",
                    "Remote cancellation is not supported; driver exit is not worker stop.",
                )
            if not self.active or self.active[0] != identity:
                raise ApiError(409, "ownership_unknown", "Execution ownership cannot be verified.")
            process = self.active[1]
            # Hold publication until the entire owned tree has confirmed exit.
            stop_process_tree(process)
            self.stopping[identity] = "cancelled"
        for thread in self.threads:
            thread.join(timeout=15)
        return self.status(identity)

    def close(self) -> None:
        """Stop owned local calls on API shutdown; never claim remote cancellation."""
        if self.active:
            identity, process = self.active
            if self.status(identity)["cancellation"]:
                self.cancel(identity)
            else:
                self.stopping[identity] = "ownership_unknown"
                stop_process_tree(process)
        for thread in self.threads:
            thread.join(timeout=15)
