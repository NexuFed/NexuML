"""Authenticated existing-operation controls and disk-backed WebSocket replay."""

import asyncio
import json
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from nexuml.api.config_routes import ConfigInput
from nexuml.api.errors import ApiError
from nexuml.api.files import Files
from nexuml.api.operations import MAX_EVENT, TERMINAL, Operations
from nexuml.api.process import call_worker
from nexuml.api.security import authenticate_websocket
from nexuml.core.log_paths import resolve_logs_root


class BuildInput(ConfigInput):
    """Explicit executable check with a bounded deadline."""

    timeout: float = Field(default=120, gt=0, le=3600)


class TrainInput(ConfigInput):
    """Freeze a scenario, or use existing Trainer checkpoint resume semantics."""

    trainer_checkpoint: str | None = None
    template_revision: str | None = None


class DiscoveryInput(ConfigInput):
    """Explicit selected-scope discovery outside the resource-consuming slot."""

    timeout: float = Field(default=10, gt=0, le=30)


class ExportInput(BaseModel):
    """Export from a completed operation and actual checkpoint, not placeholder weights."""

    model_config = ConfigDict(extra="forbid")
    source_id: str
    kind: Literal["train_package", "safetensors", "onnx"] = "train_package"
    output: str


def operation_router(operations: Operations) -> APIRouter:
    """Expose lightweight observation without a training loop or event queue.

    Returns:
        REST/WebSocket router over actual invocation ownership and persisted observations.
    """
    settings = operations.settings
    router = APIRouter(prefix="/api/v1")
    files = Files(settings.directory)
    logs_root = (settings.directory / resolve_logs_root(".experiments")).resolve()

    def artifact_path(value: str) -> Path:
        path = Path(value)
        candidate = (settings.directory / path).resolve()
        if not any(candidate.is_relative_to(root) for root in (settings.directory, logs_root)):
            raise ApiError(
                403, "forbidden_path", "Artifact is outside authorized working/log roots."
            )
        return candidate

    def checked_payload(body: ConfigInput) -> dict:
        payload = body.model_dump()
        # Transport-selected checkpoint reads follow the file boundary too. Scenario
        # dataset/library imports retain the existing trusted local Python boundary.
        if payload.get("trainer_checkpoint"):
            payload["trainer_checkpoint"] = str(artifact_path(payload["trainer_checkpoint"]))
        return payload

    @router.post("/build", status_code=202)
    def build(body: BuildInput):
        return operations.start("build", checked_payload(body), timeout=body.timeout)

    @router.post("/train", status_code=202)
    def train(body: TrainInput):
        return operations.start("train", checked_payload(body))

    @router.post("/train/prepare")
    def prepare_train(body: TrainInput):
        checked = call_worker("prepare_train", checked_payload(body), settings.directory)
        checked.pop(
            "launch_plan", None
        )  # Raw infrastructure content never crosses the browser boundary.
        return checked

    @router.post("/execution/discover")
    def discover(body: DiscoveryInput):
        return call_worker("discover_execution", checked_payload(body), settings.directory)

    @router.get("/operations")
    def list_operations():
        return {"operations": operations.list()}

    @router.get("/operations/{identity}")
    def status(identity: str):
        return operations.status(identity)

    @router.get("/operations/{identity}/config")
    def config(identity: str):
        text = (operations.folder(identity) / "config.yaml").read_text(encoding="utf-8")
        return call_worker("validate", {"yaml": text}, settings.directory)

    @router.post("/operations/{identity}/cancel")
    def cancel(identity: str):
        return operations.cancel(identity)

    @router.post("/operations/{identity}/inspect")
    def inspect(identity: str):
        return operations.inspect(identity)

    @router.get("/operations/{identity}/artifacts/{index}")
    def download(identity: str, index: int):
        artifacts = operations.status(identity)["artifacts"]
        if index < 0 or index >= len(artifacts):
            raise ApiError(404, "not_found", "Artifact not found.")
        path = artifact_path(artifacts[index]["path"])
        if not path.is_file():
            raise ApiError(404, "not_found", "Artifact no longer exists.")
        return FileResponse(path, filename=path.name)

    @router.post("/export", status_code=202)
    def export(body: ExportInput):
        source = operations.status(body.source_id)
        if source["kind"] != "train" or source["status"] != "succeeded":
            raise ApiError(409, "unsupported", "Export requires a completed local training source.")
        checkpoints = [a for a in source["artifacts"] if a["kind"] == "checkpoint"]
        if not checkpoints:
            raise ApiError(
                409, "unsupported", "No actual Trainer checkpoint is available to export."
            )
        text = (operations.folder(body.source_id) / "config.yaml").read_text(encoding="utf-8")
        return operations.start(
            "export",
            {
                "yaml": text,
                "trainer_checkpoint": str(artifact_path(checkpoints[-1]["path"])),
                "export_kind": body.kind,
                "output": str(files.path(body.output)),
            },
        )

    @router.websocket("/operations/{identity}/events")
    async def events(socket: WebSocket, identity: str):
        if not await authenticate_websocket(socket, settings):
            return
        try:
            folder = operations.folder(identity)
            after = int(socket.query_params.get("after", "0"))
            if after < 0:
                raise ValueError("Invalid sequence")
            status = operations.status(identity)
            if after > status["sequence"]:
                await socket.send_json(
                    {
                        "kind": "replay_gap",
                        "payload": {"message": "Cursor exceeds recorded history."},
                    }
                )
                after = 0
            snapshot = {"kind": "snapshot", "payload": status}
            if len(json.dumps(snapshot).encode()) > MAX_EVENT:
                await socket.send_json(
                    {
                        "kind": "replay_gap",
                        "payload": {
                            "message": (
                                "Snapshot exceeds frame limit; fetch complete status over REST."
                            ),
                        },
                    }
                )
                snapshot = {
                    "kind": "snapshot",
                    "partial": True,
                    "payload": {key: status[key] for key in ("id", "status", "sequence")},
                }
            await socket.send_json(snapshot)
            # Replay directly from disk. Awaited sends bound subscriber memory and apply
            # backpressure without another in-memory event bus/queue.
            with (folder / "events.jsonl").open(encoding="utf-8") as stream:
                while True:
                    position = stream.tell()
                    line = stream.readline(65536)
                    if line and line.endswith("\n"):
                        event = json.loads(line)
                        if event["sequence"] > after:
                            await asyncio.wait_for(socket.send_json(event), timeout=5)
                            after = event["sequence"]
                    else:
                        stream.seek(position)
                        status = operations.status(identity)
                        if status["status"] in TERMINAL:
                            if after < status["sequence"]:
                                await socket.send_json(
                                    {
                                        "kind": "replay_gap",
                                        "payload": {
                                            "message": (
                                                "Recorded history is incomplete. "
                                                "Inspect the terminal snapshot/logs."
                                            ),
                                        },
                                    }
                                )
                            await socket.close(code=1000)
                            return
                        await asyncio.sleep(0.1)
        except (ApiError, ValueError, TimeoutError):
            await socket.close(code=1008)
        except WebSocketDisconnect:
            return

    return router
