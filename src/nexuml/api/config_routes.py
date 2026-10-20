"""HTTP presentation of existing discovery and configuration boundaries."""

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from nexuml.api.errors import ApiError
from nexuml.api.files import Files
from nexuml.api.process import call_worker
from nexuml.api.security import MAX_BODY, Settings


class ConfigInput(BaseModel):
    """Editable portable config; stage_order is transport metadata only."""

    model_config = ConfigDict(extra="forbid")
    data: dict[str, Any] | None = None
    yaml: str | None = None
    stage_order: list[str] | None = None


class SaveInput(ConfigInput):
    """Revision-protected file save; new files require a null base_revision."""

    path: str
    base_revision: str | None


class PathInput(BaseModel):
    """Explicit authorized path, not arbitrary filesystem browsing."""

    model_config = ConfigDict(extra="forbid")
    path: str


class LayoutInput(PathInput):
    """Visual metadata remains separate from executable configuration."""

    base_revision: str | None
    semantic_revision: str
    layout: dict[str, Any]


def config_router(settings: Settings) -> APIRouter:
    """Register thin synchronous routes (FastAPI runs them outside its event loop).

    Returns:
        Router over the selected working directory and existing framework calls.
    """
    from nexuml.core.discovery import LibraryConfig

    router = APIRouter(prefix="/api/v1")
    files = Files(settings.directory)

    @router.get("/registry")
    def registry():
        return call_worker("catalog", {}, settings.directory)

    @router.get("/scenarios")
    def scenarios():
        result = call_worker("catalog", {}, settings.directory)
        return {"scenarios": result["scenarios"], "errors": result["errors"]}

    @router.get("/libraries")
    def libraries():
        return call_worker("catalog", {}, settings.directory)["libraries"]

    @router.post("/libraries")
    def add_library(body: PathInput):
        # Explicit root authorization retains the existing library CLI policy.
        path = Path(body.path).expanduser()
        root = (settings.directory / path).resolve()
        if not root.is_dir():
            raise ApiError(422, "validation", "Library root must be an existing directory.")
        with files.lock:
            config = LibraryConfig.load()
            config.add_root(str(root))
            config.save()
        return {"roots": config.roots}

    @router.delete("/libraries")
    def remove_library(body: PathInput):
        root = (settings.directory / Path(body.path).expanduser()).resolve()
        with files.lock:
            config = LibraryConfig.load()
            if str(root) not in config.roots:
                raise ApiError(404, "not_found", "Library root is not configured.")
            config.remove_root(str(root))
            config.save()
        return {"roots": config.roots}

    @router.post("/scenarios/{name}/resolve")
    def resolve_scenario(name: str):
        return call_worker("resolve", {"name": name}, settings.directory)

    @router.post("/scenarios/resolve-file")
    def resolve_file(body: PathInput):
        return call_worker("resolve_file", {"path": str(files.path(body.path))}, settings.directory)

    @router.post("/config/validate")
    def validate(body: ConfigInput):
        return call_worker("validate", body.model_dump(), settings.directory)

    @router.post("/config/load")
    def load(body: PathInput):
        path = files.path(body.path)
        if not path.is_file():
            raise ApiError(404, "not_found", "Configuration file not found.")
        if path.stat().st_size > MAX_BODY:
            raise ApiError(413, "too_large", "Configuration file exceeds the transport limit.")
        raw = path.read_bytes()
        result = call_worker("validate", {"yaml": raw.decode("utf-8")}, settings.directory)
        import hashlib

        return {**result, "path": str(path), "base_revision": hashlib.sha256(raw).hexdigest()}

    @router.post("/config/save")
    def save(body: SaveInput):
        path = files.path(body.path)
        result = call_worker(
            "validate",
            body.model_dump(include={"data", "yaml", "stage_order"}),
            settings.directory,
        )
        revision = files.save(path, result["yaml"], body.base_revision)
        return {**result, "path": str(path), "base_revision": revision}

    @router.post("/config/layout/load")
    def load_layout(body: PathInput):
        path = files.path(body.path + ".studio.json")
        if not path.exists():
            return {"base_revision": None, "layout": None}
        if path.stat().st_size > MAX_BODY:
            raise ApiError(413, "too_large", "Layout sidecar exceeds the transport limit.")
        try:
            return {
                **json.loads(path.read_text(encoding="utf-8")),
                "base_revision": files.revision(path),
            }
        except (ValueError, TypeError):
            raise ApiError(422, "validation", "Invalid layout sidecar; config is unchanged.")

    @router.post("/config/layout/save")
    def save_layout(body: LayoutInput):
        path = files.path(body.path + ".studio.json")
        text = json.dumps({"semantic_revision": body.semantic_revision, "layout": body.layout})
        return {"base_revision": files.save(path, text, body.base_revision)}

    return router
