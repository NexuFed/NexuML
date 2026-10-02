"""FastAPI presentation of the selected local NexuML installation."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from nexuml.api.security import INTERFACE_VERSION, Settings, TransportSecurity
from nexuml.api.errors import ApiError
from nexuml.api.operations import Operations


def create_app(settings: Settings) -> FastAPI:
    """Create an isolated local API application.

    Returns:
        Authenticated FastAPI app for this selected runtime and directory.
    """
    operations = Operations(settings)

    @asynccontextmanager
    async def lifespan(app):
        yield
        from starlette.concurrency import run_in_threadpool

        await run_in_threadpool(operations.close)

    app = FastAPI(title="NexuML local API", version=str(INTERFACE_VERSION), lifespan=lifespan)
    app.state.settings = settings
    app.state.operations = operations
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.origin],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-NexuML-Interface"],
    )
    app.add_middleware(TransportSecurity, settings=settings)

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError):
        return JSONResponse(
            {
                "error": {
                    "code": exc.code,
                    "message": exc.message.replace(settings.token, "[redacted]"),
                    "fields": exc.fields,
                }
            },
            status_code=exc.status,
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        # Never echo inputs/context: those may include credentials or arbitrary Python objects.
        fields = [
            {"loc": list(e["loc"]), "message": e["msg"], "type": e["type"]} for e in exc.errors()
        ]
        return await api_error(request, ApiError(422, "validation", "Invalid request.", fields))

    @app.exception_handler(OSError)
    async def file_error(request: Request, exc: OSError):
        return await api_error(request, ApiError(422, "file_error", "File could not be accessed."))

    @app.exception_handler(UnicodeError)
    async def encoding_error(request: Request, exc: UnicodeError):
        return await api_error(
            request, ApiError(422, "validation", "Configuration must be UTF-8 text.")
        )

    @app.get("/api/v1/runtime")
    def runtime():
        return {
            "interface_version": INTERFACE_VERSION,
            "nexuml_version": version("nexuml"),
            "python_executable": sys.executable,
            "working_directory": str(settings.directory),
        }

    from nexuml.api.config_routes import config_router
    from nexuml.api.operation_routes import operation_router

    app.include_router(config_router(settings))
    app.include_router(operation_router(operations))
    return app
