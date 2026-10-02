"""Loopback trust boundary shared by HTTP and WebSocket transports."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import WebSocket, WebSocketDisconnect
from starlette.responses import JSONResponse

INTERFACE_VERSION = 1
MAX_BODY = 2 * 1024 * 1024
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def read_token(path: Path | None) -> str:
    """Read an explicit private file or the runtime environment.

    Returns:
        Private token, never included in errors.

    Raises:
        ValueError: If a token file is not private on POSIX.
    """
    if path is None:
        return os.environ.get("NEXUML_API_TOKEN", "")
    if os.name != "nt" and path.stat().st_mode & 0o077:
        raise ValueError("Token file must be private (chmod 600).")
    return path.read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class Settings:
    """Explicit selected-runtime settings; credentials are never serialized."""

    directory: Path
    origin: str
    token: str

    def __post_init__(self):
        directory = self.directory.expanduser().resolve()
        if not directory.is_dir():
            raise ValueError("Working directory must be an existing directory.")
        object.__setattr__(self, "directory", directory)
        origin = urlsplit(self.origin)
        if (
            origin.scheme != "http"
            or origin.hostname not in LOOPBACK_HOSTS
            or origin.username
            or origin.password
            or origin.path
            or origin.query
            or origin.fragment
        ):
            raise ValueError("Studio origin must be an exact loopback HTTP origin without a path.")
        try:
            origin.port
        except ValueError as exc:
            raise ValueError("Studio origin has an invalid port.") from exc
        if len(self.token.encode()) < 32:
            raise ValueError(
                "Set NEXUML_API_TOKEN or --token-file to a random token of >=32 bytes."
            )

    def matches(self, value: str) -> bool:
        return secrets.compare_digest(value.encode(), self.token.encode())

    def permits(self, headers) -> bool:
        try:
            host = urlsplit("//" + headers.get("host", ""))
            host.port
        except ValueError:
            return False
        return (
            host.hostname in LOOPBACK_HOSTS
            and not host.username
            and not host.password
            and not host.path
            and not host.query
            and not host.fragment
            and headers.get("origin") in (None, self.origin)
        )


class TransportSecurity:
    """Authenticate before routing, including docs and WebSocket upgrades."""

    def __init__(self, app, settings: Settings):
        self.app = app
        self.settings = settings

    async def __call__(self, scope, receive, send):
        from starlette.datastructures import Headers

        if scope["type"] not in {"http", "websocket"}:
            return await self.app(scope, receive, send)
        headers = Headers(scope=scope)
        if not self.settings.permits(headers):
            if scope["type"] == "websocket":
                return await send({"type": "websocket.close", "code": 1008})
            return await self.reject(
                scope, receive, send, 403, "forbidden", "Host/origin rejected."
            )
        if scope["type"] == "websocket":
            # Browser upgrades require the exact origin. Auth follows in a bounded first frame.
            if headers.get("origin") != self.settings.origin:
                return await send({"type": "websocket.close", "code": 1008})
            return await self.app(scope, receive, send)
        if scope["method"] == "OPTIONS":
            return await self.app(scope, receive, send)
        authorization = headers.get("authorization", "")
        if not authorization.startswith("Bearer ") or not self.settings.matches(authorization[7:]):
            return await self.reject(scope, receive, send, 401, "unauthorized", "Token required.")
        if headers.get("x-nexuml-interface", "1") != str(INTERFACE_VERSION):
            return await self.reject(
                scope,
                receive,
                send,
                409,
                "interface_mismatch",
                "Studio/API interface mismatch.",
            )
        length = headers.get("content-length", "0")
        if not length.isdecimal() or int(length) > MAX_BODY:
            return await self.reject(scope, receive, send, 413, "too_large", "Request too large.")
        # Buffer once to bound chunked bodies too, before framework parsing/validation.
        messages, size = [], 0
        while True:
            message = await receive()
            messages.append(message)
            size += len(message.get("body", b""))
            if size > MAX_BODY:
                return await self.reject(
                    scope, receive, send, 413, "too_large", "Request too large."
                )
            if not message.get("more_body", False):
                break

        async def bounded_receive():
            return messages.pop(0) if messages else await receive()

        return await self.app(scope, bounded_receive, send)

    @staticmethod
    async def reject(scope, receive, send, status, code, message):
        response = JSONResponse({"error": {"code": code, "message": message}}, status_code=status)
        await response(scope, receive, send)


async def authenticate_websocket(socket: WebSocket, settings: Settings) -> bool:
    """Accept only a bounded, timely first authentication frame.

    Returns:
        Whether authentication succeeded; failures close the socket without data.
    """
    await socket.accept()
    try:
        frame = await asyncio.wait_for(socket.receive_text(), timeout=5)
        if len(frame.encode()) > 4096:
            await socket.close(code=1008)
            return False
        data = json.loads(frame)
        if not isinstance(data, dict) or not isinstance(data.get("token"), str):
            await socket.close(code=1008)
            return False
        if not settings.matches(data["token"]):
            await socket.close(code=1008)
            return False
        return True
    except (TimeoutError, ValueError, WebSocketDisconnect, KeyError, RuntimeError):
        await socket.close(code=1008)
        return False
