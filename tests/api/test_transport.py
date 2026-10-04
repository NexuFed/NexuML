"""Runnable checks for the optional local trust boundary and CLI startup."""

import builtins
import json
import sys

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from typer.testing import CliRunner

from nexuml.api.app import create_app
from nexuml.api.security import Settings, authenticate_websocket
from nexuml.cli.main import app as cli

TOKEN = "private-test-token-" * 4
ORIGIN = "http://127.0.0.1:3000"
AUTH = {"Authorization": f"Bearer {TOKEN}", "Origin": ORIGIN}


def test_runtime_and_http_protection(tmp_path):
    settings = Settings(tmp_path, ORIGIN, TOKEN)
    with TestClient(create_app(settings), base_url="http://127.0.0.1:8000") as client:
        assert client.get("/api/v1/runtime").status_code == 401
        assert client.get("/docs").status_code == 401
        response = client.get("/api/v1/runtime", headers=AUTH)
        assert response.status_code == 200
        assert response.json()["python_executable"] == sys.executable
        assert response.json()["working_directory"] == str(tmp_path.resolve())
        assert TOKEN not in response.text
        assert (
            client.get(
                "/api/v1/runtime", headers={**AUTH, "Origin": "https://evil.test"}
            ).status_code
            == 403
        )
        assert (
            client.get("/api/v1/runtime", headers={**AUTH, "Host": "evil.test"}).status_code == 403
        )
        assert (
            client.get("/api/v1/runtime", headers={**AUTH, "X-NexuML-Interface": "99"}).json()[
                "error"
            ]["code"]
            == "interface_mismatch"
        )
        assert (
            client.post(
                "/api/v1/runtime", headers=AUTH, content="x" * (2 * 1024 * 1024 + 1)
            ).status_code
            == 413
        )
        preflight = client.options(
            "/api/v1/runtime", headers={"Origin": ORIGIN, "Access-Control-Request-Method": "GET"}
        )
        assert preflight.headers["access-control-allow-origin"] == ORIGIN


def test_interface_version_header(tmp_path):
    settings = Settings(tmp_path, ORIGIN, TOKEN)
    with TestClient(create_app(settings), base_url="http://127.0.0.1:8000") as client:
        assert (
            client.get("/api/v1/runtime", headers={**AUTH, "X-NexuML-Interface": "1"}).status_code
            == 409
        )
        assert (
            client.get("/api/v1/runtime", headers={**AUTH, "X-NexuML-Interface": "2"}).status_code
            == 200
        )


def test_websocket_authentication(tmp_path):
    settings = Settings(tmp_path, ORIGIN, TOKEN)
    api = create_app(settings)

    @api.websocket("/probe")
    async def probe(socket: WebSocket):
        if await authenticate_websocket(socket, settings):
            await socket.send_json({"authenticated": True})
            await socket.close()

    with TestClient(api, base_url="http://127.0.0.1:8000") as client:
        with client.websocket_connect(
            "ws://127.0.0.1:8000/probe", headers={"Origin": ORIGIN}
        ) as socket:
            socket.send_json({"token": TOKEN})
            assert socket.receive_json() == {"authenticated": True}
        for value in ({"token": "wrong"}, [], {"token": TOKEN * 100}):
            with client.websocket_connect(
                "ws://127.0.0.1:8000/probe", headers={"Origin": ORIGIN}
            ) as socket:
                socket.send_text(json.dumps(value))
                with pytest.raises(WebSocketDisconnect) as exc:
                    socket.receive_json()
                assert exc.value.code == 1008
        for headers in (
            {},
            {"Origin": "http://evil.test"},
            {"Origin": ORIGIN, "Host": "evil.test"},
        ):
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect("ws://127.0.0.1:8000/probe", headers=headers):
                    pass


def test_settings_and_lazy_cli(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="random token"):
        Settings(tmp_path, ORIGIN, "short")
    for origin in (
        "http://0.0.0.0:3000",
        "http://127.0.0.1:3000/path",
        "*",
        "http://localhost:bad",
    ):
        with pytest.raises(ValueError):
            Settings(tmp_path, origin, TOKEN)
    runner = CliRunner()
    assert runner.invoke(cli, ["serve", "--help"]).exit_code == 0
    original_import = builtins.__import__

    def no_api(name, *args, **kwargs):
        if name in {"uvicorn", "fastapi", "websockets"}:
            raise ImportError("test missing API extra")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_api)
    assert runner.invoke(cli, ["--help"]).exit_code == 0
    result = runner.invoke(cli, ["serve", "--directory", str(tmp_path)])
    assert result.exit_code == 1
    assert "nexuml[api]" in result.output


def test_cli_uses_selected_directory_and_loopback(tmp_path, monkeypatch):
    import uvicorn

    calls = []
    monkeypatch.setenv("NEXUML_API_TOKEN", TOKEN)
    monkeypatch.setattr(uvicorn, "run", lambda api, **options: calls.append((api, options)))
    result = CliRunner().invoke(
        cli, ["serve", "--directory", str(tmp_path), "--port", "8123", "--origin", ORIGIN]
    )
    assert result.exit_code == 0, result.output
    assert calls[0][0].state.settings.directory == tmp_path.resolve()
    assert calls[0][1]["host"] == "127.0.0.1"
    assert calls[0][1]["port"] == 8123
    assert calls[0][1]["access_log"] is False
    assert TOKEN not in result.output
