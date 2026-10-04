"""Selected-runtime extension/remote observation tests; never contact a cluster."""

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from nexuml.api.app import create_app
from nexuml.api.security import Settings
from nexuml.api.operations import TERMINAL, Operations

TOKEN = "private-execution-test-token-" * 3
ORIGIN = "http://127.0.0.1:3000"


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    home = tmp_path / "home"
    config = home / ".config" / "nexuml"
    config.mkdir(parents=True)
    package = tmp_path / "library" / "execution_fixture"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "backend.py").write_text("""
import time
from pathlib import Path
from nexuml.core.components import ExecutionBackendDefinition
from nexuml.core.discovery import execution_backend
from nexuml.execution.schemas import NativeReference
@execution_backend("fixture-remote")
class Remote(ExecutionBackendDefinition):
    label = "Fixture remote backend"
    seconds: float = 0.2
    outcome: str = "succeeded"
    capabilities = {"status": True, "logs": True, "metrics": False,
                    "cancellation": True, "native_reference": True,
                    "process_ownership": False, "resume": False}
    def preflight(self, scenario, **kwargs):
        super().preflight(scenario, **kwargs)
        return {"reviewed": self.outcome}
    def run(self, scenario, **kwargs):
        assert kwargs["review"] == {"reviewed": self.outcome}
        ref = NativeReference(backend="fixture-remote", context="fixture", namespace="test",
            api_version="fixture/v1", kind="Fixture", name="owned-fixture", uid="owned-uid")
        observer = kwargs.get("observer")
        observer.record("native_reference", reference=ref.model_dump())
        observer.record("native_status", status="pending")
        observer.record("log", text="actual fixture log")
        end = time.monotonic() + self.seconds
        while time.monotonic() < end:
            if Path("cancelled-fixture").exists():
                return {"status": "cancelled", "native_reference": ref.model_dump()}
            time.sleep(0.05)
        observer.record("native_status", status="running")
        return {"status": self.outcome, "native_reference": ref.model_dump(), "metrics": {}}
    def inspect(self, reference):
        return {"status": "cancelled" if Path("cancelled-fixture").exists() else "running",
                "native_reference": reference.model_dump()}
    def cancel(self, reference):
        assert reference.uid == "owned-uid"
        Path("cancelled-fixture").touch()
        return {"status": "cancelled", "native_reference": reference.model_dump()}
""")
    (config / "libraries.json").write_text(json.dumps({"roots": [str(package.parent)]}))
    monkeypatch.setenv("HOME", str(home))
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("PYTHONPATH", f"{root / 'src'}:{root / 'library/src'}")
    return tmp_path


def body(seconds=0.2, outcome="succeeded"):
    return {
        "data": {
            "name": "remote-fixture",
            "execution": {
                "type": "fixture-remote",
                "version": "1",
                "params": {"seconds": seconds, "outcome": outcome},
            },
        }
    }


def wait(client, identity, predicate):
    deadline = time.monotonic() + 15
    while True:
        value = client.get(f"/api/v1/operations/{identity}").json()
        if predicate(value):
            return value
        assert time.monotonic() < deadline, value
        time.sleep(0.05)


@pytest.mark.parametrize("outcome", ["succeeded", "failed"])
def test_extension_review_native_terminal_and_no_resubmission(runtime, outcome):
    api = create_app(Settings(runtime, ORIGIN, TOKEN))
    with TestClient(
        api, base_url="http://127.0.0.1:8000", headers={"Authorization": f"Bearer {TOKEN}"}
    ) as client:
        catalog = client.get("/api/v1/registry").json()
        row = next(row for row in catalog["execution_backends"] if row["type"] == "fixture-remote")
        assert row["schema"]["properties"]["outcome"]["type"] == "string"
        prepared = client.post("/api/v1/train/prepare", json=body(outcome=outcome))
        assert prepared.status_code == 200, prepared.text
        assert prepared.json()["launch_review"] == {"reviewed": outcome}
        assert "launch_plan" not in prepared.json()
        launched = client.post("/api/v1/train", json=body(outcome=outcome))
        assert launched.status_code == 202, launched.text
        identity = launched.json()["id"]
        done = wait(client, identity, lambda row: row["status"] in TERMINAL)
        assert done["status"] == outcome and done["native_reference"]["uid"] == "owned-uid"
        assert done["capabilities"]["metrics"] is False
        assert not (api.state.operations.folder(identity) / "worker-config.yaml").exists()
        assert TOKEN not in json.dumps(done)
        assert client.post(f"/api/v1/operations/{identity}/inspect", json={}).status_code == 200
        assert len(client.get("/api/v1/operations").json()["operations"]) == 1


def test_verified_remote_cancel_and_unknown_after_shutdown(runtime):
    settings = Settings(runtime, ORIGIN, TOKEN)
    api = create_app(settings)
    with TestClient(
        api, base_url="http://127.0.0.1:8000", headers={"Authorization": f"Bearer {TOKEN}"}
    ) as client:
        launched = client.post("/api/v1/train", json=body(seconds=30)).json()
        identity = launched["id"]
        pending = wait(client, identity, lambda row: bool(row.get("native_reference")))
        assert pending["status"] == "pending"
        assert client.post("/api/v1/train", json=body()).status_code == 409
        # Discovery does not consume the active training slot.
        assert client.post("/api/v1/execution/discover", json=body()).status_code == 200
        cancelled = client.post(f"/api/v1/operations/{identity}/cancel", json={})
        assert cancelled.status_code == 200, cancelled.text
        assert cancelled.json()["status"] == "cancelled"
        (runtime / "cancelled-fixture").unlink()
        second = client.post("/api/v1/train", json=body(seconds=30)).json()["id"]
        wait(client, second, lambda row: bool(row.get("native_reference")))
    assert not (runtime / "cancelled-fixture").exists()  # shutdown detaches, not native cancel
    restarted = Operations(settings)
    assert restarted.status(second)["status"] == "ownership_unknown"
    assert restarted.status(second)["native_reference"]["uid"] == "owned-uid"


def test_discovery_rejects_unauthorized_paths_and_credential_urls(runtime):
    api = create_app(Settings(runtime, ORIGIN, TOKEN))
    with TestClient(
        api, base_url="http://127.0.0.1:8000", headers={"Authorization": f"Bearer {TOKEN}"}
    ) as client:
        secret = "must-not-be-echoed"
        outside = runtime.parent / "unauthorized-private.yaml"
        outside.write_text(secret)
        unsafe = {
            "data": {
                "name": "unsafe",
                "execution": {
                    "type": "kubernetes-job",
                    "version": "1",
                    "params": {"context": "do-not-probe", "template": str(outside)},
                },
            }
        }
        rejected = client.post("/api/v1/execution/discover", json=unsafe)
        assert rejected.status_code == 422 and secret not in rejected.text
        for target in (
            f"http://user:{secret}@host:8265",
            f"http://host:8265/?token={secret}",
            f"user:{secret}@host:10001",
        ):
            unsafe["data"]["execution"] = {
                "type": "ray-cluster",
                "version": "1",
                "params": {"target": {"address": target}},
            }
            rejected = client.post("/api/v1/execution/discover", json=unsafe)
            assert rejected.status_code == 422 and secret not in rejected.text
            assert rejected.json()["error"]["fields"][0]["loc"][:2] == ["execution", "params"]
