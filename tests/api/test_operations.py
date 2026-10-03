"""CPU-only proof of native calls, observation/reconnect, timeout and stop ownership."""

import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from nexuml.api.app import create_app
from nexuml.api.errors import ApiError
from nexuml.api.operations import Operations, TERMINAL, stop_process_tree, write_json
from nexuml.api.security import Settings
from nexuml.api.worker import dispatch
from nexuml.core.serialization import lower_model
from nexuml.core.export import load_package
from nexuml.core.types import RayExecutionSpec
from nexuml_library.scenarios.asd.synthetic_linear_ae import synthetic_linear_ae_reconstruction

TOKEN = "test-operation-private-token-" * 3
ORIGIN = "http://127.0.0.1:3000"


def scenario():
    value = synthetic_linear_ae_reconstruction(
        feature_shape=(8,),
        num_samples=32,
        hidden_dims=[8],
        latent_dim=2,
        batch_size=8,
        max_epochs=1,
    )
    value.training.accelerator = "cpu"
    value.training.devices = 1
    return value


def wait_status(client, identity, deadline=30):
    end = time.monotonic() + deadline
    while True:
        status = client.get(f"/api/v1/operations/{identity}").json()
        if status["status"] in TERMINAL:
            return status
        assert time.monotonic() < end, status
        time.sleep(0.1)


def test_native_cpu_build_train_replay_and_checkpoint_export(tmp_path):
    api = create_app(Settings(tmp_path, ORIGIN, TOKEN))
    with TestClient(
        api, base_url="http://127.0.0.1:8000", headers={"Authorization": f"Bearer {TOKEN}"}
    ) as client:
        body = {"data": lower_model(scenario()), "stage_order": list(scenario().pipeline.stages)}
        build = client.post("/api/v1/build", json=body)
        assert build.status_code == 202, build.text
        checked = wait_status(client, build.json()["id"])
        assert checked["status"] == "succeeded", checked
        assert checked["result"]["shapes"]["reconstructed"] == [8]
        original = copy.deepcopy(body)
        launched = client.post("/api/v1/train", json=body)
        assert launched.status_code == 202, launched.text
        identity = launched.json()["id"]
        assert client.post("/api/v1/train", json=body).status_code == 409
        body["data"]["training"]["max_epochs"] = 99
        stages = body["data"]["pipeline"]["stages"]
        first, last = body["stage_order"][0], body["stage_order"][-1]
        stages[last].append(stages[first].pop(0))
        body["stage_order"].reverse()
        # Editor transfers/order edits stay separate from the already accepted native run.
        assert client.post("/api/v1/config/validate", json=body).status_code == 200
        frozen = client.get(f"/api/v1/operations/{identity}/config").json()
        assert frozen["data"]["training"]["max_epochs"] == 1
        assert frozen["data"] == original["data"]
        assert frozen["stage_order"] == original["stage_order"]
        endpoint = f"ws://127.0.0.1:8000/api/v1/operations/{identity}/events"
        with client.websocket_connect(endpoint, headers={"Origin": ORIGIN}) as socket:
            socket.send_json({"token": TOKEN})
            assert socket.receive_json()["kind"] == "snapshot"

            first = socket.receive_json()
            sequence = first["sequence"]
        # Detaching observation must not launch/cancel/modify training.
        done = wait_status(client, identity)
        assert done["status"] == "succeeded", done
        assert done["result"]["test_results"]
        assert any(a["kind"] == "checkpoint" for a in done["artifacts"])
        events = []
        with client.websocket_connect(
            endpoint + f"?after={sequence}", headers={"Origin": ORIGIN}
        ) as socket:
            socket.send_json({"token": TOKEN})
            assert socket.receive_json()["payload"]["id"] == identity
            while True:
                event = socket.receive_json()
                events.append(event)
                if event["kind"] == "terminal":
                    break
        assert all(e["sequence"] > sequence for e in events)
        assert any(e["kind"] == "metrics" and e["payload"]["values"] for e in events)
        reported = done["result"]["test_results"][0]
        observed = [e["payload"]["values"] for e in events if e["kind"] == "metrics"][-1]
        common = set(reported) & set(observed)
        assert common, (reported, observed)
        for key in common:
            assert observed[key] == pytest.approx(reported[key])
        phases = [e["payload"]["phase"] for e in events if e["kind"] == "progress"]
        assert "test" in phases
        batches = [
            e["payload"]
            for e in events
            if e["kind"] == "progress" and e["payload"].get("phase") == "train"
        ]
        assert batches and batches[-1]["batch"] == batches[-1]["total"] > 0
        assert batches[-1]["max_epochs"] == 1
        assert TOKEN not in json.dumps(done)
        assert (
            client.post(
                "/api/v1/export",
                json={
                    "source_id": checked["id"],
                    "output": "unsupported",
                    "kind": "train_package",
                },
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/api/v1/export",
                json={
                    "source_id": identity,
                    "output": "unsupported",
                    "kind": "unknown-format",
                },
            ).status_code
            == 422
        )
        exported = client.post(
            "/api/v1/export",
            json={"source_id": identity, "output": "exported-package", "kind": "train_package"},
        )
        assert exported.status_code == 202, exported.text
        export_done = wait_status(client, exported.json()["id"])
        assert export_done["status"] == "succeeded", export_done
        assert (tmp_path / "exported-package" / "resolved_config.yaml").is_file()
        pipeline, config, metadata = load_package(tmp_path / "exported-package")
        assert config.name == scenario().name
        assert pipeline.input_sizes["reconstructed"] == (8,)
        assert metadata
        from nexuml.training.lightning import create_runtime_artifacts_from_trainer_checkpoint

        checkpoint_source = next(a for a in done["artifacts"] if a["kind"] == "checkpoint")
        expected = create_runtime_artifacts_from_trainer_checkpoint(
            checkpoint_source["path"]
        ).pipeline
        for key, tensor in expected.state_dict().items():
            torch.testing.assert_close(pipeline.state_dict()[key], tensor, rtol=0, atol=0)
        index = next(
            i for i, a in enumerate(export_done["artifacts"]) if a["kind"] == "train_package"
        )
        assert (
            client.get(f"/api/v1/operations/{export_done['id']}/artifacts/{index}").status_code
            == 200
        )
    # Completed records remain real, inspectable terminal results after server restart.
    restarted = Operations(Settings(tmp_path, ORIGIN, TOKEN))
    assert restarted.status(identity)["status"] == "succeeded"
    assert len(restarted.list()) == 3


def test_local_resume_uses_checkpoint_scenario_and_absent_artifacts(tmp_path):
    value = scenario()
    api = create_app(Settings(tmp_path, ORIGIN, TOKEN))
    with TestClient(
        api, base_url="http://127.0.0.1:8000", headers={"Authorization": f"Bearer {TOKEN}"}
    ) as client:
        launched = client.post("/api/v1/train", json={"data": lower_model(value)}).json()
        source = wait_status(client, launched["id"])
        assert source["status"] == "succeeded", source
        checkpoint = next(a for a in source["artifacts"] if a["kind"] == "checkpoint")
        value.name = "not-the-checkpoint-scenario"
        prepared = client.post(
            "/api/v1/train/prepare",
            json={
                "data": lower_model(value),
                "trainer_checkpoint": checkpoint["path"],
            },
        )
        assert prepared.status_code == 200, prepared.text
        assert prepared.json()["data"]["name"] == scenario().name
        resumed = client.post(
            "/api/v1/train",
            json={
                "data": lower_model(value),
                "trainer_checkpoint": checkpoint["path"],
            },
        ).json()
        assert wait_status(client, resumed["id"])["status"] == "succeeded"
        assert client.get(f"/api/v1/operations/{source['id']}/artifacts/999").status_code == 404
        index = source["artifacts"].index(checkpoint)
        Path(checkpoint["path"]).unlink()
        assert client.get(f"/api/v1/operations/{source['id']}/artifacts/{index}").status_code == 404


def test_cancel_and_owned_shutdown_confirm_terminal_state(tmp_path):
    api = create_app(Settings(tmp_path, ORIGIN, TOKEN))
    body = {"data": lower_model(scenario())}
    with TestClient(
        api, base_url="http://127.0.0.1:8000", headers={"Authorization": f"Bearer {TOKEN}"}
    ) as client:
        launched = client.post("/api/v1/train", json=body).json()
        stopped = client.post(f"/api/v1/operations/{launched['id']}/cancel", json={})
        assert stopped.status_code == 200, stopped.text
        assert stopped.json()["status"] == "cancelled"
        assert api.state.operations.active is None
        active = client.post("/api/v1/train", json=body).json()
    assert api.state.operations.status(active["id"])["status"] == "cancelled"
    assert api.state.operations.active is None


def test_logs_root_and_explicit_replay_gap(tmp_path, monkeypatch):
    logs = tmp_path / "external-logs"
    working = tmp_path / "work"
    working.mkdir()
    monkeypatch.setenv("NEXUML_LOGS_ROOT", str(logs))
    api = create_app(Settings(working, ORIGIN, TOKEN))
    assert api.state.operations.root == logs / ".experiments/.studio/operations"
    with TestClient(
        api, base_url="http://127.0.0.1:8000", headers={"Authorization": f"Bearer {TOKEN}"}
    ) as client:
        launched = client.post("/api/v1/build", json={"data": lower_model(scenario())}).json()
        done = wait_status(client, launched["id"])
        assert done["status"] == "succeeded"
        endpoint = f"ws://127.0.0.1:8000/api/v1/operations/{launched['id']}/events?after=999999"
        with client.websocket_connect(endpoint, headers={"Origin": ORIGIN}) as socket:
            socket.send_json({"token": TOKEN})
            assert socket.receive_json()["kind"] == "replay_gap"
            assert socket.receive_json()["kind"] == "snapshot"

        folder = api.state.operations.folder(launched["id"])
        oversized = {**done, "result": {"long": "a" * 70000}}
        write_json(folder / "status.json", oversized)
        with client.websocket_connect(
            endpoint.replace("999999", str(done["sequence"])), headers={"Origin": ORIGIN}
        ) as socket:
            socket.send_json({"token": TOKEN})
            assert "frame limit" in socket.receive_json()["payload"]["message"]
            assert socket.receive_json()["partial"] is True


def test_terminal_overwrite_bytes_survive_observation_and_raw_logs(tmp_path):
    operations = Operations(Settings(tmp_path, ORIGIN, TOKEN))
    folder = operations.root / ("b" * 32)
    folder.mkdir()
    write_json(folder / "status.json", {"id": folder.name, "status": "running", "sequence": 0})
    write_json(folder / "response.json", {"fixture": "completed"})
    raw = b"Download 1%\rDownload 2%\x1b[K\rDownload 100%\r\nReady\n"
    (folder / "logs.txt").write_bytes(raw)
    (folder / "events.jsonl").touch()
    operations.observe(folder, SimpleNamespace(poll=lambda: 0, returncode=0), None)
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    assert (
        "".join(event["payload"]["text"] for event in events if event["kind"] == "log").encode()
        == raw
    )
    assert (folder / "logs.txt").read_bytes() == raw
    assert operations.status(folder.name)["status"] == "succeeded"


def test_build_failure_is_reported_without_changing_field_validation(tmp_path):
    value = scenario()
    value.pipeline.stages["Encoder"][0].keys_in = ["missing-runtime-input"]
    with TestClient(
        create_app(Settings(tmp_path, ORIGIN, TOKEN)),
        base_url="http://127.0.0.1:8000",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        body = {"data": lower_model(value)}
        assert client.post("/api/v1/config/validate", json=body).status_code == 200
        launched = client.post("/api/v1/build", json=body)
        assert launched.status_code == 202
        done = wait_status(client, launched.json()["id"])
        assert done["status"] == "failed", done
        assert "missing-runtime-input" in json.dumps(done["result"])


def test_deadline_releases_owned_operation(tmp_path):
    api = create_app(Settings(tmp_path, ORIGIN, TOKEN))
    with TestClient(
        api, base_url="http://127.0.0.1:8000", headers={"Authorization": f"Bearer {TOKEN}"}
    ) as client:
        body = {"data": lower_model(scenario()), "timeout": 0.001}
        response = client.post("/api/v1/build", json=body)
        assert response.status_code == 202
        done = wait_status(client, response.json()["id"])
        assert done["status"] == "timeout"
        assert api.state.operations.active is None


def test_restart_unknown_and_remote_stop_is_not_simulated(tmp_path):
    settings = Settings(tmp_path, ORIGIN, TOKEN)
    operations = Operations(settings)
    folder = operations.root / ("a" * 32)
    folder.mkdir()
    state = {
        "id": folder.name,
        "status": "running",
        "cancellation": False,
        "sequence": 0,
        "created_at": "2026-01-01",
        "artifacts": [],
    }
    write_json(folder / "status.json", state)
    with pytest.raises(ApiError) as exc:
        operations.cancel(folder.name)
    assert exc.value.code == "unsupported"
    restarted = Operations(settings)
    assert restarted.status(folder.name)["status"] == "ownership_unknown"
    assert "cannot be verified" in restarted.status(folder.name)["message"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX descendant/process-group proof")
def test_cancel_owns_descendants_even_after_leader_exit(tmp_path):
    child_pid = tmp_path / "child.pid"
    code = (
        "import subprocess,sys,time; child=subprocess.Popen([sys.executable,'-c',"
        "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)']); "
        f"open({str(child_pid)!r},'w').write(str(child.pid)); time.sleep(60)"
    )
    process = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    try:
        end = time.monotonic() + 5
        while not child_pid.exists():
            assert time.monotonic() < end
            time.sleep(0.05)
        pid = int(child_pid.read_text())
        time.sleep(0.1)  # Let the descendant install its SIGTERM handler.
        stop_process_tree(process)
        assert process.poll() is not None
        proc_status = Path(f"/proc/{pid}/stat")
        if proc_status.exists():
            assert proc_status.read_text().split()[2] == "Z"  # Stopped, awaiting OS reaping.
    finally:
        if process.poll() is None:
            stop_process_tree(process)


def test_ray_delegation_and_resume_guard(tmp_path, monkeypatch):
    value = scenario()
    value.execution = RayExecutionSpec()
    payload = {"data": lower_model(value), "observations": str(tmp_path / "observations.jsonl")}
    calls = []
    monkeypatch.setattr(
        "nexuml.execution.run_ray",
        lambda s: calls.append(s) or SimpleNamespace(metrics={"loss": 1.25}),
    )
    result = dispatch("train", payload)
    assert result["metrics"] == {"loss": 1.25}
    assert len(calls) == 1
    assert calls[0].execution.kind == "ray"
    with pytest.raises(ValueError, match="local-only"):
        dispatch("prepare_train", {**payload, "trainer_checkpoint": "anything.ckpt"})
