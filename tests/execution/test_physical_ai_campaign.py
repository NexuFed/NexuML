"""Hermetic campaign monitoring; no cluster, model, native runtime or GPU."""

import json
from pathlib import Path
import runpy
import signal
import sys
import time
from types import SimpleNamespace
from uuid import UUID

import pytest


@pytest.fixture
def driver(monkeypatch):
    functions = runpy.run_path(str(Path("scripts/physical_ai_acceptance.py")))
    clock = SimpleNamespace(now=0.0)

    def sleep(seconds):
        clock.now += seconds

    monkeypatch.setitem(
        functions["run_campaign"].__globals__,
        "time",
        SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep),
    )
    return functions, clock


def attempts(count):
    execution = SimpleNamespace(
        component_name="ray-jobs-sdk",
        entrypoint_num_gpus=1,
        entrypoint_resources={"pool_gpu_workers": 1},
    )
    return [
        {
            "case": "preflight",
            "scenario": SimpleNamespace(execution=execution),
            "attempt_root": f"/mnt/data/nexuml-qualification/campaign/{index}",
            "wall_seconds": 10,
            "options": {"source_revision": "revision", "source_tree_sha256": "tree"},
        }
        for index in range(count)
    ]


class Client:
    def __init__(self, outcome="SUCCEEDED", pending_polls=1, submission_error=None):
        self.jobs = {}
        self.outcome = outcome
        self.pending_polls = pending_polls
        self.peak = 0
        self.stopped = []
        self.requests = []
        self.submission_error = submission_error

    def submit(self, scenario, **kwargs):
        assert kwargs["operation"] == "acceptance"
        self.requests.append(kwargs)
        identity = kwargs["submission_id"]
        assert UUID(identity).version == 4
        assert identity not in self.jobs
        self.jobs[identity] = 0
        self.peak = max(
            self.peak,
            sum(
                value <= self.pending_polls and name not in self.stopped
                for name, value in self.jobs.items()
            ),
        )
        if self.submission_error:
            raise self.submission_error
        return identity

    def get_job_status(self, identity):
        if identity in self.stopped:
            return "STOPPED"
        self.jobs[identity] += 1
        return "PENDING" if self.jobs[identity] <= self.pending_polls else self.outcome

    def get_job_logs(self, identity):
        assert identity in self.jobs
        return 'NEXUML_CASE_RESULT={"status":"PASS","source_tree_sha256":"tree"}\n'

    def stop_job(self, identity):
        assert identity in self.jobs
        self.stopped.append(identity)


def test_pending_jobs_occupy_slots_and_success_requires_terminal_receipt(
    driver, monkeypatch, tmp_path
):
    functions, _ = driver
    client = Client()
    monkeypatch.setattr("nexuml.execution.run", client.submit)
    result = functions["run_campaign"](
        attempts(5),
        client,
        tmp_path / "result",
        concurrency=2,
        poll_seconds=1,
    )
    assert client.peak == 2
    assert len(client.requests) == 5
    assert len(result["results"]) == 5
    assert all(row["status"] == "passed" for row in result["results"])
    assert not client.stopped


def test_submission_id_and_successful_looking_log_cannot_hide_failure(
    driver, monkeypatch, tmp_path
):
    functions, _ = driver
    client = Client(outcome="FAILED", pending_polls=0)
    monkeypatch.setattr("nexuml.execution.run", client.submit)
    result = functions["run_campaign"](attempts(1), client, tmp_path / "result")
    assert result["results"][0]["status"] == "failed"
    assert len(client.requests) == 1


def test_deadline_waits_for_owned_native_stop_before_releasing_slot(driver, monkeypatch, tmp_path):
    functions, _ = driver
    client = Client(pending_polls=100)
    monkeypatch.setattr("nexuml.execution.run", client.submit)
    result = functions["run_campaign"](
        attempts(2),
        client,
        tmp_path / "result",
        queue_seconds=1,
        poll_seconds=1,
    )
    assert client.stopped == [request["submission_id"] for request in client.requests]
    assert client.peak == 1
    assert all(row["native_status"] == "STOPPED" for row in result["results"])
    assert all(row["status"] == "failed" and row["deadline_exceeded"] for row in result["results"])


@pytest.mark.parametrize("concurrency", [0, 5, True, 1.5])
def test_invalid_concurrency_never_submits(driver, monkeypatch, tmp_path, concurrency):
    functions, _ = driver
    monkeypatch.setattr("nexuml.execution.run", lambda *_args, **_kwargs: pytest.fail("submitted"))
    with pytest.raises(ValueError, match="concurrency"):
        functions["run_campaign"](attempts(1), Client(), tmp_path / "out", concurrency=concurrency)


def test_duplicate_attempt_root_fails_before_submission(driver, monkeypatch, tmp_path):
    functions, _ = driver
    rows = attempts(2)
    rows[1]["attempt_root"] = rows[0]["attempt_root"]
    monkeypatch.setattr("nexuml.execution.run", lambda *_args, **_kwargs: pytest.fail("submitted"))
    with pytest.raises(ValueError, match="share output roots"):
        functions["run_campaign"](rows, Client(), tmp_path / "out")


def test_missing_or_changed_source_receipt_is_not_a_pass(driver, monkeypatch, tmp_path):
    functions, _ = driver
    client = Client(pending_polls=0)
    monkeypatch.setattr(
        client,
        "get_job_logs",
        lambda _: 'NEXUML_CASE_RESULT={"status":"PASS","source_tree_sha256":"other"}',
    )
    monkeypatch.setattr("nexuml.execution.run", client.submit)
    result = functions["run_campaign"](attempts(1), client, tmp_path / "out")
    assert result["results"][0]["status"] == "failed"


def test_control_plane_timeout_restores_alarm_ownership(driver):
    functions, _ = driver
    previous = signal.getsignal(signal.SIGALRM)
    with pytest.raises(TimeoutError, match="control-plane"):
        functions["_bounded_rpc"](time.sleep, 0.2, timeout_seconds=0.01)
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
    assert signal.getsignal(signal.SIGALRM) is previous


def test_worker_options_exclude_driver_metadata_and_unset_inputs(driver, monkeypatch, tmp_path):
    functions, _ = driver
    client = Client(pending_polls=0)
    monkeypatch.setattr("nexuml.execution.run", client.submit)
    rows = attempts(1)
    rows[0]["options"].update(data_root=None, suite="full", source_dirty=False)
    functions["run_campaign"](rows, client, tmp_path / "out")
    assert client.requests[0]["case_options"] == {}


def test_owned_uuid_and_identity_are_durable_before_submission(driver, monkeypatch, tmp_path):
    functions, _ = driver
    client = Client(pending_polls=0)
    output = tmp_path / "out"

    def submit(scenario, **kwargs):
        summary = json.loads((output / "summary.json").read_text())
        identity = kwargs["submission_id"]
        assert summary["active"] == [identity]
        assert summary["active_attempts"] == [
            {
                "job_id": identity,
                "case": "preflight",
                "attempt_root": attempts(1)[0]["attempt_root"],
                "source_revision": "revision",
                "source_dirty": None,
                "source_tree_sha256": "tree",
            }
        ]
        return client.submit(scenario, **kwargs)

    monkeypatch.setattr("nexuml.execution.run", submit)
    result = functions["run_campaign"](attempts(1), client, output)
    assert result["results"][0]["job_id"] == client.requests[0]["submission_id"]


def test_shared_backend_sdk_response_loss_keeps_preassigned_durable_identity(
    driver, monkeypatch, tmp_path
):
    from nexuml.core.types import ScenarioSpec
    from nexuml.execution.definitions import RayJobsSDKExecution

    functions, _ = driver
    output = tmp_path / "out"
    client = Client()

    class SDK:
        def __init__(self, *_args, **_kwargs):
            pass

        def submit_job(self, **kwargs):
            identity = kwargs["submission_id"]
            summary = json.loads((output / "summary.json").read_text())
            assert summary["active"] == [identity]
            assert summary["active_attempts"][0]["source_tree_sha256"] == "tree"
            client.requests.append(kwargs)
            client.jobs[identity] = 0
            raise TimeoutError("accepted but response lost")

    monkeypatch.setitem(sys.modules, "ray.job_submission", SimpleNamespace(JobSubmissionClient=SDK))
    rows = attempts(1)
    rows[0]["scenario"] = ScenarioSpec(
        name="acceptance",
        execution=RayJobsSDKExecution(entrypoint_resources={"pool_gpu_workers": 1}),
    )
    result = functions["run_campaign"](rows, client, output)
    assert len(client.requests) == 1
    assert result["results"][0]["job_id"] == client.requests[0]["submission_id"]
    assert result["results"][0]["status"] == "passed"
    assert not client.stopped


@pytest.mark.parametrize("outcome", ["SUCCEEDED", "succeeded"])
@pytest.mark.parametrize("pending_polls", [0, 1])
def test_lost_submission_response_reconciles_same_owned_job_without_retry(
    driver, monkeypatch, tmp_path, outcome, pending_polls
):
    functions, _ = driver
    error = TimeoutError("response lost with secret credentials")
    client = Client(outcome=outcome, pending_polls=pending_polls, submission_error=error)
    monkeypatch.setattr("nexuml.execution.run", client.submit)
    output = tmp_path / "out"
    result = functions["run_campaign"](attempts(1), client, output, poll_seconds=1)
    assert len(client.requests) == 1
    row = result["results"][0]
    assert row["job_id"] == client.requests[0]["submission_id"]
    assert row["status"] == "passed"
    assert row["submission_error_type"] == "TimeoutError"
    assert row["reconciliation"] == ("PENDING" if pending_polls else "SUCCEEDED")
    assert not client.stopped
    assert "secret credentials" not in (output / "summary.json").read_text()


@pytest.mark.parametrize("observed", [TimeoutError("unreachable"), RuntimeError("HTTP 404"), None])
def test_uncertain_submission_preserves_primary_and_owned_id_without_reusing_slot(
    driver, monkeypatch, tmp_path, observed
):
    functions, _ = driver
    error = TimeoutError("submission response lost")
    client = Client(submission_error=error)
    client.jobs["unrelated"] = 0

    def status(identity):
        assert identity != "unrelated"
        if isinstance(observed, Exception):
            raise observed
        return observed

    monkeypatch.setattr(client, "get_job_status", status)
    monkeypatch.setattr("nexuml.execution.run", client.submit)
    output = tmp_path / "out"
    with pytest.raises(TimeoutError) as caught:
        functions["run_campaign"](attempts(2), client, output)
    assert caught.value is error
    assert len(client.requests) == 1
    identity = client.requests[0]["submission_id"]
    assert client.stopped == [identity]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["active"] == [identity]
    assert summary["remaining"] == 1
    assert summary["results"][0]["job_id"] == identity
    assert summary["active_attempts"][0]["cleanup"]["status"] == "unknown"
    assert (output / f"{identity}.log").exists()


def test_foreign_returned_id_is_never_adopted_or_cancelled(driver, monkeypatch, tmp_path):
    functions, _ = driver
    client = Client()
    client.jobs["foreign"] = 0

    def submit(scenario, **kwargs):
        client.submit(scenario, **kwargs)
        return "foreign"

    monkeypatch.setattr("nexuml.execution.run", submit)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="different ID"):
        functions["run_campaign"](attempts(2), client, output)
    identity = client.requests[0]["submission_id"]
    assert client.stopped == [identity]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["active"] == [identity]
    assert summary["active_attempts"][0]["cleanup"] == {
        "status": "confirmed_terminal",
        "native_status": "STOPPED",
    }
    assert summary["remaining"] == 1


def test_duplicate_uuid_never_overwrites_an_owned_slot(driver, monkeypatch, tmp_path):
    functions, _ = driver
    client = Client()
    identity = "123456781234423482341234567890ab"
    monkeypatch.setitem(
        functions["run_campaign"].__globals__, "uuid4", lambda: SimpleNamespace(hex=identity)
    )
    monkeypatch.setattr("nexuml.execution.run", client.submit)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="must not reuse"):
        functions["run_campaign"](attempts(2), client, output, concurrency=2)
    assert len(client.requests) == 1
    assert client.stopped == [identity]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["active_attempts"][0]["attempt_root"] == attempts(1)[0]["attempt_root"]


def test_cleanup_and_summary_failures_do_not_replace_submission_exception(
    driver, monkeypatch, tmp_path
):
    functions, _ = driver
    error = TimeoutError("original submission exception")
    client = Client(submission_error=error)
    original_write = Path.write_text

    def write(path, *args, **kwargs):
        if client.requests and path.name == "summary.json.tmp":
            raise OSError("secret credential")
        return original_write(path, *args, **kwargs)

    def unavailable(_identity):
        raise RuntimeError("secret credential")

    monkeypatch.setattr(client, "get_job_status", unavailable)
    monkeypatch.setattr(client, "stop_job", unavailable)
    monkeypatch.setattr(client, "get_job_logs", unavailable)
    monkeypatch.setattr(Path, "write_text", write)
    monkeypatch.setattr("nexuml.execution.run", client.submit)
    with pytest.raises(TimeoutError) as caught:
        functions["run_campaign"](attempts(1), client, tmp_path / "out")
    assert caught.value is error
    assert caught.value.__notes__ == ["Campaign receipt also failed: OSError"]
