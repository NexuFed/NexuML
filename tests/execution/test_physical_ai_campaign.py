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
        self.info_requests = []
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

    def get_job_info(self, identity):
        self.info_requests.append(identity)
        raise AssertionError("Unexpected diagnostic lookup")


@pytest.mark.parametrize(
    "error_type,summary",
    [
        ("RUNTIME_ENV_SETUP_FAILURE", "Ray runtime environment setup failed."),
        (
            "JOB_SUPERVISOR_ACTOR_START_TIMEOUT",
            "Ray job start timed out before the workload started.",
        ),
        ("JOB_SUPERVISOR_ACTOR_START_FAILURE", "Ray job supervisor failed to start."),
        ("JOB_SUPERVISOR_ACTOR_UNSCHEDULABLE", "Ray job supervisor could not be scheduled."),
        ("JOB_SUPERVISOR_ACTOR_UNKNOWN_FAILURE", "Ray job supervisor failed unexpectedly."),
        ("JOB_SUPERVISOR_ACTOR_DIED", "Ray job supervisor died."),
        ("JOB_ENTRYPOINT_COMMAND_START_ERROR", "Ray job entrypoint could not be started."),
        ("JOB_ENTRYPOINT_COMMAND_ERROR", "Ray job entrypoint exited with an error."),
    ],
)
def test_failed_without_worker_receipt_reports_only_safe_native_cause(
    driver, monkeypatch, tmp_path, capsys, error_type, summary
):
    functions, _ = driver
    client = Client(outcome="FAILED", pending_polls=0)
    logs = "original private worker log: private-log-token\n"

    def info(identity):
        client.info_requests.append(identity)
        assert 0 < signal.getitimer(signal.ITIMER_REAL)[0] <= 30
        return SimpleNamespace(error_type=error_type, message="secret-SDK-message")

    monkeypatch.setattr(client, "get_job_info", info)
    monkeypatch.setattr(client, "get_job_logs", lambda _: logs)
    monkeypatch.setattr("nexuml.execution.run", client.submit)
    output = tmp_path / "out"
    result = functions["run_campaign"](attempts(1), client, output)
    row = result["results"][0]
    cause = {"type": error_type, "summary": summary}
    assert row["native_failure"] == cause
    assert row["status"] == "failed" and row["native_status"] == "FAILED"
    assert row["receipt"] is None
    assert client.info_requests == [row["job_id"]]
    assert len(client.requests) == 1 and not client.stopped
    assert (output / f"{row['job_id']}.log").read_text() == logs
    assert json.loads((output / f"{row['job_id']}.json").read_text()) == row
    assert json.loads((output / "summary.json").read_text()) == result
    text = capsys.readouterr().out
    terminal = next(
        json.loads(line.partition("=")[2])
        for line in text.splitlines()
        if line.startswith("NEXUML_CASE_TERMINAL=")
    )
    assert terminal["native_failure"] == cause
    assert f"{row['job_id']}: failed (FAILED): {error_type}: {summary}" in text
    for secret in ("secret-SDK-message", "private-log-token"):
        assert secret not in text + json.dumps(result)


@pytest.mark.parametrize(
    "details",
    [
        None,
        {},
        SimpleNamespace(),
        SimpleNamespace(error_type=None),
        SimpleNamespace(error_type="secret-unknown-code"),
        SimpleNamespace(error_type=["secret-malformed-code"]),
        SimpleNamespace(error_type={"secret-malformed-code": True}),
        SimpleNamespace(error_type=42),
        SimpleNamespace(error_type="UNAVAILABLE"),
    ],
)
def test_unknown_or_malformed_native_code_is_not_echoed(
    driver, monkeypatch, tmp_path, capsys, details
):
    functions, _ = driver
    client = Client(outcome="FAILED", pending_polls=0)
    monkeypatch.setattr(client, "get_job_logs", lambda _: "no worker receipt\n")
    monkeypatch.setattr(client, "get_job_info", lambda _: details)
    monkeypatch.setattr("nexuml.execution.run", client.submit)
    output = tmp_path / "out"
    result = functions["run_campaign"](attempts(1), client, output)
    assert result["results"][0]["native_failure"] == {
        "type": "UNKNOWN",
        "summary": "Ray failure cause is unknown; no worker receipt was available.",
    }
    text = capsys.readouterr().out + "".join(path.read_text() for path in output.glob("*.json"))
    assert "secret-" not in text
    assert "before the workload started" not in text


def test_native_diagnostic_exception_cannot_replace_failed_status_or_leak(
    driver, monkeypatch, tmp_path, capsys
):
    functions, _ = driver
    client = Client(outcome="FAILED", pending_polls=0)

    def unavailable(identity):
        client.info_requests.append(identity)
        raise type("secret-exception-type", (RuntimeError,), {})("secret-exception-message")

    monkeypatch.setattr(client, "get_job_info", unavailable)
    monkeypatch.setattr(client, "get_job_logs", lambda _: "original private log\n")
    monkeypatch.setattr("nexuml.execution.run", client.submit)
    output = tmp_path / "out"
    result = functions["run_campaign"](attempts(1), client, output)
    row = result["results"][0]
    assert row["status"] == "failed" and row["native_status"] == "FAILED"
    assert row["native_failure"] == {
        "type": "UNAVAILABLE",
        "summary": "Ray failure details unavailable; no worker receipt was available.",
    }
    assert client.info_requests == [row["job_id"]]
    assert not client.stopped and len(client.requests) == 1
    assert (output / f"{row['job_id']}.log").read_text() == "original private log\n"
    text = capsys.readouterr().out + "".join(path.read_text() for path in output.glob("*.json"))
    assert "secret-" not in text
    assert "before the workload started" not in text


@pytest.mark.parametrize(
    "native_status,logs",
    [
        ("SUCCEEDED", 'NEXUML_CASE_RESULT={"status":"PASS","source_tree_sha256":"tree"}\n'),
        ("FAILED", 'NEXUML_CASE_RESULT={"status":"FAIL"}\n'),
        ("FAILED", 'NEXUML_CASE_RESULT={"status":"PASS","source_tree_sha256":"tree"}\n'),
        ("SUCCEEDED", "no worker receipt\n"),
        ("STOPPED", "no worker receipt\n"),
    ],
)
def test_other_terminal_cases_make_no_diagnostic_lookup(
    driver, monkeypatch, tmp_path, capsys, native_status, logs
):
    functions, _ = driver
    client = Client(outcome=native_status, pending_polls=0)
    monkeypatch.setattr(client, "get_job_logs", lambda _: logs)
    monkeypatch.setattr("nexuml.execution.run", client.submit)
    result = functions["run_campaign"](attempts(1), client, tmp_path / "out")
    assert client.info_requests == []
    assert "native_failure" not in result["results"][0]
    assert "native_failure" not in capsys.readouterr().out


@pytest.mark.parametrize("native_status", ["SUCCEEDED", "FAILED", "STOPPED"])
def test_console_links_case_attempt_candidate_and_terminal_files_without_inventing_phase(
    driver, monkeypatch, tmp_path, capsys, native_status
):
    functions, _ = driver
    client = Client(outcome=native_status)
    monkeypatch.setattr("nexuml.execution.run", client.submit)
    case = attempts(1)[0]
    case["scenario"].execution.runtime_env = {"env_vars": {"TOKEN": "never-print-secret"}}
    case["options"]["unrelated"] = "never-print-option"
    output = tmp_path / "result"
    summary = functions["run_campaign"]([case], client, output)
    text = capsys.readouterr().out
    submitted = [
        json.loads(line.partition("=")[2])
        for line in text.splitlines()
        if line.startswith("NEXUML_CASE_SUBMITTED=")
    ]
    terminal = [
        json.loads(line.partition("=")[2])
        for line in text.splitlines()
        if line.startswith("NEXUML_CASE_TERMINAL=")
    ]
    assert len(submitted) == len(terminal) == 1
    for record in (submitted[0], terminal[0]):
        assert record["job_id"] == summary["results"][0]["job_id"]
        assert record["case"] == case["case"]
        assert record["attempt_root"] == case["attempt_root"]
        assert record["source_tree_sha256"] == "tree"
    assert submitted[0]["execution_phase"] == "unknown until observed in worker logs"
    assert submitted[0]["requested_pool"] == "pool_gpu_workers"
    assert submitted[0]["requested_gpus"] == 1
    assert submitted[0]["wall_seconds"] == 10
    assert "native_status" not in submitted[0]
    assert terminal[0]["native_status"] == native_status
    assert terminal[0]["status"] == ("passed" if native_status == "SUCCEEDED" else "failed")
    assert json.loads(Path(terminal[0]["receipt_path"]).read_text()) == summary["results"][0]
    assert Path(terminal[0]["log_path"]).read_text() == client.get_job_logs(terminal[0]["job_id"])
    assert "never-print" not in text and "TOKEN" not in text


@pytest.mark.parametrize("logger_mode", [None, "separate", "both"])
def test_browser_forwards_only_qualified_producer_identity(
    driver, monkeypatch, tmp_path, logger_mode
):
    functions, _ = driver
    client = Client()
    monkeypatch.setattr("nexuml.execution.run", client.submit)
    case = attempts(1)[0]
    case["case"] = "browser"
    producer = {
        "source_attempt": "/approved/producer",
        "source_case_sha256": "a" * 64,
        "source_guard_sha256": "b" * 64,
    }
    if logger_mode is not None:
        producer["logger_mode"] = logger_mode
    case["options"].update(**producer, command="must never reach the worker")
    functions["run_campaign"]([case], client, tmp_path / "result")
    assert client.requests[0]["case_options"] == producer


@pytest.mark.parametrize("mode", ["separate", "both"])
def test_browser_mode_cli_is_explicit_and_dry_run_stays_inert(
    driver, monkeypatch, tmp_path, capsys, mode
):
    from nexuml.execution.definitions import RayJobsSDKExecution

    functions, _ = driver
    certificate = tmp_path / "ca.pem"
    certificate.write_text("inert CA fixture")
    output = tmp_path / "not-created"
    monkeypatch.setattr("nexuml.execution.load_ray_job_profile", lambda _: RayJobsSDKExecution())
    monkeypatch.setitem(functions["main"].__globals__, "source_identity", lambda _: "tree")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "physical_ai_acceptance.py",
            "--case",
            "browser",
            "--browser-mode",
            mode,
            "--ca",
            str(certificate),
            "--output",
            str(output),
        ],
    )
    functions["main"]()
    plan = json.loads(capsys.readouterr().out)
    assert plan["attempts"][0]["options"]["logger_mode"] == mode
    assert not output.exists()


def test_browser_mode_cli_cannot_select_other_cases(driver, monkeypatch, tmp_path, capsys):
    functions, _ = driver
    certificate = tmp_path / "ca.pem"
    certificate.write_text("inert CA fixture")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "physical_ai_acceptance.py",
            "--case",
            "preflight",
            "--browser-mode",
            "both",
            "--ca",
            str(certificate),
            "--output",
            str(tmp_path / "output"),
        ],
    )
    with pytest.raises(SystemExit) as result:
        functions["main"]()
    assert result.value.code == 2
    assert "requires a browser case" in capsys.readouterr().err


def test_logging_failure_forwards_only_authenticated_source_inputs(driver, monkeypatch, tmp_path):
    functions, _ = driver
    client = Client()
    monkeypatch.setattr("nexuml.execution.run", client.submit)
    case = attempts(1)[0]
    case["case"] = "logging-failure"
    producer = {
        "source_attempt": "/approved/producer",
        "source_case_sha256": "a" * 64,
        "source_guard_sha256": "b" * 64,
    }
    case["options"].update(**producer, command="never forward", logger_mode="both")
    functions["run_campaign"]([case], client, tmp_path / "result")
    assert client.requests[0]["case"] == "logging-failure"
    assert client.requests[0]["case_options"] == producer


def test_recipe_forwards_only_fixed_recording_selection(driver, monkeypatch, tmp_path):
    functions, _ = driver
    client = Client()
    monkeypatch.setattr("nexuml.execution.run", client.submit)
    case = attempts(1)[0]
    case["case"] = "recipe"
    case["options"].update(recipe="mujoco-reacher-ppo", recording="both", command="forbidden")
    functions["run_campaign"]([case], client, tmp_path / "result")
    assert client.requests[0]["case_options"] == {
        "recipe": "mujoco-reacher-ppo",
        "recording": "both",
    }


@pytest.mark.parametrize("owner", ["package", "external"])
@pytest.mark.parametrize("family", ["rgb", "depth", "segmentation", "mixed"])
def test_sensor_owner_cli_resolves_the_existing_recipe_without_launch(
    driver, monkeypatch, tmp_path, capsys, family, owner
):
    from nexuml.execution.definitions import RayJobsSDKExecution
    from nexuml.core.scenario_registry import get_scenario_registry

    functions, _ = driver
    name = f"isaac-{family}-cartpole-ppo"
    factory = get_scenario_registry().get(name)
    selected = []

    def build(**kwargs):
        selected.append(kwargs)
        return factory(**kwargs)

    monkeypatch.setattr(
        "nexuml.core.scenario_registry.get_scenario_registry",
        lambda: SimpleNamespace(get=lambda recipe: build if recipe == name else None),
    )
    certificate = tmp_path / "ca.pem"
    certificate.write_text("inert fixture")
    output = tmp_path / "not-created"
    monkeypatch.setattr("nexuml.execution.load_ray_job_profile", lambda _: RayJobsSDKExecution())
    monkeypatch.setitem(functions["main"].__globals__, "source_identity", lambda _: "tree")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "physical_ai_acceptance.py",
            "--case",
            "recipe",
            "--recipe",
            f"isaac-{family}-cartpole-ppo",
            "--processing-owner",
            owner,
            "--ca",
            str(certificate),
            "--output",
            str(output),
        ],
    )
    functions["main"]()
    plan = json.loads(capsys.readouterr().out)
    assert plan["attempts"][0]["options"]["processing_owner"] == owner
    assert selected == [{"processing_owner": owner}]
    assert not output.exists()
    assert "isaaclab" not in sys.modules and "isaacsim" not in sys.modules


@pytest.mark.parametrize(
    "case,recipe",
    [
        ("preflight", "isaac-rgb-cartpole-ppo"),
        ("recipe", "isaac-categorical-cartpole-ppo"),
        ("recipe", "cartpole-ppo"),
    ],
)
def test_sensor_owner_rejects_other_cases_before_profile_loading(
    driver, monkeypatch, tmp_path, capsys, case, recipe
):
    functions, _ = driver
    certificate = tmp_path / "ca.pem"
    certificate.write_text("inert fixture")
    monkeypatch.setattr(
        "nexuml.execution.load_ray_job_profile", lambda _: pytest.fail("loaded profile")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "physical_ai_acceptance.py",
            "--case",
            case,
            "--recipe",
            recipe,
            "--processing-owner",
            "external",
            "--ca",
            str(certificate),
            "--output",
            str(tmp_path / "not-created"),
        ],
    )
    with pytest.raises(SystemExit) as result:
        functions["main"]()
    assert result.value.code == 2
    assert "requires one fixed I2 camera recipe" in capsys.readouterr().err


@pytest.mark.parametrize("owner", ["package", "external"])
def test_sensor_owner_is_the_only_additional_recipe_option_forwarded(
    driver, monkeypatch, tmp_path, owner
):
    functions, _ = driver
    client = Client()
    monkeypatch.setattr("nexuml.execution.run", client.submit)
    case = attempts(1)[0]
    case["case"] = "recipe"
    case["options"].update(
        recipe="isaac-mixed-cartpole-ppo", processing_owner=owner, command="forbidden"
    )
    functions["run_campaign"]([case], client, tmp_path / "result")
    assert client.requests[0]["case_options"] == {
        "recipe": "isaac-mixed-cartpole-ppo",
        "processing_owner": owner,
    }


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
