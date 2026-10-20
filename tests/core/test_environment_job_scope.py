"""Generic CLI lifetime ownership without a simulator, server, or live environment."""

import asyncio
import builtins
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
from typer.testing import CliRunner

from nexuml.cli.main import app
from nexuml.core.components import ActionAdapterDefinition, EnvironmentDefinition
from nexuml.core.config import ResolvedConfig
from nexuml.core.registry import ComponentRegistry
from nexuml.core.serialization import lower_model
from nexuml.core.types import (
    ExportSpec,
    InteractionContract,
    InteractionSpec,
    PolicyEvaluationSpec,
    PolicySpec,
    ScenarioSpec,
    TensorFieldContract,
)
from nexuml.evaluation.policy import evaluate_policy
from nexuml.training.lightning import NexuSession

runner = CliRunner()


class ScopeEnvironment(EnvironmentDefinition):
    component_name = "test.job_scope.environment"
    label: str = "training"

    def describe(self):
        return InteractionContract(
            observations={"observation": TensorFieldContract(shape=(1,), dtype="float32")},
            actions={"action": TensorFieldContract(shape=(1,), dtype="float32")},
            action_space="continuous",
            control_period_s=0.1,
        )

    def build(self, context):
        pytest.fail("CLI scope ownership must not pre-build an environment")


class ScopeAdapter(ActionAdapterDefinition):
    component_name = "test.job_scope.adapter"

    def build(self, contract):
        pytest.fail("CLI lifetime tests must not build policy adapters")


class ReplayPolicy:
    pipeline = object()
    action_adapter = object()
    contract = object()

    def __call__(self, observation):
        pytest.fail("CLI lifetime tests must not execute inference")

    def parameters(self):
        return iter(())


@pytest.fixture(scope="module")
def scope_registry():
    registry = ComponentRegistry()
    registry.scan(["nexuml.data.loaders.definitions"])
    for definition in (ScopeEnvironment, ScopeAdapter):
        registry.register(definition.component_name, definition)
    return registry


@pytest.fixture
def lifetime(monkeypatch, scope_registry):
    monkeypatch.setattr("nexuml.core.serialization.get_component_registry", lambda: scope_registry)
    state = SimpleNamespace(events=[], active=None, error=None)

    @contextmanager
    def scope(definition):
        assert state.active is None
        state.active = definition.label
        state.events.append(f"enter:{definition.label}")
        try:
            yield
        except BaseException as exc:
            state.error = exc
            raise
        finally:
            state.events.append(f"exit:{definition.label}")
            state.active = None

    monkeypatch.setattr(ScopeEnvironment, "job_scope", scope)
    return state


def scenario(label="training"):
    return ScenarioSpec(
        name="scope-job",
        interaction=InteractionSpec(environment=ScopeEnvironment(label=label)),
        policy=PolicySpec(action_adapter=ScopeAdapter()),
        policy_evaluation=PolicyEvaluationSpec(episodes=1),
        logging=None,
    )


def train_result():
    return SimpleNamespace(
        policy=object(),
        pipeline=object(),
        lightning_module=object(),
        trainer=object(),
        interaction_results={},
        test_results=[],
    )


class FinalizationLogger:
    def __init__(self, error=None):
        self.statuses = []
        self.error = error

    def finalize(self, status):
        self.statuses.append(status)
        if self.error is not None:
            raise self.error


@pytest.fixture
def unbuilt_session(monkeypatch):
    session = NexuSession(ScenarioSpec(name="unbuilt"))

    def no_lazy_resources(*args, **kwargs):
        pytest.fail("finalization must not create loggers, runtime, or trainer")

    for name in ("_build_logging_components", "build_runtime", "build_trainer"):
        monkeypatch.setattr(session, name, no_lazy_resources)
    monkeypatch.setattr(NexuSession, "trainer_loggers", property(no_lazy_resources))
    return session


@pytest.mark.parametrize("initialized", [None, False, []])
def test_finalize_loggers_does_not_initialize_resources(initialized, unbuilt_session):
    unbuilt_session._trainer_loggers = initialized
    unbuilt_session.finalize_loggers("failed")
    assert unbuilt_session._trainer_loggers is initialized
    assert unbuilt_session._runtime is None and unbuilt_session._trainer is None


def test_finalize_loggers_attempts_every_logger_and_retains_cleanup_errors(unbuilt_session):
    first = RuntimeError("first cleanup failure")
    second = SystemExit(93)
    loggers = [FinalizationLogger(first), FinalizationLogger(second), FinalizationLogger()]
    unbuilt_session._trainer_loggers = loggers
    with pytest.raises(RuntimeError) as caught:
        unbuilt_session.finalize_loggers("failed")
    assert caught.value is first
    assert any(repr(second) in note for note in first.__notes__)
    assert all(logger.statuses == ["failed"] for logger in loggers)


@pytest.mark.parametrize(
    "error_type", [RuntimeError, SystemExit, KeyboardInterrupt, asyncio.CancelledError]
)
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_real_session_run_preserves_failure_after_initialized_logger_cleanup(
    error_type, cleanup_fails, unbuilt_session, monkeypatch
):
    error = error_type(143 if error_type is SystemExit else "primary failure")
    cleanup_errors = [RuntimeError("first cleanup failure"), KeyboardInterrupt("second cleanup")]
    loggers = [FinalizationLogger(item if cleanup_fails else None) for item in cleanup_errors]
    loggers.append(FinalizationLogger())
    unbuilt_session._trainer_loggers = loggers

    def fail_fit():
        raise error

    monkeypatch.setattr(unbuilt_session, "fit", fail_fit)
    with pytest.raises(error_type) as caught:
        unbuilt_session.run()
    assert caught.value is error
    if error_type is SystemExit:
        assert caught.value.code == 143
    assert all(logger.statuses == ["failed"] for logger in loggers)
    assert unbuilt_session._runtime is None and unbuilt_session._trainer is None
    if cleanup_fails:
        assert all(any(repr(item) in note for note in error.__notes__) for item in cleanup_errors)
    else:
        assert not getattr(error, "__notes__", ())


def test_default_job_scope_is_synchronous_noop_without_optional_imports(monkeypatch):
    original_import = builtins.__import__

    def no_native(name, *args, **kwargs):
        assert name.split(".")[0] not in {
            "isaaclab",
            "isaacsim",
            "omni",
            "pxr",
            "viser",
            "gymnasium",
            "torchrl",
        }
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_native)
    scope = ScopeEnvironment().job_scope()
    assert isinstance(scope, nullcontext)
    with scope as value:
        assert value is None
    error = RuntimeError("unchanged")
    with pytest.raises(RuntimeError) as caught, ScopeEnvironment().job_scope():
        raise error
    assert caught.value is error


@pytest.mark.parametrize("original_file", [False, True])
def test_train_scopes_effective_resume_and_its_exports(
    original_file, lifetime, monkeypatch, tmp_path
):
    initialized_logger = FinalizationLogger()
    effective = scenario("resumed")
    effective.exports = [ExportSpec(output=str(tmp_path / "effective-export"))]
    checkpoint = tmp_path / "resume.ckpt"
    torch.save({"hyper_parameters": {"scenario": lower_model(effective)}}, checkpoint)
    command = ["train", "--trainer-checkpoint", str(checkpoint)]
    if original_file:
        ignored = scenario("ignored")
        ignored.exports = [ExportSpec(output=str(tmp_path / "ignored-export"))]
        loaded = SimpleNamespace(scenario=ignored)
        monkeypatch.setattr("nexuml.core.scenario_loader.load_scenario_file", lambda _: loaded)
        command += ["--scenario-file", "ignored.py", "--artifact-dir", str(tmp_path)]

    def run(session):
        assert lifetime.active == session.scenario.interaction.environment.label == "resumed"
        assert session._runtime is None and session._trainer is None
        session._trainer_loggers = [initialized_logger]
        lifetime.events.extend(["run", "checkpoint"])
        return train_result()

    def snapshot(*args, **kwargs):
        assert lifetime.active == "resumed"
        lifetime.events.append("provenance")

    def export(policy, path, **kwargs):
        assert lifetime.active == "resumed"
        assert path == tmp_path / "effective-export"
        lifetime.events.append("export")

    monkeypatch.setattr(NexuSession, "run", run)
    monkeypatch.setattr("nexuml.core.provenance.snapshot_scenario_file_run", snapshot)
    monkeypatch.setattr("nexuml.core.export.export_package", export)
    result = runner.invoke(app, command)
    assert result.exit_code == 0, (result.output, result.exception)
    assert lifetime.events == [
        "enter:resumed",
        "run",
        "checkpoint",
        *(["provenance"] if original_file else []),
        "export",
        "exit:resumed",
    ]
    assert initialized_logger.statuses == []


@pytest.mark.parametrize("stage", ["run", "provenance", "export"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_train_failure_unwinds_only_after_session_cleanup_and_outputs(
    stage, cleanup_fails, lifetime, monkeypatch, tmp_path
):
    spec = scenario()
    spec.exports = [ExportSpec(output=str(tmp_path / "export"))]
    monkeypatch.setattr(
        "nexuml.core.scenario_loader.load_scenario_file",
        lambda _: SimpleNamespace(scenario=spec),
    )
    error = RuntimeError(f"{stage} failed")

    class Logger:
        def __init__(self, name):
            self.name = name

        def finalize(self, status):
            assert lifetime.active == "training"
            lifetime.events.append(f"{self.name}:{status}")
            if cleanup_fails:
                raise RuntimeError(f"{self.name} cleanup failed")

    def operation(name):
        assert lifetime.active == "training"
        lifetime.events.append(name)
        if stage == name:
            raise error

    def run(session):
        session._trainer_loggers = [Logger("first"), Logger("second")]
        operation("run")
        return train_result()

    # Run failures exercise the real session guard; output failures exercise the CLI guard.
    monkeypatch.setattr(NexuSession, "fit" if stage == "run" else "run", run)
    monkeypatch.setattr(
        "nexuml.core.provenance.snapshot_scenario_file_run", lambda *a, **k: operation("provenance")
    )
    monkeypatch.setattr("nexuml.core.export.export_package", lambda *a, **k: operation("export"))
    result = runner.invoke(
        app, ["train", "--scenario-file", "job.py", "--artifact-dir", str(tmp_path)]
    )
    assert result.exit_code == 1 and result.exception is error
    assert lifetime.error is error
    assert lifetime.events == [
        "enter:training",
        "run",
        *(["provenance"] if stage != "run" else []),
        *(["export"] if stage == "export" else []),
        *(["first:failed", "second:failed"] * (2 if stage == "run" else 1)),
        "exit:training",
    ]
    if cleanup_fails:
        assert all(
            any(f"{name} cleanup failed" in note for note in error.__notes__)
            for name in ("first", "second")
        )


@pytest.mark.parametrize("error", [SystemExit(143), KeyboardInterrupt(), asyncio.CancelledError()])
def test_train_cancellation_preserves_scope_exception_after_session_cleanup(
    error, lifetime, monkeypatch
):
    monkeypatch.setattr("nexuml.cli.main._load_scenario", lambda *args: scenario())

    class Logger:
        def finalize(self, status):
            assert lifetime.active == "training"
            lifetime.events.append(f"session-loggers:{status}")

    def fit(session):
        session._trainer_loggers = [Logger()]
        raise error

    monkeypatch.setattr(NexuSession, "fit", fit)
    if isinstance(error, asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            runner.invoke(app, ["train", "scope-job"])
    else:
        result = runner.invoke(app, ["train", "scope-job"])
        assert result.exit_code == (143 if isinstance(error, SystemExit) else 130)
    assert lifetime.error is error
    assert lifetime.events == [
        "enter:training",
        "session-loggers:failed",
        "session-loggers:failed",
        "exit:training",
    ]


def test_train_without_interaction_needs_no_environment_scope(monkeypatch):
    monkeypatch.setattr("nexuml.cli.main._load_scenario", lambda *args: ScenarioSpec(name="plain"))
    monkeypatch.setattr(NexuSession, "run", lambda self: train_result())
    result = runner.invoke(app, ["train", "plain"])
    assert result.exit_code == 0, (result.output, result.exception)


@pytest.fixture
def replay(lifetime, monkeypatch):
    state = SimpleNamespace(scenario=scenario(), error=None, stage="evaluate", finalize_error=None)
    contract = state.scenario.interaction.environment.describe().model_dump(mode="json")
    monkeypatch.setattr(
        "nexuml.core.export.load_inference_package",
        lambda _: (
            ReplayPolicy(),
            ResolvedConfig.from_scenario(state.scenario),
            {"artifact_kind": "policy", "interaction_contract": contract},
        ),
    )

    def operation(name):
        assert lifetime.active is not None
        lifetime.events.append(name)
        if state.stage == name and state.error is not None:
            raise state.error

    class Logger:
        def log_metrics(self, metrics, step):
            operation("metrics")

        def finalize(self, status):
            operation(f"finalize:{status}")
            if state.finalize_error is not None:
                raise state.finalize_error

    def create_loggers(*args, **kwargs):
        operation("loggers")
        return [Logger(), Logger()]

    def evaluate(policy, restored, **kwargs):
        selected = restored.interaction.evaluation_environment or restored.interaction.environment
        assert lifetime.active == selected.label
        operation("evaluate")
        return {}

    monkeypatch.setattr("nexuml.tracking.logger.create_loggers", create_loggers)
    monkeypatch.setattr(
        "nexuml.tracking.logger.log_text_artifact", lambda loggers, text, name: operation(name)
    )
    monkeypatch.setattr("nexuml.evaluation.policy.evaluate_policy", evaluate)
    return state


@pytest.mark.parametrize("selection", ["training", "explicit", "overridden"])
def test_evaluate_scopes_resolved_selected_definition_through_logger_finalization(
    selection, lifetime, replay
):
    command = ["evaluate", "package"]
    if selection != "training":
        replay.scenario.interaction.evaluation_environment = ScopeEnvironment(label="explicit")
    if selection == "overridden":
        command += ["-O", "interaction.evaluation_environment.label=overridden"]
    result = runner.invoke(app, command)
    assert result.exit_code == 0, (result.output, result.exception)
    assert lifetime.events == [
        f"enter:{selection}",
        "loggers",
        "source_package.json",
        "config.yaml",
        "evaluate",
        "metrics",
        "metrics",
        "finalize:success",
        "finalize:success",
        f"exit:{selection}",
    ]


@pytest.mark.parametrize("stage", ["source_package.json", "evaluate", "metrics"])
def test_evaluate_failure_finalizes_all_loggers_before_scope_exit(stage, lifetime, replay):
    replay.stage = stage
    replay.error = RuntimeError("primary failure")
    replay.finalize_error = RuntimeError("logger cleanup failed")
    result = runner.invoke(app, ["evaluate", "package"])
    assert result.exit_code == 1
    assert lifetime.error.__cause__ is replay.error
    assert lifetime.events[-3:] == ["finalize:failed", "finalize:failed", "exit:training"]
    assert "Evaluation complete" not in result.output


@pytest.mark.parametrize("error", [SystemExit(143), KeyboardInterrupt(), asyncio.CancelledError()])
def test_evaluate_cancellation_finalizes_before_scope_exit(error, lifetime, replay):
    replay.error = error
    replay.finalize_error = RuntimeError("logger cleanup failed")
    if isinstance(error, asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            runner.invoke(app, ["evaluate", "package"])
        assert lifetime.error is error
    else:
        result = runner.invoke(app, ["evaluate", "package"])
        assert result.exit_code == (143 if isinstance(error, SystemExit) else 1)
        if isinstance(error, SystemExit):
            assert lifetime.error is error
        else:
            assert lifetime.error.__cause__ is error
    assert lifetime.events[-3:] == ["finalize:failed", "finalize:failed", "exit:training"]


def test_evaluate_logger_failure_still_exits_scope_and_fails(lifetime, replay):
    replay.finalize_error = RuntimeError("logger cleanup failed")
    result = runner.invoke(app, ["evaluate", "package"])
    assert result.exit_code == 1
    assert lifetime.error.__cause__ is replay.finalize_error
    assert lifetime.events[-3:] == ["finalize:success", "finalize:success", "exit:training"]
    assert "Evaluation complete" not in result.output


def test_disabled_evaluation_allocates_no_environment(lifetime, replay, monkeypatch):
    replay.scenario.policy_evaluation.episodes = 0
    monkeypatch.setattr(
        ScopeEnvironment, "describe", lambda self: pytest.fail("disabled evaluation described env")
    )
    # Loading uses the already-recorded contract; disabled replay must not inspect/build the env.
    monkeypatch.setattr("nexuml.evaluation.policy.evaluate_policy", evaluate_policy)
    result = runner.invoke(app, ["evaluate", "package"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert "evaluate" not in lifetime.events
    assert lifetime.events[-3:] == ["finalize:success", "finalize:success", "exit:training"]
