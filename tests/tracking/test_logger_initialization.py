from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from nexuml.core.types import DVCLiveSpec, LoggingSpec, MLflowSpec, TensorBoardSpec
from nexuml.tracking import logger as tracking


@pytest.fixture
def backends(monkeypatch, tmp_path):
    import lightning.pytorch.loggers as native

    state = SimpleNamespace(
        events=[], failures={}, finalization_error=None, prepare=tracking._prepare_mlflow_run
    )

    class Logger:
        def __init__(self, backend, **kwargs):
            self.name = backend
            self.log_dir = str(tmp_path / backend)
            self._initialized = False
            state.events.append((backend, "construct"))
            if (error := state.failures.get((backend, "construct"))) is not None:
                raise error

        @property
        def experiment(self):
            state.events.append((self.name, "experiment"))
            if (error := state.failures.get((self.name, "experiment"))) is not None:
                raise error
            self._initialized = True
            return self

        def finalize(self, status):
            state.events.append((self.name, status))
            if state.finalization_error is not None:
                raise state.finalization_error

    monkeypatch.setattr(native, "TensorBoardLogger", lambda **kw: Logger("TensorBoard", **kw))
    monkeypatch.setattr(native, "MLFlowLogger", lambda **kw: Logger("MLflow", **kw))
    monkeypatch.setitem(
        sys.modules,
        "dvclive.lightning",
        SimpleNamespace(DVCLiveLogger=lambda **kw: Logger("DVCLive", **kw)),
    )
    client = SimpleNamespace(set_terminated=lambda run, status: state.events.append((run, status)))
    state.client = client
    monkeypatch.setitem(sys.modules, "mlflow", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "mlflow.tracking", SimpleNamespace(MlflowClient=lambda **kw: client)
    )
    monkeypatch.setattr(tracking, "_configure_mlflow_tracking_uri", lambda uri: uri)
    monkeypatch.setattr(tracking, "_resolve_mlflow_artifact_location", lambda *args: None)
    monkeypatch.setattr(tracking, "_augment_mlflow_tags", lambda tags: tags)
    monkeypatch.setattr(
        tracking, "_prepare_mlflow_run", lambda **kw: (kw["run_id"] or "owned-child", "run")
    )
    monkeypatch.setenv("NEXUML_TEMP_ARTIFACT_DIR", str(tmp_path))
    return state


@pytest.mark.parametrize("selection", ["TensorBoard", "MLflow", "both", "DVCLive", "disabled"])
def test_only_selected_backends_are_initialized(backends, selection, tmp_path):
    spec = LoggingSpec(
        tensorboard=TensorBoardSpec() if selection in {"TensorBoard", "both"} else None,
        mlflow=MLflowSpec() if selection in {"MLflow", "both"} else None,
        dvclive=DVCLiveSpec() if selection == "DVCLive" else None,
        temp_artifact_dir=str(tmp_path),
    )
    loggers = tracking.create_loggers(spec)
    expected = ["TensorBoard", "MLflow"] if selection == "both" else [selection]
    if selection == "disabled":
        expected = []
    assert [native.name for native in loggers] == expected
    assert all(native._initialized for native in loggers)
    assert not any(stage in {"failed", "FAILED"} for _, stage in backends.events)
    backends.events.clear()
    assert tracking.create_loggers(None) == [] and backends.events == []


@pytest.mark.parametrize("backend", ["TensorBoard", "MLflow", "DVCLive"])
@pytest.mark.parametrize("stage", ["construct", "experiment"])
@pytest.mark.parametrize("error_type", [ImportError, OSError])
def test_requested_backend_failure_is_not_silently_skipped(backends, backend, stage, error_type):
    original = error_type("unavailable destination")
    backends.failures[(backend, stage)] = original
    spec = LoggingSpec.model_validate({backend.lower(): {}})
    with pytest.raises(RuntimeError, match=f"{backend} logger initialization failed") as caught:
        tracking.create_loggers(spec)
    assert caught.value.__cause__ is original
    if backend == "MLflow":
        assert ("owned-child", "FAILED") in backends.events
        assert any("owned-child" in note for note in original.__notes__)


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_partial_initialization_cleans_all_loggers_without_masking_error(backends, error_type):
    original = error_type("MLflow unavailable")
    backends.failures[("MLflow", "construct")] = original
    backends.finalization_error = RuntimeError("cleanup also failed")
    spec = LoggingSpec(tensorboard=TensorBoardSpec(), dvclive=DVCLiveSpec(), mlflow=MLflowSpec())
    with pytest.raises(error_type, match="MLflow unavailable") as caught:
        tracking.create_loggers(spec)
    cause = caught.value.__cause__ if error_type is RuntimeError else caught.value
    assert cause is original
    assert ("TensorBoard", "failed") in backends.events
    assert ("DVCLive", "failed") in backends.events
    assert any("cleanup also failed" in note for note in original.__notes__)


def test_failed_initialization_does_not_terminate_a_supplied_mlflow_run(backends):
    backends.failures[("MLflow", "construct")] = RuntimeError("bad constructor")
    with pytest.raises(RuntimeError, match="bad constructor"):
        tracking.create_loggers(LoggingSpec(mlflow=MLflowSpec(run_id="existing")))
    assert not any(stage == "FAILED" for _, stage in backends.events)


@pytest.mark.parametrize("reinforcement", [False, True])
def test_session_logging_failure_propagates_without_allocating_runtime(backends, reinforcement):
    from nexuml.core.types import ScenarioSpec
    from nexuml.training.lightning import NexuSession
    from nexuml_library.scenarios.reinforcement.gymnasium import cartpole_ppo

    scenario = cartpole_ppo() if reinforcement else ScenarioSpec(name="supervised")
    scenario.logging = LoggingSpec(mlflow=MLflowSpec())
    backends.failures[("MLflow", "construct")] = RuntimeError("tracking rejected")
    session = NexuSession(scenario)
    with pytest.raises(
        RuntimeError, match="MLflow logger initialization failed: tracking rejected"
    ):
        _ = session.trainer_loggers
    assert session._runtime is None and session._trainer is None


def test_requested_system_metrics_failure_is_visible(backends, monkeypatch):
    def fail():
        raise RuntimeError("system metrics unavailable")

    monkeypatch.setattr(sys.modules["mlflow"], "enable_system_metrics_logging", fail, raising=False)
    with pytest.raises(RuntimeError, match="MLflow.*system metrics unavailable"):
        tracking.create_loggers(LoggingSpec(mlflow=MLflowSpec(), log_system_metrics=True))
    assert backends.events == []


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_auto_parent_preparation_failure_retains_identity_and_cleans_owned_run(
    backends, monkeypatch, error_type, cleanup_fails
):
    original = error_type("parent finalization rejected")
    monkeypatch.setattr(tracking, "_prepare_mlflow_run", backends.prepare)
    monkeypatch.setattr(sys.modules["mlflow"], "active_run", lambda: None, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "mlflow.utils.mlflow_tags",
        SimpleNamespace(MLFLOW_PARENT_RUN_ID="parent", MLFLOW_RUN_NAME="run_name"),
    )
    backends.client.get_experiment_by_name = lambda name: SimpleNamespace(
        lifecycle_stage="active", experiment_id="experiment"
    )
    backends.client.search_runs = lambda **kw: []
    backends.client.create_run = lambda **kw: SimpleNamespace(
        info=SimpleNamespace(run_id="owned-parent")
    )

    def terminate(run_id, status):
        backends.events.append((run_id, status))
        if status == "FINISHED":
            raise original
        if cleanup_fails:
            raise RuntimeError("parent cleanup rejected")

    backends.client.set_terminated = terminate
    with pytest.raises(error_type, match="parent finalization rejected") as caught:
        tracking.create_loggers(LoggingSpec(mlflow=MLflowSpec()))
    assert backends.events == [("owned-parent", "FINISHED"), ("owned-parent", "FAILED")]
    cause = caught.value.__cause__ if error_type is RuntimeError else caught.value
    assert cause is original
    assert any("owned-parent" in note for note in original.__notes__)
    if cleanup_fails:
        assert any("parent cleanup rejected" in note for note in original.__notes__)


def test_preparation_failure_does_not_finalize_a_preexisting_auto_parent(backends, monkeypatch):
    monkeypatch.setattr(tracking, "_prepare_mlflow_run", backends.prepare)
    monkeypatch.setattr(sys.modules["mlflow"], "active_run", lambda: None, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "mlflow.utils.mlflow_tags",
        SimpleNamespace(MLFLOW_PARENT_RUN_ID="parent", MLFLOW_RUN_NAME="run_name"),
    )
    backends.client.get_experiment_by_name = lambda name: SimpleNamespace(
        lifecycle_stage="active", experiment_id="experiment"
    )
    existing = SimpleNamespace(
        info=SimpleNamespace(run_id="existing-parent"),
        data=SimpleNamespace(tags={"nexuml.auto_parent": "true", "run_name": "run"}),
    )
    searches = []

    def search(**kwargs):
        if searches:
            raise RuntimeError("child version lookup failed")
        searches.append(True)
        return [existing]

    backends.client.search_runs = search
    with pytest.raises(RuntimeError, match="child version lookup failed"):
        tracking.create_loggers(LoggingSpec(mlflow=MLflowSpec()))
    assert backends.events == []


@pytest.mark.requires_optional("tensorboard")
def test_real_tensorboard_writer_closes_after_later_backend_failure(monkeypatch, tmp_path):
    original = tracking._create_tensorboard_logger
    created = []

    def tensorboard(**kwargs):
        instance = original(**kwargs)
        created.append(instance)
        return instance

    def fail(**kwargs):
        raise RuntimeError("MLflow setup rejected")

    monkeypatch.setattr(tracking, "_create_tensorboard_logger", tensorboard)
    monkeypatch.setattr(tracking, "_create_mlflow_logger", fail)
    with pytest.raises(RuntimeError, match="MLflow setup rejected"):
        tracking.create_loggers(
            LoggingSpec(tensorboard=TensorBoardSpec(log_dir=str(tmp_path)), mlflow=MLflowSpec())
        )
    assert len(created) == 1 and created[0].experiment.file_writer is None


def test_partial_artifact_delivery_retains_successful_output(tmp_path, caplog):
    first = type("TensorBoardLogger", (), {"log_dir": str(tmp_path / "tb")})()

    def fail(*args):
        raise OSError("store unavailable")

    second = type(
        "MLFlowLogger",
        (),
        {"run_id": "failed-run", "experiment": SimpleNamespace(log_artifact=fail)},
    )()
    source = tmp_path / "receipt.json"
    source.write_text("{}")
    with pytest.raises(OSError, match="store unavailable") as caught:
        tracking.log_artifact([first, second], source)
    assert (tmp_path / "tb/artifacts/receipt.json").read_text() == "{}"
    assert source.read_text() == "{}"
    note = " ".join(caught.value.__notes__)
    assert "MLFlowLogger" in note and "failed-run" in note and "TensorBoardLogger" in note
    assert "Artifact delivery failed" in caplog.text
