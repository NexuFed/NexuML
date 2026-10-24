"""Hermetic export restoration and initialized-logger lifecycle regressions."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from nexuml.cli.main import _export_local
from nexuml.execution.definitions import LocalExecution
from nexuml.training.lightning import NexuSession


@pytest.fixture
def export_session(monkeypatch):
    loggers = [Mock(), Mock()]
    session = Mock(
        setup=Mock(),
        runtime=SimpleNamespace(policy=object()),
        pipeline=object(),
        lightning_module=object(),
        trainer=object(),
        _trainer_loggers=loggers,
    )
    session.finalize_loggers = Mock(
        side_effect=lambda status: NexuSession.finalize_loggers(session, status)
    )
    factory = Mock(return_value=session)
    factory.from_scenario.return_value = session
    exporter = Mock()
    monkeypatch.setattr("nexuml.training.lightning.NexuSession", factory)
    monkeypatch.setattr("nexuml.core.export.export_package", exporter)
    return SimpleNamespace(
        session=session,
        factory=factory,
        exporter=exporter,
        loggers=loggers,
        scenario=SimpleNamespace(name="export", checkpoint=None),
    )


def test_explicit_checkpoint_restores_full_policy_adapter(export_session, tmp_path):
    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = nn.Parameter(torch.tensor(1.0))
            self.register_buffer("offset", torch.tensor(0.0))

    def module():
        policy = nn.Module()
        policy.pipeline = nn.Linear(1, 1)
        policy.action_adapter = Adapter()
        learner = nn.Module()
        learner.policy = policy
        learner.critic = nn.Linear(1, 1)
        learner.register_buffer("frames", torch.tensor(321))
        return learner

    original = module()
    with torch.no_grad():
        original.policy.action_adapter.gain.fill_(7.0)
        original.policy.action_adapter.offset.fill_(11.0)
    checkpoint = tmp_path / "trainer.ckpt"
    torch.save({"state_dict": original.state_dict(), "optimizer_states": [{}]}, checkpoint)
    selective = SimpleNamespace(source="selective.pt")
    export_session.scenario.checkpoint = selective

    def restore():
        assert export_session.factory.call_args.kwargs == {
            "scenario": export_session.scenario,
            "trainer_checkpoint": checkpoint,
        }
        restored = module()
        restored.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
        )
        export_session.session.runtime.policy = restored.policy
        export_session.session.lightning_module = restored
        export_session.session.pipeline = restored.policy.pipeline

    export_session.session.setup.side_effect = restore

    def capture_policy(policy, output, **kwargs):
        output.mkdir()
        torch.save(policy.action_adapter.state_dict(), output / "action_adapter_state.pt")

    export_session.exporter.side_effect = capture_policy
    output = tmp_path / "export"
    result = _export_local(export_session.scenario, output=output, checkpoint=checkpoint)

    exported = export_session.exporter.call_args.args[0]
    assert exported.action_adapter.gain.item() == 7.0
    assert exported.action_adapter.offset.item() == 11.0
    assert not any("critic" in key or "frames" in key for key in exported.state_dict())
    kwargs = export_session.exporter.call_args.kwargs
    assert kwargs["lightning_module"].frames.item() == 321
    assert kwargs["checkpoint_path"] == checkpoint
    assert kwargs["source_metadata"] == {"source": {"checkpoint": str(checkpoint)}}
    assert export_session.scenario.checkpoint is selective
    assert selective.source == "selective.pt"
    export_session.factory.from_scenario.assert_not_called()
    adapter_state = output / "action_adapter_state.pt"
    saved = torch.load(adapter_state, map_location="cpu", weights_only=True)
    assert saved["gain"].item() == 7.0 and saved["offset"].item() == 11.0
    assert result == {"artifacts": [{"path": str(adapter_state), "kind": "train_package"}]}
    for operation in ("fit", "run", "validate", "test", "predict"):
        getattr(export_session.session, operation).assert_not_called()
    for logger in export_session.loggers:
        logger.finalize.assert_called_once_with("success")


@pytest.mark.parametrize("configured", [False, True])
def test_no_explicit_checkpoint_preserves_selective_config(export_session, tmp_path, configured):
    selective = SimpleNamespace(source="selective.pt") if configured else None
    export_session.scenario.checkpoint = selective
    export_session.session.runtime.policy = None
    artifact = tmp_path / "model.pt"
    export_session.exporter.side_effect = lambda *args, **kwargs: artifact.write_bytes(b"model")

    result = _export_local(export_session.scenario, output=tmp_path)

    export_session.factory.from_scenario.assert_called_once_with(export_session.scenario)
    export_session.factory.assert_not_called()
    assert export_session.scenario.checkpoint is selective
    if selective is not None:
        assert selective.source == "selective.pt"
    assert export_session.exporter.call_args.args[0] is export_session.session.pipeline
    assert result == {"artifacts": [{"path": str(artifact), "kind": "train_package"}]}
    for logger in export_session.loggers:
        logger.finalize.assert_called_once_with("success")


@pytest.mark.parametrize("phase", ["setup", "export"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_export_failure_finalizes_without_replacing_primary(
    export_session, tmp_path, phase, cleanup_fails
):
    primary = RuntimeError(f"{phase} failed")
    target = export_session.session.setup if phase == "setup" else export_session.exporter
    target.side_effect = primary
    if cleanup_fails:
        cleanup = RuntimeError("cleanup failed")
        cleanup.add_note("destination detail")
        export_session.loggers[0].finalize.side_effect = cleanup

    with pytest.raises(RuntimeError) as raised:
        _export_local(export_session.scenario, output=tmp_path)

    assert raised.value is primary
    export_session.session.finalize_loggers.assert_called_once_with("failed")
    for logger in export_session.loggers:
        logger.finalize.assert_called_once_with("failed")
    if cleanup_fails:
        assert "cleanup failed" in primary.__notes__[0]
        assert "destination detail" in primary.__notes__


def test_success_cleanup_failure_is_visible_without_retry(export_session, tmp_path):
    cleanup = RuntimeError("cleanup failed")
    export_session.loggers[0].finalize.side_effect = cleanup

    with pytest.raises(RuntimeError) as raised:
        _export_local(export_session.scenario, output=tmp_path)

    assert raised.value is cleanup
    export_session.session.finalize_loggers.assert_called_once_with("success")
    for logger in export_session.loggers:
        logger.finalize.assert_called_once_with("success")


@pytest.mark.parametrize("kind", ["onnx", "safetensors"])
def test_other_export_formats_keep_requests_and_cleanup(
    export_session, tmp_path, monkeypatch, kind
):
    artifacts = [{"path": "actual", "kind": kind}]
    requests = Mock(return_value=artifacts)
    monkeypatch.setattr("nexuml.execution.artifacts.export_requests", requests)
    checkpoint = tmp_path / "trainer.ckpt"

    assert _export_local(
        export_session.scenario, output=tmp_path, checkpoint=checkpoint, export_kind=kind
    ) == {"artifacts": artifacts}

    source, specs, source_checkpoint = requests.call_args.args
    assert source is export_session.session
    assert specs[0].kind == kind and specs[0].output == str(tmp_path)
    assert source_checkpoint == str(checkpoint)
    export_session.exporter.assert_not_called()
    for logger in export_session.loggers:
        logger.finalize.assert_called_once_with("success")


def test_observer_trainer_failure_finalizes_inside_job_scope(monkeypatch):
    events = []
    primary = RuntimeError("trainer construction failed")
    cleanup = RuntimeError("logger cleanup failed")
    initialized_logger = Mock()

    @contextmanager
    def job_scope():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    scenario = SimpleNamespace(
        interaction=SimpleNamespace(environment=SimpleNamespace(job_scope=job_scope))
    )

    class Session:
        def __init__(self, **kwargs):
            self.scenario = kwargs["scenario"]
            self._trainer_loggers = [initialized_logger]

        @property
        def trainer(self):
            assert events == ["enter"]
            events.append("trainer")
            raise primary

        def finalize_loggers(self, status):
            assert events == ["enter", "trainer"]
            initialized_logger.finalize(status)
            raise cleanup

    monkeypatch.setattr("nexuml.training.lightning.NexuSession", Session)

    with pytest.raises(RuntimeError) as raised:
        LocalExecution().run(scenario, observer=object())

    assert raised.value is primary
    assert events == ["enter", "trainer", "exit"]
    initialized_logger.finalize.assert_called_once_with("failed")
    assert "logger cleanup failed" in primary.__notes__[0]
