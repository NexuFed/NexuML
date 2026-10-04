"""Python, CLI and optional transport consume the same registered definitions."""

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from nexuml.cli.main import app
from nexuml.core.components import ExecutionBackendDefinition
from nexuml.core.config import ResolvedConfig
from nexuml.core.registry import ComponentRegistry
from nexuml.core.types import ScenarioSpec
from nexuml.execution import catalog, discover, preflight, run
from nexuml.execution.definitions import RayClusterExecution
from nexuml.execution.schemas import ResourceSnapshot


class SixthExecution(ExecutionBackendDefinition):
    component_name = "sixth-execution"
    label = "Sixth installation backend"
    capabilities = {"process_ownership": True, "status": True}
    endpoint: str = "explicit"

    def preflight(self, scenario, **kwargs):
        return {"target": self.endpoint}

    def run(self, scenario, **kwargs):
        assert kwargs["review"] == {"target": self.endpoint}
        return {"status": "succeeded", "target": self.endpoint}


@pytest.fixture
def registry(monkeypatch):
    registry = ComponentRegistry()
    registry.scan([])
    monkeypatch.setattr("nexuml.core.registry._default_registry", registry)
    registry.register("sixth-execution", SixthExecution)
    return registry


def test_sixth_definition_uses_shared_review_and_dispatch(registry, tmp_path):
    value = ScenarioSpec(name="shared", execution=SixthExecution(endpoint="chosen"))
    rows = catalog()
    assert len(rows) == 6
    selected = next(row for row in rows if row["type"] == "sixth-execution")
    assert selected["schema"]["properties"]["endpoint"]["type"] == "string"
    assert preflight(value) == {"target": "chosen"}
    assert discover(value.execution).allocatable is None
    assert run(value) == {"status": "succeeded", "target": "chosen"}
    config = tmp_path / "sixth.yaml"
    ResolvedConfig.from_scenario(value).save(config)
    cli = CliRunner()
    result = cli.invoke(app, ["backend", "catalog"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == rows
    result = cli.invoke(app, ["backend", "preflight", "--config", str(config)])
    assert result.exit_code == 0 and json.loads(result.output) == {"target": "chosen"}
    result = cli.invoke(app, ["train", "--config", str(config)])
    assert result.exit_code == 0, (result.output, result.exception)
    assert '"target": "chosen"' in result.output


def test_local_uses_one_session_observer_and_configured_exports(monkeypatch, registry):
    calls = []
    result = SimpleNamespace(artifacts=[])
    observer = object()

    class Session:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.trainer = SimpleNamespace(callbacks=[])

        def run(self):
            assert self.trainer.callbacks == [observer]
            calls.append("canonical lifecycle")
            return result

    monkeypatch.setattr("nexuml.training.lightning.NexuSession", Session)
    monkeypatch.setattr(
        "nexuml.execution.artifacts.export_requests",
        lambda source, requests: (
            calls.append("exports") or [{"path": "actual", "kind": "train_package"}]
        ),
    )
    value = ScenarioSpec(name="local", exports=[{"kind": "train_package", "output": "actual"}])
    assert run(value, observer=observer, trainer_checkpoint="native.ckpt") is result
    assert calls[0]["trainer_checkpoint"] == "native.ckpt"
    assert calls[1:] == ["canonical lifecycle", "exports"]
    assert result.artifacts == [{"path": "actual", "kind": "train_package"}]


def test_missing_dependency_is_described_not_substituted(registry, monkeypatch):
    monkeypatch.setattr("nexuml.execution.service.importlib.util.find_spec", lambda _: None)
    row = next(row for row in catalog() if row["type"] == "ray-cluster")
    assert not row["available"] and row["diagnostics"]
    value = ScenarioSpec(name="missing", execution=RayClusterExecution())
    assert discover(value.execution).allocatable is None
    with pytest.raises(ValueError, match="Missing dependency ray"):
        run(value)
    assert isinstance(value.execution, RayClusterExecution)


def test_resource_contract_rejects_unknown_fields_and_nonfinite_values():
    value = ResourceSnapshot(backend="local", source="limits", visible_nodes=0)
    assert value.model_dump()["visible_nodes"] == 0 and value.allocatable is None
    with pytest.raises(ValueError):
        ResourceSnapshot(backend="local", source="limits", token="secret")
    with pytest.raises(ValueError):
        ResourceSnapshot(backend="local", source="limits", allocatable={"CPU": float("inf")})
