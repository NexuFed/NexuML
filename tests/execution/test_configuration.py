"""Registered execution settings share NexuML's ordinary config boundary."""

import pytest
from pydantic import ValidationError

from nexuml.core.config import ResolvedConfig
from nexuml.core.registry import get_component_registry
from nexuml.core.serialization import lower_model
from nexuml.core.types import ScenarioSpec
from nexuml.execution import catalog
from nexuml.execution.definitions import LocalExecution, RayClusterExecution


def test_default_local_identity():
    value = ScenarioSpec(name="default")
    assert isinstance(value.execution, LocalExecution)
    assert lower_model(value)["execution"] == {"type": "local", "version": "1", "params": {}}


@pytest.mark.parametrize(
    "name", ["local", "ray-cluster", "ray-job", "kubernetes-job", "kubernetes-pytorchjob"]
)
def test_builtin_round_trip(name):
    definition = get_component_registry().get_type("execution_backend", name)()
    config = ResolvedConfig.from_scenario(ScenarioSpec(name=name, execution=definition))
    restored = ResolvedConfig.from_yaml(config.to_yaml())
    assert type(restored.execution) is type(definition)
    assert lower_model(restored) == lower_model(config)
    assert "kind" not in lower_model(config)["execution"]
    row = next(row for row in catalog() if row["type"] == name)
    assert row["schema"] == type(definition).model_json_schema()


def test_ray_execution_field_errors_keep_document_location():
    with pytest.raises(ValidationError) as error:
        ResolvedConfig.from_yaml("""
name: invalid
execution:
  type: ray-cluster
  version: '1'
  params:
    workers: 0
""")
    assert error.value.errors()[0]["loc"] == ("execution", "params", "workers")


@pytest.mark.parametrize(
    "identity",
    [
        {"type": "not-installed", "version": "1", "params": {}},
        {"type": "local", "version": "99", "params": {}},
    ],
)
def test_unknown_backend_is_not_replaced(identity):
    with pytest.raises(KeyError, match="Unknown component"):
        ScenarioSpec.model_validate({"name": "unknown", "execution": identity})


def test_legacy_execution_has_explicit_conversion_error():
    with pytest.raises(ValueError, match="replace legacy execution.kind"):
        ResolvedConfig.from_yaml("name: old\nexecution:\n  kind: local\n")
    with pytest.raises(ValidationError, match="replace legacy execution.kind"):
        ScenarioSpec.model_validate({"name": "old", "execution": {"kind": "ray"}})


def test_ray_settings_survive_registered_config():
    configured = RayClusterExecution(workers=(2, 4), resources_per_worker={"CPU": 3, "GPU": 1})
    config = ResolvedConfig.from_scenario(ScenarioSpec(name="ray", execution=configured))
    assert ResolvedConfig.from_yaml(config.to_yaml()).execution == configured


@pytest.mark.parametrize(
    "address", ["https://user:secret@host", "http://host?token=secret", "local"]
)
def test_credential_bearing_ray_targets_are_rejected(address):
    with pytest.raises(ValidationError):
        RayClusterExecution(target={"address": address})
