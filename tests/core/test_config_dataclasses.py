"""Config lowering keeps dataclass parameters portable without building a runtime."""

from dataclasses import dataclass
import json
from pathlib import Path

from pydantic import BaseModel
import pytest

from nexuml.core.config import ResolvedConfig
from nexuml.core.serialization import lower_model
from nexuml_library.data.synthetic import SyntheticDataset
from nexuml_library.scenarios.asd.synthetic_linear_ae import (
    synthetic_linear_ae_multiclass,
    synthetic_linear_ae_multilabel,
    synthetic_linear_ae_regression,
)


def test_nested_dataclass_fields_use_recursive_plain_value_lowering():
    @dataclass(frozen=True)
    class Values:
        path: Path
        labels: tuple[int, ...]

    @dataclass(frozen=True)
    class Settings:
        values: list[Values]

    class Config(BaseModel):
        settings: Settings

    config = Config(settings=Settings(values=[Values(Path("worker/input"), (1, 2))]))
    lowered = lower_model(config)
    assert lowered == {"settings": {"values": [{"path": "worker/input", "labels": [1, 2]}]}}
    assert Config.model_validate(json.loads(json.dumps(lowered))) == config


@pytest.mark.parametrize(
    "factory",
    [
        synthetic_linear_ae_multiclass,
        synthetic_linear_ae_multilabel,
        synthetic_linear_ae_regression,
    ],
)
def test_synthetic_target_dataclasses_round_trip_through_yaml_without_runtime(factory):
    config = ResolvedConfig.from_scenario(factory())
    text = config.to_yaml()
    restored = ResolvedConfig.from_yaml(text)
    assert lower_model(restored) == lower_model(config)
    restored_source = restored.data.datasets[0].source
    original_source = config.data.datasets[0].source
    assert isinstance(restored_source, SyntheticDataset)
    assert isinstance(original_source, SyntheticDataset)
    assert restored_source.targets == original_source.targets
