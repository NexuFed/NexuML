"""Closed choices validate before runtime construction and retain ordinary strings."""

import pytest
from pydantic import ValidationError

from nexuml.core.serialization import lower_component, restore_component
from nexuml_library.evaluation.visualizers.latent import LatentVisualizer
from nexuml_library.layers.model.resnet import ResNet


@pytest.mark.parametrize("choice", ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
def test_resnet_choices_round_trip_as_strings(choice):
    value = ResNet(resnet_type=choice)
    lowered = lower_component(value)
    assert lowered["params"]["resnet_type"] == choice
    assert restore_component(kind="layer", value=lowered) == value
    assert set(ResNet.model_json_schema()["properties"]["resnet_type"]["enum"]) == set(
        ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    )


@pytest.mark.parametrize(
    "definition,params",
    [
        (ResNet, {"resnet_type": "not-a-model"}),
        (LatentVisualizer, {"method": "not-a-projection"}),
    ],
)
def test_unsupported_choices_and_unknown_fields_fail_before_build(definition, params, monkeypatch):
    def never_build(*args, **kwargs):
        raise AssertionError("Invalid configuration reached runtime construction")

    monkeypatch.setattr(definition, "build", never_build)
    with pytest.raises(ValidationError, match="literal_error"):
        definition(**params)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        definition(unexpected=True)
    value = definition()
    with pytest.raises(ValidationError, match="frozen_instance"):
        value.unexpected = True
