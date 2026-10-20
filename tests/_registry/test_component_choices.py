"""Closed choices validate before runtime construction and retain ordinary strings."""

from typing import get_type_hints

import pytest
from pydantic import ValidationError

from nexuml.core.serialization import lower_component, restore_component
from nexuml.core.types import PipelineSpec
from nexuml_library.evaluation.visualizers.latent import LatentVisualizer
from nexuml_library.layers.model.resnet import ResNet
from nexuml_library.scenarios.model.resnet_classifier import resnet_classifier
from nexuml_library.scenarios.vision.cifar_resnet import cifar_resnet
from nexuml_library.scenarios.vision.mnist_resnet import fashionmnist_resnet, mnist_resnet
from nexuml_library.scenarios.vision.mnist_resnet_shards import mnist_resnet_shards


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
    "factory",
    [
        resnet_classifier,
        cifar_resnet,
        mnist_resnet,
        fashionmnist_resnet,
        mnist_resnet_shards,
    ],
)
def test_resnet_callers_share_closed_choices_without_building(factory):
    assert get_type_hints(factory)["resnet_type"] == get_type_hints(ResNet)["resnet_type"]
    configured = factory(resnet_type="resnet34")
    pipeline = configured if isinstance(configured, PipelineSpec) else configured.pipeline
    definition = pipeline.stages["Encoder"][0].component
    assert isinstance(definition, ResNet)
    assert definition.resnet_type == "resnet34"


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
