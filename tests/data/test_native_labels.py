"""Fixed image-source labels agree with their runtime data, without downloads."""

import sys
from functools import partial

import pytest
import torch
from pydantic import ValidationError
from tensordict import TensorDict

from nexuml.core.config import ResolvedConfig
from nexuml.core.types import DataSpec
from nexuml_library.data.image.cifar import (
    CIFAR10Dataset,
    CIFAR100Dataset,
    _CIFAR10DatasetRuntime,
    _CIFAR100DatasetRuntime,
)
from nexuml_library.data.image.fashionmnist import (
    FashionMNISTDataset,
    _FashionMNISTDatasetRuntime,
)
from nexuml_library.data.image.mnist import MNISTDataset, _MNISTDatasetRuntime


@pytest.fixture(
    params=[
        pytest.param(
            (CIFAR10Dataset, _CIFAR10DatasetRuntime, "class_labels", (3, 32, 32), "data/cifar10"),
            id="cifar10",
        ),
        pytest.param(
            (
                CIFAR100Dataset,
                _CIFAR100DatasetRuntime,
                "class_labels",
                (3, 32, 32),
                "data/cifar100",
            ),
            id="cifar100",
        ),
        pytest.param(
            (MNISTDataset, _MNISTDatasetRuntime, "digit", (28, 28), "data/mnist"),
            id="mnist",
        ),
        pytest.param(
            (
                FashionMNISTDataset,
                _FashionMNISTDatasetRuntime,
                "category",
                (28, 28),
                "data/fashionmnist",
            ),
            id="fashionmnist",
        ),
    ]
)
def source_case(request, monkeypatch):
    monkeypatch.setitem(sys.modules, "torchvision", None)
    return request.param


@pytest.mark.parametrize(
    ("train", "split", "expected_split"),
    [(True, None, "fit"), (False, None, "test"), (True, "custom", "custom")],
)
@pytest.mark.parametrize("tensor_targets", [False, True], ids=["list", "tensor"])
def test_native_labels_match_runtime_metadata(
    source_case, monkeypatch, train, split, expected_split, tensor_targets
):
    definition_type, runtime_type, label_key, shape, _ = source_case
    data = torch.stack(
        [torch.zeros(shape, dtype=torch.uint8), torch.full(shape, 255, dtype=torch.uint8)]
    )
    targets = torch.tensor([1, 2]) if tensor_targets else [1, 2]
    monkeypatch.setattr(
        sys.modules[runtime_type.__module__],
        runtime_type.__name__,
        partial(runtime_type, data=data, targets=targets),
    )

    definition = definition_type(train=train, split=split)
    dataset = definition.build()

    assert definition.fixed_label_keys == (label_key,)
    assert tuple(dataset.label_names) == definition.fixed_label_keys
    assert dataset.meta.columns.tolist() == [*definition.fixed_label_keys, "split"]
    assert dataset.meta[label_key].tolist() == [1, 2]
    assert dataset.meta["split"].tolist() == [expected_split, expected_split]
    for index, target in enumerate([1, 2]):
        x, y = dataset[index]
        assert isinstance(y, TensorDict)
        assert tuple(y.keys()) == definition.fixed_label_keys
        assert y[label_key].item() == target
        assert x["features"].shape == ((1, *shape) if len(shape) == 2 else shape)
        assert x["features"].dtype == torch.float32
        assert torch.all(x["features"] == index)


def test_runtime_consumes_definition_label_keys(source_case, monkeypatch):
    definition_type, runtime_type, _, shape, _ = source_case
    monkeypatch.setattr(definition_type, "fixed_label_keys", ("renamed_label",))

    dataset = runtime_type(data=torch.zeros(2, *shape), targets=[1, 2])

    assert tuple(dataset.label_names) == definition_type.fixed_label_keys
    assert dataset.meta.columns.tolist() == [*definition_type.fixed_label_keys, "split"]
    _, y = dataset[0]
    assert isinstance(y, TensorDict)
    assert tuple(y.keys()) == definition_type.fixed_label_keys
    assert y[definition_type.fixed_label_keys[0]].item() == 1


def test_native_labels_stay_out_of_parameters_and_yaml(source_case):
    definition_type, _, label_key, _, root = source_case
    definition = definition_type()
    expected = {"root": root, "train": True, "download": False, "split": None}
    if definition_type is MNISTDataset:
        expected["normalize"] = False

    assert definition.model_dump() == expected
    assert "fixed_label_keys" not in definition_type.model_fields
    schema = definition_type.model_json_schema()
    assert set(schema["properties"]) == set(expected)
    assert "x-nexuml-outputs" not in schema
    with pytest.raises(ValidationError, match="Extra inputs"):
        definition_type(fixed_label_keys=[label_key])
    with pytest.raises(ValidationError, match="frozen"):
        definition.train = False

    config = ResolvedConfig(name="native_labels", data=DataSpec(source=definition))
    yaml = config.to_yaml()
    assert "fixed_label_keys" not in yaml
    assert ResolvedConfig.from_yaml(yaml) == config
