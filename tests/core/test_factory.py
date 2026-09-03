"""Tests for portable typed factory configuration."""

from __future__ import annotations

import torch
import pytest
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import SingleDeviceStrategy
from pydantic import ValidationError

from nexuml import callback, optimizer, scheduler, strategy, writer
from nexuml.core.config import ResolvedConfig
from nexuml.core.types import (
    CallbackSpec,
    DataSpec,
    DistanceEstimatorSpec,
    OptimizerSpec,
    PreprocessingSpec,
    ScenarioSpec,
    SchedulerSpec,
    TrainingSpec,
)
from nexuml.data.export.tensor_shards import TensorShardsBackend
from nexuml_library.layers.head.decision_rule import DecisionRulePipelineLayer
from nexuml_library.scenarios.training.defaults import default_training


def test_training_factory_helpers_build_runtime_objects() -> None:
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer_spec = optimizer(torch.optim.Adam, betas=(0.8, 0.9))
    built_optimizer = optimizer_spec.build([parameter], lr=0.02)

    assert isinstance(built_optimizer, torch.optim.Adam)
    assert built_optimizer.defaults["lr"] == 0.02
    assert optimizer_spec.kwargs == {"betas": [0.8, 0.9]}

    scheduler_spec = scheduler(torch.optim.lr_scheduler.ConstantLR, factor=0.5)
    assert isinstance(
        scheduler_spec.build(built_optimizer),
        torch.optim.lr_scheduler.ConstantLR,
    )


def test_factory_specs_round_trip_with_scenario() -> None:
    scenario = ScenarioSpec(
        name="factories",
        training=TrainingSpec(
            optimizer=optimizer(torch.optim.Adam, weight_decay=0.1),
            scheduler=scheduler(torch.optim.lr_scheduler.ConstantLR, factor=0.5),
            strategy=strategy(SingleDeviceStrategy, device="cpu"),
        ),
        callbacks=[callback(ModelCheckpoint, monitor="val/loss")],
        data=DataSpec(
            preprocessing=PreprocessingSpec(
                writer=writer(TensorShardsBackend, samples_per_shard=8)
            ),
        ),
    )

    yaml_text = ResolvedConfig.from_scenario(scenario).to_yaml()
    restored = ResolvedConfig.from_yaml(yaml_text)

    assert restored.model_dump() == scenario.model_dump()
    assert restored.callbacks[0].factory.endswith(":ModelCheckpoint")
    assert restored.data.preprocessing.writer.backend_name() == "tensor_shards"
    assert isinstance(restored.data.preprocessing.writer.build(), TensorShardsBackend)


def test_default_training_accepts_typed_optimizer_factory() -> None:
    spec = default_training(optimizer_factory=torch.optim.SGD)
    parameter = torch.nn.Parameter(torch.ones(1))

    assert isinstance(spec.optimizer.build([parameter]), torch.optim.SGD)


@pytest.mark.parametrize(
    ("model_type", "legacy"),
    [
        (OptimizerSpec, {"type": "torch.optim.Adam", "params": {"lr": 0.1}}),
        (
            SchedulerSpec,
            {"type": "torch.optim.lr_scheduler.ConstantLR", "params": {"factor": 0.5}},
        ),
        (CallbackSpec, {"type": "checkpoint", "params": {"monitor": "val/loss"}}),
        (TrainingSpec, {"strategy": "fsdp", "strategy_params": {"cpu_offload": False}}),
        (PreprocessingSpec, {"writer": "tensor_shards", "writer_params": {}}),
        (DistanceEstimatorSpec, {"type": "mahalanobis", "params": {}}),
        (DecisionRulePipelineLayer, {"rule_type": "percentile", "rule_params": {}}),
    ],
)
def test_legacy_authoring_parameter_bags_are_rejected(model_type, legacy) -> None:
    with pytest.raises(ValidationError):
        model_type.model_validate(legacy)
