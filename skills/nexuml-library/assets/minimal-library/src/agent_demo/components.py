"""Minimal data, loss and consumer-evaluation contracts."""

import torch
from pydantic import Field
from tensordict import TensorDict

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import (
    DataSourceDefinition,
    EvalAlgorithmDefinition,
    EvalBuildContext,
    LayerBuildContext,
    LayerDefinition,
)
from nexuml.core.discovery import data_source, eval_algorithm, layer
from nexuml.data.dataset import NexuDataset
from nexuml.evaluation.algorithm import EvalAlgorithm


@data_source("agent_demo_pairs")
class PairedVectors(DataSourceDefinition):
    """Configure deterministic vectors and their regression targets."""

    samples: int = Field(default=64, gt=0)
    width: int = Field(default=4, gt=0)
    seed: int = 7

    def build(self) -> NexuDataset:
        return _PairedVectors(**self.model_dump())


class _PairedVectors(NexuDataset):
    def __init__(self, samples: int, width: int, seed: int):
        super().__init__(label_names=["target"])
        self.features = torch.randn(samples, width, generator=torch.Generator().manual_seed(seed))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        features = self.features[index]
        return (
            TensorDict({"features": features}, batch_size=[]),
            TensorDict({"target": 2 * features}, batch_size=[]),
        )


@layer("agent_demo_mse")
class SquaredError(LayerDefinition):
    """Build a label-dependent per-sample squared-error loss."""

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        return _SquaredError(**context.runtime_kwargs())


class _SquaredError(PipelineLayer):
    def forward_tensor(self, x, y=None):
        if y is None:
            if getattr(self, "_shape_propagation_mode", False):
                return x.new_zeros(x.shape[:-1])
            raise ValueError("SquaredError requires the routed target label.")
        return (x - y).square().mean(dim=-1)


@eval_algorithm("agent_demo_rmse")
class RootMeanSquaredError(EvalAlgorithmDefinition):
    """Build an element-weighted consumer evaluator."""

    def build(self, context: EvalBuildContext) -> EvalAlgorithm:
        return _RootMeanSquaredError(
            feature_key=context.feature_key or "prediction",
            label_key=context.label_key or "target",
        )


class _RootMeanSquaredError(EvalAlgorithm):
    def __init__(self, feature_key: str, label_key: str):
        self.feature_key = feature_key
        self.label_key = label_key
        self.squared_sum = 0.0
        self.elements = 0

    def eval_batch(self, x, y):
        if y is None:
            raise ValueError("RMSE requires target labels.")
        error = x[self.feature_key] - y[self.label_key]
        self.squared_sum += error.square().sum().item()
        self.elements += error.numel()

    def results(self):
        if not self.elements:
            raise ValueError("RMSE requires at least one evaluated element.")
        return {"rmse": (self.squared_sum / self.elements) ** 0.5}
