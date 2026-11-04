"""Small policy/value heads consuming explicit TensorDict observation fields."""

from math import prod
from typing import Literal

import torch
from pydantic import Field
from torch import nn

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer
from nexuml.core.types import LayerSpec, PipelineSpec


@layer("PolicyHead")
class PolicyHead(LayerDefinition):
    """Dense multimodal head with no simulator or algorithm dependencies."""

    output_dim: int = Field(default=1, gt=0)
    hidden_dims: tuple[int, ...] = (64, 64)
    output_kind: Literal["gaussian", "logits", "value"] = "value"

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        return _PolicyHeadRuntime(self, context)


class _PolicyHeadRuntime(PipelineLayer):
    def __init__(self, definition, context):
        super().__init__(**context.runtime_kwargs())
        self.output_kind = definition.output_kind
        in_dim = sum(prod(context.input_sizes[key]) for key in context.keys_in)
        out_dim = definition.output_dim * (2 if self.output_kind == "gaussian" else 1)
        dims = (in_dim, *definition.hidden_dims, out_dim)
        layers = []
        for index, (left, right) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(left, right))
            if index < len(dims) - 2:
                layers.append(nn.Tanh())
        # ponytail: dense pixels scale poorly; use a convolutional encoder for large cameras.
        self.model = nn.Sequential(*layers)

    def forward(self, x, y=None):
        tensors = []
        for key in self.keys_in:
            value = x[key]
            tensor = value.float().reshape(x.batch_size[0], -1)
            tensors.append(tensor / 255 if value.dtype == torch.uint8 else tensor)
        output = self.model(torch.cat(tensors, dim=-1))
        if self.output_kind == "gaussian":
            loc, scale = output.chunk(2, dim=-1)
            x[self.keys_out[0]] = loc
            x[self.keys_out[1]] = torch.nn.functional.softplus(scale) + 1e-4
        else:
            x[self.keys_out[0]] = output
        return x, y


def continuous_policy_mlp(
    observation_keys=("observation",),
    action_dim=1,
    hidden_dims=(64, 64),
    loc_key="action_loc",
    scale_key="action_scale",
) -> PipelineSpec:
    """Compose a bounded-Gaussian parameter head.

    Returns:
        A neural graph producing the Gaussian adapter's parameter keys.
    """
    return PipelineSpec(
        stages={
            "policy": [
                LayerSpec(
                    component=PolicyHead(
                        output_dim=action_dim, hidden_dims=hidden_dims, output_kind="gaussian"
                    ),
                    keys_in=list(observation_keys),
                    keys_out=[loc_key, scale_key],
                )
            ]
        }
    )


def discrete_policy_mlp(
    observation_keys=("observation",),
    num_actions=2,
    hidden_dims=(64, 64),
    logits_key="action_logits",
) -> PipelineSpec:
    """Compose a categorical-logit head.

    Returns:
        A neural graph producing the categorical adapter's logits key.
    """
    return PipelineSpec(
        stages={
            "policy": [
                LayerSpec(
                    component=PolicyHead(
                        output_dim=num_actions, hidden_dims=hidden_dims, output_kind="logits"
                    ),
                    keys_in=list(observation_keys),
                    keys_out=[logits_key],
                )
            ]
        }
    )
