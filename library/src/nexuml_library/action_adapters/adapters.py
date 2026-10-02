"""Direct, bounded Gaussian and categorical deployment semantics."""

import torch
from pydantic import Field
from tensordict import TensorDict
from torch import nn
from torch.distributions import (
    AffineTransform,
    Categorical,
    Independent,
    Normal,
    TanhTransform,
    TransformedDistribution,
)

from nexuml.core.components import ActionAdapterDefinition
from nexuml.core.discovery import action_adapter
from nexuml.core.types import InteractionContract


class BoundedNormal(Independent):
    """Tanh-squashed Gaussian shared by inference and TorchRL probability losses."""

    def __init__(self, loc, scale, low, high, min_scale=1e-4):
        low, high = low.to(loc), high.to(loc)
        self.loc = loc
        self.low = low
        self.high = high
        base = Normal(loc, scale.clamp_min(min_scale))
        transformed = TransformedDistribution(
            base,
            [
                TanhTransform(cache_size=1),
                AffineTransform((low + high) / 2, (high - low) / 2),
            ],
        )
        super().__init__(transformed, low.ndim)

    @property
    def deterministic_sample(self):
        return (self.low + self.high) / 2 + (self.high - self.low) / 2 * self.loc.tanh()

    @property
    def mode(self):
        return self.deterministic_sample


@action_adapter("DirectActionAdapter")
class DirectActionAdapter(ActionAdapterDefinition):
    """Expose an already-produced action tensor under its contract key."""

    source_key: str = "action"
    action_key: str = "action"

    def build(self, contract: InteractionContract) -> nn.Module:
        if self.action_key not in contract.actions:
            raise ValueError(f"Unknown action field {self.action_key!r}")
        return _DirectAdapter(self.source_key, self.action_key)


class _DirectAdapter(nn.Module):
    def __init__(self, source_key, action_key):
        super().__init__()
        self.source_key, self.action_key = source_key, action_key

    def forward(self, output, *, deterministic=True):
        return TensorDict({self.action_key: output[self.source_key]}, batch_size=output.batch_size)


@action_adapter("GaussianActionAdapter")
class GaussianActionAdapter(ActionAdapterDefinition):
    """Convert location/positive scale tensors into bounded continuous actions."""

    loc_key: str = "action_loc"
    scale_key: str = "action_scale"
    action_key: str = "action"
    min_scale: float = Field(default=1e-4, gt=0)

    def build(self, contract: InteractionContract) -> nn.Module:
        field = contract.actions[self.action_key]
        if contract.action_space != "continuous" or field.low is None or field.high is None:
            raise ValueError("Gaussian actions require a continuous contract with finite bounds")
        if not field.dtype.startswith("float") or not field.shape:
            raise ValueError("Gaussian actions require a floating-point tensor action shape")
        return _GaussianAdapter(self, field)


class _GaussianAdapter(nn.Module):
    distribution_class = BoundedNormal

    def __init__(self, definition, field):
        super().__init__()
        self.parameter_keys = {"loc": definition.loc_key, "scale": definition.scale_key}
        self.action_key = definition.action_key
        self.min_scale = definition.min_scale
        dtype = getattr(torch, field.dtype)
        for key in ("low", "high"):
            bound = torch.as_tensor(getattr(field, key), dtype=dtype)
            if bound.numel() == 1:
                bound = bound.expand(field.shape).clone()
            else:
                bound = bound.reshape(field.shape)
            self.register_buffer(key, bound)

    @property
    def distribution_kwargs(self):
        return {"low": self.low, "high": self.high, "min_scale": self.min_scale}

    def forward(self, output, *, deterministic=True):
        distribution = BoundedNormal(
            loc=output[self.parameter_keys["loc"]],
            scale=output[self.parameter_keys["scale"]].clamp_min(self.min_scale),
            **self.distribution_kwargs,
        )
        action = distribution.mode if deterministic else distribution.sample()
        return TensorDict({self.action_key: action}, batch_size=output.batch_size)


@action_adapter("CategoricalActionAdapter")
class CategoricalActionAdapter(ActionAdapterDefinition):
    """Select the mode or sample a scalar categorical action from logits."""

    logits_key: str = "action_logits"
    action_key: str = "action"

    def build(self, contract: InteractionContract) -> nn.Module:
        field = contract.actions[self.action_key]
        if contract.action_space != "discrete" or field.dtype != "int64" or field.shape != ():
            raise ValueError("Categorical actions require a scalar int64 discrete contract")
        if field.num_values is None:
            raise ValueError("Categorical action contract requires num_values")
        return _CategoricalAdapter(self, field.num_values)


class _CategoricalAdapter(nn.Module):
    distribution_class = Categorical
    distribution_kwargs = {}

    def __init__(self, definition, num_values):
        super().__init__()
        self.parameter_keys = {"logits": definition.logits_key}
        self.action_key = definition.action_key
        self.num_values = num_values

    def forward(self, output, *, deterministic=True):
        logits = output[self.parameter_keys["logits"]]
        if logits.shape[-1] != self.num_values:
            raise ValueError(f"Categorical logits require {self.num_values} classes")
        distribution = Categorical(logits=logits)
        action = distribution.mode if deterministic else distribution.sample()
        return TensorDict({self.action_key: action}, batch_size=output.batch_size)
