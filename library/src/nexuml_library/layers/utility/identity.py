"""Identity and noise layers."""

from __future__ import annotations

from typing import cast

import torch
from tensordict import TensorDict

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer


@layer("IdentityLayer")
class IdentityLayer(LayerDefinition):
    """Passthrough layer — returns input unchanged."""

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        return _IdentityLayerRuntime(**context.runtime_kwargs())


class _IdentityLayerRuntime(PipelineLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return x


@layer("MergeLayer")
class MergeLayer(LayerDefinition):
    """Merge multiple input tensors into one by concatenation along the last axis."""

    dim: int = -2

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        return _MergeLayerRuntime(dim=self.dim, **context.runtime_kwargs())


class _MergeLayerRuntime(PipelineLayer):
    def __init__(self, dim: int = -2, **kwargs):
        super().__init__(**kwargs)

        self.dim = dim

        assert len(self.keys_out) == 1, "MergeLayer requires exactly one output key"

    def forward(
        self,
        x: TensorDict | torch.Tensor,
        y: TensorDict | None = None,
    ) -> tuple[TensorDict | torch.Tensor, TensorDict | None]:
        if not self.check_update():
            return x, y

        if isinstance(x, TensorDict):
            x[self.keys_out[0]] = torch.cat(
                [cast(torch.Tensor, x[k]) for k in self.keys_in], dim=self.dim
            )
            x_out = x
        else:
            x_out = torch.cat([x], dim=-1)

        return x_out, y


@layer("AdditiveNoise")
class AdditiveNoise(LayerDefinition):
    """Adds Gaussian noise scaled by the input's standard deviation.

    Useful as a data augmentation layer (noise only applied during training).

    Attributes:
        noise_level: Noise scale relative to the input std.
    """

    noise_level: float = 0.1

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        return _AdditiveNoiseRuntime(noise_level=self.noise_level, **context.runtime_kwargs())


class _AdditiveNoiseRuntime(PipelineLayer):
    def __init__(self, noise_level: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.noise_level = noise_level

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        if not self.training:
            return x
        noise = torch.randn_like(x) * self.noise_level * x.std()
        return x + noise
