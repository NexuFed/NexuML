"""Time-conditioned U-Net for image-valued vector fields."""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
from pydantic import Field
from tensordict import TensorDict

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class _TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, output_dim: int):
        super().__init__()
        if embedding_dim % 2:
            raise ValueError("time_embedding_dim must be even")
        half = embedding_dim // 2
        frequencies = 2.0 * math.pi * torch.exp(
            torch.linspace(0.0, math.log(1000.0), half)
        )
        self.register_buffer("frequencies", frequencies)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        angles = time.reshape(time.shape[0], 1) * cast(torch.Tensor, self.frequencies)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        return self.mlp(embedding)


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time = nn.Linear(time_dim, out_channels)
        self.norm2 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        x = x + self.time(time)[:, :, None, None]
        x = self.conv2(torch.nn.functional.silu(self.norm2(x)))
        return x + residual


@layer("TimeConditionedUNet")
class TimeConditionedUNet(LayerDefinition):
    """Small time-conditioned U-Net for image-valued continuous flows."""

    base_channels: int = Field(default=32, gt=0)
    time_embedding_dim: int = Field(default=128, gt=0, multiple_of=2)

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        return _TimeConditionedUNetRuntime(
            **context.runtime_kwargs(),
            **self.model_dump(),
        )


class _TimeConditionedUNetRuntime(PipelineLayer):
    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        base_channels: int,
        time_embedding_dim: int,
        **kwargs,
    ):
        super().__init__(
            input_sizes=input_sizes,
            keys_in=keys_in,
            keys_out=keys_out,
            **kwargs,
        )
        shape = input_sizes[keys_in[0]]
        if len(shape) != 3:
            raise ValueError("TimeConditionedUNet expects image state [C, H, W]")
        channels, height, width = shape
        if height % 4 or width % 4:
            raise ValueError("TimeConditionedUNet requires height and width divisible by four")

        c1, c2, c3 = base_channels, 2 * base_channels, 4 * base_channels
        time_dim = 4 * base_channels
        self.time_embedding = _TimeEmbedding(time_embedding_dim, time_dim)

        self.stem = nn.Conv2d(channels, c1, kernel_size=3, padding=1)
        self.encoder1 = _ResidualBlock(c1, c1, time_dim)
        self.down1 = nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1)
        self.encoder2 = _ResidualBlock(c2, c2, time_dim)
        self.down2 = nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1)

        self.middle1 = _ResidualBlock(c3, c3, time_dim)
        self.middle2 = _ResidualBlock(c3, c3, time_dim)

        self.up1 = nn.Conv2d(c3, c2, kernel_size=3, padding=1)
        self.decoder1 = _ResidualBlock(c2 + c2, c2, time_dim)
        self.up2 = nn.Conv2d(c2, c1, kernel_size=3, padding=1)
        self.decoder2 = _ResidualBlock(c1 + c1, c1, time_dim)

        self.output = nn.Sequential(
            _group_norm(c1),
            nn.SiLU(),
            nn.Conv2d(c1, channels, kernel_size=3, padding=1),
        )

    def velocity(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Evaluate the image vector field.

        Returns:
            Predicted velocity with the same shape as the image state.
        """
        time_embedding = self.time_embedding(time)

        skip1 = self.encoder1(self.stem(state), time_embedding)
        skip2 = self.encoder2(self.down1(skip1), time_embedding)
        x = self.down2(skip2)

        x = self.middle1(x, time_embedding)
        x = self.middle2(x, time_embedding)

        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up1(x)
        x = self.decoder1(torch.cat((x, skip2), dim=1), time_embedding)

        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up2(x)
        x = self.decoder2(torch.cat((x, skip1), dim=1), time_embedding)
        return self.output(x)

    def forward(
        self,
        x: TensorDict | torch.Tensor,
        y: TensorDict | None = None,
    ) -> tuple[TensorDict | torch.Tensor, TensorDict | None]:
        if not self.check_update():
            return x, y
        if not isinstance(x, TensorDict):
            raise TypeError("TimeConditionedUNet requires TensorDict input")

        keys_in = cast(list[str], self.keys_in)
        state = cast(torch.Tensor, x[keys_in[0]])
        time = cast(torch.Tensor, x[keys_in[1]])
        x[self.keys_out[0]] = self.velocity(state, time)
        return x, y

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError
