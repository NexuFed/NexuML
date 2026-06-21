"""Convolutional autoencoder building blocks."""

from __future__ import annotations
from nexuml.core.discovery import layer

import math
from typing import Sequence, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from nexuml.core.base_layer import PipelineLayer


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.2, inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    return nn.ReLU(inplace=True)


def _normalise_pair(value: int | Sequence[int]) -> int | tuple[int, int]:
    if isinstance(value, int):
        return value
    values = tuple(int(v) for v in value)
    if len(values) != 2:
        raise ValueError(f"Expected int or pair, got {value!r}.")
    return values[0], values[1]


def _conv_block(
    in_channels: int,
    out_channels: int,
    activation: str,
    kernel_size: int | tuple[int, int] = 3,
    stride: int | tuple[int, int] = 2,
) -> nn.Sequential:
    kernel_size = _normalise_pair(kernel_size)
    stride = _normalise_pair(stride)
    if isinstance(kernel_size, int):
        padding = kernel_size // 2
    else:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        ),
        nn.BatchNorm2d(out_channels),
        _activation(activation),
    )


def _up_block(
    in_channels: int,
    out_channels: int,
    activation: str,
    kernel_size: int | tuple[int, int] = 3,
    scale_factor: int | tuple[int, int] = 2,
) -> nn.Sequential:
    kernel_size = _normalise_pair(kernel_size)
    scale_factor = _normalise_pair(scale_factor)
    if isinstance(kernel_size, int):
        padding = kernel_size // 2
    else:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    return nn.Sequential(
        nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False),
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        _activation(activation),
    )


@layer("ConvolutionalEncoder")
class ConvolutionalEncoder(PipelineLayer):
    """Encode a 2D tensor into a latent vector."""

    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        output_dim: int = 128,
        channel_schedule: Sequence[int] | None = None,
        kernel_sizes: Sequence[int | tuple[int, int]] | None = None,
        strides: Sequence[int | tuple[int, int]] | None = None,
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__(input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out, **kwargs)
        input_shape = tuple(self.input_sizes[keys_in[0]])
        if len(input_shape) != 3:
            raise ValueError(f"ConvolutionalEncoder expects (C, H, W) input, got {input_shape!r}.")

        channel_schedule = list(channel_schedule or [16, 32, 64])
        self.channel_schedule = channel_schedule
        self.output_dim = output_dim

        n_layers = len(channel_schedule)
        kernel_sizes = list(kernel_sizes) if kernel_sizes else [3] * n_layers
        strides = list(strides) if strides else [2] * n_layers
        self._kernel_sizes = kernel_sizes
        self._strides = strides

        encoder_layers: list[nn.Module] = []
        current_channels = input_shape[0]
        for i, out_channels in enumerate(channel_schedule):
            encoder_layers.append(
                _conv_block(
                    current_channels,
                    out_channels,
                    activation,
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                )
            )
            current_channels = out_channels
        self.encoder = nn.Sequential(*encoder_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            encoded = self.encoder(dummy)

        self.decoder_shape = tuple(encoded.shape[1:])
        self.flattened_dim = int(math.prod(self.decoder_shape))
        self.projection = nn.Linear(self.flattened_dim, output_dim)

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encoder(x)
        return self.projection(encoded.flatten(1))


@layer("VariationalLatent")
class VariationalLatent(PipelineLayer):
    """Variational bottleneck layer emitting latent sample, moments, and KL loss."""

    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        latent_dim: int = 32,
        beta: float = 1.0,
        **kwargs,
    ):
        super().__init__(input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out, **kwargs)
        input_dim = int(math.prod(self.input_sizes[keys_in[0]]))
        self.latent_dim = latent_dim
        self.beta = beta
        self.mu_proj = nn.Linear(input_dim, latent_dim)
        self.logvar_proj = nn.Linear(input_dim, latent_dim)

    def forward(
        self,
        x: TensorDict | torch.Tensor,
        y: TensorDict | None = None,
    ) -> tuple[TensorDict | torch.Tensor, TensorDict | None]:
        assert isinstance(x, TensorDict)
        keys_in_fwd: list[str] = cast(list[str], self.keys_in)
        features = cast(torch.Tensor, x[keys_in_fwd[0]]).reshape(x.batch_size[0], -1)
        mu = self.mu_proj(features)
        logvar = self.logvar_proj(features)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            latent = mu + eps * std
        else:
            latent = mu
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

        x[self.keys_out[0]] = latent
        if len(self.keys_out) > 1:
            x[self.keys_out[1]] = mu
        if len(self.keys_out) > 2:
            x[self.keys_out[2]] = logvar
        if len(self.keys_out) > 3:
            x[self.keys_out[3]] = (self.beta * kl).expand(x.batch_size)
        return x, y

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


@layer("ConvolutionalDecoder")
class ConvolutionalDecoder(PipelineLayer):
    """Decode a latent vector back into a 2D tensor."""

    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        decoder_shape: Sequence[int],
        output_shape: Sequence[int],
        channel_schedule: Sequence[int] | None = None,
        kernel_sizes: Sequence[int | tuple[int, int]] | None = None,
        strides: Sequence[int | tuple[int, int]] | None = None,
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__(input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out, **kwargs)
        latent_dim = int(math.prod(self.input_sizes[keys_in[0]]))
        self.decoder_shape = tuple(int(v) for v in decoder_shape)
        self.output_shape = tuple(int(v) for v in output_shape)
        channel_schedule = list(channel_schedule or [16, 32, 64])

        n_layers = len(channel_schedule)
        kernel_sizes = list(kernel_sizes) if kernel_sizes else [3] * n_layers
        strides = list(strides) if strides else [2] * n_layers

        self.projection = nn.Linear(latent_dim, int(math.prod(self.decoder_shape)))

        decode_layers: list[nn.Module] = []
        current_channels = self.decoder_shape[0]
        reversed_schedule = list(reversed(channel_schedule[:-1]))
        reversed_kernels = list(reversed(kernel_sizes[:-1]))
        reversed_strides = list(reversed(strides[:-1]))
        for i, out_channels in enumerate(reversed_schedule):
            decode_layers.append(
                _up_block(
                    current_channels,
                    out_channels,
                    activation,
                    kernel_size=reversed_kernels[i],
                    scale_factor=reversed_strides[i],
                )
            )
            current_channels = out_channels
        self.decoder = nn.Sequential(*decode_layers)
        self.output_head = nn.Conv2d(
            current_channels, self.output_shape[0], kernel_size=3, padding=1
        )

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        decoded = self.projection(x).view(x.shape[0], *self.decoder_shape)
        decoded = self.decoder(decoded)
        decoded = F.interpolate(
            decoded, size=self.output_shape[-2:], mode="bilinear", align_corners=False
        )
        return self.output_head(decoded)
