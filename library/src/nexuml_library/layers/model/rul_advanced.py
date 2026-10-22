"""Paper-structured advanced prognostics models with NexuML tensor contracts.

DMM-JA: Niu et al., ICASSP 2026, DOI 10.1109/ICASSP55912.2026.11464170.
PatchFormer: Liu et al., J. Power Sources 631 (2025) 236187.
RUL-Mamba: Huang et al., J. Energy Storage 120 (2025) 116376.

The original PatchFormer/RUL-Mamba papers forecast battery *capacity* and
derive end-of-life crossing. Here the forecast head emits a scalar supervised
RUL for cross-model turbine comparisons; this is an architecture adaptation,
not a replication of their battery RUL scoring protocol. DMM-JA is likewise
adapted from its oil-temperature/load ETDataset formulation.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from pydantic import Field
from torch import Tensor, nn
from torch.nn import functional as F

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer
from nexuml_library.layers.model.rul_selective_ssm import SelectiveSSMBlock


class _JumpAwareFusion(nn.Module):
    """Dynamic two-modality fusion, multiscale convolution and jump gating."""

    def __init__(
        self, length: int, channels: int, width: int, dropout: float,
        jump_threshold: float,
    ) -> None:
        super().__init__()
        first = channels // 2
        self.first = first
        self.encode_a = nn.Linear(first, width)
        self.encode_b = nn.Linear(channels - first, width)
        self.conv3 = nn.Conv1d(width, width, 3, padding=1)
        self.conv7 = nn.Conv1d(width, width, 7, padding=3)
        self.conv11 = nn.Conv1d(width, width, 11, padding=10, dilation=2)
        self.merge = nn.Conv1d(3 * width, width, 1)
        self.bilstm = nn.LSTM(
            width, width // 2, num_layers=2, bidirectional=True,
            batch_first=True, dropout=dropout,
        )
        self.ssm = SelectiveSSMBlock(width, dropout=dropout)
        self.jumps = nn.Linear(width + 2, width)
        self.temporal_attention = nn.Linear(width, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(width, 1),
        )
        self.jump_threshold = jump_threshold
        self.length = length

    def forward(self, x: Tensor) -> Tensor:
        """Return one scalar RUL from multimodal sensor-history windows.

        Returns:
            Predictions with shape [batch, 1].
        """
        a = self.encode_a(x[..., :self.first])
        b = self.encode_b(x[..., self.first:])
        tokens = torch.stack((a, b), dim=2)
        scores = tokens @ tokens.transpose(-1, -2) / math.sqrt(a.shape[-1])
        weights = scores.softmax(dim=-1).mean(dim=2)
        fused = weights[..., 0:1] * a + weights[..., 1:2] * b
        y = fused.transpose(1, 2)
        scales = torch.cat(
            (self.conv3(y), self.conv7(y), self.conv11(y)), dim=1
        )
        y = F.silu(self.merge(scales)).transpose(1, 2) + fused
        recurrent, _ = self.bilstm(y)
        t = torch.arange(self.length, device=x.device, dtype=x.dtype)
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, recurrent.shape[-1], 2, device=x.device)
            / recurrent.shape[-1]
        )
        phase = t[:, None] * freqs[None, :]
        pe = torch.zeros_like(recurrent)
        pe[..., 0::2] = phase.sin()
        pe[..., 1::2] = phase.cos()[..., :pe[..., 1::2].shape[-1]]
        encoded = self.ssm(recurrent + pe)
        jumps = torch.zeros_like(x[..., :1])
        jumps[:, 1:, 0] = (
            (x[:, 1:] - x[:, :-1]).square().mean(dim=-1).sqrt()
            > self.jump_threshold
        ).to(x.dtype)
        time = (t / max(self.length - 1, 1)).view(1, -1, 1)
        gate_input = torch.cat(
            (encoded, time.expand(x.shape[0], -1, -1), jumps), dim=-1
        )
        corrected = encoded * (1.0 - self.jumps(gate_input).sigmoid())
        attention = self.temporal_attention(corrected).softmax(dim=1)
        return self.head((corrected * attention).sum(dim=1))


class _PatchCSE(nn.Module):
    """Convolutional squeeze/excitation over patch or intra-patch tokens."""

    def __init__(self, count: int) -> None:
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(count, max(1, count // 2)),
            nn.ReLU(), nn.Linear(max(1, count // 2), count), nn.Sigmoid(),
        )

    def forward(self, pooled: Tensor) -> Tensor:
        """Normalize learned token excitations with softmax.

        Returns:
            Attention weights over the final dimension.
        """
        return self.bottleneck(pooled).softmax(dim=-1)


class _DualPatchAttention(nn.Module):
    """Intra-patch token selection plus inter-patch CSE aggregation."""

    def __init__(self, length: int, width: int, patch_size: int) -> None:
        super().__init__()
        self.size = patch_size
        self.padding = (-length) % patch_size
        self.patches = (length + self.padding) // patch_size
        self.local_cse = _PatchCSE(patch_size + 1)
        self.global_cse = _PatchCSE(self.patches)
        self.context = nn.Parameter(torch.zeros(1, 1, 1, width))
        self.fusion = nn.Sequential(nn.Linear(2 * width, 2 * width), nn.GELU())

    def forward(self, x: Tensor) -> Tensor:
        """Fuse short-range details and global patch correlations.

        Returns:
            Feature-wise representation with shape [batch, length, 2*width].
        """
        if self.padding:
            x = torch.cat((x, x[:, -1:].expand(-1, self.padding, -1)), dim=1)
        batch, length, width = x.shape
        patches = x.reshape(batch, self.patches, self.size, width)
        context = self.context.expand(batch, self.patches, -1, -1)
        augmented = torch.cat((context, patches), dim=2)
        local_weights = self.local_cse(augmented.mean(dim=-1))
        local = (augmented * local_weights.unsqueeze(-1)).sum(dim=2)
        local = local.unsqueeze(2).expand(-1, -1, self.size, -1)
        inter_weights = self.global_cse(patches.mean(dim=(2, 3)))
        global_features = patches * inter_weights[:, :, None, None]
        fused = self.fusion(torch.cat((local, global_features), dim=-1))
        return fused.reshape(batch, length, 2 * width)[:, :length - self.padding]


class _FeatureAttention(nn.Module):
    """PatchFormer's feature-wise self-attention over temporal embeddings."""

    def __init__(self, length: int, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        if length % heads:
            raise ValueError("PatchFormer FAN heads must divide the time length")
        self.attention = nn.MultiheadAttention(
            length, heads, dropout=dropout, batch_first=True
        )
        self.norm_a = nn.LayerNorm(length)
        self.norm_b = nn.LayerNorm(length)
        self.ffn = nn.Sequential(
            nn.Linear(length, 2 * length), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2 * length, length),
        )
        self.out = nn.Sequential(nn.Flatten(), nn.Linear(width * length, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Mix feature tokens with temporal projections and predict scalar RUL.

        Returns:
            RUL with shape [batch, 1].
        """
        features = x.transpose(1, 2)
        attended, _ = self.attention(features, features, features, need_weights=False)
        features = self.norm_a(features + attended)
        return self.out(self.norm_b(features + self.ffn(features)))


class _PatchFormerModel(nn.Module):
    """Variable embedding, DPAN, FAN and a one-step scalar regression head."""

    def __init__(
        self, length: int, channels: int, width: int, patch_size: int,
        heads: int, dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(channels, width)
        self.dpan = _DualPatchAttention(length, width, patch_size)
        self.fan = _FeatureAttention(length, 2 * width, heads, dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Regress RUL from dual patch-wise/feature-wise attention.

        Returns:
            Predictions with shape [batch, 1].
        """
        return self.fan(self.dpan(self.embedding(x)))


class _FeatureSE(nn.Module):
    """RUL-Mamba feature attention over learned variable embeddings."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.weights = nn.Sequential(
            nn.Linear(width, max(1, width // 2)), nn.ReLU(),
            nn.Linear(max(1, width // 2), width), nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Select useful learned feature channels before Mamba encoding.

        Returns:
            Weighted [batch,time,variable,hidden] tensor.
        """
        pooled = x.mean(dim=(1, 2))
        return x * self.weights(pooled)[:, None, None, :]


class _GatedResidual(nn.Module):
    """LayerNorm(a+GLU(Linear(ELU(Linear([a,context])))))."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(2 * width, 2 * width), nn.ELU(),
            nn.Linear(2 * width, 2 * width), nn.Dropout(dropout),
        )
        self.glu = nn.GLU(dim=-1)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        """Apply context-conditioned nonlinear gated residual correction.

        Returns:
            Gated decoder features with same shape as the primary input.
        """
        joined = torch.cat((x, context.expand_as(x)), dim=-1)
        return self.norm(x + self.glu(self.body(joined)))


class _RULMambaModel(nn.Module):
    """Variable embedding, feature attention, encoder/decoder SSM and GRN."""

    def __init__(
        self, channels: int, width: int, depth: int, dropout: float,
    ) -> None:
        super().__init__()
        self.variable_embed = nn.ModuleList(
            nn.Linear(1, width) for _ in range(channels)
        )
        self.fan = _FeatureSE(width)
        self.encoder = SelectiveSSMBlock(width, dropout=dropout)
        self.decoder = nn.ModuleList(
            SelectiveSSMBlock(width, dropout=dropout) for _ in range(depth)
        )
        self.grn = nn.ModuleList(_GatedResidual(width, dropout) for _ in range(depth))
        self.head = nn.Linear(width, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Regress scalar RUL using the FAN-Mamba-GRN encoder-decoder.

        Returns:
            Predictions with shape [batch, 1].
        """
        embedded = torch.stack(
            [layer(x[..., i:i + 1]) for i, layer in enumerate(self.variable_embed)],
            dim=2,
        )
        encoded = self.encoder(self.fan(embedded).sum(dim=2))
        context = encoded[:, -1:, :]
        decoded = context
        for block, gate in zip(self.decoder, self.grn):
            decoded = gate(block(decoded), context)
        return self.head(decoded[:, -1])


class _AdvancedRULRuntime(PipelineLayer):
    """Native NexuML [batch,time,features] to [batch,1] model adapter."""

    def __init__(
        self, input_sizes: dict[str, tuple], keys_in: list[str], keys_out: list[str],
        architecture: str, hidden_dim: int, dropout: float, patch_size: int,
        attention_heads: int, decoder_depth: int, jump_threshold: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out, **kwargs
        )
        length, channels = input_sizes[keys_in[0]]
        if length < 2 or channels < 1:
            raise ValueError("Advanced backbones require [length>=2,channels>=1]")
        if architecture == "dmm_ja":
            if channels < 2:
                raise ValueError("DMM-JA dynamic fusion requires at least two modalities")
            if hidden_dim % 2:
                raise ValueError("DMM-JA bidirectional LSTM requires even hidden_dim")
            self.backbone = _JumpAwareFusion(
                length, channels, hidden_dim, dropout, jump_threshold
            )
        elif architecture == "patchformer":
            self.backbone = _PatchFormerModel(
                length, channels, hidden_dim, patch_size, attention_heads, dropout
            )
        elif architecture == "rul_mamba":
            self.backbone = _RULMambaModel(
                channels, hidden_dim, decoder_depth, dropout
            )
        else:
            raise ValueError("Unknown advanced RUL architecture")

    def forward_tensor(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Forecast the one-step scalar RUL target using the selected backbone.

        Returns:
            Batch-aligned [batch,1] remaining useful life estimates.

        Raises:
            ValueError: If input tensor dimensions are incorrect.
        """
        if x.ndim != 3:
            raise ValueError("Advanced RUL backbone requires a [B,L,C] tensor")
        return self.backbone(x)


@layer("AdvancedRULForecaster")
class AdvancedRULForecaster(LayerDefinition):
    """Select DMM-JA, PatchFormer or RUL-Mamba architecture adaptations."""

    architecture: str = Field(pattern="^(dmm_ja|patchformer|rul_mamba)$")
    hidden_dim: int = Field(default=64, gt=0)
    dropout: float = Field(default=0.15, ge=0, lt=1)
    patch_size: int = Field(default=5, gt=0)
    attention_heads: int = Field(default=2, gt=0)
    decoder_depth: int = Field(default=1, gt=0)
    jump_threshold: float = Field(default=2.5, gt=0)
    initialization_seed: int = 42

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        """Build a deterministic paper-structured temporal prognostics layer.

        Returns:
            Regular NexuML PipelineLayer with scalar RUL output.
        """
        params = self.model_dump()
        seed = int(params.pop("initialization_seed"))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return _AdvancedRULRuntime(**context.runtime_kwargs(), **params)
