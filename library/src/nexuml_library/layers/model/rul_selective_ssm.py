"""Pure-PyTorch selective state-space block for industrial sequence models.

Implements content-dependent B, C and delta, learned diagonal continuous-time A,
causal depthwise convolution, a selective recurrence, SiLU gating and residual.
This is a readable unfused Mamba-style implementation, not the optimized CUDA
`mamba_ssm` implementation used for published throughput comparisons.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SelectiveSSMBlock(nn.Module):
    """Input-dependent selective SSM with linear sequence/state complexity."""

    def __init__(
        self,
        d_model: int,
        *,
        state_dim: int = 8,
        expand: int = 2,
        conv_kernel: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(d_model, state_dim, expand, conv_kernel) < 1:
            raise ValueError("Selective SSM dimensions and kernel must be positive")
        self.d_inner = d_model * expand
        self.state_dim = state_dim
        self.dt_rank = max(1, math.ceil(d_model / 16))
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.depthwise = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=conv_kernel,
            groups=self.d_inner,
            bias=True,
        )
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * state_dim, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        self.a_log = nn.Parameter(
            torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32))
            .repeat(self.d_inner, 1)
        )
        self.skip = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        self.conv_kernel = conv_kernel

    def forward(self, x: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        """Run causal input-dependent discretization and selective recurrence.

        Args:
            x: Sequence with shape [batch, time, d_model].

        Returns:
            Residually updated sequence with original dimensions.

        Raises:
            ValueError: If the sequence dimensionality is invalid.
        """
        if x.ndim != 3:
            raise ValueError("Selective SSM expects [batch, time, feature] input")
        residual = x
        u, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        u = self.depthwise(F.pad(u.transpose(1, 2), (self.conv_kernel - 1, 0)))
        u = F.silu(u.transpose(1, 2))
        dt_raw, b_values, c_values = torch.split(
            self.x_proj(u), [self.dt_rank, self.state_dim, self.state_dim], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt_raw))
        a = -torch.exp(self.a_log).to(dtype=u.dtype)
        state = u.new_zeros((u.shape[0], self.d_inner, self.state_dim))
        outputs: list[Tensor] = []
        for t in range(u.shape[1]):
            dt = delta[:, t]
            value = u[:, t]
            transition = torch.exp(dt.unsqueeze(-1) * a.unsqueeze(0))
            injected = dt.unsqueeze(-1) * value.unsqueeze(-1)
            state = transition * state + injected * b_values[:, t].unsqueeze(1)
            output = (state * c_values[:, t].unsqueeze(1)).sum(dim=-1)
            outputs.append(output + self.skip * value)
        sequence = torch.stack(outputs, dim=1)
        sequence = sequence * F.silu(gate)
        return residual + self.dropout(self.out_proj(sequence))
