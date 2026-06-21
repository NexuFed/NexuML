"""Pooling layers for token/sequence aggregation."""

from __future__ import annotations

from typing import Literal, cast

import torch

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.discovery import layer

#: Aliases for the canonical pooling types — accept the spec spelling
#: (e.g. ``"mean+std"``) alongside the existing concatenated form.
_POOLING_ALIASES: dict[str, str] = {
    "mean+std": "mean_std",
    "mean_plus_std": "mean_std",
}


@layer("TokenPool")
class TokenPool(PipelineLayer):
    """Pool tokens along a specified dimension.

    Args:
        dim: Dimension to pool over (negative indexing supported, e.g. -2 for time).
        pooling_type:
            - "mean": average over the pool dim
            - "max": max over the pool dim
            - "std": standard deviation over the pool dim
            - "mean_std" (alias: "mean+std", "mean_plus_std"): concatenate
              mean and std along the last axis
            - "cls": select the first index along the pool dim (CLS-style)
            - "multilayer": when the input is multi-layer hidden states
              (e.g. ``[B, L, N, D]`` from an encoder that exposes all
              hidden states), concatenate per-layer mean+std along the
              last axis, preserving the per-layer ordering.
        remove_dim: Whether to squeeze the pooled dimension.
        skip_first: Skip the first token (e.g. CLS token) before pooling.
    """

    def __init__(
        self,
        dim: int = -2,
        pooling_type: Literal["mean", "max", "std", "mean_std", "cls", "multilayer"] = "mean",
        remove_dim: bool = True,
        skip_first: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        # Resolve aliases to the canonical pooling-type spelling so the
        # forward_tensor dispatch stays a single switch.
        self.pooling_type = _POOLING_ALIASES.get(pooling_type, pooling_type)
        self.remove_dim = remove_dim
        self.skip_first = skip_first

        if self.pooling_type == "mean":
            self.pool = torch.nn.AdaptiveAvgPool1d(output_size=1)
        elif self.pooling_type == "max":
            self.pool = torch.nn.AdaptiveMaxPool1d(output_size=1)
        elif self.pooling_type in ("std", "mean_std", "cls", "multilayer"):
            self.pool = None
        else:
            raise ValueError(f"Unsupported pooling type: {pooling_type}")

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        ndim = x.ndim
        dim = self.dim if self.dim >= 0 else ndim + self.dim

        if self.pooling_type == "cls":
            return self._cls_pool(x, dim)

        if self.pooling_type == "multilayer":
            return self._multilayer_pool(x, dim)

        if self.skip_first and ndim >= 2:
            slices = [slice(None)] * ndim
            slices[dim] = slice(1, None)
            x = x[tuple(slices)]

        if self.pooling_type == "std":
            return x.std(dim=dim, keepdim=not self.remove_dim)

        if self.pooling_type == "mean_std":
            mean = x.mean(dim=dim, keepdim=True)
            std = x.std(dim=dim, keepdim=True)
            pooled = torch.cat([mean, std], dim=-1)
            if self.remove_dim:
                pooled = pooled.squeeze(dim)
            return pooled

        x = x.movedim(dim, -1)
        x = self.pool(x)  # ty: ignore[call-non-callable]
        if self.remove_dim:
            x = x.squeeze(-1)
        else:
            x = x.movedim(-1, dim)
        return x

    def _cls_pool(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Select the first index along ``dim`` (CLS-style pooling).

        Returns:
            torch.Tensor: Tensor with ``dim`` removed via index selection.

        Raises:
            ValueError: If ``x`` has fewer than 2 dimensions or the pool
                dimension is empty.
        """
        if x.ndim < 2:
            raise ValueError(f"TokenPool 'cls' requires ndim >= 2, got shape {tuple(x.shape)}")
        size = x.size(dim)
        if size < 1:
            raise ValueError(f"TokenPool 'cls' requires the pool dim to be non-empty, got {size}")
        return x.select(dim, 0)

    def _multilayer_pool(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Concat per-layer mean+std over the multi-layer hidden-states axis.

        For ``x`` of shape ``[B, L, N, D]`` (L = number of layers, N = time
        tokens) with ``dim`` pointing at L, this emits ``[B, 2 * L * D]``
        where each layer contributes a ``[B, 2D]`` block (mean+std over
        the N token axis) so per-layer ordering is preserved along the
        embedding axis. For 3-D ``[B, L, D]`` inputs (no token axis) each
        layer is concatenated as ``[B, D]`` to give ``[B, L * D]``.

        Returns:
            torch.Tensor: Concatenated per-layer statistics tensor.

        Raises:
            ValueError: If ``x`` has fewer than 3 dimensions or ``dim``
                points at the last axis.
        """
        if x.ndim < 3:
            raise ValueError(
                f"TokenPool 'multilayer' requires ndim >= 3 (multi-layer "
                f"hidden states), got shape {tuple(x.shape)}"
            )
        if dim >= x.ndim - 1 or dim < -1:
            raise ValueError(
                f"TokenPool 'multilayer' expects the pool dim to be a "
                f"layer axis (not the last axis), got dim={dim} for "
                f"ndim={x.ndim}."
            )
        # Move the layer axis to position 0 so we can iterate per layer.
        layer_dim = dim if dim >= 0 else x.ndim + dim
        x = x.movedim(layer_dim, 0)
        L = x.size(0)
        per_layer: list[torch.Tensor] = []
        for i in range(L):
            t = x[i]  # [..., D] where the axes preserved from the original
            # (other than the L axis and the last embedding axis) are
            # stacked here. For [B, L, N, D] inputs the per-layer tensor is
            # [B, N, D] and the "token" axis to mean+std over is N (which
            # is the second axis of t, i.e. position 1).
            if t.ndim >= 3:
                # [B, ..., N, D]: pool over the time-token axis (N) at
                # position 1 of t (the axis right after the batch).
                mean = t.mean(dim=1)
                std = t.std(dim=1)
                per_layer.append(torch.cat([mean, std], dim=-1))
            elif t.ndim == 2:
                # [B, D] (no inner token axis): concatenate as-is.
                per_layer.append(t)
            else:
                # t is [D], no meaningful structure — pass through.
                per_layer.append(t)
        # Concatenate the per-layer blocks along the last axis to produce
        # the multilayer feature preserving per-layer ordering.
        return torch.cat(per_layer, dim=-1)


@layer("AttentionPool")
class AttentionPool(PipelineLayer):
    """Learned attention pooling using a query token.

    Args:
        dim: Token dimension to pool over.
        remove_dim: Whether to remove the pooled dimension.
        n_heads: Number of attention heads.
    """

    def __init__(
        self,
        dim: int = -2,
        remove_dim: bool = True,
        n_heads: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.remove_dim = remove_dim
        keys_in: list[str] = cast(list[str], self.keys_in)
        self.E = self.input_sizes[keys_in[0]][-1]

        self.attention = torch.nn.MultiheadAttention(
            embed_dim=self.E, num_heads=n_heads, batch_first=True
        )
        self.q = torch.nn.Parameter(torch.zeros(1, 1, self.E))

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        B = x.size(0)
        q_batch = self.q.expand(B, -1, -1)
        pooled, _ = self.attention(q_batch, x, x)  # (B, 1, E)
        if self.remove_dim:
            pooled = pooled.movedim(self.dim, -1).squeeze(-1)
        return pooled


@layer("AttentiveStatisticsPool")
class AttentiveStatisticsPool(PipelineLayer):
    """Attentive Statistics Pooling (Okabe et al., 2018).

    Computes attention-weighted mean and std, then projects to input dim.

    Args:
        dim: Dimension to pool over.
        remove_dim: Whether to remove the pooled dimension.
        hidden_dim: Hidden dimension for the attention MLP.
    """

    def __init__(
        self,
        dim: int = -2,
        remove_dim: bool = True,
        hidden_dim: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.remove_dim = remove_dim
        keys_in: list[str] = cast(list[str], self.keys_in)
        self.E = self.input_sizes[keys_in[0]][-1]

        self.lin = torch.nn.Linear(self.E, hidden_dim, bias=True)
        self.v = torch.nn.Linear(hidden_dim, 1, bias=True)
        self.activation = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=-2)
        self.projection = torch.nn.Linear(self.E * 2, self.E, bias=True)

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, E)
        h = self.activation(self.lin(x))  # (B, T, hidden_dim)
        scores = self.v(h).squeeze(-1)  # (B, T)
        weights = self.softmax(scores.unsqueeze(-1)).transpose(-2, -1)  # (B, 1, T)

        mean = torch.bmm(weights, x).squeeze(1)  # (B, E)
        diff = x - mean.unsqueeze(1)
        var = torch.bmm(weights, diff**2).squeeze(1)
        std = torch.sqrt(var + 1e-9)

        pooled = self.projection(torch.cat([mean, std], dim=-1))  # (B, E)
        if not self.remove_dim:
            pooled = pooled.unsqueeze(self.dim)
        return pooled
