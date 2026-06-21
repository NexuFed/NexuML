"""Anomaly scoring layers."""

from __future__ import annotations
from nexuml.core.discovery import layer

import torch
from typing import cast
from tensordict import TensorDict

from nexuml.core.base_layer import PipelineLayer


@layer("AnomalyScore")
class AnomalyScore(PipelineLayer):
    """Compute a per-sample anomaly score from original and reconstructed tensors."""

    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        reduction: str = "mean",
        **kwargs,
    ):
        super().__init__(input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out, **kwargs)
        # TODO: More reduction should be supported. Look at /workspaces/prisma-baseline
        if reduction not in {"mean", "max"}:
            raise ValueError(f"Unsupported reduction {reduction!r}. Expected 'mean' or 'max'.")
        self.reduction = reduction

    def forward(
        self,
        x: "TensorDict",
        y: "TensorDict | None" = None,
    ) -> tuple["TensorDict", "TensorDict | None"]:  # ty: ignore[invalid-method-override]
        keys_in: list[str] = cast(list[str], self.keys_in)
        original = cast(torch.Tensor, x[keys_in[0]]).reshape(x.batch_size[0], -1)
        reconstructed = cast(torch.Tensor, x[keys_in[1]]).reshape(x.batch_size[0], -1)
        # Align lengths when encoder/decoder introduce rounding differences
        if original.shape[1] != reconstructed.shape[1]:
            min_len = min(original.shape[1], reconstructed.shape[1])
            original = original[:, :min_len]
            reconstructed = reconstructed[:, :min_len]
        error = (reconstructed - original).pow(2)
        if self.reduction == "max":
            score = error.max(dim=1).values
        else:
            score = error.mean(dim=1)
        x[self.keys_out[0]] = score
        return x, y

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError
