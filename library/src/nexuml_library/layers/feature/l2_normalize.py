"""L2-normalization pipeline layer.

A trivial stateless postproc: applies L2 normalization to the last
dimension of the input. Has no trainable state and no ``fit()``/``is_fitted``
contract — it is an ordinary pipeline composition primitive, not a
train-fitted postprocessor.

This is the canonical "normalize embeddings" step the DCASE 2026 P3
target pipeline uses. It is intentionally a separate layer from
:class:`FeaturePostproc` so the target scenario does not need a
train-fitted postproc / pre-materialization bridge contract.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.discovery import layer


@layer("L2Normalize")
class L2Normalize(PipelineLayer):
    """L2-normalize the last dimension of the input.

    Stateless — no trainable parameters and no fitted state. Equivalent
    to ``torch.nn.functional.normalize(x, p=2, dim=-1, eps=eps)``.

    Args:
        eps: Small epsilon added to the denominator to avoid division by
            zero. Defaults to ``1e-12`` (matches ``F.normalize`` default).
    """

    def __init__(self, eps: float = 1e-12, **kwargs) -> None:
        super().__init__(**kwargs)
        self.eps = float(eps)

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1, eps=self.eps)
