"""Regression loss layer."""

from __future__ import annotations
from nexuml.core.discovery import layer

import torch
import torch.nn as nn
from typing import cast
from tensordict import TensorDict

from nexuml.core.base_layer import PipelineLayer


@layer("RegressionLoss")
class RegressionLoss(PipelineLayer):
    """Computes MSE regression loss between predictions and targets.

    keys_in: [predictions_key]
    keys_out: [loss_key]
    Reads targets from y TensorDict using label_key.
    """

    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        label_key: str = "regression_targets",
        **kwargs,
    ):
        super().__init__(
            input_sizes=input_sizes,
            keys_in=keys_in,
            keys_out=keys_out,
            label_key=label_key,
            **kwargs,
        )
        self.criterion = nn.MSELoss()

    def forward(
        self,
        x: "TensorDict",
        y: "TensorDict | None" = None,
    ) -> tuple["TensorDict", "TensorDict | None"]:  # ty: ignore[invalid-method-override]
        predictions = cast(torch.Tensor, x[cast(list[str], self.keys_in)[0]])
        target = self.get_label(x, y)

        if target is None:
            zero = torch.tensor(0.0, device=predictions.device, requires_grad=True)
            x[self.keys_out[0]] = zero.expand(x.batch_size)
        else:
            loss = self.criterion(predictions, target.float())
            x[self.keys_out[0]] = loss.expand(x.batch_size)

        return x, y

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError
