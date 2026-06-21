"""Classification loss layer."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import cast
from tensordict import TensorDict

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.discovery import layer


@layer("ClassificationLoss")
class ClassificationLoss(PipelineLayer):
    """Computes classification loss from logits and labels.

    keys_in: [logits_key]
    keys_out: [loss_key]
    Reads labels from y TensorDict using label_key.
    """

    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        loss_type: str = "cross_entropy",
        label_key: str = "class_labels",
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            input_sizes=input_sizes,
            keys_in=keys_in,
            keys_out=keys_out,
            label_key=label_key,
            **kwargs,
        )
        if loss_type in ("bce", "binary_cross_entropy"):
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing

    def forward(
        self,
        x: TensorDict | torch.Tensor,
        y: TensorDict | None = None,
    ) -> tuple[TensorDict | torch.Tensor, TensorDict | None]:
        assert isinstance(x, TensorDict)
        logits = cast(torch.Tensor, x[cast(list[str], self.keys_in)[0]])
        label = self.get_label(x, y)

        if label is None:
            # No labels available — output zero loss
            zero = torch.tensor(0.0, device=logits.device, requires_grad=True)
            x[self.keys_out[0]] = zero.expand(x.batch_size)
        else:
            if self.loss_type in ("bce", "binary_cross_entropy"):
                label = label.float()
            else:
                label = label.long()
            loss = self.criterion(logits, label)
            x[self.keys_out[0]] = loss.expand(x.batch_size)

        return x, y

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


@layer("LogitsToClass")
class LogitsToClass(PipelineLayer):
    """Converts class logits to predicted class integers via argmax."""

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return x.argmax(dim=-1).float()
