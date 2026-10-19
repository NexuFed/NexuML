"""Base pipeline layer definitions for NexuML."""

from __future__ import annotations

import logging
from enum import Enum

import torch
from tensordict import TensorDict

from nexuml.core.storage import SharedStorage

logger = logging.getLogger("model." + __name__)

# Hooks
# https://lightning.ai/docs/pytorch/stable/common/hooks.html


class LightningMode(Enum):
    """Lightning trainer phase reported to pipeline layers."""

    NONE = "none"
    TRAINING = "training"
    VALIDATING = "validating"
    TESTING = "testing"
    PREDICTING = "predicting"


class PipelineLayer(torch.nn.Module):
    """Base class for all NexuML pipeline layers.

    Subclasses implement ``forward_tensor`` and optionally override lifecycle
    hooks (``on_fit_start``, ``on_fit_end``).  The base ``forward`` method
    routes each ``keys_in`` key through ``forward_tensor`` and writes results
    to the corresponding ``keys_out`` key in the output ``TensorDict``.
    """

    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str] | dict[str, str],
        keys_out: list[str],
        label_key: str | list[str] | None = None,
        label_in_x: bool = False,
        num_classes: int | None = None,
        output_sizes: dict[str, tuple] | None = None,
        shared_memory: SharedStorage | None = None,
        shared_outputs: list[str] | str | None = None,
        shared_inputs: list[str] | str | None = None,
        delay_epochs: int = 0,
        update_every_n_epochs: int = 1,
        **kwargs,
    ):
        super().__init__()

        self.keys_in = keys_in
        self.keys_out = keys_out
        self.label_key = label_key
        self.label_in_x = label_in_x
        self.num_classes = num_classes
        self.input_sizes = input_sizes
        self.output_sizes = output_sizes

        self.delay_epochs = delay_epochs
        self.update_every_n_epochs = update_every_n_epochs

        self.shared_memory = shared_memory
        self.shared_outputs = shared_outputs
        self.shared_inputs = shared_inputs

        self.epoch = 0
        self.fit_mode = False
        self.lightning_mode = LightningMode.NONE

    def merge_td(self, x: TensorDict | torch.Tensor) -> torch.Tensor:
        """Concatenate multiple keys_in tensors along the token (dim=1).

        Returns:
            Concatenated tensor from the TensorDict keys, or *x* unchanged
            when it is already a plain tensor.
        """
        keys = self.keys_in
        if isinstance(keys, dict):
            keys = list(keys.keys())
        if isinstance(x, TensorDict) and len(keys) > 1:
            return torch.cat([x[k] for k in keys], dim=1)  # ty: ignore[no-matching-overload]
        elif isinstance(x, TensorDict):
            return x[keys[0]]  # ty: ignore[invalid-return-type]
        return x

    def check_update(self) -> bool:
        if self.epoch < self.delay_epochs:
            return False

        # Only update once after delay_epochs
        if self.update_every_n_epochs <= 0:
            return self.epoch == self.delay_epochs

        return (self.epoch - self.delay_epochs) % self.update_every_n_epochs == 0

    def get_label(
        self,
        x: TensorDict | torch.Tensor,
        y: TensorDict | None,
    ) -> torch.Tensor | None:
        if self.label_in_x:
            return x[self.label_key] if isinstance(x, TensorDict) else None  # ty: ignore[invalid-return-type]
        else:
            return None if y is None else y[self.label_key]  # ty: ignore[invalid-return-type]

    def forward(
        self,
        x: TensorDict | torch.Tensor,
        y: TensorDict | None = None,
    ) -> tuple[TensorDict | torch.Tensor, TensorDict | None]:
        if not self.check_update():
            return x, y

        x_out = x

        keys_in = self.keys_in
        keys_out = self.keys_out
        if isinstance(keys_in, dict):
            keys_in_list: list[str] = list(keys_in.keys())
        else:
            keys_in_list = keys_in

        if not keys_out:
            keys_out = [None] * len(keys_in_list)

        for key_in, key_out in zip(keys_in_list, keys_out):
            x_mod = x[key_in] if isinstance(x, TensorDict) else x

            if self.label_key is not None:
                y_input = self.get_label(x, y)
                x_mod = self.forward_tensor(x_mod, y_input)  # ty: ignore[invalid-argument-type]
            else:
                x_mod = self.forward_tensor(x_mod)  # ty: ignore[invalid-argument-type]

            if key_out is None:
                pass
            elif isinstance(x_out, TensorDict):
                x_out[key_out] = x_mod
            else:
                x_out = x_mod

        return x_out, y

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

    def on_fit_start(self):
        self.fit_mode = True

    def on_fit_end(self):
        self.fit_mode = False

    def on_train_start(self):
        self.lightning_mode = LightningMode.TRAINING

    def on_train_epoch_end(self):
        self.epoch += 1

    def on_train_end(self):
        self.lightning_mode = LightningMode.NONE

    def on_validation_start(self):
        self.lightning_mode = LightningMode.VALIDATING

    def on_validation_end(self):
        self.lightning_mode = LightningMode.NONE

    def on_test_start(self):
        self.lightning_mode = LightningMode.TESTING

    def on_test_end(self):
        self.lightning_mode = LightningMode.NONE

    def on_predict_start(self):
        self.lightning_mode = LightningMode.PREDICTING

    def on_predict_end(self):
        self.lightning_mode = LightningMode.NONE
