"""Multi-label mAP metric layer."""

from __future__ import annotations
from nexuml.core.discovery import layer

import torch
from typing import cast
from tensordict import TensorDict
from torchmetrics.classification import MultilabelAveragePrecision

from nexuml.core.base_layer import LightningMode, PipelineLayer


@layer("MultiLabelMAPMetrics")
class MultiLabelMAPMetrics(PipelineLayer):
    """Pipeline layer computing multi-label mean Average Precision.

    Consumes logits from keys_in[0] and multi-hot labels from y[label_key].
    Applies sigmoid to logits before updating the wrapped
    torchmetrics MultilabelAveragePrecision metric.
    """

    val_metric: MultilabelAveragePrecision
    test_metric: MultilabelAveragePrecision

    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        label_key: str = "class_logits",
        num_labels: int | None = None,
        **kwargs,
    ):
        inferred_num_labels = num_labels or input_sizes.get(keys_in[0], (None,))[-1]
        super().__init__(
            input_sizes=input_sizes,
            keys_in=keys_in,
            keys_out=keys_out,
            label_key=label_key,
            **kwargs,
        )
        self.num_labels = inferred_num_labels
        if self.num_labels is None:
            raise ValueError("MultiLabelMAPMetrics requires num_labels or inferable logits size.")

        self.val_metric = self._build_metric()
        self.test_metric = self._build_metric()

    def _build_metric(self) -> MultilabelAveragePrecision:
        return MultilabelAveragePrecision(num_labels=self.num_labels)

    def _active_metric(self) -> MultilabelAveragePrecision | None:
        if self.lightning_mode == LightningMode.VALIDATING:
            return self.val_metric
        if self.lightning_mode == LightningMode.TESTING:
            return self.test_metric
        return None

    def forward(
        self,
        x: TensorDict | torch.Tensor,
        y: TensorDict | None = None,
    ) -> tuple[TensorDict | torch.Tensor, TensorDict | None]:
        if not self.check_update():
            return x, y
        assert isinstance(x, TensorDict)

        # Missing labels or not in val/test: emit zero mAP without updating state.
        assert isinstance(self.label_key, str)
        if y is None or self.label_key not in y.keys():
            zero = torch.tensor(0.0, device=x.device if hasattr(x, "device") else None)
            for key in self.keys_out:
                x[key] = zero.expand(x.batch_size)
            return x, y

        metric = self._active_metric()
        if metric is None:
            return x, y

        logits = cast(torch.Tensor, x[cast(list[str], self.keys_in)[0]])
        assert isinstance(logits, torch.Tensor)
        labels = self.get_label(x, y)
        if labels is None:
            for key in self.keys_out:
                x[key] = torch.tensor(0.0).expand(x.batch_size)
            return x, y

        probs = torch.sigmoid(logits)
        # mAP: for each of the classes, compute Average Precision over
        # the batch (precision-recall curve area), then average across classes.
        # MultilabelAveragePrecision handles per-class thresholding internally.
        labels = labels.to(probs.device)
        metric.to(probs.device)
        metric.update(probs, labels.long())  # ty: ignore[invalid-argument-type]
        for key in self.keys_out:
            x[key] = metric.compute().detach().expand(x.batch_size)  # ty: ignore[missing-argument]
        return x, y

    def get_epoch_metrics(self, stage: str) -> dict[str, torch.Tensor]:
        metric = (
            self.val_metric if stage == "val" else self.test_metric if stage == "test" else None
        )
        if metric is None:
            return {}
        return {"mAP": metric.compute().detach()}  # ty: ignore[missing-argument]

    def on_validation_start(self):
        super().on_validation_start()
        self.val_metric.reset()

    def on_test_start(self):
        super().on_test_start()
        self.test_metric.reset()

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError
