"""Classification metrics layer.

Supports both multiclass and multilabel classification metrics via the
``multi_label`` flag.  In multilabel mode, sigmoid is applied to logits
before updating metrics and labels are kept as multi-hot vectors.
"""

from __future__ import annotations
from typing import Literal, cast

from nexuml.core.discovery import layer

import torch
import torch.nn as nn
from tensordict import TensorDict

from nexuml.core.base_layer import LightningMode, PipelineLayer


@layer("ClassificationMetrics")
class ClassificationMetrics(PipelineLayer):
    """Accumulate classification metrics with torchmetrics.

    When ``multi_label`` is ``False`` (default) the layer uses multiclass
    metrics (``MulticlassAccuracy``, ``MulticlassF1Score``,
    ``MulticlassAveragePrecision``).  Labels are expected as integer class
    indices with shape ``[batch]``.

    When ``multi_label`` is ``True`` the layer uses multilabel metrics
    (``MultilabelAccuracy``, ``MultilabelF1Score``,
    ``MultilabelAveragePrecision``).  Sigmoid is applied to logits before
    passing them to the metrics and labels are expected as multi-hot float
    tensors with shape ``[batch, num_labels]``.
    """

    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        label_key: str = "class_labels",
        metrics: list[str] | None = None,
        average: str = "macro",
        top_k: int = 1,
        num_classes: int | None = None,
        multi_label: bool = False,
        **kwargs,
    ):
        inferred_num_classes = num_classes or input_sizes.get(keys_in[0], (None,))[-1]
        super().__init__(
            input_sizes=input_sizes,
            keys_in=keys_in,
            keys_out=keys_out,
            label_key=label_key,
            num_classes=inferred_num_classes,
            **kwargs,
        )
        if self.num_classes is None:
            raise ValueError("ClassificationMetrics requires num_classes or inferable logits size.")

        self.multi_label = multi_label
        self.metric_names = metrics or list(keys_out)
        self.average: str = average
        self.top_k = top_k
        self.val_metrics = self._build_metrics()
        self.test_metrics = self._build_metrics()

    def _build_metrics(self) -> nn.ModuleDict:
        metrics = nn.ModuleDict()
        for name in self.metric_names:
            lowered = name.lower()
            if self.multi_label:
                metrics[name] = self._build_multilabel_metric(lowered, name)
            else:
                metrics[name] = self._build_multiclass_metric(lowered, name)
        return metrics

    def _build_multiclass_metric(self, lowered: str, name: str) -> nn.Module:
        from torchmetrics.classification import (
            MulticlassAccuracy,
            MulticlassAveragePrecision,
            MulticlassF1Score,
        )

        num_classes = self.num_classes
        assert num_classes is not None
        average = cast(Literal["micro", "macro", "weighted", "none"], self.average)

        if lowered == "accuracy":
            return MulticlassAccuracy(num_classes=num_classes, top_k=self.top_k)
        if lowered == "f1":
            return MulticlassF1Score(num_classes=num_classes, average=average)
        if lowered in ("map", "average_precision"):
            ap_average = cast(Literal["macro", "weighted", "none"], self.average)
            return MulticlassAveragePrecision(num_classes=num_classes, average=ap_average)
        raise ValueError(f"Unsupported classification metric '{name}'.")

    def _build_multilabel_metric(self, lowered: str, name: str) -> nn.Module:
        from torchmetrics.classification import (
            MultilabelAccuracy,
            MultilabelAveragePrecision,
            MultilabelF1Score,
        )

        num_labels = self.num_classes
        assert num_labels is not None
        average = cast(Literal["micro", "macro", "weighted", "none"], self.average)

        if lowered == "accuracy":
            return MultilabelAccuracy(num_labels=num_labels, average=average)
        if lowered == "f1":
            return MultilabelF1Score(num_labels=num_labels, average=average)
        if lowered in ("map", "average_precision"):
            return MultilabelAveragePrecision(num_labels=num_labels)
        raise ValueError(f"Unsupported classification metric '{name}'.")

    def _active_metrics(self) -> nn.ModuleDict | None:
        if self.lightning_mode == LightningMode.VALIDATING:
            return self.val_metrics
        if self.lightning_mode == LightningMode.TESTING:
            return self.test_metrics
        return None

    def forward(
        self,
        x: TensorDict | torch.Tensor,
        y: TensorDict | None = None,
    ) -> tuple[TensorDict | torch.Tensor, TensorDict | None]:
        if not self.check_update():
            return x, y
        assert isinstance(x, TensorDict)

        # Shape propagation / missing labels: emit placeholder tensors.
        if y is None or self.label_key not in y.keys():  # ty: ignore[unsupported-operator]
            zero = torch.tensor(0.0, device=x.device if hasattr(x, "device") else None)
            for key in self.keys_out:
                x[key] = zero.expand(x.batch_size)
            return x, y

        metrics = self._active_metrics()
        if metrics is None:
            return x, y

        logits = cast(torch.Tensor, x[cast(list[str], self.keys_in)[0]])
        labels = self.get_label(x, y)
        if labels is None:
            return x, y

        if self.multi_label:
            # Multi-label: apply sigmoid, keep labels as multi-hot.
            preds = torch.sigmoid(logits)
            labels = labels.long()
            if preds.ndim > 2:
                preds = preds.reshape(-1, preds.shape[-1])
        else:
            # Multiclass: raw logits, labels as 1-D class indices.
            preds = logits
            if preds.ndim > 2:
                preds = preds.reshape(-1, preds.shape[-1])
            labels = labels.long().reshape(-1)

        if preds.shape[0] != labels.shape[0]:
            return x, y

        labels = labels.to(preds.device)
        for key, metric in metrics.items():
            metric.to(preds.device)
            metric.update(preds, labels)  # ty: ignore[call-non-callable]
            x[key] = metric.compute().detach().expand(x.batch_size)  # ty: ignore[call-non-callable]
        return x, y

    def get_epoch_metrics(self, stage: str) -> dict[str, torch.Tensor]:
        metrics = (
            self.val_metrics if stage == "val" else self.test_metrics if stage == "test" else None
        )
        if metrics is None:
            return {}
        return {key: metric.compute().detach() for key, metric in metrics.items()}  # ty: ignore[call-non-callable]

    def on_validation_start(self):
        super().on_validation_start()
        for metric in self.val_metrics.values():
            metric.reset()  # ty: ignore[call-non-callable]

    def on_test_start(self):
        super().on_test_start()
        for metric in self.test_metrics.values():
            metric.reset()  # ty: ignore[call-non-callable]

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError
