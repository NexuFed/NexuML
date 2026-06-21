"""Class histogram visualizer."""

from __future__ import annotations
from nexuml.core.discovery import eval_algorithm

import logging
from collections import Counter
from typing import Any

import numpy as np
from tensordict import TensorDict

from nexuml.evaluation.algorithm import EvalAlgorithm
from nexuml_library.evaluation.visualizers._plotting import (
    apply_axis_style,
    format_label,
    log_figure,
)

logger = logging.getLogger(__name__)


@eval_algorithm("class_histogram")
class ClassHistogramVisualizer(EvalAlgorithm):
    """Bar chart of class label distribution in train and test sets."""

    type_key = "class_histogram"

    def __init__(self, label_key: str = "y_true", title: str | None = None) -> None:
        self.label_key = label_key
        self.title = title
        self._train_counts: Counter = Counter()
        self._test_counts: Counter = Counter()

    def fit_batch(self, x: TensorDict, y: TensorDict | None) -> None:
        if y is not None and self.label_key in y.keys():
            for lbl in y[self.label_key].cpu().flatten().tolist():
                self._train_counts[int(lbl)] += 1

    def fit_end(self) -> None:
        pass

    def eval_batch(self, x: TensorDict, y: TensorDict | None) -> None:
        if y is not None and self.label_key in y.keys():
            for lbl in y[self.label_key].cpu().flatten().tolist():
                self._test_counts[int(lbl)] += 1

    def eval_end(self) -> None:
        pass

    def results(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        all_counts = self._train_counts + self._test_counts
        if not all_counts:
            return metrics
        metrics["n_classes"] = float(len(all_counts))
        metrics["train_samples"] = float(sum(self._train_counts.values()))
        metrics["test_samples"] = float(sum(self._test_counts.values()))
        counts = list(all_counts.values())
        if len(counts) > 1:
            metrics["imbalance_ratio"] = float(max(counts)) / float(min(counts))
        return metrics

    def visualize(self, logger_obj: Any) -> None:  # ty: ignore[invalid-method-override]
        if not self._train_counts and not self._test_counts:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            all_labels = sorted(
                set(list(self._train_counts.keys()) + list(self._test_counts.keys()))
            )
            x = np.arange(len(all_labels))
            width = 0.35

            train_vals = [self._train_counts.get(label, 0) for label in all_labels]
            test_vals = [self._test_counts.get(label, 0) for label in all_labels]
            train_total = max(sum(train_vals), 1)
            test_total = max(sum(test_vals), 1)

            fig, axes = plt.subplots(
                1,
                2,
                figsize=(max(10, len(all_labels) * 1.1), 4.8),
                gridspec_kw={"width_ratios": [2.2, 1.2]},
            )
            count_ax, ratio_ax = axes

            count_ax.bar(
                x - width / 2, train_vals, width, label="Train", alpha=0.85, color="#4c72b0"
            )
            count_ax.bar(x + width / 2, test_vals, width, label="Test", alpha=0.85, color="#dd8452")
            count_ax.set_xticks(x)
            count_ax.set_xticklabels(
                [format_label(label) for label in all_labels],
                rotation=45,
                ha="right",
            )
            count_ax.set_xlabel(self.label_key)
            count_ax.set_ylabel("Count")
            count_ax.legend(frameon=False)
            apply_axis_style(count_ax)

            train_ratio = np.asarray(train_vals, dtype=float) / train_total
            test_ratio = np.asarray(test_vals, dtype=float) / test_total
            ratio_ax.barh(
                x + width / 2,
                train_ratio,
                height=width,
                label="Train",
                alpha=0.85,
                color="#4c72b0",
            )
            ratio_ax.barh(
                x - width / 2,
                test_ratio,
                height=width,
                label="Test",
                alpha=0.85,
                color="#dd8452",
            )
            ratio_ax.set_yticks(x)
            ratio_ax.set_yticklabels([format_label(label) for label in all_labels])
            ratio_ax.set_xlabel("Share")
            apply_axis_style(ratio_ax)

            fig.suptitle(self.title or f"{self.label_key} Distribution", y=1.02)
            fig.tight_layout()
            log_figure(logger_obj, f"eval/class_histogram/{self.label_key}", fig)
            plt.close(fig)
        except Exception as e:
            logger.warning(f"ClassHistogramVisualizer.visualize() failed: {e}")
