"""Flow sample and trajectory visualizer."""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
import torch
from pydantic import Field
from tensordict import TensorDict

from nexuml.core.components import EvalAlgorithmDefinition, EvalBuildContext
from nexuml.core.discovery import eval_algorithm
from nexuml.evaluation.algorithm import EvalAlgorithm
from nexuml.evaluation.storage import ReservoirTensorDictBuffer
from nexuml_library.evaluation.visualizers._plotting import apply_axis_style, log_figure


@eval_algorithm("flow_visualizer")
class FlowVisualizer(EvalAlgorithmDefinition):
    """Visualize target, initial, generated, and optional trajectory samples."""

    initial_key: str = "flow_initial"
    generated_key: str = "flow_sample"
    trajectory_key: str = "flow_trajectory"
    max_samples: int = Field(default=512, gt=0)
    max_trajectories: int = Field(default=32, gt=0)
    image_range: Literal["zero_one", "minus_one_one"] = "zero_one"

    def build(self, context: EvalBuildContext) -> EvalAlgorithm:
        return _FlowVisualizerRuntime(
            feature_key=context.feature_key or "features",
            **self.model_dump(),
        )


class _FlowVisualizerRuntime(EvalAlgorithm):
    def __init__(
        self,
        feature_key: str,
        initial_key: str,
        generated_key: str,
        trajectory_key: str,
        max_samples: int,
        max_trajectories: int,
        image_range: Literal["zero_one", "minus_one_one"],
    ):
        self.feature_key = feature_key
        self.initial_key = initial_key
        self.generated_key = generated_key
        self.trajectory_key = trajectory_key
        self.max_trajectories = max_trajectories
        self.image_range = image_range
        self._storage = ReservoirTensorDictBuffer(max_samples=max_samples)

    def eval_batch(self, x: TensorDict, y: TensorDict | None) -> None:
        required = [self.feature_key, self.initial_key, self.generated_key]
        if any(key not in x.keys() for key in required):
            return

        target = cast(torch.Tensor, x[self.feature_key]).detach().cpu()
        payload = {
            "target": target,
            "initial": cast(torch.Tensor, x[self.initial_key]).detach().cpu(),
            "generated": cast(torch.Tensor, x[self.generated_key]).detach().cpu(),
        }
        if self.trajectory_key in x.keys():
            payload["trajectory"] = (
                cast(torch.Tensor, x[self.trajectory_key]).detach().cpu()
            )
        self._storage.add_batch(TensorDict(payload, batch_size=[target.shape[0]]))

    def visualize(self, logger_obj: Any) -> None:  # ty: ignore[invalid-method-override]
        data = self._storage.get()
        if data is None:
            return

        target = cast(torch.Tensor, data["target"])
        if target.ndim == 2 and target.shape[1] == 2:
            self._visualize_2d(logger_obj, data)
        elif target.ndim == 4 and target.shape[1] in {1, 3}:
            self._visualize_images(logger_obj, data)

    def _visualize_2d(self, logger_obj: Any, data: TensorDict) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        target = cast(torch.Tensor, data["target"]).numpy()
        initial = cast(torch.Tensor, data["initial"]).numpy()
        generated = cast(torch.Tensor, data["generated"]).numpy()

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(target[:, 0], target[:, 1], s=16, alpha=0.55, label="Target")
        ax.scatter(initial[:, 0], initial[:, 1], s=16, alpha=0.45, label="Initial")
        ax.scatter(generated[:, 0], generated[:, 1], s=16, alpha=0.65, label="Generated")
        ax.set_title("Flow Matching samples")
        ax.legend(frameon=False)
        apply_axis_style(ax)
        fig.tight_layout()
        log_figure(logger_obj, "eval/flow/samples", fig)
        plt.close(fig)

        if "trajectory" not in data.keys():
            return

        trajectory = cast(torch.Tensor, data["trajectory"]).numpy()
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(target[:, 0], target[:, 1], s=12, alpha=0.25, label="Target")
        for path in trajectory[: self.max_trajectories]:
            ax.plot(path[:, 0], path[:, 1], linewidth=1.0, alpha=0.65)
        ax.set_title("Flow Matching trajectories")
        apply_axis_style(ax)
        fig.tight_layout()
        log_figure(logger_obj, "eval/flow/trajectories", fig)
        plt.close(fig)

    def _visualize_images(self, logger_obj: Any, data: TensorDict) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        target = cast(torch.Tensor, data["target"])
        initial = cast(torch.Tensor, data["initial"])
        generated = cast(torch.Tensor, data["generated"])
        count = min(8, len(target))

        fig, axes = plt.subplots(3, count, figsize=(1.5 * count, 4.5), squeeze=False)
        generated = self._display_images(generated)
        target = self._display_images(target)
        initial = (initial.clamp(-2.0, 2.0) + 2.0) / 4.0
        rows = (
            ("Initial", initial),
            ("Generated", generated),
            ("Target", target),
        )
        for row, (label, values) in enumerate(rows):
            for column in range(count):
                image = values[column].squeeze(0).numpy()
                axes[row, column].imshow(
                    np.clip(image, 0.0, 1.0),
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                )
                axes[row, column].axis("off")
                if column == 0:
                    axes[row, column].set_title(label, loc="left", fontsize=9)

        fig.suptitle("Flow Matching samples")
        fig.tight_layout()
        log_figure(logger_obj, "eval/flow/images", fig)
        plt.close(fig)

    def _display_images(self, images: torch.Tensor) -> torch.Tensor:
        if self.image_range == "minus_one_one":
            return ((images + 1.0) / 2.0).clamp(0.0, 1.0)
        return images.clamp(0.0, 1.0)

    def results(self) -> dict[str, float]:
        data = self._storage.get()
        return {} if data is None else {"n_visualized_samples": float(data.batch_size[0])}
