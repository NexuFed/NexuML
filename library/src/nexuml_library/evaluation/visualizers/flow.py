"""Flow sample and trajectory visualizer."""

from __future__ import annotations

from typing import Any, cast

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
    """Visualize 2-D target, initial, generated, and trajectory samples."""

    initial_key: str = "flow_initial"
    generated_key: str = "flow_sample"
    trajectory_key: str = "flow_trajectory"
    max_samples: int = Field(default=512, gt=0)
    max_trajectories: int = Field(default=32, gt=0)

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
    ):
        self.feature_key = feature_key
        self.initial_key = initial_key
        self.generated_key = generated_key
        self.trajectory_key = trajectory_key
        self.max_trajectories = max_trajectories
        self._storage = ReservoirTensorDictBuffer(max_samples=max_samples)

    def eval_batch(self, x: TensorDict, y: TensorDict | None) -> None:
        required = [self.feature_key, self.initial_key, self.generated_key, self.trajectory_key]
        if any(key not in x.keys() for key in required):
            return
        target = cast(torch.Tensor, x[self.feature_key]).detach().cpu()
        initial = cast(torch.Tensor, x[self.initial_key]).detach().cpu()
        generated = cast(torch.Tensor, x[self.generated_key]).detach().cpu()
        trajectory = cast(torch.Tensor, x[self.trajectory_key]).detach().cpu()
        if target.ndim != 2 or target.shape[1] != 2:
            return
        self._storage.add_batch(
            TensorDict(
                {
                    "target": target,
                    "initial": initial,
                    "generated": generated,
                    "trajectory": trajectory,
                },
                batch_size=[target.shape[0]],
            )
        )

    def visualize(self, logger_obj: Any) -> None:  # ty: ignore[invalid-method-override]
        data = self._storage.get()
        if data is None:
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        target = cast(torch.Tensor, data["target"]).numpy()
        initial = cast(torch.Tensor, data["initial"]).numpy()
        generated = cast(torch.Tensor, data["generated"]).numpy()
        trajectory = cast(torch.Tensor, data["trajectory"]).numpy()

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

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(target[:, 0], target[:, 1], s=12, alpha=0.25, label="Target")
        for path in trajectory[: self.max_trajectories]:
            ax.plot(path[:, 0], path[:, 1], linewidth=1.0, alpha=0.65)
        ax.set_title("Flow Matching trajectories")
        apply_axis_style(ax)
        fig.tight_layout()
        log_figure(logger_obj, "eval/flow/trajectories", fig)
        plt.close(fig)

    def results(self) -> dict[str, float]:
        data = self._storage.get()
        return {} if data is None else {"n_visualized_samples": float(data.batch_size[0])}
