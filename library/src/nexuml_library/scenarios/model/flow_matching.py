"""Flow Matching model fragment."""

from __future__ import annotations

from nexuml.core.types import LayerSpec, PipelineSpec
from nexuml_library.layers.generative.flow.linear_path import LinearFlowPath
from nexuml_library.layers.loss.flow_matching_loss import FlowMatchingLoss
from nexuml_library.layers.model.flow import Flow


def mlp_flow_matching(
    feature_key: str = "features",
    hidden_dims: list[int] | None = None,
) -> PipelineSpec:
    """Build the basic MLP Flow Matching pipeline.

    Returns:
        Linear path sampling, flow model, and Flow Matching loss pipeline.
    """
    return PipelineSpec(
        stages={
            "FlowPath": [
                LayerSpec(
                    component=LinearFlowPath(),
                    keys_in=[feature_key],
                    keys_out=["flow_state", "flow_time", "flow_target_velocity"],
                )
            ],
            "Flow": [
                LayerSpec(
                    component=Flow(
                        hidden_dims=hidden_dims if hidden_dims is not None else [128, 128]
                    ),
                    keys_in=["flow_state", "flow_time"],
                    keys_out=["predicted_velocity"],
                )
            ],
            "Loss": [
                LayerSpec(
                    component=FlowMatchingLoss(),
                    keys_in=["predicted_velocity", "flow_target_velocity"],
                    keys_out=["flow_matching_loss"],
                )
            ],
        }
    )
