"""Flow Matching model scenario fragments."""

from __future__ import annotations

from nexuml.core.types import LayerSpec, PipelineSpec
from nexuml_library.layers.flow.flow_matching_loss import FlowMatchingLoss
from nexuml_library.layers.flow.linear_path import LinearFlowPath
from nexuml_library.layers.flow.vector_field import TimeConditionedVectorField


def mlp_flow_matching(
    feature_key: str = "features",
    hidden_dims: list[int] | None = None,
    time_embedding_dim: int = 16,
    activation: str = "torch.nn.SiLU",
    noise_std: float = 1.0,
) -> PipelineSpec:
    """Build a modality-agnostic MLP Flow Matching pipeline.

    Returns:
        PipelineSpec: Linear path sampling, time-conditioned vector field,
            and Flow Matching loss.
    """
    return PipelineSpec(
        stages={
            "FlowPath": [
                LayerSpec(
                    component=LinearFlowPath(noise_std=noise_std),
                    keys_in=[feature_key],
                    keys_out=["flow_state", "flow_time", "flow_target_velocity"],
                )
            ],
            "VectorField": [
                LayerSpec(
                    component=TimeConditionedVectorField(
                        hidden_dims=hidden_dims if hidden_dims is not None else [128, 128],
                        time_embedding_dim=time_embedding_dim,
                        activation=activation,
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
