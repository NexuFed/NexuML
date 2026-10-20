"""Healthy-reference-conditioned vector field and Euler flow for NEX-233.

Machine context is a trainable permutation-invariant DeepSets embedding of
healthy reference observations. Neither machine ID nor faulty observations
enter the context encoder. The interpolation path and loss are reused from
the existing NexuML library.
"""

from __future__ import annotations

from typing import cast

import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor, nn

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer


class HealthySetEncoder(nn.Module):
    """Permutation-invariant mean-pooling encoder for reference distributions."""

    def __init__(self, feature_dim: int, hidden_dim: int, context_dim: int) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, context_dim),
        )
        self.rho = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
        )

    def forward(self, reference: Tensor) -> Tensor:
        """Encode [B,K,D] healthy reference embeddings to [B,C].

        Returns:
            Permutation-invariant continuous machine context.

        Raises:
            ValueError: If reference batch or set dimensions are invalid.
        """
        if reference.ndim != 3 or reference.shape[1] == 0:
            raise ValueError("Healthy reference must have shape [B,K,D], K > 0")
        return self.rho(self.phi(reference).mean(dim=1))


@layer("HealthyReferenceConditionedFlow")
class HealthyReferenceConditionedFlow(LayerDefinition):
    """Flow Matching velocity conditioned on a healthy reference distribution.

    The optional constant-context control uses an identically shaped velocity
    backbone but removes reference information. It is a context ablation, not
    an exact parameter-matched copy of the original unconditional flow.
    """

    hidden_dims: list[int] = Field(default_factory=lambda: [128, 128])
    context_dim: int = Field(default=32, gt=0)
    encoder_hidden_dim: int = Field(default=128, gt=0)
    num_steps: int = Field(default=50, gt=0)
    context_mode: str = Field(default="reference", pattern="^(reference|constant)$")

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        """Build DeepSets and a conditional vector field using typed routing.

        Returns:
            NexuML runtime layer with velocity and conditional Euler integration.

        Raises:
            ValueError: If keys, dimensions or hidden widths are incompatible.
        """
        if len(context.keys_in) != 3 or len(context.keys_out) != 1:
            raise ValueError("Conditional flow needs state, time, reference and one output")
        state, time, reference = context.keys_in
        state_shape = context.input_sizes[state]
        reference_shape = context.input_sizes[reference]
        if (
            len(state_shape) != 1
            or len(reference_shape) != 2
            or reference_shape[0] <= 0
            or reference_shape[1] != state_shape[0]
            or tuple(context.input_sizes[time]) != (1,)
        ):
            raise ValueError("Expected state [D], time [1], reference [K,D]")
        if not self.hidden_dims or any(width <= 0 for width in self.hidden_dims):
            raise ValueError("All vector-field hidden widths must be positive")
        dimension = int(state_shape[0])
        layers: list[nn.Module] = []
        widths = [dimension + 1 + self.context_dim, *self.hidden_dims, dimension]
        for index, (input_dim, output_dim) in enumerate(zip(widths, widths[1:])):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(widths) - 2:
                layers.append(nn.SiLU())
        encoder = HealthySetEncoder(
            feature_dim=dimension,
            hidden_dim=self.encoder_hidden_dim,
            context_dim=self.context_dim,
        )
        return _HealthyReferenceConditionedFlowRuntime(
            encoder=encoder,
            backbone=nn.Sequential(*layers),
            feature_dim=dimension,
            context_dim=self.context_dim,
            num_steps=self.num_steps,
            context_mode=self.context_mode,
            **context.runtime_kwargs(),
        )


class _HealthyReferenceConditionedFlowRuntime(PipelineLayer):
    """Trainable conditional flow with no mutable cross-batch context cache."""

    def __init__(
        self,
        encoder: HealthySetEncoder,
        backbone: nn.Sequential,
        feature_dim: int,
        context_dim: int,
        num_steps: int,
        context_mode: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.encoder = encoder
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.num_steps = num_steps
        self.context_mode = context_mode

    def encode_context(self, reference: Tensor, *, batch_size: int) -> Tensor:
        """Compute context once per integration, broadcasting a common set.

        Returns:
            Tensor of shape [B,C] on the reference device.

        Raises:
            ValueError: If reference dimensions mismatch the trained model.
        """
        if reference.ndim == 2:
            reference = reference.unsqueeze(0).expand(batch_size, -1, -1)
        if (
            reference.ndim != 3
            or reference.shape[0] != batch_size
            or reference.shape[1] == 0
            or reference.shape[2] != self.feature_dim
        ):
            raise ValueError("Reference must be [K,D] or [B,K,D] with matching D")
        if self.context_mode == "constant":
            return reference.new_zeros((batch_size, self.context_dim))
        return self.encoder(reference)

    def _velocity_with_context(self, state: Tensor, time: Tensor, context: Tensor) -> Tensor:
        if state.ndim != 2 or state.shape[1] != self.feature_dim:
            raise ValueError("Flow state must be [B,D] with matching feature dimension")
        if time.shape != (len(state), 1) or context.shape != (
            len(state),
            self.context_dim,
        ):
            raise ValueError("Time/context batch shapes are incompatible with flow state")
        prediction = self.backbone(torch.cat((state, time, context), dim=-1))
        if prediction.shape != state.shape:
            raise ValueError("Conditional vector field changed the state dimension")
        return prediction

    def velocity(self, state: Tensor, time: Tensor, reference: Tensor) -> Tensor:
        """Evaluate conditional velocity without integrating an entire path.

        Returns:
            Predicted velocity [B,D].
        """
        machine_context = self.encode_context(reference, batch_size=len(state))
        return self._velocity_with_context(state, time, machine_context)

    def sample(
        self, initial: Tensor, reference: Tensor, *, record_trajectory: bool = False
    ) -> Tensor:
        """Euler-integrate healthy samples with context fixed along the path.

        Returns:
            Final samples [B,D], or complete trajectory [B,steps+1,D].
        """
        machine_context = self.encode_context(reference, batch_size=len(initial))
        state = initial
        trajectory = [state] if record_trajectory else None
        dt = 1.0 / self.num_steps
        for step in range(self.num_steps):
            time = state.new_full((len(state), 1), step * dt)
            velocity = self._velocity_with_context(state, time, machine_context)
            state = state + dt * velocity
            if trajectory is not None:
                trajectory.append(state)
        if trajectory is not None:
            return torch.stack(trajectory, dim=1)
        return state

    def forward(
        self,
        x: TensorDict | Tensor,
        y: TensorDict | None = None,
    ) -> tuple[TensorDict | Tensor, TensorDict | None]:
        """Fill the NexuML predicted-velocity key during training.

        Returns:
            TensorDict containing predicted_velocity and unchanged labels.

        Raises:
            TypeError: If the pipeline does not provide TensorDict input.
        """
        if not self.check_update():
            return x, y
        if not isinstance(x, TensorDict):
            raise TypeError("HealthyReferenceConditionedFlow requires TensorDict input")
        keys = cast(list[str], self.keys_in)
        state = cast(Tensor, x[keys[0]])
        time = cast(Tensor, x[keys[1]])
        reference = cast(Tensor, x[keys[2]])
        x[self.keys_out[0]] = self.velocity(state, time, reference)
        return x, y

    def forward_tensor(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        raise NotImplementedError
