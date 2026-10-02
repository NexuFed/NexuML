"""Complete deployable policy: neural graph, action semantics and portable I/O."""

from tensordict import TensorDict
from torch import nn

from nexuml.core.pipeline import CompiledPipeline
from nexuml.core.types import InteractionContract


class CompiledPolicy(nn.Module):
    """Policy inference independent of environments, collectors and RL algorithms."""

    pipeline: CompiledPipeline
    action_adapter: nn.Module
    contract: InteractionContract

    def __init__(
        self,
        pipeline: CompiledPipeline,
        action_adapter: nn.Module,
        contract: InteractionContract,
    ):
        super().__init__()
        self.pipeline = pipeline
        self.action_adapter = action_adapter
        self.contract = contract

    def forward(self, observation: TensorDict, *, deterministic: bool = True) -> TensorDict:
        output = self.forward_parameters(observation)
        actions = self.action_adapter(output, deterministic=deterministic)
        self._validate_fields(actions, self.contract.actions, "Action")
        return actions

    def forward_parameters(self, observation: TensorDict) -> TensorDict:
        """Run the neural graph while preserving arbitrary inference batch axes.

        Returns:
            Neural outputs for deployment or probabilistic learner integration.
        """
        self._validate_fields(observation, self.contract.observations, "Observation")
        output, _ = self.pipeline(observation.reshape(-1).clone(), None)
        return output.reshape(observation.batch_size)

    @staticmethod
    def _validate_fields(tensors: TensorDict, fields, role: str) -> None:
        import torch

        for key, field in fields.items():
            if key not in tensors.keys():
                raise ValueError(f"{role} is missing required field {key!r}")
            tensor = tensors[key]
            expected_shape = (*tensors.batch_size, *field.shape)
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected_shape:
                raise ValueError(f"{role} {key!r} must have shape {expected_shape}")
            if str(tensor.dtype).removeprefix("torch.") != field.dtype:
                raise ValueError(f"{role} {key!r} must have dtype {field.dtype!r}")
            if field.low is not None:
                low = torch.as_tensor(field.low, device=tensor.device).reshape(
                    field.shape if isinstance(field.low, list) else ()
                )
                if torch.any(tensor < low):
                    raise ValueError(f"{role} {key!r} is below its lower bound")
            if field.high is not None:
                high = torch.as_tensor(field.high, device=tensor.device).reshape(
                    field.shape if isinstance(field.high, list) else ()
                )
                if torch.any(tensor > high):
                    raise ValueError(f"{role} {key!r} is above its upper bound")
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"{role} {key!r} contains nonfinite values")
            if field.num_values is not None and torch.any(
                (tensor < 0) | (tensor >= field.num_values)
            ):
                raise ValueError(f"{role} {key!r} is outside its discrete action space")
