"""Complete deployable policy: neural graph, action semantics and portable I/O."""

from tensordict import TensorDict
from torch import nn

from nexuml.core.pipeline import CompiledPipeline
from nexuml.core.types import DataSpec, InteractionContract, TrainingSpec


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

    @classmethod
    def from_pipeline(
        cls, pipeline: CompiledPipeline, contract: InteractionContract | None = None
    ) -> "CompiledPolicy":
        """Attach declared action semantics without constructing an environment.

        Returns:
            Complete policy sharing the supplied pipeline and its trained weights.

        Raises:
            ValueError: If declarations are absent or offline inputs disagree.
        """
        scenario = pipeline.resolved_config
        if scenario.interaction is None or scenario.policy is None:
            raise ValueError("Policy assembly requires scenario.interaction and scenario.policy")
        if contract is None:
            contract = scenario.interaction.environment.describe()
        if isinstance(scenario.training, TrainingSpec):
            cls._validate_dataset_contract(scenario.data, contract)
        return cls(pipeline, scenario.policy.action_adapter.build(contract), contract)

    @staticmethod
    def _validate_dataset_contract(data: DataSpec, contract: InteractionContract) -> None:
        from nexuml.core.compiler import compile_context_from_data

        inputs = compile_context_from_data(data)
        for key, field in contract.observations.items():
            if inputs.input_sizes.get(key) != field.shape:
                raise ValueError(
                    f"Dataset input {key!r} must match interaction observation shape {field.shape}"
                )
            declared = data.input_contracts.get(key)
            if declared is not None and declared != field:
                raise ValueError(
                    f"Dataset input contract {key!r} does not match the interaction observation"
                )

    def forward(self, observation: TensorDict, *, deterministic: bool = True) -> TensorDict:
        output = self.forward_parameters(observation)
        actions = self.action_adapter(output, deterministic=deterministic)
        if actions.batch_size != observation.batch_size:
            raise ValueError("Action decoding must preserve the observation batch axes")
        spec = getattr(self.pipeline.resolved_config, "policy", None)
        if getattr(spec, "batch_mode", "independent") == "preserve":
            if any(
                found is not None and found != expected
                for found, expected in zip(actions.names, observation.names, strict=True)
            ):
                raise ValueError("Action decoding changed named observation batch axes")
            actions.names = observation.names
        self._validate_fields(actions, self.contract.actions, "Action")
        return actions

    def forward_parameters(self, observation: TensorDict) -> TensorDict:
        """Run independent samples or an explicitly batch-aware neural graph.

        Returns:
            Neural outputs for deployment or probabilistic learner integration.

        Raises:
            ValueError: If semantic batch axes would be flattened or the graph changes them.
        """
        self._validate_fields(observation, self.contract.observations, "Observation")
        spec = getattr(self.pipeline.resolved_config, "policy", None)
        if getattr(spec, "batch_mode", "independent") == "preserve":
            output, _ = self.pipeline(observation.clone(), None)
            if output.batch_size != observation.batch_size or output.names != observation.names:
                raise ValueError("A batch-aware policy must preserve its observation batch axes")
            return output
        if {"time", "agent", "agents"}.intersection(observation.names):
            raise ValueError(
                "Temporal/agent batch axes require policy.batch_mode='preserve' "
                "and a batch-aware graph, or explicit tensor feature axes"
            )
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
