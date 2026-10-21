"""Complete deployable policy: neural graph, action semantics and portable I/O."""

from typing import Any, cast

import torch
from tensordict import TensorDict
from torch import nn

from nexuml.core.components import LayerBuildContext
from nexuml.core.pipeline import CompiledPipeline
from nexuml.core.types import DataSpec, InteractionContract, PolicySpec, TrainingSpec


class CompiledPolicy(nn.Module):
    """Policy inference independent of environments, collectors and RL algorithms."""

    pipeline: CompiledPipeline
    action_adapter: nn.Module
    contract: InteractionContract
    processor: nn.Module | None

    def __init__(
        self,
        pipeline: CompiledPipeline,
        action_adapter: nn.Module,
        contract: InteractionContract,
        processor: nn.Module | None = None,
    ):
        super().__init__()
        if processor is None and any(
            field.sensor is not None and field.sensor.owner == "package"
            for field in contract.observations.values()
        ):
            raise ValueError(
                "Package-owned sensor processing requires an explicit policy processor"
            )
        self.pipeline = pipeline
        self.action_adapter = action_adapter
        self.contract = contract
        self.processor = processor
        self.actor_ids: dict[str, tuple[str, ...]] = {}

    @property
    def actors(self) -> nn.ModuleDict:
        """Unique local actors, registered only on the checkpoint-owned root pipeline."""
        return cast(nn.ModuleDict, self.pipeline.group_actors)

    @classmethod
    def from_pipeline(
        cls, pipeline: CompiledPipeline, contract: InteractionContract | None = None
    ) -> "CompiledPolicy":
        """Attach declared action semantics without constructing an environment.

        Returns:
            Complete policy sharing the supplied pipeline and its trained weights.

        Raises:
            ValueError: If declarations are absent, offline inputs disagree, or package-owned
                sensors lack a processor.
        """
        scenario = pipeline.resolved_config
        if scenario.interaction is None or scenario.policy is None:
            raise ValueError("Policy assembly requires scenario.interaction and scenario.policy")
        if contract is None:
            contract = scenario.interaction.environment.describe()
        if contract.groups or scenario.policy.groups:
            return cls._from_grouped_pipeline(pipeline, contract)
        if isinstance(scenario.training, TrainingSpec):
            cls._validate_dataset_contract(scenario.data, contract)
        assert scenario.policy.action_adapter is not None
        return cls(
            pipeline,
            scenario.policy.action_adapter.build(contract),
            contract,
            cls._build_processor(scenario.policy, contract),
        )

    @staticmethod
    def _build_processor(spec: PolicySpec, contract: InteractionContract) -> nn.Module | None:
        processor = None
        if spec.processor is not None:
            keys = list(contract.observations)
            processor = spec.processor.build(
                LayerBuildContext(
                    input_sizes={key: field.shape for key, field in contract.observations.items()},
                    keys_in=keys,
                    keys_out=keys,
                    metadata={"interaction_contract": contract.model_dump(mode="json")},
                )
            )
            if not isinstance(processor, nn.Module) or any(
                not callable(getattr(processor, method, None))
                for method in ("initialize_state", "step", "reset_state")
            ):
                raise TypeError(
                    "A policy processor must build an nn.Module with explicit state methods"
                )
        return processor

    @classmethod
    def _from_grouped_pipeline(
        cls, pipeline: CompiledPipeline, contract: InteractionContract
    ) -> "CompiledPolicy":
        from nexuml.core.compiler import compile_context_from_interaction, compile_pipeline

        scenario = pipeline.resolved_config
        assert scenario.policy is not None
        if not contract.groups or list(scenario.policy.groups) != list(contract.groups):
            raise ValueError("Policy group order must match the interaction contract")
        if scenario.pipeline.stages:
            raise ValueError("Grouped policies declare their pipelines on each group only")
        signatures = {}
        actors = getattr(pipeline, "group_actors", None)
        if actors is None:
            actors = nn.ModuleDict()
        for name, group in scenario.policy.groups.items():
            local_contract = contract.group_contract(name)
            if len(group.actor_ids) != len(contract.groups[name].agents):
                raise ValueError(f"Group {name!r} requires one actor_id per ordered agent")
            local_spec = PolicySpec(
                action_adapter=group.action_adapter,
                batch_mode=scenario.policy.batch_mode,
                processor=group.processor,
            )
            signature = (local_contract, group.pipeline, local_spec)
            for actor_id in group.actor_ids:
                if actor_id in signatures and signatures[actor_id] != signature:
                    raise ValueError(
                        f"Shared actor {actor_id!r} requires identical local semantics, "
                        "pipeline, adapter and processor"
                    )
                signatures[actor_id] = signature
                if actor_id in actors:
                    actor = cast(CompiledPolicy, actors[actor_id])
                    saved = actor.pipeline.resolved_config
                    if (actor.contract, saved.pipeline, saved.policy) != signature:
                        raise ValueError(f"Existing actor {actor_id!r} has incompatible semantics")
                    continue
                local_config = scenario.model_copy(
                    update={"pipeline": group.pipeline, "policy": local_spec}
                )
                local_pipeline = compile_pipeline(
                    group.pipeline,
                    context=compile_context_from_interaction(local_contract),
                    resolved_config=local_config,
                )
                actors[actor_id] = cls(
                    local_pipeline,
                    group.action_adapter.build(local_contract),
                    local_contract,
                    cls._build_processor(local_spec, local_contract),
                )
        if list(actors) != list(signatures):
            raise ValueError("Existing actor topology does not match the grouped policy")
        pipeline.group_actors = actors
        policy = cls(pipeline, nn.Identity(), contract)
        policy.actor_ids = {name: group.actor_ids for name, group in scenario.policy.groups.items()}
        return policy

    def initialize_state(self, batch_size=(), *, device=None) -> TensorDict:
        """Create caller-owned processing state, never live policy buffers.

        Returns:
            Fresh processing state on the requested or policy device.

        Raises:
            ValueError: If this policy has no processor.
        """
        if device is None:
            parameter = next(self.pipeline.parameters(), None)
            if parameter is not None:
                device = parameter.device
        if self.contract.groups:
            self._require_group_processor()
            if type(batch_size) is int:
                batch_size = (batch_size,)
            if not isinstance(batch_size, (tuple, list, torch.Size)) or any(
                type(size) is not int or size < 0 for size in batch_size
            ):
                raise ValueError("State batch_size must contain nonnegative integers")
            if not batch_size:
                raise ValueError("Grouped state requires an environment batch axis")
            state = TensorDict({}, batch_size=batch_size, device=device or "cpu")
            for name, actor_ids in self.actor_ids.items():
                states = []
                for actor_id in actor_ids:
                    actor = cast(CompiledPolicy, self.actors[actor_id])
                    states.append(
                        actor.initialize_state(batch_size, device=device)
                        if actor.processor is not None
                        else TensorDict({}, batch_size=batch_size, device=state.device)
                    )
                state[name] = TensorDict.stack(states, dim=state.ndim)
            return state
        if self.processor is None:
            raise ValueError("Policy state requires an explicit processor")
        return cast(Any, self.processor).initialize_state(batch_size, device=device)

    def step(
        self, observation: TensorDict, state: TensorDict, *, deterministic: bool = True
    ) -> tuple[TensorDict, TensorDict]:
        """Process one raw observation using explicit caller-owned history.

        Returns:
            Decoded actions and new processing state, without mutating inputs.

        Raises:
            ValueError: If this policy has no processor.
        """
        if self.contract.groups:
            self._require_group_processor()
            if state.batch_size != observation.batch_size:
                raise ValueError("Observation and processing state batch axes must match")
            actions = TensorDict({}, batch_size=observation.batch_size, names=observation.names)
            next_state = state.clone()
            for name, actor_ids in self.actor_ids.items():
                child = self._group_child(observation, name)
                group_state = self._group_child(state, name)
                outputs, states = [], []
                for index, actor_id in enumerate(actor_ids):
                    actor = cast(CompiledPolicy, self.actors[actor_id])
                    local = self._local_observation(child, index, actor)
                    if actor.processor is None:
                        outputs.append(actor(local, deterministic=deterministic))
                        states.append(group_state[..., index].clone())
                    else:
                        output, updated = actor.step(
                            local,
                            cast(TensorDict, group_state[..., index]),
                            deterministic=deterministic,
                        )
                        outputs.append(output)
                        states.append(updated)
                actions[name] = self._stack_group(outputs, child)
                next_state[name] = TensorDict.stack(states, dim=state.ndim)
            return actions, next_state
        if self.processor is None:
            raise ValueError("Policy state requires an explicit processor")
        processed, next_state = cast(Any, self.processor).step(observation, state)
        return self(processed, deterministic=deterministic), next_state

    def reset_state(self, state: TensorDict, mask) -> TensorDict:
        """Reset only selected streams in the supplied processing state.

        Returns:
            New caller-owned state; unselected streams keep their values.

        Raises:
            ValueError: If this policy has no processor.
        """
        if self.contract.groups:
            self._require_group_processor()
            if (
                not isinstance(mask, torch.Tensor)
                or mask.dtype != torch.bool
                or mask.shape != state.batch_size
                or mask.device != state.device
            ):
                raise ValueError(
                    "Reset mask must have exact root state batch shape/device and bool"
                )
            result = state.clone()
            for name, actor_ids in self.actor_ids.items():
                child = self._group_child(state, name)
                states = []
                for index, actor_id in enumerate(actor_ids):
                    actor = cast(CompiledPolicy, self.actors[actor_id])
                    local = cast(TensorDict, child[..., index])
                    states.append(
                        actor.reset_state(local, mask)
                        if actor.processor is not None
                        else local.clone()
                    )
                result[name] = TensorDict.stack(states, dim=state.ndim)
            return result
        if self.processor is None:
            raise ValueError("Policy state requires an explicit processor")
        return cast(Any, self.processor).reset_state(state, mask)

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
        if self.contract.groups:
            return self._forward_groups(observation, decode=True, deterministic=deterministic)
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
        if self.contract.groups:
            return self._forward_groups(observation, decode=False)
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

    def _require_group_processor(self) -> None:
        if not any(actor.processor is not None for actor in self.actors.values()):
            raise ValueError("Policy state requires an explicit processor")

    def _group_child(self, tensors: TensorDict, name: str) -> TensorDict:
        child = tensors.get(name)
        expected = (*tensors.batch_size, len(self.actor_ids[name]))
        if tensors.ndim < 1 or not isinstance(child, TensorDict) or child.batch_size != expected:
            raise ValueError(
                f"Group {name!r} must have root environment batch plus agent axis {expected}"
            )
        return child

    @staticmethod
    def _local_observation(child: TensorDict, index: int, actor: "CompiledPolicy") -> TensorDict:
        local = (
            cast(TensorDict, child[..., index])
            .select(*actor.contract.observations, strict=False)
            .clone()
        )
        spec = actor.pipeline.resolved_config.policy
        assert spec is not None
        if spec.batch_mode == "independent":
            # The explicit agent view owns the axes; ordinary local graphs see independent samples.
            local.names = [None] * local.ndim
        return local

    @staticmethod
    def _stack_group(outputs: list[TensorDict], child: TensorDict) -> TensorDict:
        stacked = TensorDict.stack(outputs, dim=child.ndim - 1)
        stacked.names = child.names
        return stacked

    def _forward_groups(
        self, observation: TensorDict, *, decode: bool, deterministic: bool = True
    ) -> TensorDict:
        result = TensorDict({}, batch_size=observation.batch_size, names=observation.names)
        for name, actor_ids in self.actor_ids.items():
            child = self._group_child(observation, name)
            outputs = []
            for index, actor_id in enumerate(actor_ids):
                actor = cast(CompiledPolicy, self.actors[actor_id])
                local = self._local_observation(child, index, actor)
                outputs.append(
                    actor(local, deterministic=deterministic)
                    if decode
                    else actor.forward_parameters(local)
                )
            result[name] = self._stack_group(outputs, child)
        return result

    @staticmethod
    def _validate_fields(
        tensors: TensorDict, fields, role: str, *, require_finite: bool = True
    ) -> None:
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
            if require_finite and tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"{role} {key!r} contains nonfinite values")
            if field.num_values is not None and torch.any(
                (tensor < 0) | (tensor >= field.num_values)
            ):
                raise ValueError(f"{role} {key!r} is outside its discrete action space")
