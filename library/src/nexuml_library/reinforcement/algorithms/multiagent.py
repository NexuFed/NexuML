"""Native TorchRL IPPO/MAPPO, with NexuML graph and optimizer ownership only."""

from importlib import import_module
from math import prod
from typing import Any, Literal, cast

import torch
from pydantic import Field
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from torch import nn

from nexuml.core.components import RLAlgorithmBuildContext, RLAlgorithmDefinition
from nexuml.core.compiler import compile_context_from_interaction, compile_pipeline
from nexuml.core.discovery import rl_algorithm
from nexuml.core.types import LayerSpec, PipelineSpec
from nexuml_library.reinforcement.policies import PolicyHead


@rl_algorithm("IPPO")
class IPPO(RLAlgorithmDefinition):
    """Independent local actors/critics; normalization is deliberately unqualified."""

    critic: PipelineSpec | None = None
    critic_encoders: dict[str, PipelineSpec] = Field(default_factory=dict)
    clip_epsilon: float = Field(default=0.2, gt=0, lt=1)
    gamma: float = Field(default=0.99, gt=0, le=1)
    gae_lambda: float = Field(default=0.95, ge=0, le=1)
    entropy_coef: float = Field(default=0.01, ge=0)
    actor_lr: float = Field(default=3e-4, gt=0)
    critic_lr: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    update_epochs: int = Field(default=10, gt=0)
    minibatch_size: int = Field(default=256, gt=0)
    max_grad_norm: float = Field(default=0.5, gt=0)
    normalize_advantage: Literal[False] = False
    average_gae: Literal[False] = False
    value_norm: None = None

    def build(self, context: RLAlgorithmBuildContext) -> nn.Module:
        return _MultiAgentRuntime(self, context, centralized=False)


@rl_algorithm("MAPPO")
class MAPPO(IPPO):
    """Decentralized actors with native cross-group encoders/trunk/value heads."""

    def build(self, context: RLAlgorithmBuildContext) -> nn.Module:
        return _MultiAgentRuntime(self, context, centralized=True)


def agent_view(tensors, group, index):
    """Keep physical root axes and a singleton numerical agent axis.

    Returns:
        view (TensorDict): Shallow group slice retaining all physical centralized inputs.
    """
    view = tensors.clone(False)
    view[group] = tensors[group][..., index : index + 1]
    if "next" in tensors.keys():
        view["next"] = tensors["next"].clone(False)
        view["next", group] = tensors["next", group][..., index : index + 1]
    return view


def _local(tensors, group, keys):
    return TensorDict(
        {key: tensors[group, key].select(tensors.ndim, 0) for key in keys},
        batch_size=tensors.batch_size,
        device=tensors.device,
    )


class _ActorParameters(TensorDictModuleBase):
    def __init__(self, policy, group):
        super().__init__()
        self.policy, self.group = policy, group
        cast(Any, self).in_keys = [(group, key) for key in policy.contract.observations]
        cast(Any, self).out_keys = [
            (group, key) for key in policy.action_adapter.parameter_keys.values()
        ]

    def forward(self, tensors):
        output = self.policy.forward_parameters(
            _local(tensors, self.group, self.policy.contract.observations)
        )
        for _, key in self.out_keys:
            tensors[self.group, key] = output[key].unsqueeze(tensors.ndim)
        return tensors


class _LocalValue(TensorDictModuleBase):
    def __init__(self, pipeline, group, keys):
        super().__init__()
        self.pipeline, self.group, self.keys = pipeline, group, keys
        cast(Any, self).in_keys = [(group, key) for key in keys]
        cast(Any, self).out_keys = [(group, "state_value")]

    def forward(self, tensors):
        inputs = _local(tensors, self.group, self.keys)
        output, _ = self.pipeline(inputs.reshape(-1), None)
        tensors[self.group, "state_value"] = output["state_value"].reshape(
            *tensors.batch_size, 1, 1
        )
        return tensors


class _CentralValue(TensorDictModuleBase):
    def __init__(self, critic, group, index):
        super().__init__()
        self.critic, self.group, self.index = critic, group, index
        self.in_keys = critic.in_keys
        cast(Any, self).out_keys = [(group, "state_value")]

    def forward(self, tensors):
        self.critic(tensors)
        tensors[self.group, "state_value"] = tensors["_values", self.group][
            ..., self.index : self.index + 1, :
        ]
        return tensors


class _RolloutPolicy(TensorDictModuleBase):
    def __init__(self, actors, slots, contract):
        super().__init__()
        self.actors, self.slots = nn.ModuleList(actors), slots
        cast(Any, self).in_keys = [
            (g, key) for g, spec in contract.groups.items() for key in spec.observations
        ]
        self.out_keys = list(dict.fromkeys(key for actor in actors for key in actor.out_keys))

    def forward(self, tensors):
        children = {group: [] for group in dict.fromkeys(g for g, _, _ in self.slots)}
        for actor, (group, index, _) in zip(self.actors, self.slots, strict=True):
            view = agent_view(tensors, group, index)
            actor(view)
            children[group].append(view[group])
        for group, parts in children.items():
            tensors[group] = torch.cat(parts, dim=tensors.ndim)
        return tensors


class _MultiAgentRuntime(nn.Module):
    updates: torch.Tensor

    def __init__(self, definition, context, centralized):
        super().__init__()
        from tensordict.nn import InteractionType

        modules = import_module("torchrl.modules")
        objectives = import_module("torchrl.objectives.multiagent")
        self.definition, self.contract = definition, context.contract
        if not self.contract.groups:
            raise ValueError("IPPO/MAPPO require explicit fixed agent groups")
        self.centralized = centralized
        self.encoders = nn.ModuleDict()
        self.critics = nn.ModuleDict()
        self.losses: Any = nn.ModuleList()
        self.slots = [
            (g, i, actor_id)
            for g, ids in context.policy.actor_ids.items()
            for i, actor_id in enumerate(ids)
        ]
        self.actor_ids = list(context.policy.actors)
        self.actor_modules = nn.ModuleDict(
            {key: context.policy.actors[key] for key in self.actor_ids}
        )
        if set(definition.critic_encoders) - set(self.contract.groups):
            raise ValueError("Critic encoders reference unknown groups")
        if not centralized and definition.critic_encoders:
            raise ValueError("IPPO uses its local compiled critic graph, not centralized encoders")
        if centralized:
            if definition.critic is not None:
                raise ValueError(
                    "MAPPO uses native CrossGroupCritic; configure critic_encoders instead"
                )
            cross_group = import_module("torchrl.modules.models.cross_group_critic")

            specs = {}
            for group, spec in self.contract.groups.items():
                width = sum(prod(field.shape) for field in spec.observations.values())
                if group in definition.critic_encoders:
                    pipeline = compile_pipeline(
                        definition.critic_encoders[group],
                        context=compile_context_from_interaction(
                            self.contract.group_contract(group)
                        ),
                        resolved_config=context.policy.pipeline.resolved_config,
                    )
                    self.encoders[group] = pipeline
                    width = prod(pipeline.input_sizes["features"])
                elif any(len(field.shape) != 1 for field in spec.observations.values()):
                    raise ValueError(
                        "Image critics require an explicit compiled encoder producing features"
                    )
                specs[group] = cross_group.CrossCriticGroupSpec(
                    width, len(spec.agents), ("_critic", group), ("_values", group)
                )
            if self.contract.critic_observations:
                width = sum(
                    prod(field.shape) for field in self.contract.critic_observations.values()
                )
                specs["_state"] = cross_group.CrossCriticGroupSpec(
                    width, 1, ("_critic", "_state"), ("_values", "_state")
                )
            self.critics["central"] = cross_group.CrossGroupCritic(
                specs, d_model=32, trunk_depth=2, trunk_cells=64
            )
        actors = []
        for group, index, actor_id in self.slots:
            policy = context.policy.actors[actor_id]
            adapter = policy.action_adapter
            if not hasattr(adapter, "distribution_class"):
                raise ValueError("IPPO/MAPPO require probabilistic action adapters")
            parameters = {*adapter.parameter_keys.values(), adapter.action_key + "_log_prob"}
            if parameters.intersection(policy.contract.observations):
                raise ValueError("Distribution outputs must not overwrite actor observations")
            actor = modules.ProbabilisticActor(
                cast(Any, _ActorParameters(policy, group)),
                in_keys=cast(
                    Any, {name: (group, key) for name, key in adapter.parameter_keys.items()}
                ),
                out_keys=[(group, adapter.action_key)],
                distribution_class=adapter.distribution_class,
                distribution_kwargs=adapter.distribution_kwargs,
                default_interaction_type=InteractionType.RANDOM,
                return_log_prob=True,
                log_prob_key=(group, adapter.action_key + "_log_prob"),
            )
            actors.append(actor)
            if centralized:
                critic = _CentralValue(self.critics["central"], group, index)
            else:
                # Separate local modules avoid packed unshared optimizer momentum/decay drift.
                name = f"{group}_{index}"
                local_contract = self.contract.group_contract(group)
                spec = definition.critic or PipelineSpec(
                    stages={
                        "value": [
                            LayerSpec(
                                component=PolicyHead(),
                                keys_in=list(local_contract.observations),
                                keys_out=["state_value"],
                            )
                        ]
                    }
                )
                pipeline = compile_pipeline(
                    spec,
                    context=compile_context_from_interaction(local_contract),
                    resolved_config=context.policy.pipeline.resolved_config,
                )
                critic = _LocalValue(pipeline, group, list(local_contract.observations))
                self.critics[name] = critic
            loss_type = objectives.MAPPOLoss if centralized else objectives.IPPOLoss
            loss = loss_type(
                actor,
                cast(Any, critic),
                functional=False,
                clip_epsilon=definition.clip_epsilon,
                entropy_coeff=definition.entropy_coef,
                normalize_advantage=False,
                value_norm=None,
            )
            loss.set_keys(
                action=(group, adapter.action_key),
                sample_log_prob=(group, adapter.action_key + "_log_prob"),
                value=(group, "state_value"),
                advantage=(group, "advantage"),
                value_target=(group, "value_target"),
                reward=(group, "reward"),
                done=(group, "done"),
                terminated=(group, "terminated"),
            )
            loss.loss_mask_key = (group, "valid")
            loss.make_value_estimator(
                gamma=definition.gamma,
                lmbda=definition.gae_lambda,
                time_dim=1,
                shifted=False,
                auto_reset_env=False,
                average_gae=False,
                deactivate_vmap=True,
            )
            assert isinstance(
                loss.value_estimator, import_module("torchrl.objectives.value").MultiAgentGAE
            )
            self.losses.append(loss)
        self.rollout_policy = _RolloutPolicy(actors, self.slots, self.contract)
        self.register_buffer("updates", torch.tensor(0, dtype=torch.int64))

    def configure_optimizers(self):
        owners = [self.actor_modules[key] for key in self.actor_ids] + list(self.critics.values())
        parameters = [list(owner.parameters()) for owner in owners]
        if self.centralized:
            parameters[-1].extend(self.encoders.parameters())
        ids = [id(parameter) for group in parameters for parameter in group]
        if len(ids) != len(set(ids)):
            raise ValueError("Each actor/critic parameter must have one optimizer owner")
        return [
            torch.optim.Adam(
                params,
                lr=self.definition.actor_lr
                if i < len(self.actor_ids)
                else self.definition.critic_lr,
                weight_decay=self.definition.weight_decay,
            )
            for i, params in enumerate(parameters)
        ]

    def _critic_inputs(self, tensors):
        if not self.centralized:
            return
        for group, spec in self.contract.groups.items():
            child = tensors[group]
            if group in self.encoders:
                inputs = child.select(*spec.observations).reshape(-1).clone()
                output, _ = self.encoders[group](inputs, None)
                features = output["features"].reshape(*child.batch_size, -1)
            else:
                features = torch.cat(
                    [
                        child[key].reshape(*child.batch_size, -1).float()
                        for key in spec.observations
                    ],
                    dim=-1,
                )
            tensors["_critic", group] = features
        if self.contract.critic_observations:
            tensors["_critic", "_state"] = torch.cat(
                [
                    tensors[key].reshape(*tensors.batch_size, -1).float()
                    for key in self.contract.critic_observations
                ],
                dim=-1,
            ).unsqueeze(-2)

    def prepare(self, rollout):
        """Run native GAE on physical [N,T] before any minibatch flattening.

        Returns:
            A detached grouped rollout with native targets, preserving agent axes.

        Raises:
            ValueError: If the physical environment/time axes are missing.
        """
        if rollout.ndim != 2:
            raise ValueError("Grouped GAE requires root [environment,time] rollout axes")
        if "shifted_valid" in rollout.keys():
            raise ValueError("shifted_valid is not an agent learning-validity mask")
        rollout = rollout.detach().clone()
        with torch.no_grad():
            self._critic_inputs(rollout)
            self._critic_inputs(rollout["next"])
            for loss, (group, index, _) in zip(self.losses, self.slots, strict=True):
                view = agent_view(rollout, group, index)
                # Other agents' cached values must not become this critic's current-only cache.
                view[group] = view[group].exclude("state_value", "advantage", "value_target")
                view["next", group] = view["next", group].exclude(
                    "state_value", "advantage", "value_target"
                )
                valid = view[group, "valid"]
                # Inactive suffix rewards are unused; real preceding boundary remains intact.
                view["next", group, "reward"] = view["next", group, "reward"].masked_fill(~valid, 0)
                loss.value_estimator(view)
                for key in ("state_value", "advantage", "value_target"):
                    if key not in rollout[group].keys():
                        rollout[group, key] = torch.zeros(
                            (*rollout[group].batch_size, 1), device=rollout.device
                        )
                    rollout[group, key][..., index : index + 1, :] = view[group, key]
        return rollout

    def update(self, rollout, *, lightning_module, optimizers):
        flat = self.prepare(rollout).reshape(-1)
        metrics, count = {}, 0
        for _ in range(self.definition.update_epochs):
            for indices in torch.randperm(flat.numel(), device=flat.device).split(
                self.definition.minibatch_size
            ):
                batch = flat[indices]
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                # Encoder features need live gradients after detached target computation.
                self._critic_inputs(batch)
                total, active = None, set()
                for slot, (loss, (group, index, actor_id)) in enumerate(
                    zip(self.losses, self.slots, strict=True)
                ):
                    view = agent_view(batch, group, index)
                    if not view[group, "valid"].any():
                        continue
                    losses = loss(view)
                    value = sum(
                        losses[key] for key in ("loss_objective", "loss_critic", "loss_entropy")
                    )
                    total = value if total is None else total + value
                    active.add(self.actor_ids.index(actor_id))
                    active.add(len(self.actor_ids) + (0 if self.centralized else slot))
                    for key in ("loss_objective", "loss_critic", "loss_entropy"):
                        name = f"{group}/{key}"
                        metrics[name] = metrics.get(name, 0.0) + float(losses[key].detach())
                if total is None:
                    continue
                lightning_module.manual_backward(total)
                for index in sorted(active):
                    optimizer = optimizers[index]
                    lightning_module.clip_gradients(
                        optimizer,
                        gradient_clip_val=self.definition.max_grad_norm,
                        gradient_clip_algorithm="norm",
                    )
                    optimizer.step()
                self.updates.add_(1)
                count += 1
        return {key: value / max(count, 1) for key, value in metrics.items()}
