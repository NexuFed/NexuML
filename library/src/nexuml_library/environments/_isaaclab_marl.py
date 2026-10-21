"""Fixed-agent translation only; the existing runtime owns all native resources."""

from collections.abc import Mapping
from importlib import import_module
import math

import torch
from tensordict import TensorDict

from nexuml.core.policy import CompiledPolicy
from nexuml.interaction.runtime import InteractionStep


class IsaacMARLBoundary:
    """Keep native reset rows, raw agent facts and learning validity separate."""

    def __init__(self, owner, definition):
        self.owner = owner
        self.contract = owner.contract
        self.bindings = definition.native_bindings()
        self.agents = tuple(
            agent for group in self.contract.groups.values() for agent in group.agents
        )
        self.live = {
            name: torch.ones(
                (self.contract.num_envs, len(group.agents), 1),
                dtype=torch.bool,
                device=owner.device,
            )
            for name, group in self.contract.groups.items()
        }
        self.ended_terminated = {name: torch.zeros_like(live) for name, live in self.live.items()}
        self.ended_truncated = {name: torch.zeros_like(live) for name, live in self.live.items()}

    def _membership(self, *mappings):
        env = self.owner.native
        if tuple(env.possible_agents) != self.agents or tuple(env.agents) != self.agents:
            raise ValueError("Dynamic/reordered native agent membership is unsupported")
        for mapping in mappings:
            if not isinstance(mapping, Mapping) or set(mapping) != set(self.agents):
                raise ValueError("Native per-agent dictionaries must match fixed agent membership")

    def validate(self):
        env = self.owner.native
        if not self.contract.groups or self.contract.critic_observations:
            raise ValueError(
                "Privileged critic state has no qualified pre-reset final-state source"
            )
        if self.bindings.critic_observations:
            raise ValueError("Privileged critic bindings have no pre-reset final-state source")
        if (
            env.num_envs != self.contract.num_envs
            or torch.device(env.device) != self.owner.device
            or not math.isclose(env.step_dt, self.contract.control_period_s, rel_tol=1e-9)
            or not env.cfg.compute_final_obs
        ):
            raise ValueError("Native MARL count/device/control period/final-state mismatch")
        self._membership(env.action_spaces, env.observation_spaces)
        for name, group in self.contract.groups.items():
            binding = self.bindings.groups[name]
            if binding.processing != "processed" or binding.critic_observations:
                raise ValueError("Native grouped raw/privileged processing is unsupported")
            for field in (*group.observations.values(), *group.actions.values()):
                if field.rate_hz is not None and not math.isclose(
                    field.rate_hz * env.step_dt, 1.0, rel_tol=1e-9
                ):
                    raise ValueError("Native MARL field rate differs from the control period")
                if field.sensor is not None:
                    raise ValueError("Native MARL sensor processing/final freshness is unqualified")
            for agent in group.agents:
                for key, path in binding.observations.items():
                    space = env.observation_spaces[agent]
                    for part in path:
                        space = space[part]
                    field = group.observations[key]
                    if tuple(space.shape) != field.shape or str(space.dtype) != field.dtype:
                        raise ValueError("Native MARL observation shape/dtype mismatch")
                field = group.actions[binding.action]
                self._validate_action_space(env.action_spaces[agent], field, group.action_space)
                self._validate_controls(agent, field)

    @staticmethod
    def _validate_action_space(space, field, kind):
        from gymnasium.spaces import Box, Discrete

        if tuple(space.shape) != field.shape or str(space.dtype) != field.dtype:
            raise ValueError("Native MARL action shape/dtype mismatch")
        if kind == "discrete":
            if not isinstance(space, Discrete) or space.start != 0 or space.n != field.num_values:
                raise ValueError("Native MARL categorical action space mismatch")
        elif kind != "continuous" or not isinstance(space, Box):
            raise ValueError("Native MARL supports Box or scalar Discrete actions only")
        else:
            for key in ("low", "high"):
                bound = getattr(field, key)
                native = torch.as_tensor(getattr(space, key))
                if bound is None:
                    if torch.isfinite(native).any():
                        raise ValueError("Native action limits must be declared")
                else:
                    declared = torch.as_tensor(bound).reshape(
                        field.shape if isinstance(bound, list) else ()
                    )
                    outside = declared < native if key == "low" else declared > native
                    if outside.any():
                        raise ValueError("Declared actions exceed actual native action limits")

    def _validate_controls(self, agent, field):
        env = self.owner.native
        module = type(env).__module__
        physical = field.physical
        if module == "isaaclab_tasks.core.pendulum.pendulum_marl_env":
            implementation = import_module(module).PendulumMARLEnv
            cfg = env.cfg
            if (
                type(env) is not implementation
                or cfg.cart_dof_name != "slider_to_cart"
                or cfg.pole_dof_name != "cart_to_pole"
                or cfg.pendulum_dof_name != "pole_to_pendulum"
                or cfg.cart_action_scale != 100.0
                or cfg.pendulum_action_scale != 50.0
            ):
                raise ValueError("Native Pendulum control implementation/mapping mismatch")
            joint, scale, unit = (
                ("slider_to_cart", 100.0, "N")
                if agent == "cart"
                else ("pole_to_pendulum", 50.0, "Nm")
            )
            if env.robot.find_joints(joint)[1] != [joint]:
                raise ValueError("Native Pendulum joint order mismatch")
            if physical is not None and (
                physical.control_mode != "effort"
                or physical.joints != (joint,)
                or physical.scale != (scale,)
                or physical.offset not in (None, (0.0,))
                or physical.units != (unit,)
            ):
                raise ValueError("Declared Pendulum physical controls differ from native mapping")
        elif module == "isaaclab_tasks.core.handover.handover_env":
            implementation = import_module(module).HandoverEnv
            cfg_module = import_module("isaaclab_tasks.core.handover.handover_env_cfg")
            cfg = env.cfg
            if (
                type(env) is not implementation
                or cfg.act_moving_average != 1.0
                or tuple(cfg.actuated_joint_names) != tuple(cfg_module.JOINT_NAMES)
                or tuple(cfg.actuated_tendon_names) != tuple(cfg_module.TENDON_NAMES)
                or cfg.actuated_tendon_position_limits != cfg_module.TENDON_POSITION_LIMITS
            ):
                raise ValueError("Native Handover joint/tendon/target-EMA mapping mismatch")
            if physical is not None:
                # Joint and tendon limit decoding is not a single normalized affine joint action.
                raise ValueError("Handover joint/tendon physical bindings are not qualified")
        elif physical is not None:
            raise ValueError("Declared MARL physical controls need a maintained native binding")

    def _state(self, observations):
        self._membership(observations)
        root = TensorDict({}, batch_size=(self.contract.num_envs,), device=self.owner.device)
        for name, group in self.contract.groups.items():
            child = TensorDict(
                {}, batch_size=(*root.batch_size, len(group.agents)), device=self.owner.device
            )
            for key, path in self.bindings.groups[name].observations.items():
                values = []
                for agent in group.agents:
                    value = observations[agent]
                    for part in path:
                        value = value[part]
                    field = group.observations[key]
                    if (
                        not isinstance(value, torch.Tensor)
                        or value.device != self.owner.device
                        or tuple(value.shape) != (self.contract.num_envs, *field.shape)
                        or value.dtype != getattr(torch, field.dtype)
                    ):
                        raise ValueError("Native MARL observation shape/dtype/device mismatch")
                    values.append(value.clone())
                child[key] = torch.stack(values, dim=1)
            CompiledPolicy._validate_fields(child, group.observations, "Native MARL observation")
            child["valid"] = self.live[name].clone()
            root[name] = child
        return root

    def reset(self, *, seed=None, mask=None):
        owner = self.owner
        count = self.contract.num_envs
        if mask is None:
            requested = torch.ones(count, dtype=torch.bool, device=owner.device)
        elif mask.dtype != torch.bool or mask.numel() != count:
            raise ValueError("MARL reset mask must contain one boolean per environment")
        else:
            requested = mask.reshape(count).to(owner.device)
        reset = requested.clone()
        if mask is not None and owner._pending_reset is not None:
            reset &= ~owner._pending_reset
        if owner._observation is None or reset.any():
            if not reset.all():
                raise ValueError(
                    "Direct MARL supports full reset or cached autoreset acknowledgement"
                )
            observations, _ = owner.native.reset(seed=seed)
            for name in self.live:
                self.live[name].fill_(True)
                self.ended_terminated[name].zero_()
                self.ended_truncated[name].zero_()
            owner._observation = self._state(observations)
        if owner._pending_reset is not None:
            owner._pending_reset &= ~requested
        return owner._observation.clone()

    def _signal(self, value, dtype, name):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (self.contract.num_envs,)
            or value.dtype != dtype
            or value.device != self.owner.device
        ):
            raise ValueError(f"Native MARL {name} shape/dtype/device mismatch")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Native MARL {name} must be finite")
        return value.clone()

    def step(self, action):
        owner = self.owner
        if owner._observation is None:
            raise RuntimeError("Reset an open MARL runtime before stepping")
        self._membership()
        if tuple(action.batch_size) != (self.contract.num_envs,):
            raise ValueError("MARL actions require root environment batch [N], including N=1")
        native_actions = {}
        for name, group in self.contract.groups.items():
            child = action[name]
            if tuple(child.batch_size) != (self.contract.num_envs, len(group.agents)):
                raise ValueError("MARL actions require ordered group batch [N,A]")
            CompiledPolicy._validate_fields(child, group.actions, "Action")
            value = child[self.bindings.groups[name].action]
            if value.device != owner.device:
                raise ValueError("MARL actions must use the native device")
            for index, agent in enumerate(group.agents):
                native_actions[agent] = value[:, index].clone()
        owner._in_step = True
        try:
            observations, rewards, terminated, truncated, extras = owner.native.step(native_actions)
        finally:
            owner._in_step = False
        self._membership(observations, rewards, terminated, truncated)
        rewards = {a: self._signal(rewards[a], torch.float32, "reward") for a in self.agents}
        terminated = {a: self._signal(terminated[a], torch.bool, "terminated") for a in self.agents}
        truncated = {a: self._signal(truncated[a], torch.bool, "truncated") for a in self.agents}
        reset = self._signal(owner.native.reset_buf, torch.bool, "reset_buf")
        # The pin declares unanimous timeouts; reset_buf ALONE decides the environment boundary.
        timeout = torch.stack(list(truncated.values()), dim=1).all(dim=1)
        terminal = torch.stack(list(terminated.values()), dim=1).all(dim=1)
        root_truncated = reset & timeout
        root_terminated = reset & (terminal | ~timeout)
        final = self._state(observations)
        if reset.any():
            if not isinstance(extras, Mapping) or any(
                not isinstance(extras.get(a), Mapping) or "final_obs" not in extras[a]
                for a in self.agents
            ):
                raise ValueError("Native MARL autoreset requires per-agent pre-reset final_obs")
            terminal = self._state({a: extras[a]["final_obs"] for a in self.agents})
            final[reset] = terminal[reset]
        signals = TensorDict({}, batch_size=final.batch_size, device=owner.device)
        team_reward = None
        if self.contract.team_reward:
            team_reward = rewards[self.agents[0]]
            if any(not torch.equal(team_reward, rewards[a]) for a in self.agents[1:]):
                raise ValueError(
                    "Declared native team rewards must be genuinely equal, not averaged"
                )
            team_reward = team_reward.unsqueeze(-1)
        for name, group in self.contract.groups.items():
            raw_term = torch.stack([terminated[a] for a in group.agents], dim=1).unsqueeze(-1)
            raw_trunc = torch.stack([truncated[a] for a in group.agents], dim=1).unsqueeze(-1)
            term = self.ended_terminated[name] | raw_term | root_terminated[:, None, None]
            trunc = self.ended_truncated[name] | raw_trunc | root_truncated[:, None, None]
            signals[name] = TensorDict(
                {
                    "reward": torch.stack([rewards[a] for a in group.agents], dim=1).unsqueeze(-1),
                    "raw_terminated": raw_term,
                    "raw_truncated": raw_trunc,
                    "terminated": term,
                    "truncated": trunc,
                    "done": term | trunc,
                    "valid": self.live[name].clone(),
                },
                batch_size=(*final.batch_size, len(group.agents)),
                device=owner.device,
            )
            self.ended_terminated[name] = term.clone()
            self.ended_truncated[name] = trunc.clone()
            self.live[name] &= ~(term | trunc)
            self.live[name][reset] = True
            self.ended_terminated[name][reset] = False
            self.ended_truncated[name][reset] = False
            final[name, "valid"] = self.live[name].clone()
        owner._observation = self._state(observations)
        owner._pending_reset = reset
        return InteractionStep(
            observation=final,
            reward=team_reward,
            terminated=root_terminated.unsqueeze(-1),
            truncated=root_truncated.unsqueeze(-1),
            groups=signals,
        )
