"""Private conversion of NexuML tensor interaction semantics to TorchRL."""

import torch
from tensordict import TensorDict

try:
    from torchrl.data import Bounded, Categorical, Composite, Unbounded
    from torchrl.envs import EnvBase
except ImportError as error:
    raise ImportError("Reinforcement runtime requires nexuml[reinforcement]") from error

from nexuml.core.policy import CompiledPolicy


class _TorchRLEnvironmentAdapter(EnvBase):
    def __init__(self, runtime, device="cpu", seed=None):
        device = torch.device(device)
        self.runtime = runtime
        contract = runtime.contract
        self._grouped = bool(contract.groups)
        batch = (
            torch.Size((contract.num_envs,))
            if self._grouped or contract.num_envs != 1
            else torch.Size(())
        )
        self._seed = seed
        super().__init__(device=device, batch_size=batch, run_type_checks=True)
        if self._grouped:
            self._grouped_specs(contract, batch, device)
            self.is_closed = False
            return
        self.observation_spec = Composite(
            {
                key: Unbounded(
                    shape=batch + torch.Size(leaf.shape),
                    dtype=getattr(torch, leaf.dtype),
                    device=device,
                )
                for key, leaf in contract.observations.items()
            },
            shape=batch,
            device=device,
        )
        actions = {}
        for key, leaf in contract.actions.items():
            shape = batch + torch.Size(leaf.shape)
            dtype = getattr(torch, leaf.dtype)
            if leaf.num_values is not None:
                actions[key] = Categorical(leaf.num_values, shape=shape, dtype=dtype, device=device)
            elif leaf.low is not None and leaf.high is not None:
                low = torch.as_tensor(leaf.low, device=device).reshape(
                    leaf.shape if isinstance(leaf.low, list) else ()
                )
                high = torch.as_tensor(leaf.high, device=device).reshape(
                    leaf.shape if isinstance(leaf.high, list) else ()
                )
                actions[key] = Bounded(
                    low.expand(shape), high.expand(shape), shape=shape, dtype=dtype, device=device
                )
            else:
                actions[key] = Unbounded(shape=shape, dtype=dtype, device=device)
        self.action_spec = Composite(actions, shape=batch, device=device)
        self.reward_spec = Unbounded(shape=batch + torch.Size((1,)), device=device)
        self.done_spec = Composite(
            {
                key: Categorical(2, shape=batch + torch.Size((1,)), dtype=torch.bool, device=device)
                for key in ("done", "terminated", "truncated")
            },
            shape=batch,
            device=device,
        )
        self.is_closed = False

    def _grouped_specs(self, contract, batch, device):
        def fields(mapping, shape):
            return {
                key: Unbounded(
                    shape=shape + torch.Size(leaf.shape),
                    dtype=getattr(torch, leaf.dtype),
                    device=device,
                )
                for key, leaf in mapping.items()
            }

        def masks(shape, keys):
            return {
                key: Categorical(2, shape=shape + (1,), dtype=torch.bool, device=device)
                for key in keys
            }

        observations = fields(contract.critic_observations, batch)
        actions, rewards = {}, {}
        dones = masks(batch, ("done", "terminated", "truncated"))
        if contract.team_reward:
            rewards["reward"] = Unbounded(shape=batch + (1,), device=device)
        for name, group in contract.groups.items():
            shape = batch + (len(group.agents),)
            observations[name] = Composite(
                {
                    **fields(group.observations, shape),
                    **masks(
                        shape,
                        (
                            "valid",
                            "raw_terminated",
                            "raw_truncated",
                            "done",
                            "terminated",
                            "truncated",
                        ),
                    ),
                },
                shape=shape,
                device=device,
            )
            group_actions = {}
            for key, leaf in group.actions.items():
                event = torch.Size(leaf.shape)
                size = shape + event
                dtype = getattr(torch, leaf.dtype)
                if leaf.num_values is not None:
                    spec = Categorical(leaf.num_values, shape=size, dtype=dtype, device=device)
                elif leaf.low is not None and leaf.high is not None:
                    low = torch.as_tensor(leaf.low, device=device).reshape(
                        event if isinstance(leaf.low, list) else ()
                    )
                    high = torch.as_tensor(leaf.high, device=device).reshape(
                        event if isinstance(leaf.high, list) else ()
                    )
                    spec = Bounded(
                        low.expand(size), high.expand(size), shape=size, dtype=dtype, device=device
                    )
                else:
                    spec = Unbounded(shape=size, dtype=dtype, device=device)
                group_actions[key] = spec
            actions[name] = Composite(group_actions, shape=shape, device=device)
            rewards[name] = Composite(
                {"reward": Unbounded(shape=shape + (1,), device=device)},
                shape=shape,
                device=device,
            )
            # Collector trajectory IDs/reset bookkeeping follow only physical root endings.
            # Agent endings remain explicit observation leaves for native multi-agent GAE.
        self.observation_spec = Composite(observations, shape=batch, device=device)
        self.action_spec = Composite(actions, shape=batch, device=device)
        self.reward_spec = Composite(rewards, shape=batch, device=device)
        self.done_spec = Composite(dones, shape=batch, device=device)

    def _validate_grouped_observation(self, observations):
        contract = self.runtime.contract
        if tuple(observations.batch_size) != tuple(self.batch_size):
            raise ValueError("Grouped environment requires root batch [N], including N=1")
        CompiledPolicy._validate_fields(observations, contract.critic_observations, "Critic")
        for name, group in contract.groups.items():
            child = observations[name]
            shape = (*self.batch_size, len(group.agents))
            if tuple(child.batch_size) != shape:
                raise ValueError("Grouped observation requires child batch [N,A]")
            CompiledPolicy._validate_fields(child, group.observations, "Observation")
            self._grouped_signal(child["valid"], (*shape, 1), torch.bool, "valid")

    def _grouped_signal(self, value, shape, dtype, key):
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.dtype != dtype
            or value.device != self.device
        ):
            raise ValueError(f"Grouped {key} shape/dtype/device disagrees with the contract")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Grouped {key} must be finite")

    def _reset(self, tensordict=None, **kwargs):
        mask = None if tensordict is None else tensordict.get("_reset", None)
        observations = self.runtime.reset(seed=self._seed, mask=mask)
        self._seed = None
        if self._grouped:
            self._validate_grouped_observation(observations)
            for name in self.runtime.contract.groups:
                child = observations[name]
                for key in ("done", "terminated", "truncated", "raw_terminated", "raw_truncated"):
                    child[key] = torch.zeros(
                        (*child.batch_size, 1), device=self.device, dtype=torch.bool
                    )
            for key in ("done", "terminated", "truncated"):
                observations[key] = torch.zeros(
                    (*self.batch_size, 1), device=self.device, dtype=torch.bool
                )
            return observations
        CompiledPolicy._validate_fields(
            observations, self.runtime.contract.observations, "Observation"
        )
        if tuple(observations.batch_size) != tuple(self.batch_size):
            raise ValueError("Environment batch dimensions disagree with contract.num_envs")
        for key in ("done", "terminated", "truncated"):
            observations[key] = torch.zeros(
                (*self.batch_size, 1), device=self.device, dtype=torch.bool
            )
        return observations

    def _step(self, tensordict):
        if self._grouped:
            return self._grouped_step(tensordict)
        actions = tensordict.select(*self.runtime.contract.actions)
        transition = self.runtime.step(actions)
        observations = transition.observation
        CompiledPolicy._validate_fields(
            observations, self.runtime.contract.observations, "Observation"
        )
        if transition.reward is None:
            raise ValueError("Reinforcement environments must return a task reward")
        observations.update(
            {
                "reward": transition.reward,
                "terminated": transition.terminated,
                "truncated": transition.truncated,
                "done": transition.terminated | transition.truncated,
            }
        )
        return observations

    def _grouped_step(self, tensordict):
        contract = self.runtime.contract
        actions = TensorDict(
            {
                name: tensordict[name].select(*group.actions)
                for name, group in contract.groups.items()
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        transition = self.runtime.step(actions)
        observations = transition.observation
        self._validate_grouped_observation(observations)
        shape = (*self.batch_size, 1)
        for key in ("terminated", "truncated"):
            value = getattr(transition, key)
            self._grouped_signal(value, shape, torch.bool, key)
            observations[key] = value
        observations["done"] = transition.terminated | transition.truncated
        if contract.team_reward:
            self._grouped_signal(transition.reward, shape, torch.float32, "team reward")
            observations["reward"] = transition.reward
        elif transition.reward is not None:
            raise ValueError("Root reward requires an explicitly declared genuine team signal")
        signals = transition.groups
        if signals is None or tuple(signals.batch_size) != tuple(self.batch_size):
            raise ValueError("Grouped transitions require root [N] per-group signals")
        for name, group in contract.groups.items():
            child = signals[name]
            shape = (*self.batch_size, len(group.agents), 1)
            if tuple(child.batch_size) != shape[:-1]:
                raise ValueError("Grouped transition signals require child batch [N,A]")
            for key in (
                "reward",
                "terminated",
                "truncated",
                "done",
                "valid",
                "raw_terminated",
                "raw_truncated",
            ):
                self._grouped_signal(
                    child[key], shape, torch.float32 if key == "reward" else torch.bool, key
                )
                # Current validity belongs to the input sample, not the next live observation.
                if key != "valid":
                    observations[name, key] = child[key]
            if not torch.equal(child["done"], child["terminated"] | child["truncated"]):
                raise ValueError("Grouped done must equal terminated | truncated")
            if contract.team_reward:
                team = transition.reward.unsqueeze(-2).expand(shape)
                if not torch.equal(child["reward"], team):
                    raise ValueError("Declared team reward differs from actual per-agent rewards")
                observations[name, "reward"] = team
        return observations

    def _set_seed(self, seed):
        self._seed = seed

    def close(self, *, raise_if_closed=True):
        if not self.is_closed:
            self.runtime.close()
        self.is_closed = True
