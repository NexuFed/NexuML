"""Private conversion of NexuML tensor interaction semantics to TorchRL."""

import torch

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
        batch = torch.Size(()) if contract.num_envs == 1 else torch.Size((contract.num_envs,))
        self._seed = seed
        super().__init__(device=device, batch_size=batch, run_type_checks=True)
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

    def _reset(self, tensordict=None, **kwargs):
        mask = None if tensordict is None else tensordict.get("_reset", None)
        observations = self.runtime.reset(seed=self._seed, mask=mask)
        self._seed = None
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

    def _set_seed(self, seed):
        self._seed = seed

    def close(self, *, raise_if_closed=True):
        if not self.is_closed:
            self.runtime.close()
        self.is_closed = True
