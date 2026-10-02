"""Gymnasium reset/step/close behind the framework-neutral NexuML contract."""

from typing import Any, cast

import numpy as np
import torch
from pydantic import Field
from tensordict import TensorDict

from nexuml.core.components import EnvironmentBuildContext, EnvironmentDefinition
from nexuml.core.discovery import environment
from nexuml.core.policy import CompiledPolicy
from nexuml.core.types import InteractionContract, TensorFieldContract
from nexuml.interaction.runtime import InteractionStep


def _gym():
    try:
        import gymnasium
    except ImportError as error:
        raise ImportError("Gymnasium environments require nexuml[reinforcement]") from error
    return gymnasium


@environment("GymnasiumEnvironment")
class GymnasiumEnvironment(EnvironmentDefinition):
    """Gym configuration; explicit contracts allow simulator-free description."""

    env_name: str = "Pendulum-v1"
    kwargs: dict[str, Any] = Field(default_factory=dict)
    contract: InteractionContract | None = None
    image_shape: tuple[int, int, int] | None = None

    def _make(self):
        gym = _gym()
        kwargs = dict(self.kwargs)
        if self.image_shape is not None:
            kwargs["render_mode"] = "rgb_array"
        try:
            return gym.make(self.env_name, **kwargs)
        except gym.error.DependencyNotInstalled as error:
            extra = (
                "mujoco"
                if "mujoco" in self.env_name.lower()
                or "Humanoid" in self.env_name
                or "Reacher" in self.env_name
                else "reinforcement"
            )
            raise ImportError(
                f"{self.env_name} requires nexuml-library[{extra}]: {error}"
            ) from error

    def describe(self) -> InteractionContract:
        if self.contract is not None:
            return self.contract
        probe = self._make()
        try:
            observations = _observation_fields(probe.observation_space)
            if self.image_shape is not None:
                observations = {
                    "image": TensorFieldContract(
                        shape=self.image_shape,
                        dtype="uint8",
                        modality="image",
                        low=0,
                        high=255,
                    )
                }
            action, semantics = _action_field(probe.action_space)
            return InteractionContract(
                observations=observations,
                actions={"action": action},
                action_space=semantics,
            )
        finally:
            probe.close()

    def build(self, context: EnvironmentBuildContext):
        return _GymnasiumRuntime(self, context)


def _observation_fields(space):
    gym = _gym()
    spaces = space.spaces if isinstance(space, gym.spaces.Dict) else {"observation": space}
    fields = {}
    for key, leaf in spaces.items():
        if isinstance(leaf, gym.spaces.Box):
            fields[key] = TensorFieldContract(
                shape=leaf.shape, dtype=leaf.dtype.name, modality="vector"
            )
        elif isinstance(leaf, gym.spaces.Discrete) and leaf.start == 0:
            fields[key] = TensorFieldContract(shape=(), dtype="int64", num_values=leaf.n)
        else:
            raise ValueError(f"Unsupported observation field {key!r}: {type(leaf).__name__}")
    return fields


def _action_field(space):
    gym = _gym()
    if isinstance(space, gym.spaces.Box):
        if not np.isfinite(space.low).all() or not np.isfinite(space.high).all():
            raise ValueError("Continuous actions require finite Gymnasium bounds")
        return TensorFieldContract(
            shape=space.shape,
            dtype=space.dtype.name,
            low=space.low.flatten().tolist(),
            high=space.high.flatten().tolist(),
        ), "continuous"
    if isinstance(space, gym.spaces.Discrete) and space.start == 0:
        return TensorFieldContract(shape=(), dtype="int64", num_values=space.n), "discrete"
    raise ValueError(f"Unsupported action field 'action': {type(space).__name__}")


class _GymnasiumRuntime:
    def __init__(self, definition, context):
        self.definition, self.context = definition, context
        self.contract = definition.describe().model_copy(update={"num_envs": context.num_envs})
        self.envs = []
        self._observations = [None] * context.num_envs
        try:
            # ponytail: synchronous vectors; use Gym's async vector API when throughput matters.
            for _ in range(context.num_envs):
                self.envs.append(definition._make())
        except BaseException:
            self.close()
            raise

    def _observation(self, env, observation):
        if self.definition.image_shape is not None:
            try:
                image = env.render()
            except ImportError as error:
                raise ImportError(
                    "Visual Gymnasium environments require nexuml-library[rendering]"
                ) from error
            _, height, width = self.definition.image_shape
            rows = np.linspace(0, image.shape[0] - 1, height, dtype=int)
            cols = np.linspace(0, image.shape[1] - 1, width, dtype=int)
            observation = {"image": image[rows[:, None], cols].transpose(2, 0, 1).copy()}
        elif not isinstance(observation, dict):
            observation = {"observation": observation}
        tensors = {
            key: torch.as_tensor(value, device=self.context.device)
            for key, value in observation.items()
        }
        return TensorDict(cast(Any, tensors), batch_size=[])

    def _batch(self):
        if self.context.num_envs == 1:
            return self._observations[0].clone()
        return torch.stack(self._observations)

    def reset(self, *, seed=None, mask=None):
        selected = [True] * len(self.envs) if mask is None else mask.reshape(-1).tolist()
        for index, env in enumerate(self.envs):
            if selected[index] or self._observations[index] is None:
                observation, _ = env.reset(seed=None if seed is None else seed + index)
                self._observations[index] = self._observation(env, observation)
        return self._batch()

    def step(self, actions):
        CompiledPolicy._validate_fields(actions, self.contract.actions, "Action")
        rewards, terminated, truncated = [], [], []
        for index, env in enumerate(self.envs):
            tensor = actions["action"] if len(self.envs) == 1 else actions["action"][index]
            action = tensor.detach().cpu().numpy()
            if self.contract.action_space == "discrete":
                action = int(action)
            observation, reward, term, trunc, _ = env.step(action)
            self._observations[index] = self._observation(env, observation)
            rewards.append(reward)
            terminated.append(term)
            truncated.append(trunc)
        shape = (1,) if len(self.envs) == 1 else (len(self.envs), 1)
        device = self.context.device
        return InteractionStep(
            observation=self._batch(),
            reward=torch.tensor(rewards, device=device, dtype=torch.float32).reshape(shape),
            terminated=torch.tensor(terminated, device=device, dtype=torch.bool).reshape(shape),
            truncated=torch.tensor(truncated, device=device, dtype=torch.bool).reshape(shape),
        )

    def close(self):
        for env in self.envs:
            env.close()
        self.envs.clear()
