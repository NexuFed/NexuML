"""Distinct native categorical Cartpole; imports do not construct native resources."""

from functools import cache
from importlib import import_module
from typing import Any

import torch


ACTION_FORCES = (-100.0, 0.0, 100.0)


def decode_categorical_force(actions: torch.Tensor) -> torch.Tensor:
    """Decode one class per native environment into its declared joint effort.

    Returns:
        Float32 force commands of shape ``[N, 1]`` on the input device.

    Raises:
        ValueError: If commands are not scalar int64 classes in the declared range.
    """
    if actions.dtype != torch.int64 or actions.ndim != 1:
        raise ValueError("Categorical Cartpole requires int64 classes with native shape [N]")
    if torch.any((actions < 0) | (actions >= len(ACTION_FORCES))):
        raise ValueError("Categorical Cartpole class must be 0, 1 or 2")
    return actions.new_tensor(ACTION_FORCES, dtype=torch.float32)[actions].unsqueeze(-1)


@cache
def categorical_cartpole_type() -> type:
    """Resolve the registered native class only inside the prepared execution job.

    Returns:
        Native direct Cartpole with an explicit class-to-effort decoder.
    """
    parent = import_module("isaaclab_tasks.core.cartpole.cartpole_direct_env").CartpoleEnv

    class CategoricalCartpoleEnv(parent):
        action_forces = ACTION_FORCES

        def _pre_physics_step(self, actions: torch.Tensor) -> None:
            self.actions = decode_categorical_force(actions)

    return CategoricalCartpoleEnv


def make_categorical_cartpole_config() -> Any:
    """Build the categorical task's native configuration through Gym resolution.

    Returns:
        Direct Cartpole config with a scalar categorical action space.
    """
    from gymnasium.spaces import Discrete

    cfg: Any = import_module(
        "isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg"
    ).CartpoleEnvCfg()
    cfg.action_space = Discrete(len(ACTION_FORCES))
    cfg.episode_length_s = 2.0
    return cfg


def make_categorical_cartpole_environment(cfg, **kwargs):
    """Construct Gym's distinctly registered categorical reference.

    Returns:
        One owned native direct environment batch.
    """
    return categorical_cartpole_type()(cfg=cfg, **kwargs)
