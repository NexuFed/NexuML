"""Maintained task registration and explicit reference customization, loaded lazily."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def configure_state_reference(cfg: Any) -> None:
    """Retain the maintained manager references' explicitly selected state distribution."""
    cfg.observations.policy.enable_corruption = False
    cfg.observations.policy.history_length = 0


def configure_custom_cartpole(cfg: Any, *, episode_length_s: float, env_spacing: float) -> Any:
    """Customize the registered example while retaining its native control semantics.

    Returns:
        The configured native task instance.

    Raises:
        ValueError: If horizon or spacing is nonpositive.
    """
    if episode_length_s <= 0 or env_spacing <= 0:
        raise ValueError("Custom Cartpole horizon and environment spacing must be positive")
    configure_state_reference(cfg)
    cfg.episode_length_s = episode_length_s
    cfg.scene.env_spacing = env_spacing
    return cfg


def make_custom_cartpole_config() -> Any:
    """Build the installed custom task's native config only when Gym resolves it.

    Returns:
        Its maintained manager-based native configuration.
    """
    cfg = import_module("isaaclab_tasks.core.cartpole.cartpole_manager_env_cfg").CartpoleEnvCfg()
    cfg.episode_length_s = 3.0
    cfg.scene.env_spacing = 2.0
    return cfg


def register_reference_tasks() -> None:
    """Register maintained tasks through native Gym, not a parallel simulator registry.

    Raises:
        ValueError: If an existing registration replaced a maintained task.
    """
    import gymnasium as gym

    manager = "isaaclab.envs:ManagerBasedRLEnv"
    for task, entry, config in (
        (
            "NexuML-SO101-Joint-Reach",
            manager,
            "nexuml_library.environments._isaaclab_so101:make_so101_config",
        ),
        (
            "NexuML-Custom-Cartpole",
            manager,
            "nexuml_library.environments._isaaclab_tasks:make_custom_cartpole_config",
        ),
        (
            "NexuML-Cartpole-Categorical",
            "nexuml_library.environments._isaaclab_categorical:make_categorical_cartpole_environment",
            "nexuml_library.environments._isaaclab_categorical:make_categorical_cartpole_config",
        ),
        (
            "NexuML-Cartpole-Sensors",
            manager,
            "isaaclab_tasks.core.cartpole.cartpole_manager_camera_env_cfg:CartpoleCameraEnvCfg",
        ),
    ):
        registered = gym.registry.get(task)
        if registered is None:
            gym.register(
                id=task,
                entry_point=entry,
                disable_env_checker=True,
                kwargs={"env_cfg_entry_point": config},
            )
        elif (
            registered.entry_point != entry
            or registered.kwargs.get("env_cfg_entry_point") != config
        ):
            raise ValueError(f"The maintained task registration has been replaced: {task}")
