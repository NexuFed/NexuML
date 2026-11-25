"""The versioned SO-101 held-out joint-reach suite, composed with Arena."""

from collections import deque
from importlib import import_module
from typing import Any

import torch

from nexuml.core.types import FactorySpec, PolicyEvaluationSpec, ScenarioSpec
from nexuml_library.environments.isaaclab import IsaacLabEnvironment
from nexuml_library.evaluation.arena import ArenaBenchmarkEnvironment
from nexuml_library.scenarios.reinforcement.isaaclab import isaac_so101_joint_reach_ppo

CONDITIONS = {"near_home": 10000, "left": 10001, "right": 10002}
_TARGETS_RAD = {
    "near_home": (0.06, -0.24, 0.23, -0.05, 0.06),
    "left": (0.16, -0.19, 0.22, -0.13, 0.18),
    "right": (-0.17, -0.29, 0.28, 0.12, -0.18),
}


def make_reach_condition(*, condition: str, seed: int, device: str) -> tuple[Any, Any]:
    """Compose the stock reach controls, scene and held-success predicate in Arena.

    Returns:
        The Arena-built environment and verified SO-101 policy interface.

    Raises:
        ValueError: If Arena changes the pinned native action or observation terms.
    """
    from nexuml_library.environments._isaaclab_so101 import (
        HOME,
        JOINT_NAMES,
        TARGET_SCALE,
        make_so101_config,
        validate_asset,
    )

    target_rad = _TARGETS_RAD[condition]
    cfg = make_so101_config(device=device, seed=seed)
    configclass = import_module("isaaclab.utils").configclass
    mdp = import_module("isaaclab.envs.mdp")
    EmbodimentBase = import_module("isaaclab_arena.embodiments.embodiment_base").EmbodimentBase
    Scene = import_module("isaaclab_arena.scene.scene").Scene
    NoTask = import_module("isaaclab_arena.tasks.no_task").NoTask
    criteria = import_module("isaaclab_arena.progress_tracking.completion_criteria")
    Termination = import_module("isaaclab_arena.tasks.task_termination_cfg").TaskTerminationCfg
    Recorder = import_module(
        "isaaclab_arena.recording.episode_recorder_manager"
    ).EpisodeRecorderTermCfg
    Environment = import_module(
        "isaaclab_arena.environments.isaaclab_arena_environment"
    ).IsaacLabArenaEnvironment
    builder = import_module("isaaclab_arena.environments.arena_env_builder")
    trace: deque = deque(maxlen=128)
    stock_reset = cfg.events.reset_joint_reach.func
    stock_success = cfg.terminations.success

    def reset(env, env_ids):
        stock_reset(env, env_ids)
        target = env.joint_reach_target
        target[slice(None) if env_ids is None else env_ids] = target.new_tensor(target_rad)
        trace.clear()

    def held_success(env):
        result = stock_success.func(env, **stock_success.params)
        robot = env.scene["robot"]
        positions = robot.data.joint_pos.torch[0, : len(JOINT_NAMES)]
        trace.append(
            {
                "step": int(env.episode_length_buf[0].item()),
                "joint_error_rad": float((positions - env.joint_reach_target[0]).abs().mean()),
            }
        )
        return result

    def record(env, env_id):
        manager = env.termination_manager
        success = manager.get_term("success")[env_id].item()
        timeout = manager.time_outs[env_id].item()
        reason = "success" if success else "native_truncated" if timeout else "native_terminated"
        return {
            "termination_reason": reason,
            "start_joint_position_rad": list(HOME),
            "goal_joint_position_rad": list(target_rad),
            "joint_error_rad": float(env.joint_reach_error[env_id].item()),
            "failure_trace": None if success else list(trace),
        }

    @configclass
    class RobotCfg:
        robot = cfg.scene.robot

    @configclass
    class ReachSceneCfg:
        ground = cfg.scene.ground
        light = cfg.scene.light

    class ReachScene(Scene):
        def get_scene_cfg(self):
            return ReachSceneCfg()

    class ReachArm(EmbodimentBase):
        registry_name = "nexuml_so101_fixed_gripper"

        def __init__(self):
            super().__init__()
            self.scene_config = RobotCfg()
            self.action_config = cfg.actions
            self.observation_config = cfg.observations

    class ReachTask(NoTask):
        def get_termination_cfg(self):
            return Termination(
                timeout_s=cfg.episode_length_s,
                success=[
                    criteria.CompletionCriteria(
                        name="held_joint_goal", predicate_sequence=[held_success]
                    )
                ],
            )

        def get_events_cfg(self):
            cfg.events.reset_joint_reach.func = reset
            return cfg.events

        def get_rewards_cfg(self):
            return cfg.rewards

    def configure(config):
        config.sim.dt = cfg.sim.dt
        config.sim.render_interval = cfg.sim.render_interval
        config.sim.physics = cfg.sim.physics.isaacsim_physx
        config.decimation = cfg.decimation
        return config

    environment = Environment(
        name=f"NexuML-Arena-SO101-{condition}",
        scene=ReachScene(),
        embodiment=ReachArm(),
        task=ReachTask(),
        env_cfg_callback=configure,
        episode_recorder_terms={"nexuml_reach": Recorder(func=record)},
    )
    env = builder.ArenaEnvBuilder(
        environment,
        builder.ArenaEnvBuilderCfg(
            device=device, num_envs=1, seed=seed, env_spacing=1.5, solve_relations=False
        ),
    ).make_registered()
    native = env.unwrapped
    try:
        validate_asset(native)
        if native.action_manager.active_terms != ["arm_action"]:
            raise ValueError("Arena SO-101 must expose only its five arm joints")
        term = native.action_manager.get_term("arm_action")
        if (
            type(term) is not mdp.JointPositionAction
            or term.IO_descriptor.joint_names != list(JOINT_NAMES)
            or term.cfg.scale != TARGET_SCALE
            or term.cfg.use_default_offset
            or term.cfg.clip is not None
            or torch.count_nonzero(torch.as_tensor(term.IO_descriptor.offset))
            or native.observation_manager.active_terms.get("policy")
            != ["joint_pos", "joint_vel", "target"]
        ):
            raise ValueError("Arena SO-101 control or observation semantics differ from the policy")
    except BaseException as error:
        try:
            import_module("isaaclab_arena.evaluation.resource_cleanup").close_environment(env)
        except BaseException as cleanup:
            error.add_note(f"Arena SO-101 cleanup also failed: {cleanup}")
        raise
    return env, IsaacLabEnvironment(task="NexuML-SO101-Joint-Reach").describe()


def so101_arena_evaluation() -> ScenarioSpec:
    """Select SO-101's frozen Arena evaluation without changing training or weight ownership.

    Returns:
        Scenario with an evaluation-only Arena suite and a fixed episode budget.
    """
    scenario = isaac_so101_joint_reach_ppo()
    assert scenario.interaction is not None
    stock = scenario.interaction.environment
    assert isinstance(stock, IsaacLabEnvironment)
    scenario.interaction.evaluation_environment = ArenaBenchmarkEnvironment(
        suite="so101-joint-reach",
        revision="1",
        contract=stock.describe(),
        bindings=stock.native_bindings(),
        factory=FactorySpec.from_factory(make_reach_condition),
        conditions=CONDITIONS,
    )
    scenario.policy_evaluation = PolicyEvaluationSpec(episodes=len(CONDITIONS))
    return scenario
