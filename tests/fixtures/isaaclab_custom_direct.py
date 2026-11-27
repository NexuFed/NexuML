"""Exercise one native registration through Isaac's runner and NexuML in fresh jobs."""

from contextlib import contextmanager
from importlib import import_module
import json
from pathlib import Path
import sys
from typing import Any, cast
from unittest.mock import patch

import torch
import pytest


TASK = "NexuML-Cartpole-Direct-Centering"
MODULE = "nexuml_isaac_tasks.centering_cartpole"
DEVICE = "cuda:0"
SEED, WIDTH, STEPS = 23, 8, 60


@contextmanager
def native_job():
    import gymnasium as gym
    from isaaclab.app import launch_simulation
    from isaaclab.sim import SimulationContext
    from isaaclab_tasks.utils import parse_env_cfg

    example = import_module(MODULE)
    cfg = parse_env_cfg(TASK, device=DEVICE, num_envs=WIDTH)
    cfg.seed = SEED
    args = {
        "headless": True,
        "device": DEVICE,
        "visualizer": ["none"],
        "visualizer_explicit": True,
    }
    with launch_simulation(cfg, args):
        env = gym.make(TASK, cfg=cfg)
        try:
            assert type(env.unwrapped) is example.CenteringCartpole
            assert type(env.unwrapped.cfg) is example.CenteringCartpoleCfg
            yield env
        finally:
            env.close()
    assert SimulationContext.instance() is None
    assert "nexuml" not in sys.modules


def controls(native, runtime=None):
    """Row zero drives out of bounds; the other rows reach the time limit."""
    from tensordict import TensorDict

    if runtime is None:
        initial = native.reset(seed=SEED)[0]["policy"].clone()
    else:
        initial = runtime.reset(seed=SEED)["observation"]
        empty = torch.arange(WIDTH, device=DEVICE) == 0
        with pytest.raises(ValueError, match="not explicit partial native reset"):
            runtime.reset(mask=empty)
        assert (native.episode_length_buf == 0).all()
    assert native.step_dt == pytest.approx(1 / 60) and native.max_episode_length == STEPS
    frames = []
    action = torch.zeros((WIDTH, 1), device=DEVICE)
    action[0] = 1.0
    force = None
    for tick in range(STEPS):
        if runtime is None:
            obs, reward, term, trunc, info = native.step(action)
            done = term | trunc
            next_obs = obs["policy"]
            terminal = next_obs.clone()
            if done.any():
                terminal[done] = info["final_obs"]["policy"][done]
        else:
            transition = runtime.step(TensorDict({"action": action}, [WIDTH]))
            reward = transition.reward.squeeze(-1)
            term = transition.terminated.squeeze(-1)
            trunc = transition.truncated.squeeze(-1)
            done = term | trunc
            terminal = transition.observation["observation"]
            next_obs = runtime.reset(mask=done)["observation"] if done.any() else terminal
        if tick == 0:
            force = native.actions[0, 0].item()
            assert force == 100.0
            assert terminal[0, 2] > initial[0, 2] and not done.any()
        assert (native.episode_length_buf[done] == 0).all()
        assert (terminal[done] != next_obs[done]).any(dim=-1).all()
        frames.append(
            tuple(value.cpu().clone() for value in (terminal, next_obs, reward, term, trunc))
        )
        if term[0]:
            action[0] = 0.0
    terminal, resets, reward, terminated, truncated = map(torch.stack, zip(*frames, strict=True))
    assert terminated[:, 0].sum() == 1 and not terminated[:, 1:].any()
    assert not truncated[:-1].any() and truncated[-1, 1:].all() and not truncated[-1, 0]
    assert (terminal[..., 0][terminated].abs() > 0.5).all()
    assert torch.isfinite(reward).all()
    return dict(
        initial=initial.cpu(),
        terminal=terminal,
        resets=resets,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        asset=native.cfg.scene.cartpole.spawn.usd_path,
        force=force,
    )


def train_native(env, output):
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_rl.rsl_rl.utils import (
        check_rsl_rl_version,
        create_rsl_rl_runner,
        handle_deprecated_rsl_rl_cfg,
    )
    from isaaclab_tasks.utils import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    cfg = cast(Any, load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point"))
    cfg.device = DEVICE
    cfg.obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    cfg.algorithm.num_learning_epochs = 1
    cfg.algorithm.num_mini_batches = 2
    torch.manual_seed(SEED)
    runner = create_rsl_rl_runner(
        RslRlVecEnvWrapper(env, clip_actions=1.0),
        handle_deprecated_rsl_rl_cfg(cfg, check_rsl_rl_version()),
    )
    assert isinstance(runner, OnPolicyRunner)
    models = (runner.alg.get_policy(), runner.alg.critic)
    before = [
        torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone() for model in models
    ]
    start = env.unwrapped.common_step_counter
    with patch.object(runner.logger, "log", wraps=runner.logger.log) as logs:
        runner.learn(2)
    assert (env.unwrapped.common_step_counter - start) * env.unwrapped.num_envs == 256
    losses = [call.kwargs["loss_dict"] for call in logs.call_args_list]
    assert len(losses) == 2 and all(losses)
    for original, model in zip(before, models, strict=True):
        updated = torch.nn.utils.parameters_to_vector(model.parameters())
        assert torch.isfinite(updated).all() and not torch.equal(original, updated)
    runner.save(str(output / "native-learner.pt"))
    runner.export_policy_to_jit(str(output), "native-actor.pt")
    (output / "native-training.json").write_text(
        json.dumps({"frames": 256, "losses": losses}, allow_nan=False)
    )


def main(mode, output):
    if mode in {"native", "native-replay"}:
        with native_job() as env:
            if mode == "native":
                torch.save(controls(env.unwrapped), output / "native-controls.pt")
                train_native(env, output)
            else:
                actor = torch.jit.load(str(output / "native-actor.pt"), map_location=DEVICE)
                obs, _ = env.reset(seed=SEED + 1)
                with torch.inference_mode():
                    for _ in range(STEPS):
                        action = actor(obs["policy"]).clamp(-1, 1)
                        obs, reward, _, _, _ = env.step(action)
                        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                assert env.unwrapped.common_step_counter == STEPS
    else:
        from nexuml.core.components import EnvironmentBuildContext
        from nexuml_library.environments._isaaclab_runtime import _IsaacLabRuntime
        from nexuml_library.scenarios.reinforcement.isaaclab_custom import (
            isaac_custom_direct_cartpole_ppo,
        )

        scenario = isaac_custom_direct_cartpole_ppo()
        assert scenario.interaction is not None
        definition = scenario.interaction.environment
        with definition.job_scope():
            runtime = definition.build(
                EnvironmentBuildContext(purpose="train", device=DEVICE, num_envs=WIDTH, seed=SEED)
            )
            try:
                assert isinstance(runtime, _IsaacLabRuntime)
                torch.save(controls(runtime.native, runtime), output / "adapted-controls.pt")
            finally:
                runtime.close()
    print("CUSTOM_DIRECT_VERIFIED", mode, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
