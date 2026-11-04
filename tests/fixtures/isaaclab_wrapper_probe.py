"""Native public-wrapper diagnostic, not a passing-wrapper or learning claim."""

import json
import sys

import torch
from tensordict import TensorDict
from torchrl.envs import TransformedEnv
from torchrl.envs.libs.isaac_lab import IsaacLabWrapper
from torchrl.objectives.value.functional import generalized_advantage_estimate

from nexuml.core.components import EnvironmentBuildContext
from nexuml_library.environments._isaaclab_runtime import _IsaacLabRuntime
from nexuml_library.environments.isaaclab import IsaacLabEnvironment


def main():
    """Probe genuine native class wiring and the documented public info reader."""
    task, mode = sys.argv[1:]
    definition = IsaacLabEnvironment.model_validate({"task": task, "physics": "newton_mjwarp"})
    runtime = definition.build(
        EnvironmentBuildContext(purpose="train", device="cuda:0", num_envs=4, seed=17)
    )
    assert isinstance(runtime, _IsaacLabRuntime)
    native = runtime.native
    resets, observed = [], {}
    original_reset, original_step = native.reset, native.step

    def reset(**kwargs):
        resets.append(kwargs)
        return original_reset(**kwargs)

    def step(action):
        result = original_step(action)
        obs, reward, terminated, truncated, info = result
        observed.update(
            reset=obs["policy"].clone(),
            reward=reward.clone(),
            final=info.get("final_obs", {}).get("policy", obs["policy"]).clone(),
            terminated=terminated.clone(),
            truncated=truncated.clone(),
        )
        return result

    native.reset, native.step = reset, step
    result = {"task": task, "mode": mode, "native_ready": True}
    env = None
    try:
        env = IsaacLabWrapper(
            native, device=torch.device("cuda:0"), native_autoreset=mode.startswith("native")
        )
        if mode.endswith("reader"):

            def final_reader(info, td):
                if "final_obs" in info:
                    td["final"] = TensorDict(
                        {"policy": info["final_obs"]["policy"].clone()}, batch_size=[4]
                    )
                return td

            base = env.base_env if isinstance(env, TransformedEnv) else env
            base.set_info_dict_reader(final_reader)
        root = env.reset()
        key = "observation" if "observation" in root else "policy"
        native.episode_length_buf[1] = native.max_episode_length - 1
        before = len(resets)
        root["action"] = torch.zeros((4, 1), device="cuda:0")
        first, next_root = env.step_and_maybe_reset(root)
        obs = first["next", key]
        assert isinstance(obs, torch.Tensor)
        result["no_double_reset"] = len(resets) == before
        result["terminal_truth"] = torch.allclose(obs[1], observed["final"][1])
        result["reset_truth"] = torch.allclose(next_root[key], observed["reset"])
        expected = observed["reward"][1] + 0.9 * (observed["final"][1, 0] + 100)
        _, target = generalized_advantage_estimate(
            0.9,
            0.95,
            torch.zeros((4, 1, 1), device="cuda:0"),
            obs[:, None, :1] + 100,
            first["next", "reward"][:, None],
            first["next", "done"][:, None],
            first["next", "terminated"][:, None],
        )
        result["timeout_target_truth"] = torch.allclose(target[1, 0, 0], expected)
        snapshot = obs.clone()
        next_root["action"] = torch.zeros((4, 1), device="cuda:0")
        _, next_root = env.step_and_maybe_reset(next_root)
        result["stable_observation_buffer"] = torch.allclose(obs, snapshot, equal_nan=True)
        before_lengths = native.episode_length_buf.clone()
        next_root["_reset"] = torch.tensor([[True], [False], [False], [False]], device="cuda:0")
        env.reset(next_root)
        result["manual_partial_reset_truth"] = bool(
            native.episode_length_buf[0] == 0
            and torch.equal(native.episode_length_buf[1:], before_lengths[1:])
        )
    except (AssertionError, RuntimeError, TypeError, KeyError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            runtime.close()
    result["conformant"] = (
        all(
            result.get(key) is True
            for key in (
                "no_double_reset",
                "terminal_truth",
                "reset_truth",
                "timeout_target_truth",
                "stable_observation_buffer",
                "manual_partial_reset_truth",
            )
        )
        and "error" not in result
    )
    print("WRAPPER_ORACLE_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
