"""Measure one native transition from a fresh, seeded runtime."""

from functools import partial
import json
from pathlib import Path
import sys
from time import monotonic_ns
from typing import cast

import pytest
import torch
from tensordict import TensorDict

from nexuml.core.compiler import compile
from nexuml.core.components import EnvironmentBuildContext
from nexuml.core.policy import CompiledPolicy
from nexuml.interaction import TraceableEnvironmentRuntime
from nexuml_library.environments._isaaclab_runtime import _IsaacLabRuntime
from nexuml_library.environments.isaaclab import IsaacLabEnvironment
from nexuml_library.scenarios.reinforcement.isaaclab import isaac_cartpole_ppo


def main(phase: str, output: Path) -> None:
    device = "cuda:0"
    scenario = isaac_cartpole_ppo()
    assert scenario.interaction is not None
    definition = scenario.interaction.environment
    assert isinstance(definition, IsaacLabEnvironment)
    purpose = "eval" if phase == "policy" else "play"
    with definition.job_scope():
        traced = definition.build(EnvironmentBuildContext(purpose=purpose, device=device))
        try:
            assert isinstance(traced, TraceableEnvironmentRuntime)
            runtime = cast(_IsaacLabRuntime, traced)
            initial = traced.reset(seed=23)
            term = runtime.native.action_manager.get_term("joint_effort")
            frame = runtime.native.common_step_counter
            if phase == "policy":
                torch.manual_seed(17)
                policy = CompiledPolicy.from_pipeline(compile(scenario)).to(device).eval()
                with torch.inference_mode():
                    action = policy(initial, deterministic=True)
            else:
                values = (
                    [0.5]
                    if phase == "known"
                    else json.loads((output / "policy.json").read_text())["requested_action"]
                )
                action = TensorDict({"action": torch.tensor(values, device=device)}, [])
            if phase == "known":
                goal = cast(TensorDict, {"target": 0.5, "duration_s": 0.5})
                invalid = TensorDict({"action": torch.tensor([2.0], device=device)}, [])
                for step in (runtime.step, partial(traced.step_recorded, source="script")):
                    with pytest.raises(TypeError, match="synchronous TensorDict"):
                        step(goal)
                    with pytest.raises(ValueError, match="upper bound"):
                        step(invalid)
                assert runtime.native.common_step_counter == frame
            source = "policy:seed-17" if phase == "policy" else f"script:{phase}"
            before = monotonic_ns()
            transition = traced.step_recorded(action, source=source)
            trace = transition.trace
            assert trace is not None and trace.source == source
            assert before <= trace.issued_monotonic_ns <= monotonic_ns()
            assert trace.native_clock == "isaaclab.control_step"
            assert trace.control_period_s == definition.describe().control_period_s
            assert (trace.native_frame_before, trace.native_frame_after) == (frame, frame + 1)
            torch.testing.assert_close(trace.requested_action["action"], action["action"])
            torch.testing.assert_close(trace.native_action, action["action"].unsqueeze(0))
            force = term.processed_actions.item()
            assert force == pytest.approx(100 * trace.native_action.item())
            assert transition.reward is not None
            if phase == "known":
                assert force == 50.0
                assert transition.observation["observation"][2] > initial["observation"][2]
            row = {
                "purpose": purpose,
                "source": trace.source,
                "requested_action": trace.requested_action["action"].tolist(),
                "submitted_action": trace.native_action.tolist(),
                "force_n": force,
                "issued_monotonic_ns": trace.issued_monotonic_ns,
                "native_clock": trace.native_clock,
                "native_frames": [trace.native_frame_before, trace.native_frame_after],
                "control_period_s": trace.control_period_s,
                "initial_observation": initial["observation"].tolist(),
                "observation": transition.observation["observation"].tolist(),
                "reward": transition.reward.item(),
                "terminated": transition.terminated.item(),
                "truncated": transition.truncated.item(),
            }
            if phase == "known":
                obs = traced.reset(seed=23)["observation"]
                native_obs = runtime.native.obs_buf["policy"].squeeze(0)
                torch.testing.assert_close(obs, native_obs)
                row["native_in_process_reset_observation"] = obs.tolist()
            (output / f"{phase}.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
        finally:
            traced.close()


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
