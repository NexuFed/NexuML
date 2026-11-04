"""Native terminal-scene pixel oracle; reuse the production camera and no media writer."""

import numpy as np
import torch
from tensordict import TensorDict
from unittest.mock import patch

from nexuml.core.components import EnvironmentBuildContext
from nexuml_library.environments import _isaaclab_rgb as rgb
from nexuml_library.environments._isaaclab_runtime import _IsaacLabRuntime
from nexuml_library.environments.isaaclab import IsaacLabEnvironment


def main():
    """Read an independent public pre-reset recorder against a displaced native cart."""
    original = rgb.configure_rgb
    oracle = {}

    def with_oracle(cfg, owner) -> None:
        original(cfg, owner)
        from isaaclab.managers import RecorderTerm, RecorderTermCfg

        class PreResetOracle(RecorderTerm):
            def record_pre_reset(self, env_ids):
                if owner._in_step:
                    native = self._env
                    image = native.scene["nexuml_replay_camera"].data.output["rgb"].torch
                    oracle["image"] = image[0, ..., :3].cpu().numpy().copy()
                    robot = native.scene["robot"]
                    oracle["positions"] = robot.data.joint_pos.torch.clone()
                return None, None

        cfg.recorders.independent_pixel_oracle = RecorderTermCfg(class_type=PreResetOracle)

    # Extend only our config seam; the native recorder manager and camera are unmodified.
    with patch.object(rgb, "configure_rgb", with_oracle):
        runtime = IsaacLabEnvironment().build(
            EnvironmentBuildContext(
                purpose="eval",
                device="cuda:0",
                num_envs=1,
                seed=17,
                render_mode="rgb_array",
            )
        )
    assert isinstance(runtime, _IsaacLabRuntime)
    try:
        runtime.reset()
        native = runtime.native
        robot = native.scene["robot"]
        ids, names = robot.find_joints("slider_to_cart")
        assert names == ["slider_to_cart"]
        positions = robot.data.joint_pos.torch.clone()
        positions[:, ids] = 3.5
        robot.write_joint_position_to_sim_index(position=positions)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(positions))
        step = runtime.step(TensorDict({"action": torch.zeros(1, device="cuda:0")}, []))
        assert step.terminated.all()
        assert oracle["positions"][0, ids[0]] > 3.0
        assert robot.data.joint_pos.torch[0, ids[0]].abs() < 1.0
        terminal = runtime.render_frame()
        assert np.array_equal(terminal, oracle["image"])
        assert terminal.shape == (240, 320, 3) and terminal.dtype == np.uint8
        current = rgb.read_rgb(native)
        difference = np.abs(terminal.astype(float) - current.astype(float))
        assert (difference.max(axis=-1) > 20).sum() > 100, "terminal image shows the reset scene"
        saved = terminal.copy()
        terminal[:] = 0
        assert np.array_equal(runtime.render_frame(), saved), "caller mutated the cached frame"
        print("NATIVE_TERMINAL_PIXEL_AND_SCENE_IDENTITY_VERIFIED", flush=True)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
