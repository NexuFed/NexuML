"""Smoke-test an installed CUDA PPO learner with no native simulator packages."""

import sys
from importlib.util import find_spec
from pathlib import Path

import nexuml
import nexuml_library
import torch
from lightning.pytorch.callbacks import Callback

from nexuml.core.export import load_inference_package
from nexuml.core.types import ExportSpec
from nexuml.execution import run
from nexuml_library.reinforcement.algorithms.ppo import PPO
from nexuml_library.scenarios.reinforcement.gymnasium import pendulum_ppo


def main(output: Path) -> None:
    """Require a native-free wheel install, CUDA fit, export and fresh resume."""
    prefix = Path(sys.prefix).resolve()
    for package in (nexuml, nexuml_library):
        assert Path(package.__file__).resolve().is_relative_to(prefix)
    for name in ("isaaclab", "isaacsim", "mujoco", "newton", "warp", "rclpy", "carla"):
        assert find_spec(name) is None, f"Unexpected simulator package: {name}"
    assert torch.cuda.is_available(), "The installed learner smoke requires a CUDA GPU"

    checkpoint = output / "trainer.ckpt"
    scenario = pendulum_ppo(total_frames=32, frames_per_batch=16)
    scenario.logging = None
    scenario.training.accelerator = "gpu"
    scenario.training.devices = 1
    scenario.training.algorithm = PPO(minibatch_size=8, update_epochs=1)

    class ResumeProof(Callback):
        restored = False

        def on_train_start(self, trainer, pl_module):
            assert int(pl_module.get_buffer("frames")) == 32
            assert trainer.global_step == 8
            self.restored = True

    proof = ResumeProof()
    for frames, resume in ((32, None), (48, checkpoint)):
        policy = output / f"policy-{frames}"
        scenario.training.total_frames = frames
        scenario.exports = [ExportSpec(output=str(policy))]
        result = run(
            scenario,
            trainer_checkpoint=resume,
            enable_progress_bar=False,
            observer=proof if resume is not None else None,
            on_trained=lambda fit: fit.trainer.save_checkpoint(checkpoint),
        )
        assert result.lightning_module.fit_device == "cuda:0"
        assert result.trainer.global_step == frames // 4
        _, _, metadata = load_inference_package(policy)
        assert metadata["training_frames"] == frames
    assert proof.restored
    print("Installed CUDA learner verified: fit/export 32 frames, resume/export 48 frames")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
