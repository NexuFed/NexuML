"""Fresh-process helpers for opt-in Isaac integration tests, not a training entrypoint."""

import importlib.abc
import json
from pathlib import Path
import sys


def _equal(left, right):
    import torch

    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def _infer(output):
    class BlockOptional(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            first = fullname.split(".")[0]
            if first.startswith(("isaaclab", "isaacsim")) or first in {
                "omni",
                "pxr",
                "torchrl",
                "PIL",
                "imageio",
                "imageio_ffmpeg",
                "moviepy",
                "av",
            }:
                raise ImportError("optional inference runtime blocked: " + fullname)

    sys.meta_path.insert(0, BlockOptional())
    import torch
    from tensordict import TensorDict
    from nexuml.core.export import load_inference_package

    observation = TensorDict(
        torch.load(output / "observation.pt", weights_only=True), batch_size=[2]
    )
    expected = torch.load(output / "expected.pt", weights_only=True)
    policy, _, metadata = load_inference_package(output / "policy")
    with torch.no_grad():
        actual = policy(observation, deterministic=True)["action"]
    assert torch.equal(actual, expected)
    assert torch.isfinite(actual).all() and (actual.abs() <= 1).all()
    assert not metadata["training_state_available"]
    assert not any(
        name.split(".")[0]
        in {
            "isaaclab",
            "isaacsim",
            "omni",
            "pxr",
            "torchrl",
            "PIL",
            "imageio",
            "imageio_ffmpeg",
            "moviepy",
            "av",
        }
        for name in sys.modules
    )
    print("EXACT_PORTABLE_CPU_INFERENCE_VERIFIED", flush=True)


def _train(task, stage, root):
    assert stage in {"train", "rounded", "interrupt", "terminate", "resume", "complete"}
    from lightning.pytorch.callbacks import Callback
    import torch
    from tensordict import TensorDict

    from nexuml.core.export import export_package
    from nexuml.reinforcement.collector import RolloutDataset
    from nexuml.training.lightning import NexuSession
    from nexuml.training.reinforcement import NexuRLLightningModule
    from nexuml_library.environments import _isaaclab_runtime as runtime
    from nexuml_library.scenarios.reinforcement.isaaclab import (
        isaac_cartpole_ppo,
        isaac_franka_reach_ppo,
        isaac_so101_joint_reach_ppo,
    )

    output = root / stage
    output.mkdir(parents=True, exist_ok=True)
    factory = {
        "cartpole": isaac_cartpole_ppo,
        "cartpole-newton": isaac_cartpole_ppo,
        "cartpole-direct": isaac_cartpole_ppo,
        "franka": isaac_franka_reach_ppo,
        "so101": isaac_so101_joint_reach_ppo,
    }[task]
    saved = None
    if stage in {"resume", "complete"}:
        checkpoint = root / (
            "interrupt/intermediate.ckpt" if stage == "resume" else "resume/complete.ckpt"
        )
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        session = NexuSession.from_trainer_checkpoint(
            checkpoint, log_dir=output, enable_progress_bar=False
        )
    else:
        scenario = factory(
            total_frames=193 if stage == "rounded" else 256, frames_per_batch=128, num_envs=8
        )
        if task in {"cartpole-newton", "cartpole-direct"}:
            from nexuml_library.environments.isaaclab import IsaacLabEnvironment

            scenario.interaction.environment = IsaacLabEnvironment(
                task="Isaac-Cartpole-Direct" if task == "cartpole-direct" else "Isaac-Cartpole",
                physics="newton_mjwarp",
            )
        scenario.training.algorithm = scenario.training.algorithm.model_copy(
            update={"update_epochs": 1, "minibatch_size": 64}
        )
        session = NexuSession(scenario, log_dir=output, enable_progress_bar=False)
    module = session.lightning_module
    assert isinstance(module, NexuRLLightningModule)
    before = torch.nn.utils.parameters_to_vector(module.policy.parameters()).detach().cpu().clone()
    initial_frames = int(module.frames)
    shapes, collected_after, finalized = [], [], []

    class Boundary(Callback):
        def on_train_start(self, trainer, pl_module):
            """Prove learning state is restored before another rollout is updated."""
            if saved is not None:
                assert _equal(pl_module.state_dict(), saved["state_dict"])
                assert _equal(
                    [optimizer.state_dict() for optimizer in trainer.optimizers],
                    saved["optimizer_states"],
                )
                print("ACTOR_CRITIC_OPTIMIZERS_COUNTERS_RESTORED", flush=True)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            """Save an intermediate state and intentionally fail the first process.

            Raises:
                RuntimeError: To test checkpoint recovery and failed-job cleanup.
            """
            if stage in {"interrupt", "terminate"}:
                assert int(pl_module.frames) == 128
                trainer.save_checkpoint(output / "intermediate.ckpt")
                if stage == "terminate":
                    import os
                    import signal

                    os.kill(os.getpid(), signal.SIGTERM)
                    raise AssertionError("SIGTERM did not interrupt the Python job")
                raise RuntimeError("intentional checkpoint interruption")

    getattr(session.trainer, "callbacks").append(Boundary())
    loggers = session.trainer_loggers
    assert isinstance(loggers, list) and loggers
    for logger in loggers:
        original_finalize = logger.finalize

        def finalize(status, original=original_finalize):
            original(status)
            finalized.append(status)

        logger.finalize = finalize
    original_update = module.algorithm.update
    original_iteration = RolloutDataset.__iter__

    def iterate(dataset):
        for rollout in original_iteration(dataset):
            collected_after.append(int(module.batches))
            yield rollout

    setattr(RolloutDataset, "__iter__", iterate)

    def update(rollout, **kwargs):
        assert tuple(rollout.batch_size) == (8, 16)
        assert rollout.device == torch.device("cuda:0")
        assert rollout["action"].is_cuda and rollout["next", "observation"].is_cuda
        assert (rollout["action"].abs() <= 1).all()
        learner = list(module.algorithm.rollout_policy.parameters())
        active_collector = module._collector
        assert active_collector is not None
        collector = list(active_collector.policy.parameters())
        assert len(learner) == len(collector)
        assert all(torch.equal(a, b) for a, b in zip(learner, collector))
        with torch.no_grad():
            distribution = module.algorithm.rollout_policy.get_dist(rollout.clone())
            torch.testing.assert_close(
                distribution.log_prob(rollout["action"]),
                rollout["action_log_prob"],
                rtol=1e-4,
                atol=1e-5,
            )
        shapes.append(list(rollout.batch_size))
        return original_update(rollout, **kwargs)

    module.algorithm.update = update
    try:
        result = session.run()
    finally:
        assert module._collector is None and finalized
        if stage in {"interrupt", "terminate"}:
            assert finalized == ["failed"]
            assert int(module.frames) == 128 and len(shapes) == 1
            print("INTERRUPTED_CHECKPOINT_AND_FAILED_LOGGER_FINALIZED", flush=True)
    assert finalized == ["success"]
    assert int(module.frames) == 256 and int(module.batches) == 2
    assert all(torch.isfinite(value).all() for value in result.trainer.logged_metrics.values())
    if stage == "complete":
        assert not shapes and runtime._process_launcher is None and runtime._active_runtime is None
        print("COMPLETED_BUDGET_NO_SIMULATOR", flush=True)
    else:
        assert len(shapes) == (1 if stage == "resume" else 2)
        assert collected_after == ([1] if stage == "resume" else [0, 1])
        after = torch.nn.utils.parameters_to_vector(module.policy.parameters()).detach().cpu()
        assert not torch.equal(before, after)
        result.trainer.save_checkpoint(output / "complete.ckpt")
        policy = result.policy
        assert policy is not None
        policy.cpu()
        observation = TensorDict(
            {
                key: torch.zeros(2, *field.shape)
                for key, field in policy.contract.observations.items()
            },
            batch_size=[2],
        )
        with torch.no_grad():
            expected = policy(observation, deterministic=True)["action"]
        torch.save(observation.to_dict(), output / "observation.pt")
        torch.save(expected, output / "expected.pt")
        export_package(policy, output / "policy", lightning_module=module, trainer=result.trainer)
    print(
        json.dumps(
            {
                "task": task,
                "stage": stage,
                "initial_frames": initial_frames,
                "frames": int(module.frames),
                "batches": int(module.batches),
                "rollouts": shapes,
                "collected_after_batches": collected_after,
                "logger_status": finalized,
                "process_launcher_allocated": runtime._process_launcher is not None,
            }
        ),
        flush=True,
    )
    print("ROUNDTRIP_STAGE_VERIFIED", flush=True)


def _cancel_replay(root):
    import os
    import runpy
    import signal

    from nexuml_library.environments._isaaclab_runtime import _IsaacLabRuntime

    original = _IsaacLabRuntime.render_frame
    calls = 0

    def render(runtime):
        nonlocal calls
        frame = original(runtime)
        calls += 1
        if calls == 2:
            os.kill(os.getpid(), signal.SIGTERM)
        return frame

    setattr(_IsaacLabRuntime, "render_frame", render)
    logging = {
        "tensorboard": None,
        "mlflow": {
            "tracking_uri": f"sqlite:///{root / 'cancel-replay.db'}",
            "artifact_location": str(root / "cancel-artifacts"),
            "auto_nested_runs": False,
        },
    }
    sys.argv = [
        "nexuml",
        "evaluate",
        str(root / "train/policy"),
        "--device",
        "cuda:0",
        "-O",
        "training.evaluation.episodes=1",
        "-O",
        "training.evaluation.max_steps_per_episode=8",
        "-O",
        "training.evaluation.video.enabled=true",
        "-O",
        "logging=" + json.dumps(logging),
    ]
    runpy.run_module("nexuml", run_name="__main__", alter_sys=True)


def _cancel_native():
    import asyncio

    from nexuml.core.components import EnvironmentBuildContext
    from nexuml_library.environments.isaaclab import IsaacLabEnvironment

    runtime = IsaacLabEnvironment().build(
        EnvironmentBuildContext(purpose="eval", device="cuda:0", num_envs=1, seed=17)
    )
    try:
        runtime.reset()
    finally:
        runtime.close()
    print("NATIVE_RUNTIME_CLOSED_BEFORE_ASYNC_CANCEL", flush=True)
    raise asyncio.CancelledError()


if __name__ == "__main__":
    reference, phase, directory = sys.argv[1:]
    task_root = Path(directory) / reference
    if phase == "infer":
        _infer(task_root / "resume")
    elif phase == "cancel-replay":
        _cancel_replay(task_root)
    elif phase == "cancel-native":
        _cancel_native()
    else:
        _train(reference, phase, task_root)
