"""Run the actual Lightning/TorchRL learner under torchrun; retain local receipts."""

import os
from contextlib import ExitStack
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest.mock import patch

import torch
from lightning.pytorch.callbacks import Callback

from nexuml.core.types import ProcessEvaluationSpec, RLEvaluationSpec
from nexuml.execution.worker import launch_options
from nexuml.training.lightning import NexuSession
from nexuml.training.reinforcement import NexuRLLightningModule
from nexuml_library.environments.gymnasium import GymnasiumEnvironment
from nexuml_library.scenarios.reinforcement.gymnasium import cartpole_ppo


def scenario():
    result = cartpole_ppo(total_frames=32, frames_per_batch=16)
    result.training = result.training.model_copy(
        update={
            "accelerator": "cpu",
            "devices": 2,
            "strategy": "ddp",
            "seed": 17,
            "algorithm": result.training.algorithm.model_copy(
                update={"minibatch_size": 8, "update_epochs": 2}
            ),
            "evaluation": RLEvaluationSpec(episodes=0),
        }
    )
    result.logging = None
    return result


def run(directory, mode):
    from lightning.pytorch.utilities.exceptions import SIGTERMException

    config = scenario()
    checkpoint = directory / "partial.ckpt"
    rank = int(os.environ["RANK"])
    receipts: dict[str, Any] = {"seeds": [], "rollouts": [], "rng": [], "updates": []}
    original_build = GymnasiumEnvironment.build
    request = NexuRLLightningModule._request_fit_evaluation
    evaluator = None
    if mode == "cancel":
        config.training.evaluation = RLEvaluationSpec(
            episodes=1,
            every_n_frames=16,
            process=ProcessEvaluationSpec(scheduling="skip_if_busy"),
        )
    if mode == "overbudget":
        config.training.total_frames = 16
        checkpoint = directory / "full.ckpt"

    def cancel_snapshot(module):
        nonlocal evaluator
        request(module)
        assert rank == 0 and module._pending_evaluation is not None
        evaluator = module._pending_evaluation[1].worker
        raise SIGTERMException()

    def build(self, context):
        receipts["seeds"].append((context.worker_rank, context.seed))
        runtime = original_build(self, context)
        if mode == "failure" and rank == 1:

            def fail(*args, **kwargs):
                raise RuntimeError("injected rank-local step failure")

            runtime.step = fail
        return runtime

    class Trace(Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            receipts["rollouts"].append(batch.clone())
            receipts["rng"].append(torch.random.get_rng_state())

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            receipts["updates"].append(
                {name: value.detach().clone() for name, value in pl_module.named_parameters()}
            )
            if mode == "fit" and batch_idx == 0:
                trainer.save_checkpoint(checkpoint)
            if mode == "fit" and batch_idx == 1:
                trainer.save_checkpoint(directory / "full.ckpt")

    with ExitStack() as patches:
        patches.enter_context(patch.object(GymnasiumEnvironment, "build", build))
        if mode == "cancel":
            patches.enter_context(
                patch.object(NexuRLLightningModule, "_request_fit_evaluation", cancel_snapshot)
            )
        session = NexuSession(
            config,
            log_dir=directory / mode,
            enable_progress_bar=False,
            trainer_checkpoint=checkpoint if mode in ("resume", "overbudget") else None,
            **launch_options(config, dict(os.environ)),
        )
        getattr(session.trainer, "callbacks").append(Trace())
        if mode == "overbudget":
            try:
                session.fit()
            except ValueError as error:
                assert "smaller than the checkpoint" in str(error) and not receipts["seeds"]
                return
            raise AssertionError("training accepted a budget below its restored count")
        if mode == "cancel":
            try:
                session.fit()
            except (SIGTERMException, RuntimeError) as error:
                if rank == 0:
                    assert isinstance(error, SIGTERMException) and error.code == 143
                    assert evaluator is not None and evaluator.poll() is not None
                else:
                    assert "periodic evaluation failed on rank 0" in str(error)
                return
            raise AssertionError("training ignored cancellation")
        if mode == "failure":
            # TorchRL may step during its constructor; either way both ranks must abort.
            try:
                session.fit()
            except RuntimeError as error:
                assert ("injected rank-local" if rank else "failed on rank 1") in str(error)
                module = session.lightning_module
                assert isinstance(module, NexuRLLightningModule)
                assert int(module.frames) == 0 and module._collector is None
                return
            raise AssertionError("training ignored the failed rank")
        result = session.run()
    module = result.lightning_module
    assert isinstance(module, NexuRLLightningModule)
    assert module._collector is None
    receipts.update(
        frames=int(module.frames),
        batches=int(module.batches),
        optimizer_steps=[
            {int(state["step"]) for state in opt.state.values()}
            for opt in result.trainer.optimizers
        ],
    )
    torch.save(receipts, directory / f"{mode}-{rank}.pt")


def run_ray_test(directory, *, native=False):
    import ray
    from ray.train import RunConfig
    from ray.train.torch import TorchConfig, TorchTrainer

    from nexuml.core.serialization import lower_model
    from nexuml.core.export import PACKAGE_FILENAME
    from nexuml.core.types import ExportSpec
    from nexuml.execution.definitions import RayClusterExecution
    from nexuml.execution.ray import _scaling_config, train_loop_per_worker

    export = directory / "export"
    config = scenario()
    gpu_count = int(native)
    if native:
        from nexuml_library.scenarios.reinforcement.isaaclab import isaac_cartpole_learning_ppo

        config = isaac_cartpole_learning_ppo(total_frames=32, frames_per_batch=16, num_envs=4)
        config.training.algorithm = config.training.algorithm.model_copy(
            update={"update_epochs": 2}
        )
        config.logging = None
    config.training.devices = 1
    config.exports = [ExportSpec(kind="train_package", output=str(export))]
    config.execution = RayClusterExecution(
        workers=1 if native else 2,
        resources_per_worker={"CPU": 1, "GPU": 1} if native else {"CPU": 1},
    )
    config.execution.preflight(config)
    with TemporaryDirectory(prefix="nr-") as ray_dir:
        ray.init(
            address="local",
            num_cpus=4,
            num_gpus=gpu_count,
            object_store_memory=140 * 1024**2,
            include_dashboard=False,
            _temp_dir=ray_dir,
        )
        try:
            result = TorchTrainer(
                train_loop_per_worker,
                train_loop_config={"scenario": lower_model(config)},
                scaling_config=_scaling_config(config.execution),
                run_config=cast(Any, RunConfig(storage_path=str(directory), name="cpu-rl")),
                torch_config=TorchConfig(timeout_s=90),
            ).fit()
            assert result.checkpoint is not None
            with result.checkpoint.as_directory() as path:
                receipt = torch.load(Path(path) / "checkpoint.ckpt", weights_only=False)
                assert receipt.get("rl_world_size", 1) == config.execution.workers
                assert int(receipt["state_dict"]["frames"]) == 32
                config.training.total_frames = 48
                config.exports = []
                resumed = TorchTrainer(
                    train_loop_per_worker,
                    train_loop_config={
                        "scenario": lower_model(config),
                        "trainer_checkpoint": ray.put(
                            (Path(path) / "checkpoint.ckpt").read_bytes()
                        ),
                    },
                    scaling_config=_scaling_config(config.execution),
                    run_config=cast(
                        Any, RunConfig(storage_path=str(directory), name="cpu-rl-resumed")
                    ),
                    torch_config=TorchConfig(timeout_s=90),
                ).fit()
                assert resumed.metrics is not None and resumed.metrics["train/rl/frames"] == 48
                assert resumed.metrics["train/rl/batches"] == 3
                assert resumed.metrics["train/rl/optimizer_steps"] == (48 if native else 24)
        finally:
            ray.shutdown()
        assert result.metrics is not None and result.metrics["train/rl/frames"] == 32
        assert result.metrics["train/rl/batches"] == 2
        assert result.metrics["train/rl/optimizer_steps"] == (32 if native else 16)
        assert (export / PACKAGE_FILENAME).is_file()


if __name__ == "__main__":
    root, mode = Path(sys.argv[1]), sys.argv[2]
    if mode in ("ray", "ray-native"):
        run_ray_test(root, native=mode == "ray-native")
    else:
        run(root, mode)
