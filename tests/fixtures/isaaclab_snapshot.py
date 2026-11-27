"""A simulator-free owner dispatches an actual native CUDA snapshot evaluator."""

import os
import sys
from contextlib import ExitStack
from tempfile import TemporaryDirectory

from nexuml.core.types import ProcessEvaluationSpec, RayEvaluationSpec, RLEvaluationSpec
from nexuml.evaluation._process import ProcessEvaluation
from nexuml.training.reinforcement import NexuRLLightningModule
from nexuml_library.scenarios.reinforcement.isaaclab import isaac_cartpole_ppo


def _probe(resources):
    scenario = isaac_cartpole_ppo()
    scenario.policy_evaluation = RLEvaluationSpec(
        episodes=1,
        seed=51,
        max_steps_per_episode=6,
        process=ProcessEvaluationSpec(device="cuda:0", timeout_s=120),
    )
    owner = NexuRLLightningModule(scenario)
    owner.frames.fill_(16)
    owner.batches.fill_(2)
    if "--ray" in sys.argv:
        import ray

        from nexuml.evaluation._ray import RayEvaluation

        temporary = resources.enter_context(TemporaryDirectory(prefix="nrs-"))
        ray.init(
            address="local",
            num_cpus=2,
            num_gpus=1,
            include_dashboard=False,
            object_store_memory=140 * 1024**2,
            _temp_dir=temporary,
        )
        resources.callback(ray.shutdown)
        worker = RayEvaluation(
            owner.policy, scenario, owner=owner, spec=RayEvaluationSpec(num_gpus=1)
        )
        receipt = worker.result(wait=True, check_cancelled=lambda: None)
        assert receipt is not None
        assert receipt["evaluator_resources"]["GPU"] == 1
    else:
        worker = ProcessEvaluation(
            owner.policy, scenario, owner=owner, device="cuda:0", timeout_s=120
        )
        resources.callback(worker.close)
        receipt = worker.result(wait=True)
    assert receipt is not None
    assert receipt["evaluator_pid"] != os.getpid()
    assert receipt["evaluator_device"] == "cuda:0"
    assert receipt["source_training_frames"] == 16
    assert receipt["source_training_batches"] == 2
    assert receipt["metrics"]["eval/rl/episodes"] == 1
    assert str(owner.device) == "cpu"


if __name__ == "__main__":
    with ExitStack() as stack:
        _probe(stack)
    print("NATIVE_SNAPSHOT_RECEIPT_VERIFIED", flush=True)
