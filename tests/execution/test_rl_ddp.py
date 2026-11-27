"""Numerical and lifecycle proof across two real Lightning/Gloo ranks."""

import os
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from lightning.pytorch.callbacks import Callback

from nexuml.reinforcement.collector import RolloutDataset
from nexuml.training.lightning import NexuSession
from nexuml.training.reinforcement import NexuRLLightningModule
from nexuml_library.reinforcement.algorithms import ppo

pytestmark = [pytest.mark.requires_optional("torchrl"), pytest.mark.requires_optional("gymnasium")]
WORKER = Path(__file__).resolve().parents[1] / "fixtures" / "rl_ddp_worker.py"
scenario = runpy.run_path(str(WORKER))["scenario"]


def _torchrun(directory, mode):
    entrypoint = [str(WORKER), str(directory), mode]
    if mode in ("public", "public-resume"):
        from nexuml.core.config import ResolvedConfig
        from nexuml.core.types import ExportSpec

        config = scenario()
        if mode == "public-resume":
            config.training.total_frames = 48
        config.exports = [ExportSpec(kind="train_package", output=str(directory / "policy"))]
        ResolvedConfig.from_scenario(config).save(directory / "worker.yaml")
        entrypoint = ["-m", "nexuml.execution.worker", "--config", str(directory / "worker.yaml")]
        if mode == "public-resume":
            entrypoint.extend(["--trainer-checkpoint", str(directory / "full.ckpt")])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            *entrypoint,
        ],
        cwd=directory,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-20000:]
    assert "backend=gloo" in result.stderr
    if mode in ("fit", "resume"):
        return tuple(torch.load(directory / f"{mode}-{i}.pt", weights_only=False) for i in range(2))


def test_ddp_ppo_matches_global_minibatches_and_resumes(tmp_path):
    ranks = _torchrun(tmp_path, "fit")
    assert [rank["seeds"] for rank in ranks] == [[(0, 17)], [(1, 18)]]
    assert not torch.equal(
        ranks[0]["rollouts"][0]["observation"], ranks[1]["rollouts"][0]["observation"]
    )
    for rank in ranks:
        assert (rank["frames"], rank["batches"], rank["optimizer_steps"]) == (32, 2, [{8}, {8}])

    # Replay the same on-policy transitions, preserving separate GAE trajectories, then
    # use the exact union of rank-local sample indices for each global minibatch.
    rollouts = [torch.stack(pair) for pair in zip(*(r["rollouts"] for r in ranks), strict=True)]
    for rollout in rollouts:
        rollout.names = [None, "time"]  # ty: ignore[invalid-assignment]  # TensorDict permits unnamed axes.
    permutations = []
    for states in zip(*(r["rng"] for r in ranks), strict=True):
        generators = [torch.Generator().set_state(state) for state in states]
        for _ in range(2):
            first, second = (torch.randperm(8, generator=gen).reshape(2, 4) for gen in generators)
            permutations.append(torch.cat([first, second + 8], dim=1).flatten())

    class Replay(NexuRLLightningModule):
        def train_dataloader(self):
            return torch.utils.data.DataLoader(
                RolloutDataset(rollouts, 2), batch_size=None, collate_fn=lambda rollout: rollout
            )

    class ReferenceTrace(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            metrics = trainer.callback_metrics
            assert {"train/rl/reward", "train/rl/loss_objective"} <= metrics.keys()
            torch.testing.assert_close(
                metrics, ranks[0]["metrics"][batch_idx], atol=2e-6, rtol=2e-5
            )
            torch.testing.assert_close(
                ranks[0]["metrics"][batch_idx], ranks[1]["metrics"][batch_idx], atol=0, rtol=0
            )
            for name, value in pl_module.named_parameters():
                torch.testing.assert_close(
                    value, ranks[0]["updates"][batch_idx][name], atol=2e-6, rtol=2e-5
                )
                torch.testing.assert_close(
                    ranks[0]["updates"][batch_idx][name],
                    ranks[1]["updates"][batch_idx][name],
                    atol=0,
                    rtol=0,
                )

    class ReplayRandom:
        def randperm(self, n, *, device):
            assert n == 16
            return permutations.pop(0).to(device)

        def __getattr__(self, name):
            return getattr(torch, name)

    config = scenario()
    config.training = config.training.model_copy(update={"devices": 1, "strategy": "auto"})
    reference = Replay(config)
    session = NexuSession(config, enable_loggers=False, enable_progress_bar=False, log_dir=tmp_path)
    with (
        patch.object(ppo, "torch", ReplayRandom()),
        patch("nexuml.training.reinforcement.synchronize_policy"),
    ):
        getattr(session.trainer, "callbacks").append(ReferenceTrace())
        session.trainer.fit(reference)
    assert not permutations

    resumed = _torchrun(tmp_path, "resume")
    assert [rank["seeds"] for rank in resumed] == [[(0, 19)], [(1, 20)]]
    for rank in resumed:
        assert (rank["frames"], rank["batches"], rank["optimizer_steps"]) == (32, 2, [{8}, {8}])
        assert len(rank["rollouts"]) == 1
    for key in resumed[0]["updates"][0]:
        torch.testing.assert_close(
            resumed[0]["updates"][0][key], resumed[1]["updates"][0][key], atol=0, rtol=0
        )
    _torchrun(tmp_path, "overbudget")
    from nexuml.core.export import load_inference_package

    _torchrun(tmp_path, "public-resume")
    _, _, metadata = load_inference_package(tmp_path / "policy")
    assert metadata["training_frames"] == 48


def test_ddp_rank_local_collection_failure_aborts_both_ranks(tmp_path):
    _torchrun(tmp_path, "failure")


def test_ddp_rank_zero_cancellation_reaps_pending_snapshot(tmp_path):
    _torchrun(tmp_path, "cancel")


def test_public_ddp_worker_exports_one_completed_rank_zero_policy(tmp_path):
    from nexuml.core.export import load_inference_package

    _torchrun(tmp_path, "public")
    _, _, metadata = load_inference_package(tmp_path / "policy")
    assert metadata["training_frames"] == 32


def test_ddp_checkpoint_rejects_a_changed_actor_graph():
    config = scenario()
    checkpoint = {
        "rl_world_size": 2,
        "hyper_parameters": dict(NexuRLLightningModule(config).hparams),
    }
    changed = config.model_copy(deep=True)
    changed.pipeline.stages["changed"] = changed.pipeline.stages.pop("policy")
    with pytest.raises(ValueError, match="unchanged actor graph"):
        NexuRLLightningModule(changed).on_load_checkpoint(checkpoint)


@pytest.mark.requires_optional("ray")
@pytest.mark.skipif(
    os.getenv("NEXUML_TEST_RAY_TRAIN") != "1", reason="opt-in real Ray Train cluster"
)
def test_ray_train_round_trip(tmp_path):
    result = subprocess.run(
        [sys.executable, str(WORKER), str(tmp_path), "ray"],
        cwd=tmp_path,
        env={
            **os.environ,
            "RAY_ADDRESS": "local",
            "RAY_TRAIN_V2_ENABLED": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-20000:]
