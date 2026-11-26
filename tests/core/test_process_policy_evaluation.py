"""Fit evaluation owns one immutable worker snapshot at a time."""

import json
import os
import subprocess
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic
from types import SimpleNamespace

import lightning as L
import pytest
import torch

from nexuml.core.compiler import compile
from nexuml.core.policy import CompiledPolicy
from nexuml.core.types import (
    ProcessEvaluationSpec,
    RayEvaluationSpec,
    RLEvaluationSpec,
    RLVideoSpec,
)
from nexuml.evaluation import _process
from nexuml.training.lightning import NexuSession
from nexuml_library.scenarios.reinforcement.gymnasium import pendulum_ppo


@pytest.fixture
def gated_evaluation(monkeypatch, tmp_path):
    """Hold the first real evaluator until the learner releases its input pipe."""
    sandbox = tmp_path / "workers"
    sandbox.mkdir()
    monkeypatch.setattr(
        _process, "TemporaryDirectory", lambda **kwargs: TemporaryDirectory(dir=sandbox, **kwargs)
    )
    launch = subprocess.Popen
    workers = []

    def gate_first(args, *pos, **kwargs):
        assert args[1:3] == ["-m", _process.__name__]
        if not workers:
            gate = 'if sys.stdin.buffer.read(1) != b"e": raise SystemExit(17)\n'
            if args[-1].startswith("cuda:"):
                gate = (
                    "from pathlib import Path\n"
                    "from nexuml_library.environments.gymnasium import _GymnasiumRuntime\n"
                    f"witness = Path({str(tmp_path / 'gpu-step.json')!r})\n"
                    "step = _GymnasiumRuntime.step\n"
                    "def observed_step(self, actions):\n"
                    "    result = step(self, actions)\n"
                    "    if not witness.exists():\n"
                    "        import json\n"
                    '        assert actions["action"].is_cuda and result.reward.is_cuda\n'
                    "        staged = witness.with_suffix('.tmp')\n"
                    '        staged.write_text(json.dumps(str(actions["action"].device)))\n'
                    "        staged.replace(witness)\n"
                    f"        {gate}"
                    "    return result\n"
                    "_GymnasiumRuntime.step = observed_step\n"
                )
            args = [
                args[0],
                "-c",
                "import runpy, sys\n"
                + gate
                + "runpy.run_module(sys.argv.pop(1), run_name='__main__')",
                *args[2:],
            ]
            kwargs["stdin"] = subprocess.PIPE
        worker = launch(args, *pos, **kwargs)
        workers.append(worker)
        return worker

    monkeypatch.setattr(
        _process,
        "subprocess",
        SimpleNamespace(
            Popen=gate_first,
            STDOUT=subprocess.STDOUT,
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
    )
    yield workers
    for worker in workers:
        assert worker.poll() is not None
        worker.communicate()
    assert not list(sandbox.iterdir())


def assert_gpu_step(tmp_path, worker):
    witness = tmp_path / "gpu-step.json"
    limit = monotonic() + 60
    gate = Event()
    while not witness.is_file():
        assert worker.poll() is None and monotonic() < limit
        gate.wait(0.025)
    assert json.loads(witness.read_text()) == "cuda:0"


def test_fit_evaluates_start_periodic_and_required_final_snapshots(tmp_path, monkeypatch):
    pytest.importorskip("gymnasium")
    pytest.importorskip("tensorboard")
    monkeypatch.setenv("NEXUML_LOGS_ROOT", str(tmp_path / "logs"))
    scenario = pendulum_ppo(total_frames=96, frames_per_batch=32)
    scenario.training.algorithm = scenario.training.algorithm.model_copy(
        update={"minibatch_size": 16, "update_epochs": 1}
    )
    scenario.training.evaluation = RLEvaluationSpec(
        at_start=True,
        every_n_frames=64,
        episodes=2,
        max_steps_per_episode=4,
        seed=23,
        process=ProcessEvaluationSpec(),
    )
    reports = []
    session = NexuSession(scenario, log_dir=tmp_path, enable_progress_bar=False)
    module = session.lightning_module
    publish = module._publish_snapshot
    base = _process.ProcessEvaluation

    class ObservedProcess(base):
        def __init__(self, policy, scenario, **kwargs):
            self.actor = policy
            self.before = {
                key: value.detach().clone() for key, value in policy.state_dict().items()
            }
            super().__init__(policy, scenario, **kwargs)

        def result(self, **kwargs):
            receipt = super().result(**kwargs)
            torch.testing.assert_close(self.actor.state_dict(), self.before, rtol=0, atol=0)
            return receipt

    def observe(receipt):
        reports.append(receipt)
        publish(receipt)

    monkeypatch.setattr(_process, "ProcessEvaluation", ObservedProcess)
    monkeypatch.setattr(module, "_publish_snapshot", observe)
    session.run()
    assert [report["source_training_frames"] for report in reports] == [0, 64, 96]
    assert [report["source_training_batches"] for report in reports] == [0, 2, 3]
    assert len({report["snapshot_sha256"] for report in reports}) == 3
    assert len({report["evaluator_pid"] for report in reports}) == 3
    for receipt in reports:
        assert receipt["evaluator_pid"] != os.getpid()
        report = receipt["episode_report"]
        assert report["source_training_frames"] == receipt["source_training_frames"]
        assert report["completed_episodes"] == 2
        assert [episode["seed"] for episode in report["episodes"]] == [23, 24]
        assert [episode["steps"] for episode in report["episodes"]] == [4, 4]
    files = sorted(tmp_path.rglob("evaluation-frame-*.json"))
    assert [json.loads(path.read_text()) for path in files] == reports


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.requires_hardware)]
)
def test_slow_evaluator_overlaps_ppo_and_does_not_drop_final(
    gated_evaluation, tmp_path, monkeypatch, device
):
    pytest.importorskip("gymnasium")
    pytest.importorskip("tensorboard")
    if device != "cpu":
        assert torch.cuda.is_available()
        for module_name in ("pygame", "PIL", "imageio_ffmpeg"):
            pytest.importorskip(module_name)
        monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
        monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.setenv("NEXUML_LOGS_ROOT", str(tmp_path / "logs"))
    scenario = pendulum_ppo(total_frames=96, frames_per_batch=32)
    scenario.training.accelerator = "cpu"
    scenario.training.algorithm = scenario.training.algorithm.model_copy(
        update={"minibatch_size": 16, "update_epochs": 1}
    )
    scenario.training.evaluation = RLEvaluationSpec(
        at_start=device == "cpu",
        every_n_frames=32,
        episodes=2,
        max_steps_per_episode=4,
        seed=23,
        process=ProcessEvaluationSpec(scheduling="skip_if_busy", device=device),
        video=RLVideoSpec(enabled=device != "cpu", format="both", max_frames=2, max_size=64),
    )
    progress = []

    class ReleaseAfterTraining(L.Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if device == "cpu" or batch_idx != 1:
                return
            assert_gpu_step(tmp_path, gated_evaluation[0])
            assert pl_module.device.type == "cpu"

        def on_train_batch_end(self, trainer, pl_module, *args):
            (worker,) = gated_evaluation
            assert worker.poll() is None
            progress.append(int(pl_module.frames))

        def on_train_end(self, trainer, pl_module):
            gated_evaluation[0].stdin.write(b"e")
            gated_evaluation[0].stdin.flush()

    session = NexuSession(scenario, log_dir=tmp_path, enable_progress_bar=False)
    session.trainer.callbacks.append(ReleaseAfterTraining())
    skipped = []
    module = session.lightning_module
    initial = {key: value.detach().clone() for key, value in module.policy.state_dict().items()}
    publish = module._publish_evaluation_metrics

    def capture(metrics, step):
        if metrics.get("eval/rl/snapshot_skipped"):
            skipped.append(step)
        publish(metrics, step)

    monkeypatch.setattr(module, "_publish_evaluation_metrics", capture)
    session.run()
    assert any(
        not torch.equal(value, module.policy.state_dict()[key]) for key, value in initial.items()
    )
    assert progress == [32, 64, 96]
    assert skipped == ([32, 64, 96] if device == "cpu" else [64, 96])
    assert len(gated_evaluation) == 2
    files = sorted(tmp_path.rglob("evaluation-frame-*.json"))
    reports = [json.loads(path.read_text()) for path in files]
    first_frame = 0 if device == "cpu" else 32
    assert [r["source_training_frames"] for r in reports] == [first_frame, 96]
    assert [r["source_training_batches"] for r in reports] == [first_frame // 32, 3]
    assert [r["evaluator_pid"] for r in reports] == [worker.pid for worker in gated_evaluation]
    assert reports[0]["snapshot_sha256"] != reports[1]["snapshot_sha256"]
    for receipt in reports:
        assert receipt["evaluator_device"] == device
        assert [ep["seed"] for ep in receipt["episode_report"]["episodes"]] == [23, 24]
    if device != "cpu":
        import imageio.v2 as imageio
        from PIL import Image

        baseline = scenario.model_copy(deep=True)
        baseline.training.evaluation = RLEvaluationSpec(episodes=0)
        reference = NexuSession(
            baseline, log_dir=tmp_path, enable_progress_bar=False, enable_loggers=False
        )
        reference.run()
        torch.testing.assert_close(
            reference.lightning_module.policy.state_dict(),
            module.policy.state_dict(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            [optimizer.state_dict() for optimizer in reference.trainer.optimizers],
            [optimizer.state_dict() for optimizer in session.trainer.optimizers],
            rtol=0,
            atol=0,
        )
        for frame, receipt in zip((first_frame, 96), reports, strict=True):
            (metadata_file,) = tmp_path.rglob(f"frame_{frame:012d}.json")
            media = json.loads(metadata_file.read_text())
            assert media["captured_frames"] == 2 and media["capture_capped"]
            assert media["frame_step"] == frame and media["seeds"] == [23, 24]
            assert media["provenance"]["source_training_frames"] == frame
            assert (
                media["provenance"]["source_policy_files_sha256"]
                == receipt["episode_report"]["source_policy_files_sha256"]
            )
            with imageio.get_reader(metadata_file.with_suffix(".mp4")) as reader:
                assert reader.count_frames() == 2
            with Image.open(metadata_file.with_suffix(".gif")) as gif:
                assert gif.n_frames in (1, 2)  # GIF may coalesce visually identical frames.


@pytest.mark.parametrize(
    ("failure", "device"),
    [
        ("cancel", "cpu"),
        ("worker", "cpu"),
        pytest.param("cancel", "cuda:0", marks=pytest.mark.requires_hardware),
    ],
)
def test_async_evaluator_reaped_on_cancel_or_worker_failure(
    gated_evaluation, tmp_path, monkeypatch, failure, device
):
    from lightning.pytorch.utilities.exceptions import SIGTERMException

    pytest.importorskip("gymnasium")
    pytest.importorskip("tensorboard")
    monkeypatch.setenv("NEXUML_LOGS_ROOT", str(tmp_path / "logs"))
    scenario = pendulum_ppo(total_frames=32, frames_per_batch=32)
    scenario.training.accelerator = "cpu"
    scenario.training.algorithm = scenario.training.algorithm.model_copy(
        update={"minibatch_size": 16, "update_epochs": 1}
    )
    # The only periodic request is also the required final frame.
    scenario.training.evaluation = RLEvaluationSpec(
        every_n_frames=32,
        episodes=1,
        max_steps_per_episode=4,
        process=ProcessEvaluationSpec(scheduling="skip_if_busy", device=device),
    )

    class EndFit(L.Callback):
        def on_fit_end(self, trainer, pl_module):
            if failure == "cancel":
                if device != "cpu":
                    assert_gpu_step(tmp_path, gated_evaluation[0])
                trainer._signal_connector.received_sigterm = True
            else:
                gated_evaluation[0].stdin.write(b"f")
                gated_evaluation[0].stdin.flush()

    session = NexuSession(scenario, log_dir=tmp_path, enable_progress_bar=False)
    session.trainer.callbacks.append(EndFit())
    if failure == "cancel":
        with pytest.raises(SIGTERMException) as caught:
            session.run()
        assert caught.value.code == 143
    else:
        with pytest.raises(RuntimeError, match="CPU snapshot evaluation exited 17"):
            session.run()
    (worker,) = gated_evaluation
    assert worker.poll() is not None
    assert not list(tmp_path.rglob("evaluation-frame-*.json"))


def test_timeout_reaps_owned_worker_and_removes_temporary_snapshot(tmp_path, monkeypatch):
    pytest.importorskip("gymnasium")
    scenario = pendulum_ppo()
    policy = CompiledPolicy.from_pipeline(compile(scenario))
    owner = SimpleNamespace(policy=policy, frames=torch.tensor(0), batches=torch.tensor(0))
    monkeypatch.setattr(
        _process, "TemporaryDirectory", lambda **kwargs: TemporaryDirectory(dir=tmp_path, **kwargs)
    )
    launch = _process.subprocess.Popen
    workers = []

    def record_worker(*args, **kwargs):
        worker = launch(*args, **kwargs)
        workers.append(worker)
        return worker

    monkeypatch.setattr(_process.subprocess, "Popen", record_worker)
    pending = _process.ProcessEvaluation(policy, scenario, owner=owner, timeout_s=0.001)
    try:
        with pytest.raises(RuntimeError, match="CPU snapshot evaluation exceeded"):
            pending.result(wait=True)
    finally:
        pending.close()
    (worker,) = workers
    assert worker.returncode is not None
    assert not list(tmp_path.iterdir())


def test_process_evaluation_requires_explicit_device():
    with pytest.raises(ValueError, match="explicit cuda:N"):
        ProcessEvaluationSpec(device="cuda")


def test_process_evaluation_rejects_unavailable_gpu_before_collection(monkeypatch, tmp_path):
    pytest.importorskip("gymnasium")
    scenario = pendulum_ppo(total_frames=32, frames_per_batch=32)
    scenario.training.accelerator = "cpu"
    scenario.training.evaluation = RLEvaluationSpec(process=ProcessEvaluationSpec(device="cuda:0"))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    session = NexuSession(
        scenario, log_dir=tmp_path, enable_progress_bar=False, enable_loggers=False
    )
    with pytest.raises(ValueError, match="GPU cuda:0 is unavailable"):
        session.fit()
    assert session.lightning_module._collector is None


def test_ray_placement_validation_and_missing_connection(monkeypatch):
    import sys

    from nexuml.evaluation._ray import require_ray_runtime

    with pytest.raises(ValueError, match="custom Ray resources"):
        RayEvaluationSpec(resources={"GPU": 1})
    with pytest.raises(ValueError, match="not both"):
        RLEvaluationSpec(ray=RayEvaluationSpec(), process=ProcessEvaluationSpec())
    with pytest.raises(ValueError, match="does not support video"):
        RLEvaluationSpec(ray=RayEvaluationSpec(), video=RLVideoSpec(enabled=True))
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(is_initialized=lambda: False))
    with pytest.raises(ValueError, match="existing Ray runtime"):
        require_ray_runtime()
