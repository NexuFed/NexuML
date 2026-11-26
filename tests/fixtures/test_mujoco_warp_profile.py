"""Opt-in same-geometry Cartpole timing; each lane runs in a fresh process."""

from importlib.metadata import version
import json
from math import isfinite
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tarfile
from time import perf_counter
from typing import Any, cast

import pytest


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = (
    ROOT / "openspec/changes/decouple-simulation-learning-and-evaluation/local-fit-receipts.tar.gz"
)
LOCK = "/tmp/opencode/nexuml-decouple-native-gpu.lock"
XML_SHA = "d9657349921533445007a0f83eac74d2ba274ba0cc49c585618d00bb98b75704"
BASE_SHA = "99899fb561134f94b18fcd1cccb2b8efc6b9b4a48c910b9f7e987e1ca588d4a2"


@pytest.mark.requires_simulator
@pytest.mark.requires_optional("torchrl")
@pytest.mark.parametrize(
    "lane",
    [
        pytest.param("mujoco", marks=pytest.mark.requires_optional("mujoco")),
        pytest.param(
            "warp",
            marks=[pytest.mark.requires_gpu, pytest.mark.requires_optional("mujoco_warp")],
        ),
        pytest.param(
            "isaac", marks=[pytest.mark.requires_gpu, pytest.mark.requires_optional("isaaclab")]
        ),
    ],
)
def test_paired_cartpole_profile(lane, tmp_path):
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        xml = archive.extractfile("pair/cartpole.xml")
        assert xml is not None
        (tmp_path / "cartpole.xml").write_bytes(xml.read())
    command = [sys.executable, __file__, lane, str(tmp_path)]
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "NEXUML_LOGS_ROOT": str(tmp_path / "native-logs"),
    }
    if lane == "mujoco":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        command[:0] = ["flock", "-F", "-w", "180", LOCK]
    started = perf_counter()
    with (tmp_path / "child.log").open("w") as log:
        result = subprocess.run(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=360
        )
    # Includes flock wait, process startup and exit (Kit can exit from its scope).
    (tmp_path / "process.json").write_text(
        json.dumps({"exit": result.returncode, "wall_seconds_with_lock": perf_counter() - started})
        + "\n"
    )
    assert result.returncode == 0, (tmp_path / "child.log").read_text()
    report = json.loads((tmp_path / "profile.json").read_text())
    assert report["counts"] == [1024, 8, 16, 32] and len(report["warm_windows"]) == 3
    assert report["actor_start_sha256"] == BASE_SHA


def _profile(lane, root):
    entered = perf_counter()
    from lightning.pytorch.callbacks import Callback
    import torch

    from mujoco_warp_reference import MujocoWarpCartpole
    from nexuml.core.compiler import compile
    from nexuml.core.policy import CompiledPolicy
    from nexuml.core.types import ReinforcementLearningSpec
    from nexuml.training.lightning import NexuSession
    from nexuml.training.local_fit import _state_hash
    from nexuml.training.reinforcement import NexuRLLightningModule
    from paired_cartpole import _digest, _scenario

    assert _digest(root / "cartpole.xml") == XML_SHA
    torch.set_num_threads(4)
    device = "cpu" if lane == "mujoco" else "cuda:0"
    scenario = _scenario("mujoco" if lane == "mujoco" else "isaac", root)
    assert scenario.interaction is not None and isinstance(
        scenario.training, ReinforcementLearningSpec
    )
    scenario.name = f"paired-cartpole-profile-{lane}"
    scenario.training.seed = 31
    if lane == "warp":
        scenario.interaction.environment = MujocoWarpCartpole(xml_file=str(root / "cartpole.xml"))
    definition = scenario.interaction.environment
    torch.manual_seed(17)
    base = CompiledPolicy.from_pipeline(compile(_scenario("isaac", root))).state_dict()
    assert _state_hash(base) == BASE_SHA

    def tick():
        if device != "cpu":
            torch.cuda.synchronize(device)
        return perf_counter()

    class Timing(Callback):
        def __init__(self):
            self.rows = []

        def on_train_start(self, trainer, pl_module):
            self.first = self.previous = tick()

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            self.update = tick()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            ended = tick()
            assert str(batch["action"].device) == device and batch.batch_size == (8, 16)
            losses = {
                name.removeprefix("train/rl/"): float(value)
                for name, value in trainer.callback_metrics.items()
                if name.startswith("train/rl/loss_")
            }
            assert len(losses) == 3 and all(map(isfinite, losses.values()))
            self.rows.append(
                {
                    "collect_s": self.update - self.previous,
                    "update_s": ended - self.update,
                    "terminated": int(batch["next", "terminated"].sum()),
                    "truncated": int(batch["next", "truncated"].sum()),
                    "reward_sum": float(batch["next", "reward"].sum()),
                    "losses": losses,
                }
            )
            self.last = ended
            self.previous = tick()

    timing = Timing()
    session = NexuSession(
        scenario, log_dir=root / "fit", enable_loggers=False, enable_progress_bar=False
    )
    module = session.lightning_module
    assert isinstance(module, NexuRLLightningModule)
    module.policy.load_state_dict(base, strict=True)
    trainer = cast(Any, session.trainer)
    trainer.callbacks.append(timing)
    contract = definition.describe()
    assert contract.physical is not None and contract.control_period_s == 1 / 60
    packages = ["torch", "torchrl", "tensordict", "lightning", "gymnasium"]
    gpu = None
    if device != "cpu":
        gpu = {
            "device": device,
            "name": torch.cuda.get_device_name(device),
            "sm": torch.cuda.get_device_capability(device),
            "driver": subprocess.check_output(
                ["nvidia-smi", "-i", "0", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
            ).strip(),
        }
    isaaclab_revision = None
    if lane == "isaac":
        from nexuml_library.environments.isaaclab import IsaacLabEnvironment

        assert isinstance(definition, IsaacLabEnvironment)
        isaaclab_revision = definition.revision
        packages += ["isaaclab", "isaacsim", "warp-lang"]
    else:
        packages += ["mujoco"] + (["mujoco-warp", "warp-lang"] if lane == "warp" else [])

    with definition.job_scope():
        started = tick()
        session.fit()
        finished = tick()
        assert module.fit_device == device and module._collector is None
        counts = [
            int(module.frames),
            int(module.batches),
            int(module.algorithm.updates),
            trainer.global_step,
        ]
        assert counts == [1024, 8, 16, 32] and len(timing.rows) == 8
        state = module.policy.state_dict()
        delta = sum((state[k].cpu() - v).square().sum() for k, v in base.items()).sqrt().item()
        assert isfinite(delta) and delta > 0
        warm = []
        for start in (2, 4, 6):
            window = timing.rows[start : start + 2]
            collect, update = (sum(row[key] for row in window) for key in ("collect_s", "update_s"))
            assert collect > 0 and update > 0
            warm.append(
                {
                    "frames": 256,
                    "collect_s": collect,
                    "update_s": update,
                    "collect_frames_s": 256 / collect,
                    "collect_update_frames_s": 256 / (collect + update),
                }
            )
        report = {
            "lane": lane,
            "device": device,
            "gpu": gpu,
            "python": platform.python_version(),
            "python_abi": sys.implementation.cache_tag,
            "machine": platform.machine(),
            "libc": platform.libc_ver(),
            "packages": {name: version(name) for name in packages},
            "isaaclab_revision": isaaclab_revision,
            "torch_cuda_build": torch.version.cuda,
            "threads": torch.get_num_threads(),
            "control_period_s": contract.control_period_s,
            "physical_asset_id": contract.physical.asset_id,
            "actor_start_sha256": BASE_SHA,
            "actor_delta_l2": delta,
            "counts": counts,
            "phases_s": {
                "profile_before_fit": started - entered,
                "fit_startup": timing.first - started,
                "post_last_batch": finished - timing.last,
                "fit_total": finished - started,
            },
            "peak_rss_kib_before_app_exit": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "peak_torch_cuda_bytes": torch.cuda.max_memory_allocated(device) if gpu else None,
            "rollouts": timing.rows,
            "warm_windows": warm,
        }
        (root / "profile.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    _profile(sys.argv[1], Path(sys.argv[2]))
