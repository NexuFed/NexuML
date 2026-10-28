"""Declarative/CLI boundaries only; no real fitting, S3, renderer or subprocess."""

from contextlib import contextmanager
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from unittest import mock

import pytest


def test_public_scenario_keeps_episode_splits_and_has_no_fake_native_contract(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("NEXUML_SO101_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("NEXUML_SO101_OUTPUT_ROOT", str(tmp_path / "run"))
    spec = runpy.run_path("scripts/so101_public_bc.py")["scenario"]()
    source = spec.data.datasets[0].source
    assert spec.data.datasets[0].split_type == "keep"
    assert source.split_episodes == {
        "train": tuple(range(40)),
        "val": tuple(range(40, 45)),
        "test": tuple(range(45, 50)),
    }
    assert spec.training.max_steps == 64 and spec.training.accelerator == "gpu"
    assert spec.interaction is spec.policy is None
    assert spec.data.targets[0].num_outputs == 6
    assert spec.data.input_shapes == {"observation.state": [6]}
    assert spec.callbacks[0].kwargs["dirpath"] == str(tmp_path / "run/checkpoints")


def test_public_scenario_refuses_outputs_inside_original_data(monkeypatch, tmp_path):
    monkeypatch.setenv("NEXUML_SO101_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("NEXUML_SO101_OUTPUT_ROOT", str(tmp_path / "run"))
    with pytest.raises(ValueError, match="must not modify"):
        runpy.run_path("scripts/so101_public_bc.py")["scenario"]()


def test_native_gif_uses_explicit_recording_size_without_weakening_preview_defaults(tmp_path):
    from PIL import Image

    first = Image.new("RGB", (320, 240), "black")
    second = first.copy()
    first.putpixel((0, 0), (255, 255, 255))
    second.putpixel((1, 0), (255, 255, 255))
    path = tmp_path / "recording.gif"
    first.save(path, save_all=True, append_images=[second], duration=20, loop=0)
    gif_info = runpy.run_path("tests/fixtures/physical_ai_logging.py")["_gif_info"]
    with pytest.raises(AssertionError, match="bounded animated"):
        gif_info(path)
    result = gif_info(path, max_size=320)
    assert result == {"frames": 2, "size": [320, 240], "changing_nonuniform_frames": True}


@pytest.mark.parametrize("mismatch", [False, True])
def test_public_workflow_uses_real_cli_commands_and_checkpoint_budget(tmp_path, mismatch):
    import torch

    fn = runpy.run_path("tests/fixtures/physical_ai_public_training.py")["run_workflow"]
    commands = []

    @contextmanager
    def staged():
        yield tmp_path / "data", {"total_bytes": 624607954}

    def stage(record, name, argv, env):
        commands.append((name, argv, env))
        assert argv[:3] == [sys.executable, "-m", "nexuml.cli.main"]
        assert "--scenario-file" in argv
        assert env["HF_HUB_OFFLINE"] == "1"
        if argv[3] == "export":
            output = Path(argv[argv.index("--output") + 1])
            output.mkdir()
            weight = float("--checkpoint" in argv) * (2 if mismatch else 1)
            torch.save({"weight": torch.tensor([weight])}, output / "state_dict.pt")
        elif argv[3] == "evaluate":
            (tmp_path / f"{name}.log").write_text(
                "Model evaluation (test split): {'test/mse': 1.0}"
            )
        elif argv[3] == "train":
            (tmp_path / "checkpoints").mkdir()
            torch.save(
                {"global_step": 64, "state_dict": {"pipeline.weight": torch.tensor([1.0])}},
                tmp_path / "checkpoints/last.ckpt",
            )
        else:
            pytest.fail("unexpected CLI operation")

    with mock.patch.dict(
        fn.__globals__,
        runpy=SimpleNamespace(
            run_path=lambda _: {
                "staged_dataset": staged,
                "SOURCE_URI": "public",
                "MANIFEST_SHA256": "pinned",
            }
        ),
    ):
        record = {"checks": {}}
        if mismatch:
            with pytest.raises(AssertionError, match="checkpoint export differs"):
                fn(record, tmp_path, {}, stage)
            assert record["checks"] == {}
            return
        fn(record, tmp_path, {}, stage)
    assert [row[1][3] for row in commands] == ["export", "evaluate", "train", "export", "evaluate"]
    assert record["checks"] == {"public_so101_cli": "PASS"}
    assert record["public_training"]["global_step"] == 64
    assert record["public_training"]["parameters_changed"]
    assert record["public_training"]["complete_checkpoint_state_match"]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Policy evaluation: {}",
        "Model evaluation (test split): {}",
        "Model evaluation (test split): {'test/mse': 1e999}",
    ],
)
def test_public_metrics_require_actual_finite_test_result(tmp_path, text):
    path = tmp_path / "log"
    path.write_text(text)
    fn = runpy.run_path("tests/fixtures/physical_ai_public_training.py")["model_metrics"]
    with pytest.raises(AssertionError):
        fn(path)


def test_isaac_recording_keeps_physx_and_original_scalar_metrics(tmp_path):
    from nexuml.core.config import ResolvedConfig
    from nexuml.core.types import PolicyEvaluationSpec
    from nexuml_library.environments.isaaclab import IsaacLabEnvironment
    from nexuml_library.scenarios.reinforcement.isaaclab import isaac_cartpole_ppo

    fn = runpy.run_path("tests/fixtures/physical_ai_isaac_recording.py")["run_recorded_isaac"]
    spec = isaac_cartpole_ppo(total_frames=1024, frames_per_batch=128, num_envs=2)
    spec.policy_evaluation = PolicyEvaluationSpec(episodes=2, seed=10000)
    config = ResolvedConfig.from_scenario(spec)
    config.save(tmp_path / "eval-case.yaml")
    policy = tmp_path / "checkpoint-policy"
    policy.mkdir()
    config.save(policy / "resolved_config.yaml")
    scalar = {"eval/rl/return_mean": 1.0}
    record = {
        "recipe": "isaac-cartpole-ppo",
        "assertions": {"training_frames": 1024},
        "metrics": scalar,
        "checks": {},
    }
    media = mock.Mock(return_value={"decoded": True})
    shared = {"_policy_identity": lambda *_: {"checkpoint": "fixed"}, "_gif_info": mock.Mock()}

    def stage(record, name, argv, env):
        assert name == "isaac-recorded-evaluate" and argv[-2:] == ["--device", "cuda:0"]
        (tmp_path / "logs/isaac-recorded-tb/run/artifacts/evaluation").mkdir(parents=True)
        return {"log_path": str(tmp_path / "render.log")}

    def load(path):
        return {"_media": media} if "mujoco_recording" in path else shared

    with mock.patch.dict(fn.__globals__, runpy=SimpleNamespace(run_path=load)):
        fn(record, tmp_path, {}, stage, lambda *_, **__: scalar)
    saved = ResolvedConfig.load(tmp_path / "isaac-recorded-eval.yaml")
    assert saved.interaction is not None and saved.policy_evaluation is not None
    assert isinstance(saved.interaction.environment, IsaacLabEnvironment)
    assert saved.interaction.environment.physics == "isaacsim_physx"
    assert saved.policy_evaluation.video.max_frames == 120
    assert saved.policy_evaluation.video.max_size == 320
    assert media.call_args.kwargs == {"capture_frames": 120, "capture_size": 320}
    assert record["checks"] == {"isaac_recorded_replay": "PASS"}
