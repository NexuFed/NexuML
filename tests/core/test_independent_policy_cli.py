"""Generic offline-policy CLI proof; the tiny MLP does not establish ACT/VLA support."""

import asyncio
import io
import json
from types import SimpleNamespace

import pytest
import torch
from typer.testing import CliRunner

from nexuml.cli.main import app
from nexuml.core.compiler import compile
from nexuml.core.config import ResolvedConfig
from nexuml.core.export import export_package, load_inference_package
from nexuml.core.policy import CompiledPolicy
from nexuml.core.serialization import lower_component, lower_model, restore_model_data
from nexuml.core.types import RLEvaluationSpec, ReinforcementLearningSpec, TrainingSpec
from nexuml_library.environments.gymnasium import GymnasiumEnvironment
from nexuml_library.scenarios.reinforcement.gymnasium import cartpole_ppo
from test_independent_policy_evaluation import forbid_training, forbidden, offline_scenario
from test_policy_evaluation_cli import RecordingLogger, native_logging as native_logging

runner = CliRunner()


@pytest.fixture(scope="module")
def independent_packages(tmp_path_factory):
    pytest.importorskip("gymnasium")
    root = tmp_path_factory.mktemp("independent_cli_packages")
    packages = {}
    for name in ("offline", "offline-no-evaluation", "legacy"):
        scenario = offline_scenario()
        if name != "offline":
            scenario.policy_evaluation = None
        if name == "legacy":
            scenario.training = cartpole_ppo().training
            scenario.training.seed = 91
            scenario.training.evaluation = RLEvaluationSpec(episodes=0)
        contract = scenario.interaction.environment.describe()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            policy = CompiledPolicy(
                compile(scenario), scenario.policy.action_adapter.build(contract), contract
            )
        packages[name] = export_package(policy, root / name)
    return packages


@pytest.fixture
def offline_source(independent_packages, monkeypatch):
    policy, config, metadata = load_inference_package(independent_packages["offline"])
    source = SimpleNamespace(policy=policy, config=config, metadata=metadata)
    monkeypatch.setattr(
        "nexuml.core.export.load_inference_package",
        lambda _: (source.policy, source.config, source.metadata),
    )
    return source


@pytest.fixture
def runtime_builds(monkeypatch):
    builds = []
    original = GymnasiumEnvironment.build

    def build(definition, context):
        runtime = original(definition, context)
        builds.append((definition, context, runtime))
        return runtime

    monkeypatch.setattr(GymnasiumEnvironment, "build", build)
    return builds


def test_exported_offline_policy_runs_real_cpu_cli_and_keeps_training_provenance(
    independent_packages, native_logging, runtime_builds, monkeypatch
):
    package = independent_packages["offline"]
    source_yaml = (package / "resolved_config.yaml").read_bytes()
    original = ResolvedConfig.load(package / "resolved_config.yaml")
    training = lower_model(original.training)
    forbid_training(monkeypatch)
    monkeypatch.setattr("nexuml.core.export.export_package", forbidden)
    result = runner.invoke(
        app,
        [
            "evaluate",
            str(package),
            "-O",
            "policy_evaluation.episodes=1",
            "-O",
            "policy_evaluation.max_steps_per_episode=2",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert "Applied overrides" in result.output and "Evaluation complete" in result.output
    metrics, step = native_logging.loggers[0].metrics[0]
    assert step == 0 and metrics["eval/rl/length_mean"] == metrics["eval/rl/return_mean"] == 2
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert native_logging.loggers[0].statuses == ["success"]
    assert len(runtime_builds) == 1
    _, context, runtime = runtime_builds[0]
    assert context.device == "cpu" and context.purpose == "eval" and context.seed == 7
    assert context.num_envs == 1 and context.render_mode is None and not runtime.envs
    artifacts = {name: text for _, text, name in native_logging.artifacts}
    effective = ResolvedConfig.from_yaml(artifacts["config.yaml"])
    provenance = json.loads(artifacts["source_package.json"])
    assert type(effective.training) is TrainingSpec
    assert lower_model(effective.training) == provenance["resolved_config"]["training"] == training
    assert "algorithm" not in training and "mode" not in training
    assert provenance["source_training_frames"] is None
    assert provenance["source_package"] == str(package.resolve())
    assert provenance["source_config_hash"] is not None
    assert effective.policy_evaluation is not None
    assert effective.policy_evaluation.episodes == 1
    assert effective.policy_evaluation.max_steps_per_episode == 2
    assert (package / "resolved_config.yaml").read_bytes() == source_yaml


@pytest.mark.parametrize("name", ["offline-no-evaluation", "legacy"])
@pytest.mark.parametrize("seed", [None, 0])
def test_whole_block_repairs_offline_or_explicitly_migrates_legacy_without_seed_fallback(
    name, seed, independent_packages, native_logging, runtime_builds, monkeypatch
):
    package = independent_packages[name]
    source_yaml = (package / "resolved_config.yaml").read_bytes()
    source = ResolvedConfig.load(package / "resolved_config.yaml")
    assert source.policy_evaluation is None
    block = {"episodes": 1, "max_steps_per_episode": 2, "seed": seed}
    forbid_training(monkeypatch)
    result = runner.invoke(
        app, ["evaluate", str(package), "-O", f"policy_evaluation={json.dumps(block)}"]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert len(runtime_builds) == 1
    _, context, runtime = runtime_builds[0]
    assert context.seed == seed and not runtime.envs
    assert native_logging.loggers[0].metrics[0][0]["eval/rl/length_mean"] == 2
    artifacts = {name: text for _, text, name in native_logging.artifacts}
    effective = ResolvedConfig.from_yaml(artifacts["config.yaml"])
    assert lower_model(effective.training) == lower_model(source.training)
    assert effective.policy_evaluation is not None
    assert effective.policy_evaluation.seed == seed
    if name == "legacy":
        assert isinstance(effective.training, ReinforcementLearningSpec)
        assert effective.training.seed == 91 and effective.training.evaluation.episodes == 0
    else:
        assert type(effective.training) is TrainingSpec
    assert (package / "resolved_config.yaml").read_bytes() == source_yaml


def test_actual_legacy_package_keeps_training_evaluation_override_path(
    independent_packages, native_logging, runtime_builds, monkeypatch
):
    forbid_training(monkeypatch)
    result = runner.invoke(
        app,
        [
            "evaluate",
            str(independent_packages["legacy"]),
            "-O",
            "training.evaluation.episodes=1",
            "-O",
            "training.evaluation.max_steps_per_episode=2",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert runtime_builds[0][1].seed == 91
    assert native_logging.loggers[0].metrics[0][0]["eval/rl/length_mean"] == 2
    artifacts = {name: text for _, text, name in native_logging.artifacts}
    assert ResolvedConfig.from_yaml(artifacts["config.yaml"]).policy_evaluation is None


@pytest.mark.parametrize(
    "override,diagnostic",
    [
        ("policy_evaluation.episodes=-1", "greater than or equal to 0"),
        ("policy_evaluation.every_n_frames=0", "greater than 0"),
        ("policy_evaluation.video.format=avi", "gif"),
        ("policy_evaluation.video.max_frames=0", "greater than 0"),
        ("policy_evaluation.video.fps=nan", "finite"),
        ("policy_evaluation.unknown=1", "Override path not found"),
        ("policy_evaluation..episodes=1", "empty segment"),
        ("policy_evaluation.episodes", "expected key=value"),
        ("pipeline={}", "packaged policy/pipeline"),
        ("model={}", "not supported for replay"),
        ("policy.action_adapter={}", "packaged policy/pipeline"),
        ("training.max_epochs=2", "not supported for replay"),
        ("training.mode=reinforcement", "not supported for replay"),
        ("interaction.environment.env_name=Pendulum-v1", "not supported for replay"),
    ],
)
def test_new_block_invalid_or_out_of_scope_overrides_fail_before_runtime_and_logging(
    override, diagnostic, offline_source, monkeypatch
):
    monkeypatch.setattr("nexuml.tracking.logger.create_loggers", forbidden)
    monkeypatch.setattr("nexuml.evaluation.policy.evaluate_policy", forbidden)
    monkeypatch.setattr(GymnasiumEnvironment, "build", forbidden)
    result = runner.invoke(app, ["evaluate", "package", "-O", override])
    assert result.exit_code != 0
    assert diagnostic in " ".join(result.output.split())


def test_new_block_video_with_zero_episodes_is_revalidated(offline_source, monkeypatch):
    monkeypatch.setattr("nexuml.tracking.logger.create_loggers", forbidden)
    monkeypatch.setattr("nexuml.evaluation.policy.evaluate_policy", forbidden)
    result = runner.invoke(
        app,
        [
            "evaluate",
            "package",
            "-O",
            "policy_evaluation.episodes=0",
            "-O",
            "policy_evaluation.video.enabled=true",
        ],
    )
    assert result.exit_code != 0 and "episodes > 0" in result.output


@pytest.mark.parametrize("legacy", [False, True])
def test_explicit_block_rejects_legacy_override_instead_of_ignoring_it(
    legacy, offline_source, monkeypatch
):
    if legacy:
        offline_source.config.training = cartpole_ppo().training
    monkeypatch.setattr("nexuml.tracking.logger.create_loggers", forbidden)
    monkeypatch.setattr("nexuml.evaluation.policy.evaluate_policy", forbidden)
    result = runner.invoke(app, ["evaluate", "package", "-O", "training.evaluation.episodes=0"])
    assert result.exit_code != 0
    assert "policy_evaluation" in result.output and "training.evaluation" in result.output


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("whole", [False, True])
def test_mixed_override_namespaces_fail_before_runtime_even_for_old_configuration(
    legacy, reverse, whole, offline_source, monkeypatch
):
    if legacy:
        values = lower_model(offline_source.config)
        del values["policy_evaluation"]
        values["training"] = lower_model(cartpole_ppo().training)
        offline_source.config = ResolvedConfig.model_validate(
            restore_model_data(values, ResolvedConfig)
        )
    overrides = [
        'policy_evaluation={"episodes":1}' if whole else "policy_evaluation.episodes=1",
        "training.evaluation.episodes=2",
    ]
    if reverse:
        overrides.reverse()
    monkeypatch.setattr("nexuml.tracking.logger.create_loggers", forbidden)
    monkeypatch.setattr("nexuml.evaluation.policy.evaluate_policy", forbidden)
    result = runner.invoke(app, ["evaluate", "package", "-O", overrides[0], "-O", overrides[1]])
    assert result.exit_code != 0 and "mix" in result.output.lower()
    assert "policy_evaluation" in result.output and "training.evaluation" in result.output


@pytest.mark.parametrize("legacy", [False, True])
def test_nested_new_override_does_not_initialize_missing_block(legacy, offline_source, monkeypatch):
    offline_source.config.policy_evaluation = None
    if legacy:
        offline_source.config.training = cartpole_ppo().training
    monkeypatch.setattr("nexuml.tracking.logger.create_loggers", forbidden)
    monkeypatch.setattr("nexuml.evaluation.policy.evaluate_policy", forbidden)
    result = runner.invoke(app, ["evaluate", "package", "-O", "policy_evaluation.episodes=1"])
    assert result.exit_code != 0
    assert "policy_evaluation=" in "".join(result.output.split())


@pytest.mark.parametrize("missing", ["interaction", "policy", "policy_evaluation"])
def test_missing_evaluation_inputs_have_actionable_cli_diagnostic(
    missing, offline_source, monkeypatch
):
    offline_source.config.policy_evaluation = None
    setattr(offline_source.config, missing, None)
    forbid_training(monkeypatch)
    monkeypatch.setattr("nexuml.tracking.logger.create_loggers", forbidden)
    monkeypatch.setattr(GymnasiumEnvironment, "build", forbidden)
    result = runner.invoke(app, ["evaluate", "package"])
    assert result.exit_code != 0 and missing in result.output


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("episodes", [0, 1])
@pytest.mark.parametrize("explicit_device", [False, True])
def test_policy_placement_uses_selected_episode_budget_without_disabled_allocation(
    legacy, episodes, explicit_device, offline_source, native_logging, monkeypatch
):
    if legacy:
        offline_source.config.training = cartpole_ppo().training
        offline_source.config.training.evaluation = RLEvaluationSpec(
            episodes=9 if not episodes else 0
        )
    placements = []
    monkeypatch.setattr(offline_source.policy, "to", lambda device: placements.append(device))
    monkeypatch.setattr(GymnasiumEnvironment, "describe", forbidden)
    monkeypatch.setattr(GymnasiumEnvironment, "build", forbidden)
    if episodes:

        def evaluate(policy, scenario, **kwargs):
            assert scenario.policy_evaluation.episodes == 1
            return {"eval/rl/return_mean": 1.0}

        monkeypatch.setattr("nexuml.evaluation.policy.evaluate_policy", evaluate)
    else:
        monkeypatch.setattr(offline_source.policy, "forward", forbidden)
    forbid_training(monkeypatch)
    command = ["evaluate", "package", "-O", f"policy_evaluation.episodes={episodes}"]
    if explicit_device:
        command += ["--device", "cpu"]
    result = runner.invoke(app, command)
    assert result.exit_code == 0, (result.output, result.exception)
    assert placements == (["cpu"] if explicit_device and episodes else [])
    assert native_logging.loggers[0].metrics[0][0] == (
        {"eval/rl/return_mean": 1.0} if episodes else {}
    )


@pytest.mark.parametrize("compatible", [False, True])
def test_offline_evaluation_environment_override_is_restored_and_contract_checked(
    compatible, independent_packages, native_logging, monkeypatch
):
    definition = lower_component(
        GymnasiumEnvironment(
            env_name="CartPole-v1" if compatible else "Pendulum-v1",
            kwargs={"max_episode_steps": 2},
        )
    )
    forbid_training(monkeypatch)
    result = runner.invoke(
        app,
        [
            "evaluate",
            str(independent_packages["offline"]),
            "-O",
            f"interaction.evaluation_environment={json.dumps(definition)}",
        ],
    )
    if compatible:
        assert result.exit_code == 0, (result.output, result.exception)
        assert native_logging.loggers[0].metrics[0][0]["eval/rl/length_mean"] == 2
    else:
        assert result.exit_code != 0 and "do not match the policy" in result.output
        assert native_logging.loggers[0].statuses == ["failed"]


@pytest.mark.parametrize("legacy", [False, True])
def test_selected_block_controls_media_preflight_and_native_logger_sink(
    legacy, offline_source, native_logging, monkeypatch
):
    if legacy:
        offline_source.config.training = cartpole_ppo().training
        assert not offline_source.config.training.evaluation.video.enabled
    calls = []
    frames, metadata = object(), {"captured_frames": 2}

    def preflight(loggers, video):
        assert loggers is native_logging.loggers and video.format == "both"
        calls.append("preflight")

    def media(loggers, actual_frames, actual_metadata, video, *, step):
        assert loggers is native_logging.loggers and video.format == "both"
        assert actual_frames is frames and actual_metadata is metadata and step == 0
        calls.append("media")

    def evaluate(policy, scenario, *, media_sink, **kwargs):
        assert calls == ["preflight"]
        assert scenario.policy_evaluation.video.enabled
        assert scenario.logging.tensorboard.log_dir == "unused-fake-sink"
        media_sink(frames, metadata)
        return {}

    monkeypatch.setattr("nexuml.tracking.logger.preflight_evaluation_media", preflight)
    monkeypatch.setattr("nexuml.tracking.logger.log_evaluation_media", media)
    monkeypatch.setattr("nexuml.evaluation.policy.evaluate_policy", evaluate)
    result = runner.invoke(
        app,
        [
            "evaluate",
            "package",
            "-O",
            'policy_evaluation.video={"enabled":true,"format":"both","max_frames":2}',
            "-O",
            'logging={"tensorboard":{"log_dir":"unused-fake-sink"}}',
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert calls == ["preflight", "media"]
    assert native_logging.loggers[0].statuses == ["success"]


def test_cancelled_error_preserves_cancellation_and_finalizes_every_logger_failed(
    offline_source, native_logging, monkeypatch
):
    error = asyncio.CancelledError("policy replay cancelled")
    native_logging.loggers = [RecordingLogger(RuntimeError("close failure")), RecordingLogger()]

    def evaluate(*args, **kwargs):
        raise error

    monkeypatch.setattr("nexuml.evaluation.policy.evaluate_policy", evaluate)
    with pytest.raises(asyncio.CancelledError) as caught:
        runner.invoke(app, ["evaluate", "package"])
    assert caught.value is error
    assert all(logger.statuses == ["failed"] for logger in native_logging.loggers)
    assert all(not logger.metrics for logger in native_logging.loggers)


@pytest.mark.requires_optional("pygame")
@pytest.mark.requires_optional("PIL")
@pytest.mark.requires_optional("imageio")
@pytest.mark.requires_optional("imageio_ffmpeg")
@pytest.mark.requires_optional("tensorboard")
def test_actual_offline_package_records_real_gym_rgb_gif_mp4_and_tensorboard(
    independent_packages, tmp_path, monkeypatch
):
    from PIL import Image, ImageSequence
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import imageio.v2 as imageio

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.setenv("NEXUML_TEMP_ARTIFACT_DIR", str(tmp_path / "staging"))
    logging = {
        "tensorboard": {"log_dir": str(tmp_path / "tensorboard")},
        "temp_artifact_dir": str(tmp_path / "staging"),
        "diagram": None,
    }
    block = {
        "episodes": 1,
        "seed": 0,
        # Three steps can produce identical downscaled pixels for a valid left-action policy.
        "max_steps_per_episode": 8,
        "video": {"enabled": True, "format": "both", "max_frames": 8, "max_size": 64},
    }
    forbid_training(monkeypatch)
    result = runner.invoke(
        app,
        [
            "evaluate",
            str(independent_packages["offline"]),
            "-O",
            f"policy_evaluation={json.dumps(block)}",
            "-O",
            f"logging={json.dumps(logging)}",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    metadata_paths = list((tmp_path / "tensorboard").rglob("frame_*.json"))
    assert len(metadata_paths) == 1
    metadata_path = metadata_paths[0]
    metadata = json.loads(metadata_path.read_text())
    assert metadata["formats"] == ["gif", "mp4"] and metadata["captured_frames"] == 8
    assert metadata["capture_capped"] and metadata["seeds"] == [0]
    assert metadata["source_size"] == [600, 400] and max(metadata["captured_size"]) == 64
    assert metadata["metrics"]["eval/rl/length_mean"] == 8
    assert metadata["provenance"]["source_training_frames"] is None
    assert "algorithm" not in metadata["provenance"]["resolved_config"]["training"]
    with Image.open(metadata_path.with_suffix(".gif")) as gif:
        assert max(gif.size) == 64 and sum(1 for _ in ImageSequence.Iterator(gif)) >= 2
    video = imageio.mimread(str(metadata_path.with_suffix(".mp4")))
    assert len(video) == 8 and max(video[0].shape[:2]) == 64
    run_dir = metadata_path.parents[2]
    events = EventAccumulator(str(run_dir)).Reload()
    preview = events.Images("eval/rl/episode")[0]
    assert preview.step == 0
    with Image.open(io.BytesIO(preview.encoded_image_string)) as gif:
        assert sum(1 for _ in ImageSequence.Iterator(gif)) >= 2
    training_yaml = ResolvedConfig.load(run_dir / "artifacts" / "config.yaml")
    assert type(training_yaml.training) is TrainingSpec
    assert lower_model(training_yaml.training) == lower_model(offline_scenario().training)
