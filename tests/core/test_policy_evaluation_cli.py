"""Inference-package replay stays outside the training lifecycle."""

import builtins
from contextlib import contextmanager
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest
import torch
from click import unstyle
from typer.testing import CliRunner

from nexuml.cli.main import app
from nexuml.core.compiler import compile
from nexuml.core.config import ResolvedConfig
from nexuml.core.export import export_package, load_inference_package
from nexuml.core.policy import CompiledPolicy
from nexuml.core.serialization import lower_component, lower_model
from nexuml.core.types import (
    ExportSpec,
    InteractionContract,
    MLflowSpec,
    ReinforcementLearningSpec,
    ScenarioSpec,
)
from nexuml_library.environments.gymnasium import GymnasiumEnvironment
from nexuml_library.scenarios.reinforcement.gymnasium import cartpole_ppo, pendulum_ppo

runner = CliRunner()


def fail_training(*args, **kwargs):
    pytest.fail("replay must not construct training or export artifacts")


@pytest.mark.parametrize(
    "env_device,policy_device,expected",
    [("cpu", "auto", "cpu"), ("cuda:0", "auto", "cuda:0"), ("cpu", "cuda:1", "cuda:1")],
)
@pytest.mark.parametrize("episodes", [0, 1])
@pytest.mark.parametrize("explicit_device", [False, True])
@pytest.mark.parametrize("provider", ["isaac", "gymnasium"])
@pytest.mark.requires_optional("gymnasium")
def test_replay_moves_portable_cpu_policy_only_with_explicit_device(
    monkeypatch, tmp_path, env_device, policy_device, expected, episodes, explicit_device, provider
):
    from nexuml_library.scenarios.reinforcement.isaaclab import isaac_cartpole_ppo

    scenario = isaac_cartpole_ppo() if provider == "isaac" else cartpole_ppo()
    assert isinstance(scenario.training, ReinforcementLearningSpec)
    scenario.logging = None
    scenario.training.collector.env_device = env_device
    scenario.training.collector.policy_device = policy_device
    assert scenario.interaction is not None and scenario.policy is not None
    contract = scenario.interaction.environment.describe()
    policy = CompiledPolicy(
        compile(scenario), scenario.policy.action_adapter.build(contract), contract
    )
    placements = []
    expected_placements = [expected] if episodes and explicit_device else []
    monkeypatch.setattr(policy, "to", lambda device: placements.append(device) or policy)
    monkeypatch.setattr(
        "nexuml.core.export.load_inference_package",
        lambda path: (
            policy,
            ResolvedConfig.from_scenario(scenario),
            {"artifact_kind": "policy", "interaction_contract": contract.model_dump(mode="json")},
        ),
    )

    def evaluate(loaded, restored, **kwargs):
        assert loaded is policy and placements == expected_placements
        assert restored.training.evaluation.episodes == episodes
        return {"eval/rl/return_mean": 1.0}

    monkeypatch.setattr("nexuml.evaluation.policy.evaluate_policy", evaluate)
    command = ["evaluate", str(tmp_path), "-O", f"training.evaluation.episodes={episodes}"]
    if explicit_device:
        command += ["--device", expected]
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    assert placements == expected_placements
    assert all(parameter.device.type == "cpu" for parameter in policy.parameters())


@pytest.fixture(scope="module")
def policy_packages(tmp_path_factory):
    pytest.importorskip("gymnasium")
    root = tmp_path_factory.mktemp("replay_packages")
    packages = {}
    for factory in (cartpole_ppo, pendulum_ppo):
        scenario = factory(total_frames=16, frames_per_batch=16)
        scenario.exports = [ExportSpec(output=str(root / "must_not_export"))]
        contract = scenario.interaction.environment.describe()
        policy = CompiledPolicy(
            compile(scenario), scenario.policy.action_adapter.build(contract), contract
        )
        packages[scenario.name] = export_package(policy, root / scenario.name)
    return packages


@pytest.fixture
def replay_source(policy_packages, monkeypatch):
    policy, config, metadata = load_inference_package(policy_packages["cartpole-ppo"])
    source = SimpleNamespace(policy=policy, config=config, metadata=metadata)
    monkeypatch.setattr(
        "nexuml.core.export.load_inference_package",
        lambda _: (source.policy, source.config, source.metadata),
    )
    return source


class RecordingLogger:
    def __init__(self, finalize_error=None):
        self.metrics = []
        self.statuses = []
        self.finalize_error = finalize_error

    def log_metrics(self, metrics, step):
        self.metrics.append((metrics, step))

    def finalize(self, status):
        self.statuses.append(status)
        if self.finalize_error is not None:
            raise self.finalize_error


@pytest.fixture
def native_logging(monkeypatch):
    state = SimpleNamespace(loggers=[RecordingLogger()], configs=[], artifacts=[])

    def create(spec, run_name):
        state.configs.append((spec, run_name))
        return state.loggers

    monkeypatch.setattr("nexuml.tracking.logger.create_loggers", create)
    monkeypatch.setattr(
        "nexuml.tracking.logger.log_text_artifact",
        lambda loggers, text, name: state.artifacts.append((loggers, text, name)),
    )
    return state


@pytest.mark.parametrize("factory", [cartpole_ppo, pendulum_ppo])
def test_reference_logging_defaults_round_trip(factory):
    scenario = ResolvedConfig.from_yaml(ResolvedConfig.from_scenario(factory()).to_yaml())
    assert scenario.logging is not None
    assert isinstance(scenario.training, ReinforcementLearningSpec)
    assert scenario.logging.tensorboard is not None
    assert scenario.logging.mlflow is None
    assert not scenario.training.evaluation.video.enabled
    assert not scenario.training.evaluation.at_start
    assert scenario.training.evaluation.every_n_frames is None


@pytest.mark.parametrize("color", [False, True])
def test_evaluate_help(monkeypatch, color):
    monkeypatch.setattr("typer.rich_utils.FORCE_TERMINAL", color)
    result = runner.invoke(app, ["evaluate", "--help"], color=color)
    assert result.exit_code == 0, result.output
    if color:
        assert "\x1b[" in result.output
    assert "--override" in unstyle(result.output)


@pytest.mark.requires_optional("gymnasium")
@pytest.mark.requires_optional("torchrl")
@pytest.mark.parametrize("evaluation_fails", [False, True])
def test_checkpoint_replay_materializes_weight_parity_without_trainer(
    monkeypatch, tmp_path, native_logging, evaluation_fails
):
    import lightning as L
    from tensordict import TensorDict
    from nexuml.training.reinforcement import NexuRLLightningModule

    scenario = pendulum_ppo(total_frames=32, frames_per_batch=16)
    scenario.training.evaluation.episodes = 1
    module = NexuRLLightningModule(scenario)
    module.frames.fill_(16)
    with torch.no_grad():
        for parameter in module.policy.parameters():
            parameter.fill_(0.125)
    path = tmp_path / "trusted policy.ckpt"
    torch.save(
        {
            "pytorch-lightning_version": L.__version__,
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams),
        },
        path,
    )
    digest = sha256(path.read_bytes()).hexdigest()
    package = export_package(module.policy, tmp_path / "reference", checkpoint_path=path)
    observation = TensorDict(
        {
            key: torch.zeros(field.shape, dtype=getattr(torch, field.dtype))
            for key, field in module.policy.contract.observations.items()
        },
        batch_size=[],
    )
    reference, _, metadata = load_inference_package(package)
    expected = reference(observation, deterministic=True)["action"].detach()
    exports = []

    def load(candidate):
        exports.append(candidate)
        assert candidate.is_dir() and candidate != package
        restored, config, provenance = load_inference_package(candidate)
        assert provenance["policy_files_sha256"] == metadata["policy_files_sha256"]
        return restored, config, provenance

    def evaluate(policy, scenario, *, provenance, frame_step, **_):
        assert torch.equal(policy(observation, deterministic=True)["action"], expected)
        assert frame_step == 16
        assert provenance["source_package"] is None
        assert provenance["source_checkpoint"]["source"] == str(path)
        assert provenance["source_checkpoint"]["sha256"] == digest
        if evaluation_fails:
            raise ValueError("evaluation fixture failed")
        return {"eval/rl/episodes": 1.0}

    @contextmanager
    def job_scope(_):
        assert exports[0].is_dir()
        try:
            yield
        finally:
            assert not exports[0].exists(), "cleanup must precede non-returning native teardown"

    monkeypatch.setattr("nexuml.core.export.load_inference_package", load)
    monkeypatch.setattr("nexuml.evaluation.policy.evaluate_policy", evaluate)
    monkeypatch.setattr(GymnasiumEnvironment, "job_scope", job_scope)
    monkeypatch.setattr("nexuml.training.lightning.NexuSession", fail_training)
    monkeypatch.setattr(L.Trainer, "__init__", fail_training)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", fail_training)
    monkeypatch.setattr("nexuml.training.reinforcement.build_collector", fail_training)
    commands = ["evaluate", "--trainer-checkpoint", str(path)]
    result = runner.invoke(app, commands)
    assert result.exit_code == int(evaluation_fails), (result.output, result.exception)
    if evaluation_fails:
        assert "evaluation fixture failed" in result.output
    assert native_logging.loggers[0].statuses == ["failed" if evaluation_fails else "success"]
    assert len(exports) == 1 and not exports[0].exists()
    assert sha256(path.read_bytes()).hexdigest() == digest


def test_checkpoint_replay_rejects_model_before_data_build(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    torch.save(
        {"hyper_parameters": {"scenario": lower_model(ScenarioSpec(name="model"))}}, checkpoint
    )
    monkeypatch.setattr("nexuml.training.lightning.create_data_module_from_spec", fail_training)
    result = runner.invoke(app, ["evaluate", "--trainer-checkpoint", str(checkpoint)])
    assert result.exit_code == 1
    assert "--trainer-checkpoint requires a complete policy checkpoint" in result.output


def test_mlflow_policy_replay_publishes_episode_results(policy_packages, native_logging):
    result = runner.invoke(
        app,
        [
            "evaluate",
            str(policy_packages["cartpole-ppo"]),
            "-O",
            "logging.mlflow={}",
            "-O",
            "training.evaluation.episodes=2",
            "-O",
            "training.evaluation.seed=17",
            "-O",
            "training.evaluation.max_steps_per_episode=1",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert isinstance(native_logging.configs[0][0].mlflow, MLflowSpec)
    artifacts = {name: text for _, text, name in native_logging.artifacts}
    report = json.loads(artifacts["evaluation-episodes.json"])
    assert report["completed_episodes"] == report["requested_episodes"] == 2
    assert [row["seed"] for row in report["episodes"]] == [17, 18]
    assert all(row["task_success"] is None for row in report["episodes"])
    assert report["selected_failure"] is None
    assert native_logging.loggers[0].statuses == ["success"]


@pytest.mark.parametrize("name", ["cartpole-ppo", "pendulum-ppo"])
def test_packaged_scalar_replay_without_training_or_encoders(
    name, policy_packages, native_logging, monkeypatch
):
    from nexuml_library.reinforcement.algorithms.ppo import PPO

    monkeypatch.setattr("nexuml.training.lightning.NexuSession", fail_training)
    monkeypatch.setattr(PPO, "build", fail_training)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", fail_training)
    monkeypatch.setattr("nexuml.training.reinforcement.build_collector", fail_training)
    monkeypatch.setattr("nexuml.reinforcement.collector.build_collector", fail_training)
    monkeypatch.setattr("nexuml.core.export.export_package", fail_training)
    original_import = builtins.__import__

    def no_encoders(name, *args, **kwargs):
        if name.split(".")[0] in {"PIL", "imageio", "imageio_ffmpeg", "moviepy"}:
            pytest.fail(f"scalar replay imported encoder {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_encoders)
    result = runner.invoke(
        app,
        [
            "evaluate",
            str(policy_packages[name]),
            "-O",
            "training.evaluation.episodes=1",
            "-O",
            "training.evaluation.max_steps_per_episode=3",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert "Applied overrides" in result.output and "Evaluation complete" in result.output
    metrics, step = native_logging.loggers[0].metrics[0]
    assert step == 0 and metrics["eval/rl/length_mean"] == 3
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert native_logging.loggers[0].statuses == ["success"]


@pytest.mark.parametrize(
    "override,diagnostic",
    [
        ("pipeline={}", "packaged policy/pipeline"),
        ("policy.action_adapter={}", "packaged policy/pipeline"),
        ("training.total_frames=32", "not supported for replay"),
        ("interaction.environment.env_name=Pendulum-v1", "not supported for replay"),
        ("exports=[]", "not supported for replay"),
        ("training.evaluation.unknown=1", "Override path not found"),
        ("training.evaluation..episodes=1", "empty segment"),
        ("training.evaluation.episodes", "expected key=value"),
        ("training.evaluation.episodes=-1", "greater than or equal to 0"),
        ("training.evaluation.video.format=avi", "gif"),
        ("training.evaluation.video.max_frames=0", "greater than 0"),
        ("training.evaluation.video.fps=nan", "finite"),
    ],
)
def test_invalid_replay_override_fails_before_execution(
    override, diagnostic, replay_source, monkeypatch
):
    monkeypatch.setattr("nexuml.evaluation.policy.evaluate_policy", fail_training)
    monkeypatch.setattr("nexuml.tracking.logger.create_loggers", fail_training)
    result = runner.invoke(app, ["evaluate", "package", "-O", override])
    assert result.exit_code != 0
    assert diagnostic in " ".join(result.output.split())


def test_recording_with_zero_episodes_is_revalidated(replay_source, monkeypatch):
    monkeypatch.setattr("nexuml.evaluation.policy.evaluate_policy", fail_training)
    monkeypatch.setattr("nexuml.tracking.logger.create_loggers", fail_training)
    result = runner.invoke(
        app,
        [
            "evaluate",
            "package",
            "-O",
            "training.evaluation.episodes=0",
            "-O",
            "training.evaluation.video.enabled=true",
        ],
    )
    assert result.exit_code != 0
    assert "episodes > 0" in result.output


@pytest.mark.parametrize("kind", ["pipeline", None])
def test_non_policy_package_rejected(kind, replay_source, monkeypatch):
    replay_source.metadata["artifact_kind"] = kind
    monkeypatch.setattr("nexuml.evaluation.policy.evaluate_policy", fail_training)
    result = runner.invoke(app, ["evaluate", "package"])
    assert result.exit_code != 0
    assert "complete exported policy package" in result.output


def test_incomplete_policy_rejected(replay_source):
    replay_source.policy = lambda _: None
    result = runner.invoke(app, ["evaluate", "package"])
    assert result.exit_code != 0 and "complete exported policy" in result.output


def test_missing_rl_configuration_rejected(replay_source):
    replay_source.config = ResolvedConfig(name="not-rl")
    result = runner.invoke(app, ["evaluate", "package"])
    assert result.exit_code != 0 and "embedded RL scenario" in result.output


def test_missing_package_has_focused_diagnostic(tmp_path):
    result = runner.invoke(app, ["evaluate", str(tmp_path / "absent")])
    assert result.exit_code != 0
    assert "No package artifact found" in result.output


@pytest.mark.parametrize("environment,compatible", [("CartPole-v1", True), ("Pendulum-v1", False)])
def test_evaluation_environment_override_uses_serialized_definition(
    environment, compatible, replay_source, native_logging
):
    definition = lower_component(
        GymnasiumEnvironment(env_name=environment, kwargs={"max_episode_steps": 2})
    )
    result = runner.invoke(
        app,
        [
            "evaluate",
            "package",
            "-O",
            "training.evaluation.episodes=1",
            "-O",
            f"interaction.evaluation_environment={json.dumps(definition)}",
        ],
    )
    if compatible:
        assert result.exit_code == 0, (result.output, result.exception)
        assert native_logging.loggers[0].metrics[0][0]["eval/rl/length_mean"] == 2
    else:
        assert result.exit_code != 0
        assert "do not match the policy" in result.output
        assert native_logging.loggers[0].statuses == ["failed"]


@pytest.mark.parametrize(
    "source_frames,expected",
    [(None, None), (32, 32), (0, 0), (-1, None), (True, None), (1.5, None)],
)
def test_native_logging_uses_explicit_source_frames_and_provenance(
    source_frames, expected, replay_source, native_logging, monkeypatch
):
    replay_source.metadata["global_step"] = 99
    if source_frames is not None:
        replay_source.metadata["training_frames"] = source_frames
    received = []

    def evaluate(policy, scenario, **kwargs):
        received.append(kwargs)
        return {"eval/rl/return_mean": 1.0}

    monkeypatch.setattr("nexuml.evaluation.policy.evaluate_policy", evaluate)
    result = runner.invoke(app, ["evaluate", "package"])
    assert result.exit_code == 0, result.output
    assert received[0]["frame_step"] == (expected if expected is not None else 0)
    assert native_logging.loggers[0].metrics == [
        ({"eval/rl/return_mean": 1.0}, received[0]["frame_step"])
    ]
    provenance = received[0]["provenance"]
    assert provenance["source_training_frames"] == expected
    assert provenance["source_package"].endswith("/package")
    assert provenance["source_config_hash"] == replay_source.metadata["config_hash"]
    assert (
        provenance["interaction_contract_hash"]
        == replay_source.metadata["interaction_contract_hash"]
    )
    artifacts = {
        name: text
        for loggers, text, name in native_logging.artifacts
        if loggers is native_logging.loggers
    }
    assert json.loads(artifacts["source_package.json"]) == provenance
    assert ResolvedConfig.from_yaml(artifacts["config.yaml"]).name == "cartpole-ppo"
    assert native_logging.configs[0][1] == "cartpole-ppo-evaluation"


@pytest.mark.parametrize("tensorboard", [False, True])
def test_media_uses_same_native_loggers_after_preflight(
    tensorboard, replay_source, native_logging, monkeypatch
):
    video_calls = []
    frames, metadata = object(), {"captured_frames": 2}
    replay_source.metadata["training_frames"] = 16

    def preflight(loggers, video):
        assert loggers is native_logging.loggers and video.format == "both"
        video_calls.append("preflight")

    def media(loggers, actual_frames, actual_metadata, video, *, step):
        assert loggers is native_logging.loggers
        assert actual_frames is frames and actual_metadata is metadata
        assert step == 16 and video.format == "both"
        video_calls.append("media")

    def evaluate(policy, scenario, *, media_sink, **kwargs):
        assert video_calls == ["preflight"]
        assert isinstance(policy.contract, InteractionContract)
        assert (
            policy.contract.model_dump(mode="json")
            == replay_source.metadata["interaction_contract"]
        )
        assert scenario.logging.mlflow.tracking_uri == "sqlite:///replay.db"
        assert (scenario.logging.tensorboard is not None) == tensorboard
        media_sink(frames, metadata)
        return {"eval/rl/return_mean": 1.0}

    monkeypatch.setattr("nexuml.tracking.logger.preflight_evaluation_media", preflight)
    monkeypatch.setattr("nexuml.tracking.logger.log_evaluation_media", media)
    monkeypatch.setattr("nexuml.evaluation.policy.evaluate_policy", evaluate)
    logging = {"mlflow": {"tracking_uri": "sqlite:///replay.db"}}
    if tensorboard:
        logging["tensorboard"] = {}
    result = runner.invoke(
        app,
        [
            "evaluate",
            "package",
            "-O",
            f"logging={json.dumps(logging)}",
            "-O",
            'training.evaluation.video={"enabled":true,"format":"both","max_frames":2}',
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert video_calls == ["preflight", "media"]
    assert native_logging.loggers[0].statuses == ["success"]


@pytest.mark.parametrize("error", [RuntimeError("primary replay failure"), KeyboardInterrupt()])
def test_failure_finalizes_all_loggers_without_masking_primary(
    error, replay_source, native_logging, monkeypatch
):
    native_logging.loggers = [RecordingLogger(RuntimeError("close failure")), RecordingLogger()]

    def evaluate(*args, **kwargs):
        raise error

    monkeypatch.setattr("nexuml.evaluation.policy.evaluate_policy", evaluate)
    result = runner.invoke(app, ["evaluate", "package"])
    assert result.exit_code != 0
    assert (
        "primary replay failure" if isinstance(error, RuntimeError) else "cancelled"
    ) in result.output
    assert "Evaluation complete" not in result.output
    assert all(logger.statuses == ["failed"] for logger in native_logging.loggers)
    assert result.output.index("Evaluation failed:") < result.output.index("close failure")


def test_finalization_failure_is_not_success(replay_source, native_logging):
    native_logging.loggers = [RecordingLogger(RuntimeError("close failure")), RecordingLogger()]
    result = runner.invoke(app, ["evaluate", "package", "-O", "training.evaluation.episodes=0"])
    assert result.exit_code != 0
    assert "close failure" in result.output and "Evaluation complete" not in result.output
    assert all(logger.statuses == ["success"] for logger in native_logging.loggers)


def test_signal_exit_keeps_status_and_finalizes_failed_loggers(
    replay_source, native_logging, monkeypatch
):
    native_logging.loggers = [RecordingLogger(RuntimeError("close failure")), RecordingLogger()]

    def evaluate(*args, **kwargs):
        raise SystemExit(143)

    monkeypatch.setattr("nexuml.evaluation.policy.evaluate_policy", evaluate)
    result = runner.invoke(app, ["evaluate", "package"])
    assert result.exit_code == 143
    assert "Evaluation complete" not in result.output
    assert all(logger.statuses == ["failed"] for logger in native_logging.loggers)


def test_failed_media_preflight_prevents_evaluation(replay_source, native_logging, monkeypatch):
    def preflight(*args):
        raise ValueError("missing media target")

    monkeypatch.setattr("nexuml.tracking.logger.preflight_evaluation_media", preflight)
    monkeypatch.setattr("nexuml.evaluation.policy.evaluate_policy", fail_training)
    result = runner.invoke(
        app, ["evaluate", "package", "-O", "training.evaluation.video.enabled=true"]
    )
    assert result.exit_code != 0 and "missing media target" in result.output
    assert native_logging.loggers[0].statuses == ["failed"]
