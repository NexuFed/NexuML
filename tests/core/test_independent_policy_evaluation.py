"""Trainer-independent evaluation contracts, not ACT/VLA or native-physics proof."""

import copy
import hashlib
import json
import random
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from tensordict import TensorDict
from torch import nn

from nexuml.core.compiler import compile
from nexuml.core.config import ResolvedConfig
from nexuml.core.export import export_package, load_inference_package, load_package
from nexuml.core.policy import CompiledPolicy
from nexuml.core.serialization import lower_component, lower_model, restore_model_data
from nexuml.core.types import (
    DataSpec,
    InteractionSpec,
    PolicySpec,
    RLEvaluationSpec,
    RLVideoSpec,
    ScenarioSpec,
    TrainingSpec,
)
from nexuml.training.reinforcement import (
    NexuRLLightningModule,
    evaluate_policy,
    resolve_policy_evaluation,
)
from nexuml_library.action_adapters.adapters import CategoricalActionAdapter
from nexuml_library.environments.gymnasium import GymnasiumEnvironment
from nexuml_library.reinforcement.algorithms.ppo import PPO
from nexuml_library.reinforcement.policies import discrete_policy_mlp
from nexuml_library.scenarios.reinforcement.gymnasium import cartpole_ppo
from test_rl_evaluation import FakeEnvironment, FakePolicy, FakeRuntime


def offline_scenario():
    return ScenarioSpec(
        name="offline-policy-evaluation",
        pipeline=discrete_policy_mlp(hidden_dims=(8,)),
        data=DataSpec(input_shapes={"observation": [4]}),
        training=TrainingSpec(
            max_epochs=3,
            batch_size=2,
            lr=0.002,
            loss_keys={"imitation_loss": 1.0},
            metric_keys=["accuracy"],
            accelerator="cpu",
            devices=1,
        ),
        interaction=InteractionSpec(environment=GymnasiumEnvironment(env_name="CartPole-v1")),
        policy=PolicySpec(action_adapter=CategoricalActionAdapter()),
        policy_evaluation=RLEvaluationSpec(episodes=2, seed=7, max_steps_per_episode=3),
    )


def forbidden(*args, **kwargs):
    pytest.fail("policy evaluation must not allocate a learner, collector or disabled runtime")


def forbid_training(monkeypatch):
    monkeypatch.setattr("nexuml.training.lightning.NexuSession", forbidden)
    monkeypatch.setattr(PPO, "build", forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    monkeypatch.setattr("nexuml.training.reinforcement.build_collector", forbidden)
    monkeypatch.setattr("nexuml.reinforcement.collector.build_collector", forbidden)
    monkeypatch.setattr("nexuml.training.reinforcement.compile", forbidden)


@pytest.fixture
def fake(monkeypatch):
    FakeEnvironment.instances.clear()
    FakeEnvironment.contexts.clear()
    monkeypatch.setattr(FakeEnvironment, "factory", FakeRuntime)
    scenario = offline_scenario()
    scenario.interaction.environment = FakeEnvironment()
    return scenario, FakePolicy()


@pytest.mark.parametrize("seed", [None, 0, 7])
def test_lowered_and_yaml_round_trips_keep_offline_training_and_dataset_evaluation(seed):
    scenario = offline_scenario()
    scenario.policy_evaluation = RLEvaluationSpec(
        episodes=2,
        seed=seed,
        max_steps_per_episode=3,
        every_n_frames=16,
        at_start=True,
        video=RLVideoSpec(enabled=True, format="both", fps=15, max_frames=2, max_size=8),
    )
    original = lower_model(scenario)
    lowered = ScenarioSpec.model_validate(restore_model_data(original, ScenarioSpec))
    yaml = ResolvedConfig.from_scenario(scenario).to_yaml()
    restored = ResolvedConfig.from_yaml(yaml).to_scenario()
    for result in (lowered, restored):
        assert type(result.training) is TrainingSpec
        assert result.training.model_dump() == scenario.training.model_dump()
        assert isinstance(result.policy_evaluation, RLEvaluationSpec)
        assert result.policy_evaluation == scenario.policy_evaluation
        assert lower_model(result) == original
        assert resolve_policy_evaluation(result)[2] == seed
    assert original["evaluation"] == lower_model(scenario.evaluation)
    assert "algorithm" not in original["training"] and "mode" not in original["training"]


@pytest.mark.parametrize("missing", ["interaction", "policy"])
def test_explicit_block_requires_evaluation_inputs_during_config_validation(missing):
    scenario = offline_scenario()
    values = dict(scenario)
    values[missing] = None
    with pytest.raises(ValidationError, match=rf"policy_evaluation.*scenario\.{missing}"):
        ScenarioSpec.model_validate(values)


@pytest.mark.parametrize(
    "options",
    [
        {"episodes": -1},
        {"every_n_frames": 0},
        {"video": {"fps": float("nan")}},
        {"episodes": 0, "video": {"enabled": True}},
    ],
)
def test_top_level_block_reuses_episode_media_and_schedule_validation(options):
    values = dict(offline_scenario())
    values["policy_evaluation"] = options
    with pytest.raises(ValidationError, match="policy_evaluation"):
        ScenarioSpec.model_validate(values)


@pytest.mark.parametrize(
    "mode,explicit,episode_seed,training_seed,expected",
    [
        ("offline", True, None, None, None),
        ("offline", True, 0, None, 0),
        ("rl", True, None, 91, None),
        ("rl", True, 0, 91, 0),
        ("rl", True, 7, 91, 7),
        ("rl", False, None, 91, 91),
        ("rl", False, None, None, None),
        ("rl", False, 0, 91, 0),
        ("rl", False, 7, 91, 7),
    ],
)
def test_resolution_precedence_and_seed_fallback_are_training_mode_independent(
    mode, explicit, episode_seed, training_seed, expected
):
    scenario = offline_scenario() if mode == "offline" else cartpole_ppo()
    options = RLEvaluationSpec(episodes=2, seed=episode_seed, max_steps_per_episode=3)
    if mode == "rl":
        scenario.training.seed = training_seed
        scenario.training.evaluation = (
            RLEvaluationSpec(episodes=9, seed=83, max_steps_per_episode=1) if explicit else options
        )
    scenario.policy_evaluation = options if explicit else None
    # Exercise old YAML that does not contain the new field at all.
    if not explicit:
        values = lower_model(scenario)
        del values["policy_evaluation"]
        scenario = ResolvedConfig.model_validate(restore_model_data(values, ResolvedConfig))
    before = lower_model(scenario)
    training = scenario.training
    definition, resolved, seed = resolve_policy_evaluation(scenario)
    assert scenario.interaction is not None
    assert definition is scenario.interaction.environment
    assert resolved == options and seed == expected
    assert scenario.training is training and lower_model(scenario) == before


def test_evaluation_environment_is_preferred_without_allocating_it(fake, monkeypatch):
    scenario, _ = fake
    alternate = FakeEnvironment()
    scenario.interaction.evaluation_environment = alternate
    monkeypatch.setattr(FakeEnvironment, "describe", forbidden)
    monkeypatch.setattr(FakeEnvironment, "build", forbidden)
    assert resolve_policy_evaluation(scenario)[0] is alternate
    assert not FakeEnvironment.instances


@pytest.mark.parametrize("missing", ["policy_evaluation", "interaction", "policy"])
def test_missing_inputs_fail_actionably_before_runtime_or_training_allocation(
    fake, monkeypatch, missing
):
    scenario, policy = fake
    # Ordinary offline scenarios need no evaluation block until closed-loop invocation.
    scenario.policy_evaluation = None
    scenario = ScenarioSpec.model_validate(dict(scenario))
    assert ScenarioSpec(name="ordinary").policy_evaluation is None
    if missing != "policy_evaluation":
        scenario.policy_evaluation = RLEvaluationSpec(episodes=1)
        setattr(scenario, missing, None)
    forbid_training(monkeypatch)
    monkeypatch.setattr(FakeEnvironment, "describe", forbidden)
    monkeypatch.setattr(FakeEnvironment, "build", forbidden)
    monkeypatch.setattr(policy, "forward", forbidden)
    with pytest.raises(ValueError, match=missing):
        resolve_policy_evaluation(scenario)
    with pytest.raises(ValueError, match=missing):
        evaluate_policy(policy, scenario)
    assert not FakeEnvironment.instances


@pytest.mark.parametrize("mode", ["offline", "rl"])
def test_explicit_zero_episodes_overrides_legacy_and_allocates_nothing(fake, monkeypatch, mode):
    scenario, policy = fake
    if mode == "rl":
        scenario.training = cartpole_ppo().training
        scenario.training.evaluation = RLEvaluationSpec(episodes=9)
    scenario.policy_evaluation = RLEvaluationSpec(episodes=0)
    before = lower_model(scenario)
    forbid_training(monkeypatch)
    monkeypatch.setattr(FakeEnvironment, "describe", forbidden)
    monkeypatch.setattr(FakeEnvironment, "build", forbidden)
    monkeypatch.setattr(policy, "forward", forbidden)
    assert evaluate_policy(policy, scenario) == {}
    assert not FakeEnvironment.instances and lower_model(scenario) == before


@pytest.mark.requires_optional("gymnasium")
@pytest.mark.parametrize("seed", [None, 0])
def test_manually_assembled_offline_policy_runs_real_cpu_gym_without_a_learner(monkeypatch, seed):
    scenario = offline_scenario()
    scenario.policy_evaluation.seed = seed
    contract = scenario.interaction.environment.describe()
    policy = CompiledPolicy(
        compile(scenario), scenario.policy.action_adapter.build(contract), contract
    )
    before = lower_model(scenario)
    training = scenario.training
    allocations = []
    build = GymnasiumEnvironment.build

    def observe_build(definition, context):
        runtime = build(definition, context)
        allocations.append((runtime, context))
        return runtime

    forbid_training(monkeypatch)
    monkeypatch.setattr(GymnasiumEnvironment, "build", observe_build)
    metrics = evaluate_policy(policy, scenario)
    legacy_keys = {
        "eval/rl/return_mean",
        "eval/rl/return_std",
        "eval/rl/length_mean",
        "eval/rl/terminated",
        "eval/rl/truncated",
        "eval/rl/episodes",
        "eval/rl/native_truncated",
        "eval/rl/evaluation_horizon_limits",
    }
    assert set(metrics) == legacy_keys | {
        key.replace("eval/rl/", "eval/policy/") for key in legacy_keys
    }
    assert all(
        metrics[key] == metrics[key.replace("eval/rl/", "eval/policy/")] for key in legacy_keys
    )
    assert metrics["eval/rl/length_mean"] == metrics["eval/rl/return_mean"] == 3
    assert metrics["eval/rl/terminated"] + metrics["eval/rl/truncated"] == 2
    assert metrics["eval/rl/episodes"] == 2
    assert all(np.isfinite(value) for value in metrics.values())
    assert len(allocations) == 1
    runtime, context = allocations[0]
    assert context.purpose == "eval" and context.device == "cpu" and context.num_envs == 1
    assert context.seed == seed and context.render_mode is None and not runtime.envs
    assert all(parameter.device.type == "cpu" for parameter in policy.parameters())
    assert scenario.training is training and lower_model(scenario) == before


@pytest.mark.requires_optional("gymnasium")
def test_complete_offline_package_round_trip_preserves_original_training(tmp_path, monkeypatch):
    scenario = offline_scenario()
    contract = scenario.interaction.environment.describe()
    policy = CompiledPolicy(
        compile(scenario), scenario.policy.action_adapter.build(contract), contract
    )
    original = lower_model(scenario)
    observation = TensorDict({"observation": torch.zeros(4)}, batch_size=[])
    expected = policy(observation)["action"]
    package = export_package(policy, tmp_path / "offline")
    forbid_training(monkeypatch)
    assert lower_model(ResolvedConfig.load(package / "resolved_config.yaml")) == original
    for load in (load_package, load_inference_package):
        loaded, config, metadata = load(package)
        assert metadata["artifact_kind"] == "policy"
        # Inference archives own isolated classes; check the complete-policy interface.
        assert hasattr(loaded, "action_adapter") and hasattr(loaded, "contract")
        loaded = cast(CompiledPolicy, loaded)
        assert "algorithm" not in metadata
        assert type(config.training) is TrainingSpec and lower_model(config) == original
        assert lower_model(loaded.pipeline.resolved_config) == original
        assert torch.equal(loaded(observation)["action"], expected)
        assert evaluate_policy(loaded, config.to_scenario())["eval/rl/length_mean"] == 3
        assert lower_model(config) == original
    assert scenario.training is policy.pipeline.resolved_config.training
    assert lower_model(scenario) == original


@pytest.mark.requires_optional("PIL")
@pytest.mark.parametrize("seed", [None, 0, 7])
def test_offline_media_is_first_episode_bounded_and_preserves_config_and_provenance(fake, seed):
    scenario, policy = fake
    scenario.policy_evaluation.seed = seed
    scenario.policy_evaluation.video = RLVideoSpec(
        enabled=True, format="both", max_frames=2, max_size=8, fps=15
    )
    before = lower_model(scenario)
    provenance = {"source_package": "offline.package", "source_training_frames": None}
    original_provenance = provenance.copy()
    recordings = []
    metrics = evaluate_policy(
        policy,
        scenario,
        frame_step=32,
        provenance=provenance,
        media_sink=lambda frames, meta: recordings.append((frames, meta)),
    )
    assert len(recordings) == 1
    frames, metadata = recordings[0]
    seeds = [None, None] if seed is None else [seed, seed + 1]
    runtime = FakeEnvironment.instances[0]
    assert runtime.resets == seeds and runtime.closed and runtime.renders == 2
    assert len(frames) == 2
    assert all(frame.shape == (4, 8, 3) and frame.dtype == np.uint8 for frame in frames)
    assert frames[0].max() == 0 and frames[1].min() == 50
    assert metadata["seeds"] == seeds and metadata["frame_step"] == 32
    assert metadata["captured_frames"] == 2 and metadata["capture_capped"]
    assert metadata["source_size"] == [17, 9] and metadata["fps"] == 15
    assert metadata["evaluation"] == scenario.policy_evaluation.model_dump(mode="json")
    assert metadata["environment"] == lower_component(scenario.interaction.environment)
    assert metadata["metrics"] == metrics and metrics["eval/rl/length_mean"] == 3
    assert metadata["provenance"] == provenance == original_provenance
    assert (
        metadata["config_hash"]
        == hashlib.sha256(ResolvedConfig.from_scenario(scenario).to_yaml().encode()).hexdigest()[
            :16
        ]
    )
    assert (
        metadata["contract_hash"]
        == hashlib.sha256(
            json.dumps(policy.contract.model_dump(mode="json"), sort_keys=True).encode()
        ).hexdigest()
    )
    assert lower_model(scenario) == before
    context = FakeEnvironment.contexts[0]
    assert context.purpose == "eval" and context.seed == seed and context.num_envs == 1
    assert context.render_mode == "rgb_array"


@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("seed", [None, 0])
def test_offline_evaluation_preserves_rng_weights_gradients_and_mixed_modes(
    fake, deterministic, seed
):
    scenario, policy = fake
    scenario.policy_evaluation.seed = seed
    scenario.policy_evaluation.deterministic = deterministic
    policy.net[1].eval()
    for parameter in policy.parameters():
        parameter.grad = torch.ones_like(parameter)
    weights = copy.deepcopy(policy.state_dict())
    modes = [module.training for module in policy.modules()]
    python_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
    )
    assert isinstance(numpy_state, tuple)
    first = evaluate_policy(policy, scenario)
    actions = policy.actions.copy()
    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert isinstance(after_numpy, tuple)
    assert after_numpy[0] == numpy_state[0]
    assert np.array_equal(after_numpy[1], numpy_state[1])
    assert after_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    policy.actions.clear()
    assert evaluate_policy(policy, scenario) == first and policy.actions == actions
    assert [module.training for module in policy.modules()] == modes
    assert all(torch.equal(value, policy.state_dict()[key]) for key, value in weights.items())
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in policy.parameters())
    assert all(runtime.closed for runtime in FakeEnvironment.instances)


@pytest.mark.parametrize("failure", [RuntimeError("step failure"), KeyboardInterrupt()])
def test_offline_failure_closes_runtime_without_masking_error_or_leaking_state(
    fake, monkeypatch, failure
):
    scenario, policy = fake
    runtime = FakeRuntime()
    runtime.failure, runtime.close_failure = failure, True
    monkeypatch.setattr(FakeEnvironment, "factory", lambda: runtime)
    policy.net[1].eval()
    modes = [module.training for module in policy.modules()]
    python_state, torch_state = random.getstate(), torch.get_rng_state().clone()
    with pytest.raises(type(failure)) as caught:
        evaluate_policy(policy, scenario)
    assert caught.value is failure and "close failure" in " ".join(caught.value.__notes__)
    assert runtime.closed and [module.training for module in policy.modules()] == modes
    assert random.getstate() == python_state and torch.equal(torch.get_rng_state(), torch_state)


@pytest.mark.requires_optional("PIL")
def test_rl_current_evaluation_cache_and_hooks_follow_authoritative_block(fake, monkeypatch):
    scenario, policy = fake
    scenario.training = cartpole_ppo().training
    scenario.training.evaluation = RLEvaluationSpec(episodes=0)
    scenario.policy_evaluation = RLEvaluationSpec(
        episodes=2,
        seed=None,
        max_steps_per_episode=3,
        every_n_frames=16,
        at_start=True,
        video=RLVideoSpec(enabled=True, max_frames=2),
    )
    legacy = scenario.training.evaluation
    legacy_before = legacy.model_dump()
    # Only evaluator hooks/cache are under test, not a fake PPO training proof.
    monkeypatch.setattr(PPO, "build", lambda self, context: nn.Identity())
    module = NexuRLLightningModule(scenario, policy=policy)
    monkeypatch.setattr(module, "_trainer", SimpleNamespace(loggers=[]))
    preflights, media = [], []
    monkeypatch.setattr(
        "nexuml.tracking.logger.preflight_evaluation_media",
        lambda loggers, video: preflights.append(video),
    )
    monkeypatch.setattr(
        "nexuml.tracking.logger.log_evaluation_media",
        lambda loggers, frames, metadata, video, **kwargs: media.append((metadata, video)),
    )
    module.on_fit_start()
    assert preflights == [scenario.policy_evaluation.video]
    assert module._next_evaluation_frame == 16
    first = module.evaluate_current_policy()
    module.on_fit_end()
    assert len(FakeEnvironment.instances) == len(media) == 1
    assert FakeEnvironment.instances[0].resets == [None, None]
    scenario.training.seed = 99
    assert module.evaluate_current_policy() == first
    assert len(FakeEnvironment.instances) == 1
    scenario.policy_evaluation.seed = 0
    module.evaluate_current_policy()
    assert len(FakeEnvironment.instances) == len(media) == 2
    assert FakeEnvironment.instances[-1].resets == [0, 1]
    assert media[-1][1] == scenario.policy_evaluation.video
    scenario.policy_evaluation = None
    assert module.evaluate_current_policy() == {}
    assert len(FakeEnvironment.instances) == 2
    assert scenario.training.evaluation is legacy and legacy.model_dump() == legacy_before


@pytest.mark.requires_optional("gymnasium")
@pytest.mark.requires_optional("torchrl")
def test_explicit_block_controls_real_rl_cadence_without_rewriting_legacy(tmp_path, monkeypatch):
    from nexuml.training.lightning import NexuSession
    import nexuml.training.reinforcement as module

    scenario = cartpole_ppo(total_frames=32, frames_per_batch=16)
    scenario.training.algorithm = PPO(update_epochs=1, minibatch_size=8)
    scenario.training.evaluation = RLEvaluationSpec(episodes=0)
    scenario.policy_evaluation = RLEvaluationSpec(
        episodes=1,
        seed=0,
        max_steps_per_episode=2,
        at_start=True,
        every_n_frames=16,
    )
    legacy = scenario.training.evaluation
    calls = []
    original = module.evaluate_policy

    def evaluate(policy, current, **kwargs):
        calls.append(kwargs["frame_step"])
        return original(policy, current, **kwargs)

    monkeypatch.setattr(module, "evaluate_policy", evaluate)
    result = NexuSession(
        scenario,
        log_dir=tmp_path,
        enable_loggers=False,
        enable_progress_bar=False,
    ).run()
    assert calls == [0, 16, 32]
    assert result.interaction_results["eval/rl/length_mean"] == 2
    assert scenario.training.evaluation is legacy and legacy.episodes == 0
