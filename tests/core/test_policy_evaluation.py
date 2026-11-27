"""Generic environment evaluation without an RL learner or fabricated rewards."""

from dataclasses import replace
import subprocess
import sys

import pytest
import torch

from nexuml.evaluation.policy import evaluate_policy
from test_independent_policy_evaluation import offline_scenario
from test_rl_evaluation import FakeEnvironment, FakePolicy, FakeRuntime


@pytest.fixture
def outcome_task(monkeypatch):
    FakeEnvironment.instances.clear()
    FakeEnvironment.contexts.clear()
    scenario = offline_scenario()
    scenario.interaction.environment = FakeEnvironment()

    def factory():
        runtime = FakeRuntime()
        original = runtime.step

        def step(action):
            transition = original(action)
            return replace(
                transition,
                reward=None,
                info={"task_success": torch.tensor(runtime.steps == 2)},
            )

        monkeypatch.setattr(runtime, "step", step)
        return runtime

    monkeypatch.setattr(FakeEnvironment, "factory", factory)
    return scenario, FakePolicy()


def test_rewardless_offline_task_reports_outcomes_without_inventing_returns(outcome_task):
    scenario, policy = outcome_task
    metrics = evaluate_policy(policy, scenario)
    assert metrics["eval/policy/task_success_rate"] == 1
    assert metrics["eval/policy/length_mean"] == 3
    assert metrics["eval/policy/terminated"] == 2
    assert not any("return_" in name for name in metrics)
    assert FakeEnvironment.instances[-1].closed


def test_task_specific_errors_and_violation_events_are_not_rewards(outcome_task, monkeypatch):
    scenario, policy = outcome_task
    factory = FakeEnvironment.factory

    def with_measurements():
        runtime = factory()
        original = runtime.step

        def step(action):
            transition = original(action)
            return replace(
                transition,
                info={
                    **transition.info,
                    "task_errors": {"distance_m": torch.tensor(0.1 * runtime.steps)},
                    "constraint_violations": torch.tensor(int(runtime.steps == 1)),
                    "action_violations": torch.tensor(0),
                },
            )

        monkeypatch.setattr(runtime, "step", step)
        return runtime

    monkeypatch.setattr(FakeEnvironment, "factory", with_measurements)
    metrics = evaluate_policy(policy, scenario)
    assert metrics["eval/policy/final_distance_m_mean"] == pytest.approx(0.3)
    assert metrics["eval/policy/constraint_violations"] == 2
    assert metrics["eval/policy/action_violations"] == 0
    assert not any("return_" in name for name in metrics)


@pytest.mark.parametrize(
    "change", ["rewards", "flags", "outcomes", "errors", "counter", "collision"]
)
def test_invalid_or_changing_measurements_fail_closed(outcome_task, monkeypatch, change):
    scenario, policy = outcome_task
    factory = FakeEnvironment.factory

    def invalid():
        runtime = factory()
        original = runtime.step

        def step(action):
            transition = original(action)
            if change == "rewards" and runtime.steps == 2:
                return replace(transition, reward=torch.tensor(0.0))
            if change == "flags":
                return replace(transition, terminated=torch.tensor(0))
            info = dict(transition.info)
            if change == "outcomes":
                info["task_success"] = torch.tensor(1.0)
            if change == "errors":
                info["task_errors"] = {"distance_m": torch.tensor(float("nan"))}
            if change == "counter":
                info["action_violations"] = torch.tensor(0.5)
            if change == "collision":
                info["task_errors"] = {"joint_error_rad": torch.tensor(0.1)}
            return replace(transition, info=info)

        monkeypatch.setattr(runtime, "step", step)
        return runtime

    monkeypatch.setattr(FakeEnvironment, "factory", invalid)
    with pytest.raises(ValueError):
        evaluate_policy(policy, scenario)
    assert FakeEnvironment.instances[-1].closed


@pytest.mark.parametrize("rl", [False, True])
def test_reward_is_still_required_for_rl_or_missing_task_outcomes(outcome_task, monkeypatch, rl):
    from nexuml_library.scenarios.reinforcement.gymnasium import cartpole_ppo

    scenario, policy = outcome_task
    if rl:
        scenario.training = cartpole_ppo().training
    else:
        factory = FakeEnvironment.factory

        def without_outcomes():
            runtime = factory()
            original = runtime.step
            monkeypatch.setattr(runtime, "step", lambda a: replace(original(a), info=None))
            return runtime

        monkeypatch.setattr(FakeEnvironment, "factory", without_outcomes)
    with pytest.raises(ValueError, match="reward|outcome"):
        evaluate_policy(policy, scenario)
    assert FakeEnvironment.instances[-1].closed


def test_legacy_evaluator_import_is_the_same_runner():
    from nexuml.training.reinforcement import evaluate_policy as legacy

    assert legacy is evaluate_policy


def test_evaluation_imports_no_training_framework_or_simulator():
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('nexuml.training') or fullname.split('.')[0] in {
            'lightning', 'torchrl', 'gymnasium', 'isaacsim', 'isaaclab', 'lerobot', 'ray'
        }:
            raise AssertionError('unexpected evaluation dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from nexuml.evaluation.policy import evaluate_policy, resolve_policy_evaluation
assert callable(evaluate_policy) and callable(resolve_policy_evaluation)
assert 'nexuml.training.reinforcement' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
