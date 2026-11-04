"""Portable grouped actors use the existing local compiler, checkpoints and processors."""

import json
import os
import subprocess
import sys
from typing import cast

import pytest
import torch
from tensordict import TensorDict

from nexuml.core.compiler import compile, compile_context_from_interaction
from nexuml.core.config import ResolvedConfig
from nexuml.core.export import export_package, load_inference_package, load_package, load_weights
from nexuml.core.policy import CompiledPolicy
from nexuml.core.serialization import lower_model
from nexuml.core.types import (
    AgentGroupContract,
    GroupPolicySpec,
    InteractionContract,
    InteractionSpec,
    PhysicalSystemContract,
    PhysicalTensorContract,
    PolicySpec,
    ReinforcementLearningSpec,
    ScenarioSpec,
    SensorProcessingContract,
    TensorFieldContract,
)
from nexuml_library.action_adapters.adapters import (
    CategoricalActionAdapter,
    GaussianActionAdapter,
)
from nexuml_library.environments.gymnasium import GymnasiumEnvironment
from nexuml_library.layers.processing.sensor_history import SensorHistoryProcessor
from nexuml_library.reinforcement.algorithms.ppo import PPO
from nexuml_library.reinforcement.policies import continuous_policy_mlp, discrete_policy_mlp


def _scenario(*, shared=True, processing=False, external=False):
    sensor = (
        SensorProcessingContract(
            owner="external" if external else "package",
            raw_shape=(2,),
            raw_dtype="float32",
            raw_layout="vector",
            units="m",
            scale=0.5,
            history_length=2,
        )
        if processing
        else None
    )
    runners = AgentGroupContract(
        agents=("runner_0", "runner_1"),
        observations={
            "observation": TensorFieldContract(shape=(4,), dtype="float32", sensor=sensor)
        },
        actions={
            "action": TensorFieldContract(
                shape=(2,),
                dtype="float32",
                low=-1.0,
                high=1.0,
                rate_hz=20,
                physical=PhysicalTensorContract(
                    units=("N",),
                    frame="body",
                    joints=("left", "right"),
                    control_mode="effort",
                    scale=(2.0, 3.0),
                    offset=(0.0,),
                ),
            )
        },
        action_space="continuous",
    )
    scouts = AgentGroupContract(
        agents=("scout_0",),
        observations={"observation": TensorFieldContract(shape=(3,), dtype="float32")},
        actions={"action": TensorFieldContract(shape=(), dtype="int64", num_values=3)},
        action_space="discrete",
    )
    contract = InteractionContract(
        groups={"runners": runners, "scouts": scouts},
        critic_observations={"world": TensorFieldContract(shape=(7,), dtype="float32")},
        action_space="structured",
        num_envs=2,
        control_period_s=0.05,
        physical=PhysicalSystemContract(
            embodiment_id="synthetic-pair",
            asset_source="authored-test",
            asset_id="portable",
            calibration_id="unit-test",
        ),
    )
    return ScenarioSpec(
        name="grouped-portable-policy",
        interaction=InteractionSpec(
            environment=GymnasiumEnvironment(env_name="custom_group_task:inert", contract=contract)
        ),
        policy=PolicySpec(
            groups={
                "runners": GroupPolicySpec(
                    pipeline=continuous_policy_mlp(action_dim=2, hidden_dims=(8,)),
                    action_adapter=GaussianActionAdapter(),
                    actor_ids=("shared", "shared") if shared else ("left", "right"),
                    processor=SensorHistoryProcessor() if processing else None,
                ),
                "scouts": GroupPolicySpec(
                    pipeline=discrete_policy_mlp(num_actions=3, hidden_dims=(8,)),
                    action_adapter=CategoricalActionAdapter(),
                    actor_ids=("scout",),
                ),
            }
        ),
        training=ReinforcementLearningSpec(algorithm=PPO()),
    )


def _policy(**kwargs):
    scenario = _scenario(**kwargs)
    return CompiledPolicy.from_pipeline(compile(scenario))


def _observation(batch=(2,), *, raw=False, value=1.0, named=False):
    names = ["environment", "time"][: len(batch)] if named else None
    root = TensorDict({}, batch_size=batch, names=names)
    for name, count, width in (("runners", 2, 2 if raw else 4), ("scouts", 1, 3)):
        values = torch.arange(count * width, dtype=torch.float32).reshape(count, width) + value
        root[name] = TensorDict(
            {"observation": values.expand(*batch, count, width).clone()},
            batch_size=(*batch, count),
            names=[*root.names, "agent"] if named else None,
        )
    return root


def _equal(left, right):
    assert left.batch_size == right.batch_size
    assert set(left.keys(True, True)) == set(right.keys(True, True))
    for key in left.keys(True, True):
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def test_contract_roundtrip_preserves_order_semantics_and_local_actor_inputs():
    scenario = _scenario(processing=True)
    restored = ResolvedConfig.from_yaml(ResolvedConfig.from_scenario(scenario).to_yaml())
    assert lower_model(restored) == lower_model(scenario)
    assert restored.interaction is not None
    contract = restored.interaction.environment.describe()
    assert list(contract.groups) == ["runners", "scouts"]
    assert contract.groups["runners"].agents == ("runner_0", "runner_1")
    local = contract.group_contract("runners")
    assert not local.groups and not local.critic_observations
    assert local.physical == contract.physical
    assert local.control_period_s == contract.control_period_s
    assert local.observations == contract.groups["runners"].observations
    assert local.actions == contract.groups["runners"].actions
    assert compile_context_from_interaction(contract).input_sizes == {}
    assert compile_context_from_interaction(local).input_sizes == {"observation": (4,)}
    assert "team_reward" not in contract.model_dump()
    assert "groups" not in local.model_dump()
    assert "critic_observations" not in local.model_dump()
    team = InteractionContract.model_validate({**contract.model_dump(), "team_reward": True})
    assert team.team_reward and team.model_dump()["team_reward"] is True
    assert not team.group_contract("runners").team_reward


@pytest.mark.parametrize("name", ["bad.id", "", "has space", "items", "training", "class", "_x"])
def test_module_identifiers_reject_invalid_group_and_actor_names(name):
    scenario = _scenario()
    contract = scenario.interaction.environment.describe()
    with pytest.raises(ValueError, match="module identifier"):
        InteractionContract(action_space="structured", groups={name: contract.groups["runners"]})
    group = scenario.policy.groups["runners"]
    with pytest.raises(ValueError, match="module identifier"):
        GroupPolicySpec(
            pipeline=group.pipeline, action_adapter=group.action_adapter, actor_ids=(name,)
        )


def test_contract_validation_rejects_ambiguous_identity_root_fields_and_missing_adapter():
    scenario = _scenario()
    contract = scenario.interaction.environment.describe()
    runners = contract.groups["runners"]
    with pytest.raises(ValueError, match="unique"):
        AgentGroupContract.model_validate(
            {**runners.model_dump(), "agents": ("duplicate", "duplicate")}
        )
    with pytest.raises(ValueError, match="globally unique"):
        InteractionContract(action_space="structured", groups={"first": runners, "second": runners})
    with pytest.raises(ValueError, match="empty root"):
        InteractionContract.model_validate(
            {**contract.model_dump(), "observations": runners.observations}
        )
    with pytest.raises(ValueError, match="nonempty"):
        InteractionContract(action_space="continuous")
    with pytest.raises(ValueError, match="action_adapter"):
        PolicySpec()
    with pytest.raises(ValueError, match="each group"):
        PolicySpec(groups=scenario.policy.groups, action_adapter=GaussianActionAdapter())
    with pytest.raises(ValueError, match="Critic-only"):
        InteractionContract.model_validate(
            {
                **contract.model_dump(),
                "critic_observations": {"runners": next(iter(runners.observations.values()))},
            }
        )


def test_compatible_groups_share_the_actual_local_actor_across_group_views():
    scenario = _scenario()
    contract = scenario.interaction.environment.describe()
    contract.groups["peer"] = contract.groups["runners"].model_copy(
        update={"agents": ("peer_0", "peer_1")}, deep=True
    )
    scenario.policy.groups["peer"] = scenario.policy.groups["runners"].model_copy(deep=True)
    policy = CompiledPolicy.from_pipeline(compile(scenario))
    observation = _observation()
    observation["peer"] = observation["runners"].clone()
    assert list(policy.actors) == ["shared", "scout"]
    assert policy.actor_ids["peer"] == policy.actor_ids["runners"]
    output = policy(observation)
    _equal(output["peer"], output["runners"])
    with torch.no_grad():
        for parameter in policy.actors["shared"].parameters():
            parameter.add_(0.25)
    changed = policy(observation)
    _equal(changed["peer"], changed["runners"])
    assert not torch.equal(changed["peer", "action"], output["peer", "action"])


@pytest.mark.parametrize(
    "failure", ["order", "count", "observation", "action", "pipeline", "adapter", "processor"]
)
def test_incompatible_group_bindings_and_sharing_reject(failure):
    scenario = _scenario()
    groups = scenario.policy.groups
    if failure == "order":
        scenario.policy.groups = dict(reversed(list(groups.items())))
    elif failure == "count":
        groups["runners"].actor_ids = ("shared",)
    else:
        contract = scenario.interaction.environment.describe()
        peer = contract.groups["runners"].model_copy(
            update={"agents": ("peer_0", "peer_1")}, deep=True
        )
        other = groups["runners"].model_copy(deep=True)
        if failure == "observation":
            peer.observations["observation"] = peer.observations["observation"].model_copy(
                update={"physical": PhysicalTensorContract(units=("m",), frame="body")}
            )
        elif failure == "action":
            field = peer.actions["action"]
            peer.actions["action"] = field.model_copy(
                update={"physical": field.physical.model_copy(update={"scale": (7.0, 3.0)})}
            )
        elif failure == "pipeline":
            other.pipeline = continuous_policy_mlp(action_dim=2, hidden_dims=(9,))
        elif failure == "adapter":
            other.action_adapter = GaussianActionAdapter(min_scale=0.1)
        elif failure == "processor":
            other.processor = SensorHistoryProcessor()
        # Pydantic frozen contracts still contain mutable dictionaries; author a separate copy.
        contract.groups["peer"] = peer
        groups["peer"] = other
    with pytest.raises(ValueError, match="group order|one actor_id|identical local semantics"):
        CompiledPolicy.from_pipeline(compile(scenario))


@pytest.mark.parametrize("batch", [(1,), (2,), (2, 3)])
def test_heterogeneous_nested_batches_local_outputs_and_shared_parameter_identity(batch):
    policy = _policy()
    observation = _observation(batch, named=True)
    original = observation.clone()
    parameters = policy.forward_parameters(observation)
    actions = policy(observation)
    assert actions.batch_size == parameters.batch_size == torch.Size(batch)
    assert actions.names == observation.names
    assert actions["runners"].batch_size == torch.Size((*batch, 2))
    assert actions["runners"].names == observation["runners"].names
    assert actions["runners", "action"].shape == (*batch, 2, 2)
    assert actions["scouts", "action"].shape == (*batch, 1)
    assert actions["scouts", "action"].dtype == torch.int64
    assert parameters["runners", "action_loc"].shape == (*batch, 2, 2)
    assert parameters["scouts", "action_logits"].shape == (*batch, 1, 3)
    assert policy.actor_ids == {"runners": ("shared", "shared"), "scouts": ("scout",)}
    assert list(policy.actors) == ["shared", "scout"]
    assert policy.actors is policy.pipeline.group_actors
    assert all(isinstance(actor, CompiledPolicy) for actor in policy.actors.values())
    assert not any(key.startswith("actors.") for key in policy.state_dict())
    assert any(
        key.startswith("group_actors.shared.pipeline.") for key in policy.pipeline.state_dict()
    )
    actor = cast(CompiledPolicy, policy.actors["shared"])
    assert actor.pipeline.resolved_config.policy is not None
    assert actor.pipeline.resolved_config.policy.groups == {}
    assert actor.pipeline.resolved_config.name == policy.pipeline.resolved_config.name
    assert actor.contract == policy.contract.group_contract("runners")
    local = observation["runners"][..., 0].clone()
    local.names = None
    torch.testing.assert_close(actions["runners", "action"][..., 0, :], actor(local)["action"])
    _equal(observation, original)
    again = CompiledPolicy.from_pipeline(policy.pipeline)
    assert again.actors["shared"] is actor
    # Root critic features and other agents must never reach the local actor graph.
    changed = observation.clone()
    changed["world"] = torch.full((*batch, 7), float("nan"))
    changed["scouts", "observation"] += 100
    changed["runners", "observation"][..., 1, :] += 100
    torch.testing.assert_close(
        policy(changed)["runners", "action"][..., 0, :], actions["runners", "action"][..., 0, :]
    )


def test_independent_actor_parameters_and_outputs_are_not_packed_or_aliased():
    policy = _policy(shared=False)
    left = policy.actors["left"]
    right = policy.actors["right"]
    assert left is not right
    assert not {id(p) for p in left.parameters()} & {id(p) for p in right.parameters()}
    observation = _observation((2, 3))
    before = policy.forward_parameters(observation)
    with torch.no_grad():
        for parameter in left.parameters():
            parameter.add_(0.5)
    after = policy.forward_parameters(observation)
    assert not torch.equal(
        before["runners", "action_loc"][..., 0, :], after["runners", "action_loc"][..., 0, :]
    )
    torch.testing.assert_close(
        before["runners", "action_loc"][..., 1, :],
        after["runners", "action_loc"][..., 1, :],
        rtol=0,
        atol=0,
    )
    loss = after["runners", "action_loc"].sum()
    loss.backward()
    assert all(parameter.grad is not None for parameter in left.parameters())
    assert all(parameter.grad is not None for parameter in right.parameters())


def test_group_axes_and_local_observation_validation_fail_closed():
    policy = _policy()
    wrong = _observation()
    wrong["runners"] = wrong["runners"][..., :1]
    with pytest.raises(ValueError, match="agent axis"):
        policy(wrong)
    with pytest.raises(ValueError, match="environment batch"):
        policy(_observation(())[()])
    wrong = _observation()
    wrong["runners", "observation"][0, 1, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        policy(wrong)
    with pytest.raises(ValueError, match="explicit processor"):
        policy.initialize_state((2,))


@pytest.mark.parametrize("external", [False, True])
def test_group_processing_uses_local_history_and_selected_environment_resets(external):
    policy = _policy(processing=True, external=external)
    state = policy.initialize_state((2,))
    assert state["runners"].batch_size == torch.Size((2, 2))
    if external:
        raw = _observation()
        actions, advanced = policy.step(raw, state)
        _equal(actions, policy(raw))
        _equal(advanced, state)
        return
    original = state.clone()
    for value in (1.0, 2.0):
        raw = _observation(raw=True, value=value)
        expected = []
        for index in range(2):
            actor = cast(CompiledPolicy, policy.actors["shared"])
            local, _ = actor.step(raw["runners"][..., index], state["runners"][..., index])
            expected.append(cast(torch.Tensor, local["action"]))
        actions, state = policy.step(raw, state)
        torch.testing.assert_close(
            actions["runners", "action"], torch.stack(expected, dim=1), rtol=0, atol=0
        )
    _equal(policy.initialize_state((2,)), original)
    assert not torch.equal(
        state["runners", "observation", "frames"][:, 0],
        state["runners", "observation", "frames"][:, 1],
    )
    continuing = state[1].clone()
    reset = policy.reset_state(state, torch.tensor([True, False]))
    _equal(reset[0], original[0])
    _equal(reset[1], continuing)
    assert state["runners", "observation", "initialized"].all()
    with pytest.raises(ValueError, match="root state batch"):
        policy.reset_state(state, torch.zeros(2, 2, dtype=torch.bool))


@pytest.mark.parametrize("shared", [True, False])
def test_current_exports_preserve_actor_weights_sharing_and_processing(tmp_path, shared):
    torch.manual_seed(8)
    policy = _policy(processing=True, shared=shared)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.add_(0.25)
    package = export_package(policy, tmp_path / "grouped")
    saved = torch.load(package / "state_dict.pt", weights_only=True)
    assert saved and all(key.startswith("group_actors.") for key in saved)
    assert any("action_adapter.low" in key for key in saved)
    metadata = json.loads((package / "metadata.json").read_text())
    assert metadata["schema_version"] == 2
    assert not metadata["training_state_available"]
    assert metadata["x_keys"] == ["runners", "scouts"]
    assert not (package / "lightning.ckpt").exists()
    loaded = [
        cast(CompiledPolicy, load_package(package)[0]),
        cast(CompiledPolicy, load_inference_package(package)[0]),
    ]
    fresh = _policy(processing=True, shared=shared)
    report = load_weights(
        fresh.pipeline, package / "state_dict.pt", allow_missing=False, allow_shape_mismatch=False
    )
    assert set(report.matched) == set(saved) and not report.missing
    loaded.append(fresh)
    # Also exercise the current directory reconstruction without its packaged topology.
    (package / "pipeline.package").rename(package / "held.package")
    loaded.append(cast(CompiledPolicy, load_package(package)[0]))
    (package / "held.package").rename(package / "pipeline.package")
    for restored in loaded:
        assert restored.actor_ids == policy.actor_ids
        assert list(restored.actors) == list(policy.actors)
        for key, value in policy.pipeline.state_dict().items():
            torch.testing.assert_close(restored.pipeline.state_dict()[key], value, rtol=0, atol=0)
        states = [candidate.initialize_state((2,)) for candidate in (policy, restored)]
        for value in (1.0, 3.0, 5.0):
            outputs = []
            for index, candidate in enumerate((policy, restored)):
                output, states[index] = candidate.step(
                    _observation(raw=True, value=value), states[index]
                )
                outputs.append(output)
            _equal(outputs[0], outputs[1])
            _equal(states[0], states[1])
            states = [
                candidate.reset_state(state, torch.tensor([True, False]))
                for candidate, state in zip((policy, restored), states, strict=True)
            ]


def test_static_grouped_yaml_restoration_imports_no_native_or_torchrl_modules(tmp_path):
    config = tmp_path / "grouped.yaml"
    ResolvedConfig.from_scenario(_scenario(processing=True)).save(config)
    script = r"""
import importlib.abc
import sys
from pathlib import Path
blocked = {'torchrl', 'isaaclab', 'isaacsim', 'viser', 'custom_group_task'}
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ImportError('blocked: ' + fullname)
sys.meta_path.insert(0, Deny())
from nexuml.core.config import ResolvedConfig
restored = ResolvedConfig.load(Path(sys.argv[1]))
contract = restored.interaction.environment.describe()
assert list(contract.groups) == list(restored.policy.groups) == ['runners', 'scouts']
assert restored.policy.groups['runners'].actor_ids == ('shared', 'shared')
assert contract.group_contract('runners').observations['observation'].sensor.owner == 'package'
assert not any(name.split('.')[0] in blocked for name in sys.modules)
print('static grouped restoration')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(config)],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "static grouped restoration" in result.stdout


def test_clean_grouped_inference_needs_no_workspace_native_or_learner_imports(tmp_path):
    policy = _policy(processing=True)
    package = export_package(policy, tmp_path / "portable")
    state = policy.initialize_state((2,))
    expected = []
    for value in (1.0, 2.0, 3.0):
        raw = _observation(raw=True, value=value)
        actions, state = policy.step(raw, state)
        expected.append((raw, actions, state.clone()))
        state = policy.reset_state(state, torch.tensor([True, False]))
    torch.save(expected, tmp_path / "expected.pt")
    script = r"""
import importlib.abc
import sys
import torch
from torch.package import PackageImporter
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'isaaclab', 'isaacsim', 'torchrl', 'viser', 'custom_group_task',
            'nexuml', 'nexuml_library',
        }:
            raise ImportError('blocked: ' + fullname)
sys.meta_path.insert(0, Deny())
sys.path[:] = [p for p in sys.path if '/src' not in p]
payload = PackageImporter(sys.argv[1]).load_pickle('nexuml_export', 'artifact.pkl')
policy = payload['policy']
assert list(policy.actors) == ['shared', 'scout']
assert policy.actor_ids['runners'] == ('shared', 'shared')
assert not payload['training_state']
state = policy.initialize_state((2,))
for raw, actions, reference in torch.load(sys.argv[2], weights_only=False):
    actual, state = policy.step(raw, state)
    for key in actions.keys(True, True):
        torch.testing.assert_close(actual[key], actions[key], rtol=0, atol=0)
    for key in reference.keys(True, True):
        torch.testing.assert_close(state[key], reference[key], rtol=0, atol=0)
    state = policy.reset_state(state, torch.tensor([True, False]))
blocked = {'torchrl', 'isaaclab', 'isaacsim', 'viser', 'custom_group_task'}
assert not any(name.split('.')[0] in blocked for name in sys.modules)
print('clean grouped inference/reset parity')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(package / "pipeline.package"),
            str(tmp_path / "expected.pt"),
        ],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "clean grouped inference/reset parity" in result.stdout
