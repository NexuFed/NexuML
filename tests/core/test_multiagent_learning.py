"""Native MARL numerical, dependency, optimizer and current-continuation proofs."""

import copy
from typing import Any

import pytest
import torch
from pydantic import ValidationError
from tensordict import TensorDict

from nexuml.core.compiler import compile
from nexuml.core.components import RLAlgorithmBuildContext
from nexuml.core.policy import CompiledPolicy
from nexuml_library.reinforcement.algorithms.multiagent import IPPO, MAPPO, agent_view
from nexuml_library.scenarios.reinforcement.isaaclab_marl import (
    isaac_pendulum_ippo,
    isaac_pendulum_mappo,
    isaac_handover_ippo,
    isaac_handover_mappo,
)


class ManualOptimization:
    def manual_backward(self, loss):
        loss.backward()

    def clip_gradients(self, optimizer, *, gradient_clip_val, gradient_clip_algorithm):
        assert gradient_clip_algorithm == "norm"
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]], gradient_clip_val
        )


def build(factory):
    scenario = factory(total_frames=32, frames_per_batch=16, num_envs=2)
    scenario.training.algorithm = scenario.training.algorithm.model_copy(
        update={"update_epochs": 1, "minibatch_size": 16}
    )
    policy = CompiledPolicy.from_pipeline(compile(scenario))
    algorithm = scenario.training.algorithm.build(RLAlgorithmBuildContext(policy, policy.contract))
    return policy, algorithm


def rollout(policy, algorithm, *, seed=7, n=2, t=5):
    torch.manual_seed(seed)

    def observations(batch):
        root = TensorDict({}, batch_size=batch)
        for group, declared in policy.contract.groups.items():
            root[group] = TensorDict(
                {
                    key: torch.randn(*batch, len(declared.agents), *leaf.shape)
                    for key, leaf in declared.observations.items()
                },
                batch_size=(*batch, len(declared.agents)),
            )
            root[group, "valid"] = torch.ones((*batch, len(declared.agents), 1), dtype=torch.bool)
        return root

    current = observations((n, t))
    with torch.no_grad():
        algorithm.rollout_policy(current)
    following = observations((n, t))
    following["done"] = torch.zeros(n, t, 1, dtype=torch.bool)
    following["terminated"] = torch.zeros(n, t, 1, dtype=torch.bool)
    following["truncated"] = torch.zeros(n, t, 1, dtype=torch.bool)
    for group, declared in policy.contract.groups.items():
        shape = (n, t, len(declared.agents), 1)
        following[group, "reward"] = torch.randn(shape)
        for key in ("done", "terminated", "truncated"):
            following[group, key] = torch.zeros(shape, dtype=torch.bool)
    current["next"] = following
    current.names = ["environment", "time"]
    return current


@pytest.mark.parametrize(
    "factory",
    [isaac_pendulum_ippo, isaac_pendulum_mappo, isaac_handover_ippo, isaac_handover_mappo],
)
def test_native_loss_estimator_identity_axes_and_real_owned_updates(factory):
    from torchrl.objectives.multiagent import IPPOLoss, MAPPOLoss
    from torchrl.objectives.value import MultiAgentGAE

    policy, algorithm = build(factory)
    data = rollout(policy, algorithm)
    assert data.batch_size == (2, 5) and data.numel() == 10
    prepared = algorithm.prepare(data)
    optimizers = algorithm.configure_optimizers()
    parameter_ids = [
        id(p)
        for optimizer in optimizers
        for group in optimizer.param_groups
        for p in group["params"]
    ]
    assert len(parameter_ids) == len(set(parameter_ids))
    before_actor = [p.detach().clone() for p in policy.parameters()]
    before_critic = [p.detach().clone() for p in algorithm.critics.parameters()]
    for loss, (group, index, _) in zip(algorithm.losses, algorithm.slots, strict=True):
        assert isinstance(loss, MAPPOLoss if algorithm.centralized else IPPOLoss)
        assert type(loss.value_estimator) is MultiAgentGAE
        assert (
            not loss.normalize_advantage
            and not loss.value_estimator.average_gae
            and loss.value_norm is None
        )
        assert loss.loss_mask_key == (group, "valid")
        group_data = prepared[group]
        for key in ("state_value", "advantage", "value_target"):
            assert group_data[key].shape == (2, 5, len(policy.contract.groups[group].agents), 1)
        assert data[group, "action_log_prob"].shape == (
            2,
            5,
            len(policy.contract.groups[group].agents),
        )
        flat = agent_view(prepared, group, index).reshape(-1)
        assert flat[group].batch_size == (10, 1)
        assert torch.isfinite(
            sum(loss(flat)[k] for k in ("loss_objective", "loss_critic", "loss_entropy"))
        )
    metrics = algorithm.update(data, lightning_module=ManualOptimization(), optimizers=optimizers)
    assert metrics and all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before_actor, policy.parameters(), strict=True)
    )
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before_critic, algorithm.critics.parameters(), strict=True)
    )
    assert int(algorithm.updates) == 1
    assert all(
        int(state["step"]) == 1 for optimizer in optimizers for state in optimizer.state.values()
    )


@pytest.mark.parametrize("factory", [isaac_pendulum_ippo, isaac_pendulum_mappo])
def test_local_actor_and_ippo_value_independence_and_mappo_cross_group_gradients(factory):
    policy, algorithm = build(factory)
    data = rollout(policy, algorithm)
    changed = data.clone()
    changed["pendulum", "observation"].add_(10)
    torch.testing.assert_close(
        policy(data)["cart", "action"], policy(changed)["cart", "action"], rtol=0, atol=0
    )
    before, after = algorithm.prepare(data), algorithm.prepare(changed)
    same = torch.equal(before["cart", "state_value"], after["cart", "state_value"])
    assert same is (not algorithm.centralized)
    if algorithm.centralized:
        inputs = data.detach().clone()
        inputs["pendulum", "observation"].requires_grad_()
        algorithm._critic_inputs(inputs)
        critic = algorithm.losses[0].critic_network
        critic(agent_view(inputs, "cart", 0))["cart", "state_value"].sum().backward()
        assert inputs["pendulum", "observation"].grad.abs().sum() > 0
        native = algorithm.critics["central"].module
        assert all(p.grad is not None for p in native.encoders.parameters())
        assert all(p.grad is not None for p in native.trunk.parameters())
        assert all(p.grad is not None for p in native.group_heads["cart"].parameters())


@pytest.mark.parametrize("factory", [isaac_handover_ippo, isaac_handover_mappo])
def test_homogeneous_shared_actors_remain_local_and_critic_dependencies_match_algorithm(factory):
    policy, algorithm = build(factory)
    data = rollout(policy, algorithm)
    changed = data.clone()
    changed["hands", "observation"][..., 1, :].add_(10)
    torch.testing.assert_close(
        policy(data)["hands", "action"][..., 0, :],
        policy(changed)["hands", "action"][..., 0, :],
        rtol=0,
        atol=0,
    )
    left, right = algorithm.prepare(data), algorithm.prepare(changed)
    same = torch.equal(
        left["hands", "state_value"][..., 0, :], right["hands", "state_value"][..., 0, :]
    )
    assert same is (not algorithm.centralized)


@pytest.mark.parametrize("factory", [isaac_pendulum_ippo, isaac_pendulum_mappo])
def test_native_gae_terminal_timeout_and_inactive_suffix_reward_oracle(factory):
    policy, algorithm = build(factory)
    data = rollout(policy, algorithm)
    data["next", "cart", "done"][:, 1] = True
    data["next", "cart", "terminated"][:, 1] = True
    data["cart", "valid"][:, 2:] = False
    data["next", "pendulum", "done"][:, 2] = True
    data["next", "pendulum", "truncated"][:, 2] = True
    prepared = algorithm.prepare(data)
    changed = data.clone()
    changed["next", "cart", "reward"][:, 2:] = 1e6
    other = algorithm.prepare(changed)
    torch.testing.assert_close(
        prepared["cart", "advantage"][:, :2], other["cart", "advantage"][:, :2], rtol=0, atol=0
    )
    torch.testing.assert_close(
        prepared["cart", "value_target"][:, 1], data["next", "cart", "reward"][:, 1]
    )
    # Independent test-only recurrence using actual native current/final values.
    for loss, (group, index, _) in zip(algorithm.losses, algorithm.slots, strict=True):
        view = agent_view(data, group, index)
        algorithm._critic_inputs(view)
        algorithm._critic_inputs(view["next"])
        with torch.no_grad():
            value = loss.critic_network(view)[group, "state_value"]
            following = loss.critic_network(view["next"])[group, "state_value"]
        reward = view["next", group, "reward"].masked_fill(~view[group, "valid"], 0)
        term, done = view["next", group, "terminated"], view["next", group, "done"]
        expected = torch.zeros_like(value)
        carry = torch.zeros_like(value[:, 0])
        for step in reversed(range(data.shape[1])):
            delta = (
                reward[:, step]
                + algorithm.definition.gamma * (~term[:, step]) * following[:, step]
                - value[:, step]
            )
            carry = (
                delta
                + algorithm.definition.gamma
                * algorithm.definition.gae_lambda
                * (~done[:, step])
                * carry
            )
            expected[:, step] = carry
        torch.testing.assert_close(
            prepared[group, "advantage"][..., index : index + 1, :], expected, rtol=1e-5, atol=1e-6
        )


@pytest.mark.parametrize("factory", [isaac_pendulum_ippo, isaac_pendulum_mappo])
def test_warmed_inactive_actor_and_all_invalid_batches_never_step(factory):
    policy, algorithm = build(factory)
    algorithm.definition = algorithm.definition.model_copy(update={"weight_decay": 0.1})
    optimizers = algorithm.configure_optimizers()
    data = rollout(policy, algorithm)
    algorithm.update(data, lightning_module=ManualOptimization(), optimizers=optimizers)
    actor_id = policy.actor_ids["cart"][0]
    actor_before = copy.deepcopy(policy.actors[actor_id].state_dict())
    optimizer_before = copy.deepcopy(optimizers[0].state_dict())
    critic_steps = [int(state["step"]) for state in optimizers[-1].state.values()]
    calls = []
    original_step = optimizers[-1].step

    def count_step(*args, **kwargs):
        calls.append(1)
        return original_step(*args, **kwargs)

    optimizers[-1].step = count_step
    data["cart", "valid"][:] = False
    algorithm.update(data, lightning_module=ManualOptimization(), optimizers=optimizers)
    torch.testing.assert_close(policy.actors[actor_id].state_dict(), actor_before, rtol=0, atol=0)
    torch.testing.assert_close(optimizers[0].state_dict(), optimizer_before, rtol=0, atol=0)
    assert all(p.grad is None for p in policy.actors[actor_id].parameters())
    assert len(calls) == 1
    assert all(
        int(state["step"]) in (step, step + 1)
        for state, step in zip(optimizers[-1].state.values(), critic_steps, strict=True)
    )
    for group in policy.contract.groups:
        data[group, "valid"][:] = False
    before = copy.deepcopy(algorithm.state_dict())
    optimizer_states = [copy.deepcopy(optimizer.state_dict()) for optimizer in optimizers]
    assert (
        algorithm.update(data, lightning_module=ManualOptimization(), optimizers=optimizers) == {}
    )
    torch.testing.assert_close(algorithm.state_dict(), before, rtol=0, atol=0)
    torch.testing.assert_close(
        [opt.state_dict() for opt in optimizers], optimizer_states, rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "factory", [isaac_pendulum_ippo, isaac_pendulum_mappo, isaac_handover_mappo]
)
def test_current_shared_optimizer_continuation_matches_uninterrupted(factory, tmp_path):
    policy, algorithm = build(factory)
    optimizers = algorithm.configure_optimizers()
    data = rollout(policy, algorithm)
    algorithm.update(data, lightning_module=ManualOptimization(), optimizers=optimizers)
    payload = copy.deepcopy(algorithm.state_dict())
    states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
    torch.save({"algorithm": payload, "optimizers": states}, tmp_path / "current.pt")
    saved = torch.load(tmp_path / "current.pt", weights_only=True)
    restored_policy, restored = build(factory)
    restored.load_state_dict(saved["algorithm"])
    restored_optimizers = restored.configure_optimizers()
    for optimizer, state in zip(restored_optimizers, saved["optimizers"], strict=True):
        optimizer.load_state_dict(state)
    torch.manual_seed(123)
    algorithm.update(data, lightning_module=ManualOptimization(), optimizers=optimizers)
    torch.manual_seed(123)
    restored.update(data, lightning_module=ManualOptimization(), optimizers=restored_optimizers)
    torch.testing.assert_close(algorithm.state_dict(), restored.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(
        [o.state_dict() for o in optimizers],
        [o.state_dict() for o in restored_optimizers],
        rtol=0,
        atol=0,
    )
    if "hands" in policy.contract.groups:
        assert restored_policy.actor_ids["hands"] == ("hand_actor", "hand_actor")
        assert len(restored_policy.actors) == 1 and len(restored_optimizers) == 2


@pytest.mark.parametrize("definition", [IPPO, MAPPO])
@pytest.mark.parametrize(
    "option", [{"normalize_advantage": True}, {"average_gae": True}, {"value_norm": "PopArt"}]
)
def test_unqualified_masked_normalization_rejects(definition, option):
    with pytest.raises(ValidationError):
        definition(**option)


def test_shifted_valid_native_terminal_counterexample_and_integration_denial():
    from torchrl.objectives.value import MultiAgentGAE

    data = TensorDict(
        {
            "state_value": torch.zeros(1, 2, 1, 1),
            "next": TensorDict(
                {
                    "state_value": torch.full((1, 2, 1, 1), 2.0),
                    "reward": torch.ones(1, 2, 1),
                    "done": torch.ones(1, 2, 1, dtype=torch.bool),
                    "terminated": torch.ones(1, 2, 1, dtype=torch.bool),
                },
                batch_size=(1, 2),
            ),
        },
        batch_size=(1, 2),
    )
    estimator = MultiAgentGAE(
        gamma=0.9, lmbda=0.95, value_network=None, time_dim=1, average_gae=False
    )
    genuine = estimator(data.clone())["advantage"][0, 0, 0, 0]
    wrong = data.clone()
    wrong["shifted_valid"] = torch.tensor([[True, False]])
    corrupted = estimator(wrong)["advantage"][0, 0, 0, 0]
    assert float(genuine) == 1.0
    assert float(corrupted) == pytest.approx(2.8)
    policy, algorithm = build(isaac_pendulum_mappo)
    batch = rollout(policy, algorithm)
    batch["shifted_valid"] = torch.ones(batch.batch_size, dtype=torch.bool)
    with pytest.raises(ValueError, match="not an agent"):
        algorithm.prepare(batch)


def test_native_masked_loss_matches_independent_clipped_surrogate_and_smooth_l1():
    policy, algorithm = build(isaac_pendulum_mappo)
    data = rollout(policy, algorithm)
    data["cart", "valid"][:, 3:] = False
    prepared = algorithm.prepare(data)
    flat = prepared.reshape(-1)
    with torch.no_grad():
        next(policy.actors["cart_actor"].parameters()).add_(0.1)
    view = agent_view(flat, "cart", 0)
    loss = algorithm.losses[0]
    dist = loss.actor_network.get_dist(view.clone())
    log_prob = dist.log_prob(view["cart", "action"])
    ratio = (log_prob - view["cart", "action_log_prob"]).exp().unsqueeze(-1)
    advantage = view["cart", "advantage"]
    mask = view["cart", "valid"]
    expected = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)[mask].mean()
    native = loss(view.clone())
    torch.testing.assert_close(native["loss_objective"], expected)
    prediction = loss.critic_network(view.clone())["cart", "state_value"]
    error = torch.nn.functional.smooth_l1_loss(
        prediction, view["cart", "value_target"], reduction="none"
    )
    torch.testing.assert_close(native["loss_critic"], error[mask].mean())

    def gradients(batch):
        algorithm.zero_grad(set_to_none=True)
        torch.manual_seed(8)
        values = loss(batch)
        sum(values[key] for key in ("loss_objective", "loss_entropy", "loss_critic")).backward()
        return [None if p.grad is None else p.grad.clone() for p in algorithm.parameters()]

    baseline = gradients(view.clone())
    changed = view.clone()
    changed["cart", "value_target"][~mask] = 1e6
    changed["cart", "advantage"][~mask] = -1e6
    torch.testing.assert_close(gradients(changed), baseline, rtol=0, atol=0)


def test_inactive_body_still_informs_centralized_world_state():
    policy, algorithm = build(isaac_pendulum_mappo)
    data = rollout(policy, algorithm)
    data["cart", "valid"][:] = False
    before = algorithm.prepare(data)["pendulum", "state_value"]
    data["cart", "observation"].add_(10)
    after = algorithm.prepare(data)["pendulum", "state_value"]
    assert not torch.equal(before, after)


@pytest.mark.parametrize("centralized", [False, True])
def test_grouped_image_vector_categorical_learning_and_actor_export(centralized, tmp_path):
    from nexuml.core.types import TensorFieldContract, LayerSpec, PipelineSpec, GroupPolicySpec
    from nexuml_library.action_adapters.adapters import CategoricalActionAdapter
    from nexuml_library.reinforcement.policies import PolicyHead, discrete_policy_mlp
    from nexuml.core.export import export_package, load_inference_package, load_package

    scenario = isaac_pendulum_mappo() if centralized else isaac_pendulum_ippo()
    definition = scenario.interaction.environment
    groups = dict(definition.contract.groups)
    groups["cart"] = groups["cart"].model_copy(
        update={
            "observations": {
                "observation": TensorFieldContract(
                    shape=(3, 4, 4),
                    dtype="float32",
                    modality="image",
                    description="CPU processed image routing only, not native qualified",
                )
            }
        }
    )
    groups["pendulum"] = groups["pendulum"].model_copy(
        update={
            "action_space": "discrete",
            "actions": {"action": TensorFieldContract(shape=(), dtype="int64", num_values=3)},
        }
    )
    contract = definition.contract.model_copy(update={"groups": groups})
    scenario.interaction.environment = definition.model_copy(update={"contract": contract})
    policies = dict(scenario.policy.groups)
    policies["pendulum"] = GroupPolicySpec(
        pipeline=discrete_policy_mlp(num_actions=3),
        action_adapter=CategoricalActionAdapter(),
        actor_ids=("pendulum_actor",),
    )
    scenario.policy = scenario.policy.model_copy(update={"groups": policies})
    encoders = (
        {
            "cart": PipelineSpec(
                stages={
                    "encode": [
                        LayerSpec(
                            component=PolicyHead(output_dim=6),
                            keys_in=["observation"],
                            keys_out=["features"],
                        )
                    ]
                }
            )
        }
        if centralized
        else {}
    )
    scenario.training.algorithm = (MAPPO if centralized else IPPO)(
        update_epochs=1, minibatch_size=16, critic_encoders=encoders
    )
    policy = CompiledPolicy.from_pipeline(compile(scenario))
    algorithm: Any = scenario.training.algorithm.build(RLAlgorithmBuildContext(policy, contract))
    data = rollout(policy, algorithm)
    assert data["pendulum", "action"].dtype == torch.int64 and data["pendulum", "action"].shape == (
        2,
        5,
        1,
    )
    assert (
        data["cart", "action_log_prob"].shape
        == data["pendulum", "action_log_prob"].shape
        == (2, 5, 1)
    )
    before = [p.detach().clone() for p in algorithm.encoders.parameters()]
    algorithm.update(
        data, lightning_module=ManualOptimization(), optimizers=algorithm.configure_optimizers()
    )
    if centralized:
        assert any(
            not torch.equal(a, b)
            for a, b in zip(before, algorithm.encoders.parameters(), strict=True)
        )
    inputs = data[:, 0].select(*contract.groups)
    expected = policy(inputs)
    export_package(policy, tmp_path / "policy")
    assert "torchrl" not in (tmp_path / "policy/requirements.txt").read_text().lower()
    for loader in (load_package, load_inference_package):
        loaded, _, _ = loader(tmp_path / "policy")
        torch.testing.assert_close(loaded(inputs), expected, rtol=0, atol=0)
        assert all("critic" not in key for key in loaded.state_dict())


def test_native_default_advantage_standardization_ignores_custom_loss_mask_statistics():
    policy, algorithm = build(isaac_pendulum_ippo)
    data = rollout(policy, algorithm)
    data["cart", "valid"][:, 2:] = False
    prepared = algorithm.prepare(data).reshape(-1)
    native = algorithm.losses[0]
    native.normalize_advantage = True  # Diagnose upstream; the registered definition rejects this.
    view = agent_view(prepared, "cart", 0)
    mask = view["cart", "valid"]
    view["cart", "advantage"] = torch.arange(view.numel(), dtype=torch.float32).reshape(-1, 1, 1)
    first = native(view.clone())["loss_objective"]
    view["cart", "advantage"][~mask] = 1e5
    second = native(view.clone())["loss_objective"]
    assert not torch.allclose(first, second)
    assert not algorithm.definition.normalize_advantage


def test_existing_lightning_step_counts_only_physical_frames_and_labels_group_rewards(monkeypatch):
    from types import SimpleNamespace
    from nexuml.training.reinforcement import NexuRLLightningModule

    scenario = isaac_pendulum_ippo(total_frames=16, frames_per_batch=8, num_envs=2)
    definition = scenario.interaction.environment
    contract = definition.contract.model_copy(update={"team_reward": False})
    scenario.interaction.environment = definition.model_copy(update={"contract": contract})
    scenario.training.algorithm = IPPO(update_epochs=1, minibatch_size=16)
    module = NexuRLLightningModule(scenario)
    data = rollout(module.policy, module.algorithm)
    optimizers = module.configure_optimizers()
    records, syncs = {}, []
    monkeypatch.setattr(module, "optimizers", lambda: optimizers)
    monkeypatch.setattr(module, "manual_backward", ManualOptimization().manual_backward)
    monkeypatch.setattr(module, "clip_gradients", ManualOptimization().clip_gradients)
    monkeypatch.setattr(
        module, "log", lambda name, value, **kwargs: records.__setitem__(name, value)
    )
    module._collector = SimpleNamespace(
        update_policy_weights_=lambda weights: syncs.append(weights)
    )
    module.training_step(data, 0)
    assert int(module.frames) == 10 and int(module.batches) == 1
    assert "train/rl/reward" not in records and len(syncs) == 1
    for group in contract.groups:
        torch.testing.assert_close(
            records[f"train/rl/{group}/agent_reward_mean"], data["next", group, "reward"].mean()
        )


@pytest.mark.parametrize(
    "factory",
    [isaac_pendulum_ippo, isaac_pendulum_mappo, isaac_handover_ippo, isaac_handover_mappo],
)
def test_registered_marl_actor_packages_run_cleanly_without_learner_or_native_imports(
    factory, tmp_path
):
    import subprocess
    import sys
    from pathlib import Path
    from nexuml.core.export import export_package

    policy, algorithm = build(factory)
    data = rollout(policy, algorithm)[:, 0].select(*policy.contract.groups)
    expected = policy(data)
    output = tmp_path / "resume"
    output.mkdir()
    torch.save(data, output / "observation.pt")
    torch.save(expected, output / "expected.pt")
    export_package(policy, output / "policy")
    assert "torchrl" not in (output / "policy/requirements.txt").read_text().lower()
    fixture = Path(__file__).resolve().parents[1] / "fixtures/isaaclab_marl_job.py"
    result = subprocess.run(
        [sys.executable, "-I", str(fixture), "unused", "unused", "infer", str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout
    assert "GROUPED_EXACT_ACTOR_ONLY_CLEAN_INFERENCE_AND_SHARING_VERIFIED" in result.stdout


@pytest.mark.parametrize("change", ["semantics", "count", "sharing"])
def test_grouped_checkpoint_rejects_changed_semantics_count_or_sharing_before_collection(change):
    from nexuml.training.reinforcement import NexuRLLightningModule

    scenario = isaac_handover_mappo(total_frames=32, frames_per_batch=16, num_envs=2)
    original = NexuRLLightningModule(scenario)
    checkpoint = {"hyper_parameters": dict(original.hparams)}
    original.on_load_checkpoint(checkpoint)
    changed = scenario.model_copy(deep=True)
    if change == "count":
        changed.training.collector = changed.training.collector.model_copy(
            update={"num_envs_per_collector": 1}
        )
    elif change == "sharing":
        group = changed.policy.groups["hands"].model_copy(
            update={"actor_ids": ("right_actor", "left_actor")}
        )
        changed.policy = changed.policy.model_copy(update={"groups": {"hands": group}})
    else:
        definition = changed.interaction.environment
        group = definition.contract.groups["hands"]
        field = group.observations["observation"].model_copy(
            update={"description": "changed physical meaning"}
        )
        group = group.model_copy(update={"observations": {"observation": field}})
        contract = definition.contract.model_copy(update={"groups": {"hands": group}})
        changed.interaction.environment = definition.model_copy(update={"contract": contract})
    module = NexuRLLightningModule(changed)
    with pytest.raises(ValueError, match="Grouped checkpoint"):
        module.on_load_checkpoint(checkpoint)
    assert module._collector is None


@pytest.mark.parametrize(
    "factory", [isaac_pendulum_ippo, isaac_pendulum_mappo, isaac_handover_mappo]
)
def test_serialized_lightning_grouped_counters_and_updates_continue_exactly(
    factory, tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from nexuml.training.reinforcement import NexuRLLightningModule

    scenario = factory(total_frames=64, frames_per_batch=16, num_envs=2)
    scenario.training.algorithm = scenario.training.algorithm.model_copy(
        update={"update_epochs": 1, "minibatch_size": 16}
    )

    def wire():
        module = NexuRLLightningModule(scenario)
        optimizers = module.configure_optimizers()
        monkeypatch.setattr(module, "optimizers", lambda: optimizers)
        monkeypatch.setattr(module, "manual_backward", ManualOptimization().manual_backward)
        monkeypatch.setattr(module, "clip_gradients", ManualOptimization().clip_gradients)
        monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)
        module._collector = SimpleNamespace(update_policy_weights_=lambda weights: None)
        return module, optimizers

    module, optimizers = wire()
    data = rollout(module.policy, module.algorithm)
    module.training_step(data, 0)
    torch.save(
        {
            "hyper_parameters": dict(module.hparams),
            "state_dict": module.state_dict(),
            "optimizer_states": [optimizer.state_dict() for optimizer in optimizers],
        },
        tmp_path / "current.ckpt",
    )
    checkpoint = torch.load(tmp_path / "current.ckpt", weights_only=False)
    restored, restored_optimizers = wire()
    restored.on_load_checkpoint(checkpoint)
    restored.load_state_dict(checkpoint["state_dict"])
    for optimizer, state in zip(restored_optimizers, checkpoint["optimizer_states"], strict=True):
        optimizer.load_state_dict(state)
    assert int(restored.frames) == 10 and int(restored.batches) == 1
    torch.manual_seed(99)
    module.training_step(data, 1)
    torch.manual_seed(99)
    restored.training_step(data, 1)
    torch.testing.assert_close(module.state_dict(), restored.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(
        [optimizer.state_dict() for optimizer in optimizers],
        [optimizer.state_dict() for optimizer in restored_optimizers],
        rtol=0,
        atol=0,
    )
    assert int(restored.frames) == 20 and int(restored.batches) == 2


def test_registered_grouped_recipes_restore_and_compile_without_native_or_learner_imports():
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent("""
        import importlib.abc
        import sys
        class BlockNative(importlib.abc.MetaPathFinder):
            def find_spec(self,fullname,path=None,target=None):
                if fullname.split('.')[0] in {
                    'torchrl','isaaclab','isaaclab_tasks','isaaclab_assets',
                    'isaacsim','newton','viser','warp','omni','pxr'}:
                    raise ImportError('Native/learner import blocked: '+fullname)
        sys.meta_path.insert(0,BlockNative())
        import torch
        def no_cuda():
            raise AssertionError('Static grouped authoring allocated CUDA')
        torch.cuda._lazy_init = no_cuda
        from nexuml.core.compiler import compile
        from nexuml.core.config import ResolvedConfig
        from nexuml.core.policy import CompiledPolicy
        from nexuml_library.scenarios.reinforcement import isaaclab_marl
        for family in ('pendulum','handover'):
            for algorithm in ('ippo','mappo'):
                scenario = getattr(isaaclab_marl,'isaac_'+family+'_'+algorithm)(total_frames=32)
                text = ResolvedConfig.from_scenario(scenario).to_yaml()
                restored = ResolvedConfig.from_yaml(text).to_scenario()
                policy = CompiledPolicy.from_pipeline(compile(restored))
                assert policy.actor_ids
        assert 'torchrl' not in sys.modules and 'isaaclab' not in sys.modules
        print('GROUPED_STATIC_AUTHORING_WITH_NATIVE_AND_CUDA_BLOCKED')
    """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout
    assert "GROUPED_STATIC_AUTHORING_WITH_NATIVE_AND_CUDA_BLOCKED" in result.stdout


@pytest.mark.parametrize(
    "change", ["remove_config", "config_kwargs", "physics", "presets", "bindings", "task", "viewer"]
)
def test_grouped_checkpoint_captures_native_task_configuration_but_not_viewer(change):
    from nexuml.training.reinforcement import NexuRLLightningModule
    from nexuml_library.environments.isaaclab_config import IsaacTaskConfigSpec, ViserSpec

    scenario = isaac_pendulum_ippo(total_frames=64, frames_per_batch=16, num_envs=2)
    environment = scenario.interaction.environment.model_copy(
        update={
            "task_config": IsaacTaskConfigSpec(
                factory="operator_tasks:configure_horizon", kwargs={"seconds": 0.1}
            )
        }
    )
    scenario.interaction.environment = environment
    original = NexuRLLightningModule(scenario)
    checkpoint = {"hyper_parameters": dict(original.hparams)}
    changed = scenario.model_copy(deep=True)
    updates = {
        "remove_config": {"task_config": None},
        "config_kwargs": {
            "task_config": environment.task_config.model_copy(update={"kwargs": {"seconds": 5.0}})
        },
        "physics": {"physics": "isaacsim_physx"},
        "presets": {"presets": ("other_distribution",)},
        "bindings": {"bindings": environment.bindings.model_copy(update={"processing": "raw"})},
        "task": {"task": "Other-Native-Pendulum"},
        "viewer": {"viewer": ViserSpec(port=8081)},
    }
    changed.interaction.environment = environment.model_copy(update=updates[change])
    module = NexuRLLightningModule(changed)
    assert module.policy.contract == original.policy.contract
    if change == "viewer":
        module.on_load_checkpoint(checkpoint)
    else:
        with pytest.raises(ValueError, match="task configuration"):
            module.on_load_checkpoint(checkpoint)
    assert module._collector is None


@pytest.mark.parametrize(
    "key", ["valid", "done", "reward", "state_value", "advantage", "value_target"]
)
def test_group_fields_cannot_clobber_learning_or_transition_signals(key):
    from nexuml.core.types import AgentGroupContract

    original = isaac_pendulum_ippo().interaction.environment.describe().groups["cart"]
    fields = original.model_dump()
    fields["observations"] = {key: fields["observations"]["observation"]}
    with pytest.raises(ValueError, match="overwrite learning or transition"):
        AgentGroupContract.model_validate(fields)


@pytest.mark.parametrize("name", ["next", "collector", "reward", "done"])
def test_group_names_cannot_clobber_root_transition_metadata(name):
    from nexuml.core.types import InteractionContract

    original = isaac_pendulum_ippo().interaction.environment.describe()
    fields = original.model_dump()
    fields["groups"] = {name: fields["groups"]["cart"]}
    with pytest.raises(ValueError, match="overwrite root transition"):
        InteractionContract.model_validate(fields)


@pytest.mark.parametrize("key", ["action_loc", "action_log_prob"])
def test_learner_distribution_outputs_cannot_replace_actor_observations(key):
    from nexuml_library.reinforcement.policies import continuous_policy_mlp

    scenario = isaac_pendulum_ippo()
    environment = scenario.interaction.environment
    group = environment.contract.groups["cart"]
    group = group.model_copy(update={"observations": {key: group.observations["observation"]}})
    contract = environment.contract.model_copy(
        update={"groups": {**environment.contract.groups, "cart": group}}
    )
    scenario.interaction.environment = environment.model_copy(update={"contract": contract})
    policy_spec = scenario.policy.groups["cart"].model_copy(
        update={"pipeline": continuous_policy_mlp(observation_keys=(key,))}
    )
    scenario.policy = scenario.policy.model_copy(
        update={"groups": {**scenario.policy.groups, "cart": policy_spec}}
    )
    policy = CompiledPolicy.from_pipeline(compile(scenario))
    with pytest.raises(ValueError, match="overwrite actor observations"):
        scenario.training.algorithm.build(RLAlgorithmBuildContext(policy, contract))
