"""Data/interaction compilation shares one graph and keeps deployment independent."""

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from nexuml.core.compiler import (
    compile,
    compile_context_from_data,
    compile_context_from_interaction,
    compile_pipeline,
)
from nexuml.core.config import ResolvedConfig
from nexuml.core.policy import CompiledPolicy
from nexuml.core.torch_adapter import nn_module
from nexuml.core.types import (
    DataSpec,
    InteractionContract,
    LayerSpec,
    PipelineSpec,
    ScenarioSpec,
    TensorFieldContract,
    TrainingSpec,
    PolicySpec,
    InteractionSpec,
)
from nexuml_library.layers.model.linear_encoder import LinearEncoder
from nexuml_library.action_adapters.adapters import (
    BoundedNormal,
    DirectActionAdapter,
    GaussianActionAdapter,
    CategoricalActionAdapter,
)


@pytest.mark.parametrize("bound", [1.0, 2.0])
def test_bounded_normal_saturated_samples_keep_finite_reproducible_likelihoods(bound):
    loc = torch.full((8, 2), 20.0, requires_grad=True)
    scale = torch.ones(8, 2, requires_grad=True)
    distribution = BoundedNormal(loc, scale, torch.full((2,), -bound), torch.full((2,), bound))
    torch.manual_seed(0)
    sample = distribution.rsample()
    upper = torch.nextafter(torch.tensor(bound), torch.tensor(-bound))
    assert (sample == upper).all()
    log_prob = distribution.log_prob(sample)
    assert torch.isfinite(log_prob).all()
    fresh = BoundedNormal(loc, scale, torch.full((2,), -bound), torch.full((2,), bound))
    torch.testing.assert_close(fresh.log_prob(sample.detach()), log_prob, rtol=0, atol=0)
    assert (fresh.sample() < bound).all() and (fresh.mode < bound).all()
    (-log_prob.mean()).backward()
    assert loc.grad is not None and torch.isfinite(loc.grad).all()
    assert scale.grad is not None and torch.isfinite(scale.grad).all()


def test_bounded_normal_interior_sampling_and_log_prob_match_existing_transforms():
    from torch.distributions import (
        Independent,
        Normal,
        TransformedDistribution,
        TanhTransform,
        AffineTransform,
    )

    loc, scale = torch.tensor([[0.2, -0.3]]), torch.tensor([[0.8, 0.9]])
    bounds = torch.ones(2)
    old = Independent(
        TransformedDistribution(
            Normal(loc, scale),
            [
                TanhTransform(cache_size=1),
                AffineTransform(torch.zeros(2), bounds),
            ],
        ),
        1,
    )
    current = BoundedNormal(loc, scale, -bounds, bounds)
    torch.manual_seed(6)
    reference = old.rsample()
    torch.manual_seed(6)
    actual = current.rsample()
    assert torch.equal(actual, reference)
    torch.testing.assert_close(current.log_prob(actual), old.log_prob(reference), rtol=0, atol=0)


@pytest.mark.parametrize("bounds", [(-2.0, 1.0), (0.0, 1.0), (-3.0, 9.0), (1.0, 2.0)])
@pytest.mark.parametrize("location", [-20.0, 20.0])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_bounded_normal_asymmetric_roundoff_keeps_fresh_likelihoods_finite(bounds, location, dtype):
    loc = torch.full((8, 1), location, dtype=dtype, requires_grad=True)
    scale = torch.ones(8, 1, dtype=dtype, requires_grad=True)
    low, high = (torch.tensor([value], dtype=dtype) for value in bounds)
    distribution = BoundedNormal(loc, scale, low, high)
    torch.manual_seed(0)
    sample = distribution.rsample()
    assert ((sample > low) & (sample < high)).all()
    log_prob = distribution.log_prob(sample)
    assert torch.isfinite(log_prob).all()
    fresh = BoundedNormal(loc, scale, low, high)
    torch.testing.assert_close(fresh.log_prob(sample.detach()), log_prob, rtol=0, atol=0)
    (-log_prob.mean()).backward()
    assert loc.grad is not None and torch.isfinite(loc.grad).all()
    assert scale.grad is not None and torch.isfinite(scale.grad).all()


def test_bounded_normal_rejects_unrepresentable_action_interval():
    low = torch.tensor([1.0])
    high = torch.nextafter(low, torch.full_like(low, float("inf")))
    with pytest.raises(ValueError, match="floating-point Gaussian interior"):
        BoundedNormal(torch.zeros(1), torch.ones(1), low, high)


def contract():
    return InteractionContract(
        observations={
            "camera": TensorFieldContract(shape=(3, 8, 8), dtype="uint8", modality="image"),
            "joints": TensorFieldContract(shape=(4,), dtype="float32", modality="proprioception"),
        },
        actions={"action": TensorFieldContract(shape=(2,), dtype="float32", low=-1, high=1)},
        action_space="continuous",
        num_envs=8,
    )


class BoundedDirect(nn.Module):
    def forward(self, tensors, *, deterministic):
        return TensorDict({"action": tensors["command"].tanh()}, batch_size=tensors.batch_size)


def scenario():
    return ScenarioSpec(
        name="interaction-compile",
        data=DataSpec(input_shapes={"joints": [4], "camera": [3, 8, 8]}),
        pipeline=PipelineSpec(
            stages={
                "actor": [
                    LayerSpec(
                        component=LinearEncoder(output_dim=2),
                        keys_in=["joints"],
                        keys_out=["command"],
                    )
                ],
                "image": [
                    LayerSpec(
                        component=nn_module(nn.Identity),
                        keys_in=["camera"],
                        keys_out=["camera_copy"],
                    )
                ],
            }
        ),
        training=TrainingSpec(loss_keys={"loss": 1}, metric_keys=["accuracy"], lr=0.01),
    )


def test_actor_and_critic_share_interaction_context_without_data():
    spec = scenario()
    context = compile_context_from_interaction(contract())
    resolved = ResolvedConfig.from_scenario(spec)
    actor = compile_pipeline(spec.pipeline, context=context, resolved_config=resolved)
    critic = compile_pipeline(
        PipelineSpec(
            stages={
                "value": [
                    LayerSpec(
                        component=LinearEncoder(output_dim=1),
                        keys_in=["joints"],
                        keys_out=["value"],
                    )
                ]
            }
        ),
        context=context,
        resolved_config=resolved,
    )
    assert context.input_sizes["camera"] == (3, 8, 8)
    assert context.input_metadata["camera"]["dtype"] == "uint8"
    assert context.input_metadata["camera"]["modality"] == "image"
    x = TensorDict(
        {
            "camera": torch.zeros(8, 3, 8, 8, dtype=torch.uint8),
            "joints": torch.randn(8, 4),
        },
        batch_size=[8],
    )
    out, y = actor(x.clone())
    assert y is None
    assert out["command"].shape == (8, 2)
    assert out["camera_copy"].dtype == torch.uint8
    assert critic(x.clone())[0]["value"].shape == (8, 1)
    assert actor.loss_keys == {} and actor.metric_keys == []
    with pytest.raises(RuntimeError, match="algorithm-owned"):
        actor.create_optimizer()
    with pytest.raises(RuntimeError, match="algorithm-owned"):
        actor.create_scheduler(torch.optim.Adam(actor.parameters()))


def test_supervised_wrapper_matches_explicit_compilation():
    spec = scenario()
    torch.manual_seed(123)
    ordinary = compile(spec)
    torch.manual_seed(123)
    explicit = compile_pipeline(
        spec.pipeline,
        context=compile_context_from_data(spec.data),
        resolved_config=ResolvedConfig.from_scenario(spec),
        training=spec.training,
    )
    assert ordinary.loss_keys == explicit.loss_keys == {"loss": 1}
    assert ordinary.metric_keys == explicit.metric_keys == ["accuracy"]
    assert ordinary.input_sizes == explicit.input_sizes
    assert ordinary.create_optimizer().param_groups[0]["lr"] == 0.01
    for key, value in ordinary.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[key])


def test_complete_policy_checks_contract_and_preserves_observations():
    spec = scenario()
    io = contract()
    pipeline = compile_pipeline(
        spec.pipeline,
        context=compile_context_from_interaction(io),
        resolved_config=ResolvedConfig.from_scenario(spec),
    )
    policy = CompiledPolicy(pipeline, BoundedDirect(), io)
    x = TensorDict(
        {
            "camera": torch.zeros(3, 8, 8, dtype=torch.uint8),
            "joints": torch.randn(4),
        },
        batch_size=[],
    )
    actions = policy(x)
    assert actions["action"].shape == (2,)
    assert policy(x.expand(2, 3))["action"].shape == (2, 3, 2)
    assert "command" not in x.keys()
    assert torch.equal(actions["action"], policy(x)["action"])
    assert any(key.startswith("pipeline.") for key in policy.state_dict())
    with pytest.raises(ValueError, match="missing.*joints"):
        policy(x.exclude("joints"))
    wrong_dtype = x.clone()
    wrong_dtype["camera"] = wrong_dtype["camera"].float()
    with pytest.raises(ValueError, match="camera.*dtype"):
        policy(wrong_dtype)
    wrong_shape = TensorDict(
        {
            "camera": x["camera"].unsqueeze(0),
            "joints": torch.zeros(1, 5),
        },
        batch_size=[1],
    )
    with pytest.raises(ValueError, match="joints.*shape"):
        policy(wrong_shape)


def test_unsupported_observation_names_leaf_in_diagnostic():
    io = contract()
    io = io.model_copy(
        update={
            "observations": {
                "language": TensorFieldContract(shape=(4,), dtype="string", modality="language"),
            }
        }
    )
    with pytest.raises(ValueError, match="language.*string"):
        compile_context_from_interaction(io)
    with pytest.raises(TypeError, match="describe.*InteractionContract"):
        compile_context_from_interaction(object())  # ty: ignore[invalid-argument-type]


def test_maintained_adapters_use_base_torch_and_preserve_sampling_semantics():
    continuous = contract()
    loc = torch.tensor([[0.0, 1.0], [1.0, -1.0]])
    parameters = TensorDict(
        {
            "action_loc": loc,
            "action_scale": torch.ones(2, 2),
        },
        batch_size=[2],
    )
    gaussian = GaussianActionAdapter().build(continuous)
    assert torch.equal(gaussian(parameters)["action"], loc.tanh())
    assert gaussian(parameters, deterministic=False)["action"].abs().max() <= 1
    direct = DirectActionAdapter(source_key="command").build(continuous)
    command = torch.ones(2, 2)
    output = TensorDict({"command": command}, batch_size=[2])
    assert torch.equal(direct(output)["action"], command)
    discrete = InteractionContract(
        observations={"observation": TensorFieldContract(shape=(4,), dtype="float32")},
        actions={"action": TensorFieldContract(shape=(), dtype="int64", num_values=3)},
        action_space="discrete",
    )
    categorical = CategoricalActionAdapter().build(discrete)
    logits = TensorDict({"action_logits": torch.tensor([[0.0, 2.0, 1.0]])}, batch_size=[1])
    assert categorical(logits)["action"].item() == 1
    assert 0 <= categorical(logits, deterministic=False)["action"].item() < 3
    with pytest.raises(ValueError, match="3 classes"):
        categorical(TensorDict({"action_logits": torch.zeros(1, 2)}, batch_size=[1]))


@pytest.mark.parametrize("semantic_axis", ["time", "agent", "agents"])
def test_independent_batch_mode_rejects_semantic_batch_axes(semantic_axis):
    spec = scenario()
    io = contract()
    policy = CompiledPolicy(compile(spec), BoundedDirect(), io)
    observation = TensorDict(
        {"camera": torch.zeros(2, 3, 3, 8, 8, dtype=torch.uint8), "joints": torch.zeros(2, 3, 4)},
        batch_size=[2, 3],
        names=["environment", semantic_axis],
    )
    with pytest.raises(ValueError, match="batch_mode='preserve'"):
        policy(observation)


def test_preserved_batch_layout_keeps_temporal_computation_and_package_parity(
    tmp_path, monkeypatch
):
    from nexuml.core.export import export_package, load_inference_package
    from nexuml_library.environments.gymnasium import GymnasiumEnvironment

    io = InteractionContract(
        observations={"state": TensorFieldContract(shape=(2,), dtype="float32")},
        actions={"action": TensorFieldContract(shape=(2,), dtype="float32")},
        action_space="continuous",
    )
    spec = ScenarioSpec(
        name="batch-aware-policy",
        data=DataSpec(input_shapes={"state": [2]}),
        pipeline=PipelineSpec(
            stages={
                "actor": [
                    LayerSpec(
                        component=nn_module(nn.Softmax, dim=1),
                        keys_in=["state"],
                        keys_out=["command"],
                    )
                ]
            }
        ),
        interaction=InteractionSpec(environment=GymnasiumEnvironment(contract=io)),
        policy=PolicySpec(
            action_adapter=DirectActionAdapter(source_key="command"), batch_mode="preserve"
        ),
    )
    restored = ResolvedConfig.from_yaml(ResolvedConfig.from_scenario(spec).to_yaml()).to_scenario()
    policy = CompiledPolicy.from_pipeline(compile(restored), io)
    values = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    observation = TensorDict({"state": values}, [2, 3], names=["environment", "time"])
    expected = values.softmax(dim=1)
    actions = policy(observation)
    assert torch.equal(actions["action"], expected)
    assert actions.names == observation.names
    assert policy.forward_parameters(observation).names == observation.names
    assert "command" not in observation
    package = export_package(policy, tmp_path / "temporal")
    loaded, _, _ = load_inference_package(package)
    reloaded_actions = loaded(observation)
    assert torch.equal(reloaded_actions["action"], expected)
    assert reloaded_actions.names == observation.names

    # Equal sizes must not hide an environment/time transpose.
    square = observation[:, :2]
    graph_forward = policy.pipeline.forward

    def swapped_graph(inputs, labels):
        parameters, labels = graph_forward(inputs, labels)
        return parameters.transpose(0, 1), labels

    with monkeypatch.context() as patch:
        patch.setattr(policy.pipeline, "forward", swapped_graph)
        with pytest.raises(ValueError, match="preserve.*axes"):
            policy(square)

    decode = policy.action_adapter.forward

    def swapped_decoder(parameters, **kwargs):
        return decode(parameters, **kwargs).transpose(0, 1)

    # Decoders without names are filled from the validated graph; explicit wrong names fail.
    def renamed_decoder(parameters, **kwargs):
        actions = decode(parameters, **kwargs)
        actions.names = ["time", "environment"]
        return actions

    with monkeypatch.context() as patch:
        patch.setattr(policy.action_adapter, "forward", renamed_decoder)
        with pytest.raises(ValueError, match="changed named"):
            policy(square)
    with monkeypatch.context() as patch:
        patch.setattr(policy.action_adapter, "forward", swapped_decoder)
        with pytest.raises(ValueError, match="preserve.*axes"):
            policy(observation)
