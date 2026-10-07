from __future__ import annotations

import torch
from tensordict import TensorDict

from nexuml.core.components import LayerBuildContext
from nexuml.core.compiler import compile
from nexuml.core.config import ResolvedConfig
from nexuml.training.lightning import create_runtime_artifacts
from nexuml_library.evaluation.visualizers.flow import FlowVisualizer
from nexuml_library.layers.generative.flow.euler import EulerIntegrator
from nexuml_library.layers.generative.flow.linear_path import LinearFlowPath, _linear_path
from nexuml_library.layers.generative.flow.vector_field import TimeConditionedVectorField
from nexuml_library.layers.loss.flow_matching_loss import FlowMatchingLoss
from nexuml_library.layers.model.flow import Flow
from nexuml_library.scenarios.flow.synthetic_flow_matching import synthetic_flow_matching


def test_linear_path_equations_are_exact() -> None:
    x0 = torch.tensor([[0.0, 2.0], [1.0, -1.0]])
    x1 = torch.tensor([[2.0, 4.0], [5.0, 3.0]])
    t = torch.tensor([[0.25], [0.75]])

    state, velocity = _linear_path(x0, x1, t)

    assert torch.allclose(state, torch.tensor([[0.5, 2.5], [4.0, 2.0]]))
    assert torch.equal(velocity, torch.tensor([[2.0, 2.0], [4.0, 4.0]]))


def test_linear_path_runtime_obeys_tensordict_contract() -> None:
    runtime = LinearFlowPath().build(
        LayerBuildContext(
            input_sizes={"features": (2,)},
            keys_in=["features"],
            keys_out=["flow_state", "flow_time", "flow_target_velocity"],
        )
    )
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x = TensorDict({"features": features.clone()}, batch_size=[2])

    out, _ = runtime(x)
    t = out["flow_time"]
    target_velocity = out["flow_target_velocity"]
    recovered_x0 = features - target_velocity

    assert out["flow_state"].shape == features.shape
    assert t.shape == (2, 1)
    assert target_velocity.shape == features.shape
    assert torch.all((t >= 0.0) & (t <= 1.0))
    expected_state, _ = _linear_path(recovered_x0, features, t)
    assert torch.allclose(out["flow_state"], expected_state)


def test_flow_matching_loss_matches_analytical_mse() -> None:
    runtime = FlowMatchingLoss().build(
        LayerBuildContext(
            input_sizes={"predicted": (2,), "target": (2,)},
            keys_in=["predicted", "target"],
            keys_out=["flow_matching_loss"],
        )
    )
    x = TensorDict(
        {
            "predicted": torch.tensor([[1.0, 3.0], [2.0, 2.0]]),
            "target": torch.tensor([[0.0, 1.0], [1.0, 4.0]]),
        },
        batch_size=[2],
    )

    out, _ = runtime(x)
    assert torch.allclose(out["flow_matching_loss"], torch.tensor([2.5, 2.5]))


def test_vector_field_preserves_shape_and_has_gradients() -> None:
    runtime = TimeConditionedVectorField(hidden_dims=[8]).build(
        LayerBuildContext(
            input_sizes={"flow_state": (2,), "flow_time": (1,)},
            keys_in=["flow_state", "flow_time"],
            keys_out=["predicted_velocity"],
        )
    )
    state = torch.randn(4, 2)
    x = TensorDict(
        {"flow_state": state, "flow_time": torch.rand(4, 1)},
        batch_size=[4],
    )

    out, _ = runtime(x)
    prediction = out["predicted_velocity"]
    prediction.pow(2).mean().backward()

    gradients = [parameter.grad for parameter in runtime.parameters()]
    assert prediction.shape == state.shape
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
    assert any(gradient.abs().sum() > 0 for gradient in gradients if gradient is not None)


def test_euler_integrator_is_a_pipeline_layer() -> None:
    integrator = EulerIntegrator(
        velocity=lambda state, time: torch.full_like(state, 2.0),
        num_steps=20,
        input_sizes={"state": (2,)},
        keys_in=["state"],
        keys_out=["sample"],
    )
    result = integrator.forward_tensor(torch.zeros(3, 2))

    assert torch.allclose(result, torch.full((3, 2), 2.0), atol=1e-6)


def test_flow_scenario_round_trips_and_compiles() -> None:
    scenario = synthetic_flow_matching(
        num_samples=32,
        hidden_dims=[8],
        batch_size=8,
        max_epochs=1,
    )

    restored = ResolvedConfig.from_yaml(ResolvedConfig.from_scenario(scenario).to_yaml())
    pipeline = compile(scenario)

    assert isinstance(restored.pipeline.stages["FlowPath"][0].component, LinearFlowPath)
    assert isinstance(restored.pipeline.stages["Flow"][0].component, Flow)
    assert isinstance(restored.pipeline.stages["Loss"][0].component, FlowMatchingLoss)
    assert "flow_matching_loss" in pipeline.loss_keys
    assert isinstance(scenario.evaluation.algorithms[0].algorithm, FlowVisualizer)


def test_flow_scenario_one_step_and_sampling() -> None:
    scenario = synthetic_flow_matching(
        num_samples=32,
        hidden_dims=[8],
        batch_size=8,
        max_epochs=1,
    )
    artifacts = create_runtime_artifacts(scenario)
    artifacts.data_module.setup("fit")
    x, y = next(iter(artifacts.data_module.train_dataloader()))

    x_out, _ = artifacts.lightning_module.pipeline(x, y)
    loss, _ = artifacts.lightning_module._compute_loss(x_out)
    loss.backward()

    flow = next(
        layer
        for stage_name, _layer_name, layer in artifacts.pipeline.iter_layers()
        if stage_name == "Flow"
    )
    samples = getattr(flow, "sample")(torch.randn(4, 2))
    assert flow.integrator.last_trajectory is not None

    assert torch.isfinite(loss)
    assert samples.shape == (4, 2)
    assert torch.isfinite(samples).all()
