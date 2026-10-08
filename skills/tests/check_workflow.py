"""Independent artifact checks: external .venv/bin/python -I PATH MODE, from its project."""

import argparse
import json
import math
from pathlib import Path

import torch
from tensordict import TensorDict

from nexuml.core.compiler import compile
from nexuml.core.components import LayerBuildContext
from nexuml.core.config import ResolvedConfig
from nexuml.core.registry import get_component_registry
from nexuml.training.lightning import create_data_module_from_spec


def load_scenario(path):
    """Restore portable YAML without compiling or running training.

    Returns:
        The round-tripped native scenario.
    """
    config = ResolvedConfig.from_yaml(Path(path).read_text())
    scenario = config.to_scenario()
    assert ResolvedConfig.from_scenario(scenario).to_yaml() == config.to_yaml()
    return scenario


def partitions(scenario):
    """Inspect actual native loaders; never evaluate a final-test sample.

    Returns:
        Native data module and its verified partition membership.
    """
    data = create_data_module_from_spec(scenario)
    data.setup()
    train = list(data.train_dataloader().dataset.indices)
    validation = list(data.val_dataloader().dataset.indices)
    assert len(data.test_dataloader().dataset) == 0
    assert (len(train), len(validation)) == (48, 16)
    assert not set(train) & set(validation)
    assert set(train) | set(validation) == set(range(64))
    return data, {"train": train, "validation": validation, "test": []}


def component():
    """Check a fresh-discovered label-dependent MAE, including its compiled routing.

    Raises:
        AssertionError: If real execution accepts missing labels.
    """
    from agent_demo.components import AbsoluteError

    registry = get_component_registry()
    assert "agent_demo_mae" in {item.name for item in registry.entries()}
    assert not registry.errors, [error.short() for error in registry.errors]
    runtime = AbsoluteError().build(
        LayerBuildContext(
            input_sizes={"prediction": (2,)},
            keys_in=["prediction"],
            keys_out=["mae"],
            label_key="target",
        )
    )
    prediction = torch.tensor([[1.0, 3.0], [-2.0, 5.0]], requires_grad=True)
    target = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
    original = prediction.detach().clone()
    result, _ = runtime(
        TensorDict({"prediction": prediction}, batch_size=[2]),
        TensorDict({"target": target}, batch_size=[2]),
    )
    torch.testing.assert_close(result["mae"], torch.tensor([1.5, 2.5]))
    torch.testing.assert_close(result["prediction"], original)
    result["mae"].sum().backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[0.5, 0.5], [-0.5, 0.5]]))
    try:
        runtime(TensorDict({"prediction": prediction.detach()}, batch_size=[2]))
    except (ValueError, KeyError):
        pass
    else:
        raise AssertionError("Missing real labels were accepted")
    print("PASS: numerical MAE/gradients, preserved prediction, missing real labels")
    scenario = load_scenario("results/scenario.yaml")
    assert any(
        isinstance(layer.component, AbsoluteError)
        and layer.keys_in == ["prediction"]
        and "prediction" not in layer.keys_out
        and layer.label_key == "target"
        for stage in scenario.pipeline.stages.values()
        for layer in stage
    )
    pipeline = compile(scenario)
    data, _ = partitions(scenario)
    x, y = next(iter(data.val_dataloader()))
    features = x["features"].clone()
    output, _ = pipeline(x, y)
    torch.testing.assert_close(output["features"], features)
    for stage in scenario.pipeline.stages.values():
        for layer in stage:
            if isinstance(layer.component, AbsoluteError):
                torch.testing.assert_close(
                    output[layer.keys_out[0]],
                    (output["prediction"] - y["target"]).abs().mean(dim=-1),
                )
    print("PASS: fresh discovery, portable YAML, native compiled MAE routing and actual splits")


def paper():
    """Check exact synthetic reference parity and native, non-overwriting routing."""
    from agent_demo.paper import RMSNorm

    module = RMSNorm(eps=1e-6)
    assert not list(module.parameters()), "The reference has no learned scale"
    x = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 4.0],
            [-1.0, 1.0, -1.0, 1.0],
            [1e-4, 2e-4, -1e-4, 0.0],  # distinguish epsilon placement at small scale
        ]
    )
    expected = x / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(module(x), expected, atol=1e-6, rtol=1e-6)
    differentiable = x[1:].clone().requires_grad_()
    module(differentiable).sum().backward()
    assert torch.isfinite(differentiable.grad).all()
    print("PASS: reference equation, zero/small inputs, epsilon placement, finite gradients")
    scenario = load_scenario("results/scenario.yaml")
    pipeline = compile(scenario)
    data, membership = partitions(scenario)
    batch_x, batch_y = next(iter(data.val_dataloader()))
    features = batch_x["features"].clone()
    output, _ = pipeline(batch_x, batch_y)
    torch.testing.assert_close(output["features"], features)
    torch.testing.assert_close(
        output["normalized"], features / torch.sqrt(features.square().mean(-1, keepdim=True) + 1e-6)
    )
    print("PASS: reference equation/zero/gradients, portable routing, actual 48/16/0 splits")
    print(json.dumps(membership))


def research():
    """Recompute validation scores from all three saved model/config/cohort artifacts."""
    attempts = sorted(Path("runs").iterdir())
    assert len(attempts) == 3 and any(run.name == "baseline" for run in attempts)
    baseline = json.loads(Path("runs/baseline/metrics.json").read_text())
    baseline_scenario = load_scenario("runs/baseline/scenario.yaml").model_dump()
    observed = {}
    for run in attempts:
        evidence = json.loads((run / "metrics.json").read_text())
        assert evidence["state"] == "completed" and evidence["epochs"] == 1
        assert evidence["global_step"] == 6 and 0.001 <= evidence["lr"] <= 0.03
        assert evidence["environment"] == baseline["environment"]
        scenario = load_scenario(run / "scenario.yaml")
        assert scenario.training.lr == evidence["lr"]
        dumped = scenario.model_dump()
        dumped["name"] = baseline_scenario["name"]
        dumped["training"]["lr"] = baseline_scenario["training"]["lr"]
        assert dumped == baseline_scenario, "Changed more than the permitted learning rate/name"
        pipeline = compile(scenario)
        assert all(
            group["lr"] == evidence["lr"] for group in pipeline.create_optimizer().param_groups
        )
        pipeline.load_state_dict(torch.load(run / "model.pt", weights_only=True))
        pipeline.eval()
        data, membership = partitions(scenario)
        assert membership == baseline["partitions"] == evidence["partitions"]
        squared_sum = elements = 0
        with torch.no_grad():
            for x, y in data.val_dataloader():
                output, _ = pipeline(x, y)
                error = output["prediction"] - y["target"]
                squared_sum += error.square().sum().item()
                elements += error.numel()
        measured = squared_sum / elements
        assert math.isclose(measured, evidence["validation_loss"], rel_tol=1e-6)
        observed[run.name] = measured
    print("PASS: three native attempts, unchanged protocol/cohort, recomputed validation scores")
    print(json.dumps({"metrics": observed, "best_run": min(observed, key=observed.get)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["component", "paper", "research"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    {"component": component, "paper": paper, "research": research}[args.mode]()
