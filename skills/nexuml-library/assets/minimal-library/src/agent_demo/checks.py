"""Run with the selected environment: python -m agent_demo.checks."""

import math

import torch
from pydantic import ValidationError
from tensordict import TensorDict

from nexuml.core.compiler import compile
from nexuml.core.components import EvalBuildContext, LayerBuildContext
from nexuml.core.config import ResolvedConfig
from nexuml.core.registry import get_component_registry
from nexuml.core.scenario_registry import get_scenario_registry
from nexuml.training.lightning import create_data_module_from_spec

from agent_demo.components import PairedVectors, RootMeanSquaredError, SquaredError
from agent_demo.scenarios import regression


def main():
    """Verify the installed example's numerical and persistence contracts.

    Raises:
        AssertionError: If a contract check fails.
    """
    try:
        PairedVectors(width=0)
    except ValidationError:
        pass
    else:
        raise AssertionError("Invalid width was accepted.")

    dataset = PairedVectors(samples=8).build()
    x, y = dataset[0]
    assert len(dataset) == 8 and x.batch_size == torch.Size([])
    assert x["features"].shape == (4,) and y["target"].shape == (4,)
    torch.testing.assert_close(y["target"], 2 * x["features"])
    torch.testing.assert_close(
        dataset[0][0]["features"], PairedVectors(samples=8).build()[0][0]["features"]
    )

    runtime = SquaredError().build(
        LayerBuildContext(
            input_sizes={"prediction": (2,)},
            keys_in=["prediction"],
            keys_out=["mse"],
            label_key="target",
        )
    )
    try:
        runtime.forward_tensor(torch.zeros(1, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("Missing real labels were silently accepted.")
    prediction = torch.tensor([[1.0, 3.0]], requires_grad=True)
    target = torch.tensor([[0.0, 1.0]])
    loss = runtime.forward_tensor(prediction, target)
    torch.testing.assert_close(loss, torch.tensor([2.5]))
    loss.sum().backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[1.0, 2.0]]))

    evaluator = RootMeanSquaredError().build(EvalBuildContext("prediction", "target"))
    for values in ([[1.0]], [[3.0], [3.0]]):
        value = torch.tensor(values)
        evaluator.eval_batch(
            TensorDict({"prediction": value}, batch_size=[len(value)]),
            TensorDict({"target": torch.zeros_like(value)}, batch_size=[len(value)]),
        )
    assert math.isclose(evaluator.results()["rmse"], math.sqrt(19 / 3))

    registry = get_component_registry()
    assert {"agent_demo_pairs", "agent_demo_mse", "agent_demo_rmse"} <= {
        e.name for e in registry.entries()
    }
    assert not registry.errors, [error.short() for error in registry.errors]
    assert "agent-demo-regression" in get_scenario_registry().list()
    scenario = regression()
    config = ResolvedConfig.from_scenario(scenario)
    restored = ResolvedConfig.from_yaml(config.to_yaml()).to_scenario()
    assert list(restored.pipeline.stages) == ["predict", "objective"]
    assert restored.data.test_split == 0.0
    assert ResolvedConfig.from_scenario(restored).to_yaml() == config.to_yaml()
    assert "prediction" in compile(restored).input_sizes
    data = create_data_module_from_spec(restored)
    data.setup()
    train, validation, test = (
        loader.dataset
        for loader in (data.train_dataloader(), data.val_dataloader(), data.test_dataloader())
    )
    assert (len(train), len(validation), len(test)) == (48, 16, 0)
    assert not set(train.indices) & set(validation.indices)
    assert set(train.indices) | set(validation.indices) == set(range(64))
    batch_x, batch_y = next(iter(data.train_dataloader()))
    assert batch_x["features"].shape == batch_y["target"].shape == (8, 4)
    torch.testing.assert_close(batch_y["target"], 2 * batch_x["features"])
    print("PASS: parameters, batches/splits, loss/gradients, RMSE, discovery, YAML, compile")


if __name__ == "__main__":
    main()
