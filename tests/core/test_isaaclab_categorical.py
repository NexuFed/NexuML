"""CPU decoding proof only; native categorical control requires its separate gate."""

import builtins
import importlib

import pytest
import torch
from tensordict import TensorDict

from nexuml_library.environments import _isaaclab_categorical as categorical


def test_classes_decode_to_exact_declared_effort_without_changing_inputs():
    actions = torch.tensor([0, 1, 2, 2, 0], dtype=torch.int64)
    original = actions.clone()
    forces = categorical.decode_categorical_force(actions)
    assert torch.equal(forces, torch.tensor([[-100.0], [0.0], [100.0], [100.0], [-100.0]]))
    assert forces.dtype == torch.float32 and forces.device == actions.device
    assert torch.equal(actions, original)
    assert categorical.ACTION_FORCES == (-100.0, 0.0, 100.0)


@pytest.mark.parametrize(
    "actions",
    [
        torch.tensor([0.0]),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([True]),
        torch.tensor([[0]]),
        torch.tensor(0),
        torch.tensor([-1]),
        torch.tensor([3]),
    ],
)
def test_invalid_categorical_commands_reject_before_control(actions):
    with pytest.raises(ValueError, match="Categorical Cartpole"):
        categorical.decode_categorical_force(actions)


def test_categorical_definition_import_is_native_and_cuda_inert(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("isaaclab", "isaacsim", "omni", "pxr", "viser", "newton")):
            raise AssertionError("Unexpected native import: " + name)
        return original(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Static import cannot inspect or allocate CUDA")

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    importlib.reload(categorical)
    assert callable(categorical.make_categorical_cartpole_config)
    assert categorical.categorical_cartpole_type.cache_info().currsize == 0


def test_categorical_recipe_roundtrip_and_complete_policy_compile_without_native(
    monkeypatch, tmp_path
):
    from nexuml.core.compiler import compile
    from nexuml.core.config import ResolvedConfig
    from nexuml.core.policy import CompiledPolicy
    from nexuml.core.export import export_package, load_inference_package
    from nexuml_library.scenarios.reinforcement.isaaclab_categorical import (
        isaac_categorical_cartpole_ppo,
    )

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("isaaclab", "isaacsim", "omni", "pxr", "viser", "newton")):
            raise AssertionError("Static categorical path imported " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    recipe = isaac_categorical_cartpole_ppo(total_frames=32, frames_per_batch=16, num_envs=2)
    restored = ResolvedConfig.from_yaml(
        ResolvedConfig.from_scenario(recipe).to_yaml()
    ).to_scenario()
    assert restored == recipe
    assert restored.interaction is not None
    contract = restored.interaction.environment.describe()
    physical = contract.actions["action"].physical
    assert physical is not None and physical.class_controls == ((-100.0,), (0.0,), (100.0,))
    policy = CompiledPolicy.from_pipeline(compile(restored))
    output = policy(TensorDict({"observation": torch.zeros(2, 4)}, batch_size=(2,)))
    assert output["action"].shape == (2,) and output["action"].dtype == torch.int64
    decoded = categorical.decode_categorical_force(output["action"])
    assert set(decoded.flatten().tolist()) <= {-100.0, 0.0, 100.0}
    package = export_package(policy, tmp_path / "categorical-policy")
    loaded, _, metadata = load_inference_package(package)
    actual = loaded(TensorDict({"observation": torch.zeros(2, 4)}, batch_size=(2,)))
    assert torch.equal(actual["action"], output["action"])
    assert not metadata["training_state_available"]
    dependencies = (package / "requirements.txt").read_text()
    assert not any(
        name in dependencies for name in ("isaaclab", "isaacsim", "torchrl", "viser", "newton")
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"shape": (1,)},
        {"dtype": "float32"},
        {"num_values": 2},
        {
            "physical": {
                "units": ("N",),
                "frame": "joint",
                "joints": ("slider_to_cart",),
                "control_mode": "effort",
                "scale": (1.0,),
                "class_controls": ((-100.0, 0.0),),
            }
        },
    ],
)
def test_class_control_metadata_rejects_incompatible_event_or_table(changes):
    from pydantic import ValidationError
    from nexuml.core.types import TensorFieldContract

    data = {
        "shape": (),
        "dtype": "int64",
        "num_values": 3,
        "physical": {
            "units": ("N",),
            "frame": "joint",
            "joints": ("slider_to_cart",),
            "control_mode": "effort",
            "scale": (1.0,),
            "class_controls": ((-100.0,), (0.0,), (100.0,)),
        },
    }
    with pytest.raises(ValidationError, match="Categorical"):
        TensorFieldContract.model_validate({**data, **changes})
