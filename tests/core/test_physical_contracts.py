"""Physical meaning is portable and checked before allocating an evaluation runtime."""

import json
from io import BytesIO

import pytest
from pydantic import ValidationError

from nexuml.core.serialization import lower_model, restore_model_data
from nexuml.core.types import InteractionContract, PhysicalTensorContract, TensorFieldContract
from nexuml.training.reinforcement import _validate_evaluation_contract
from nexuml_library.environments.isaaclab import IsaacLabEnvironment


def _contract():
    return IsaacLabEnvironment(task="NexuML-SO101-Joint-Reach").describe()


def test_existing_contract_owners_roundtrip_without_native_imports():
    contract = _contract()
    assert InteractionContract.model_validate_json(contract.model_dump_json()) == contract
    restored = restore_model_data(lower_model(contract), InteractionContract)
    assert InteractionContract.model_validate(restored) == contract
    assert contract.physical is not None
    assert contract.physical.asset_id.startswith("sha256:")
    action = contract.actions["action"].physical
    observation = contract.observations["observation"].physical
    assert action is not None and observation is not None
    assert action.joints == observation.joints[:5]
    assert action.control_mode == "absolute_position" and action.scale == (0.5,)
    assert observation.units == ("rad",) * 5 + ("rad/s",) * 5 + ("rad",) * 5
    assert observation.sensor_id == "simulation:joint-state-and-task-goal"


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": 2},
        {"units": ()},
        {"units": (" ",)},
        {"frame": " "},
        {"scale": (1.0,)},
        {"control_mode": "absolute_position"},
        {"control_mode": "absolute_position", "joints": ("joint",)},
        {"control_mode": "effort", "joints": ("a", "a"), "scale": (1.0,)},
        {"control_mode": "effort", "joints": ("a",), "scale": (0.0,)},
        {"control_mode": "effort", "joints": ("a",), "scale": (float("nan"),)},
        {"sensor_id": ""},
    ],
)
def test_malformed_physical_metadata_is_rejected(overrides):
    with pytest.raises(ValidationError):
        PhysicalTensorContract.model_validate({"units": ("rad",), "frame": "joint", **overrides})


@pytest.mark.parametrize("field", ["units", "joints", "scale", "offset"])
def test_per_element_metadata_matches_the_action_axis(field):
    original = _contract().actions["action"].model_dump()
    original["physical"][field] = ("rad", "rad") if field in {"units", "joints"} else (1.0, 1.0)
    with pytest.raises(ValidationError, match="Physical"):
        TensorFieldContract.model_validate(original)


def test_new_physical_rate_cannot_contradict_the_control_period():
    contract = _contract().model_dump()
    contract["actions"]["action"]["rate_hz"] = 60
    with pytest.raises(ValidationError, match="action rate"):
        InteractionContract.model_validate(contract)


@pytest.mark.parametrize(
    "change",
    [
        "joints",
        "units",
        "frame",
        "control_mode",
        "scale",
        "offset",
        "sensor_id",
        "calibration_id",
        "embodiment_id",
        "asset_id",
        "asset_source",
        "bounds",
        "rate",
        "missing_field_metadata",
        "missing_system_metadata",
    ],
)
def test_same_width_physical_mismatches_fail_compatibility(change):
    expected = _contract()
    actual = expected.model_dump()
    action = actual["actions"]["action"]
    physical = action["physical"]
    if change == "joints":
        physical[change] = list(reversed(physical[change]))
    elif change == "units":
        physical[change] = ("degree",)
    elif change == "frame":
        physical[change] = "world"
    elif change == "control_mode":
        physical[change] = "relative_position"
    elif change in {"scale", "offset"}:
        physical[change] = (1.0,)
    elif change == "sensor_id":
        actual["observations"]["observation"]["physical"][change] = "different-encoder"
    elif change in {"calibration_id", "embodiment_id", "asset_id", "asset_source"}:
        actual["physical"][change] = "different-identity"
    elif change == "bounds":
        action["high"] = 0.5
    elif change == "rate":
        # Construct a different valid rate/period pair rather than malformed metadata.
        action["rate_hz"], actual["control_period_s"] = 60, 1 / 60
    elif change == "missing_field_metadata":
        action["physical"] = None
    else:
        actual["physical"] = None
    with pytest.raises(ValueError, match="Evaluation"):
        _validate_evaluation_contract(expected, InteractionContract.model_validate(actual))


def test_legacy_absence_keeps_its_previous_checks_without_claiming_physical_qualification():
    current = _contract()
    data = json.loads(current.model_dump_json())
    data.pop("physical")
    for fields in (data["observations"], data["actions"]):
        for field in fields.values():
            field.pop("physical")
    old = InteractionContract.model_validate(data)
    assert old.physical is None and old.actions["action"].physical is None
    _validate_evaluation_contract(old, current)
    with pytest.raises(ValueError, match="physical system identity"):
        _validate_evaluation_contract(current, old)


def test_current_inference_archive_contract_is_compared_by_value(tmp_path):
    from nexuml.core.compiler import compile
    from nexuml.core.export import export_package, load_inference_package
    from nexuml.core.policy import CompiledPolicy
    from nexuml_library.scenarios.reinforcement.isaaclab import isaac_so101_joint_reach_ppo

    scenario = isaac_so101_joint_reach_ppo()
    assert scenario.policy is not None
    contract = _contract()
    policy = CompiledPolicy(
        compile(scenario), scenario.policy.action_adapter.build(contract), contract
    )
    export_package(policy, tmp_path / "policy")
    restored, _, _ = load_inference_package(tmp_path / "policy")
    assert type(restored.contract) is not InteractionContract
    _validate_evaluation_contract(restored.contract, contract)


def test_actual_legacy_archive_without_new_attributes_keeps_compatibility():
    from torch.package import PackageExporter, PackageImporter

    # A real isolated pre-metadata model, without relying on Git history or another environment.
    source = """
from pydantic import BaseModel
class TensorField(BaseModel):
    shape: tuple[int, ...]
    dtype: str
    modality: str
    low: float | list[float] | None = None
    high: float | list[float] | None = None
    num_values: int | None = None
    rate_hz: float | None = None
    description: str | None = None
class Contract(BaseModel):
    observations: dict[str, TensorField]
    actions: dict[str, TensorField]
    action_space: str
    num_envs: int
    control_period_s: float | None = None
"""
    buffer = BytesIO()
    with PackageExporter(buffer) as exporter:
        exporter.extern("pydantic.**")
        exporter.save_source_string("legacy_contract", source)
    buffer.seek(0)
    old_module = PackageImporter(buffer).import_module("legacy_contract")
    current = _contract()
    data = current.model_dump()
    data.pop("physical")
    for fields in (data["observations"], data["actions"]):
        for field in fields.values():
            field.pop("physical")
    old = old_module.Contract.model_validate(data)
    assert not hasattr(old, "physical") and not hasattr(old.actions["action"], "physical")
    _validate_evaluation_contract(old, current)
