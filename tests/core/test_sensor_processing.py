"""Portable sensor meaning, explicit history ownership and clean inference parity."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import cast

import pytest
import torch
from pydantic import ValidationError
from tensordict import TensorDict

from nexuml.core.compiler import compile
from nexuml.core.components import LayerBuildContext
from nexuml.core.config import ResolvedConfig
from nexuml.core.export import export_package, load_inference_package
from nexuml.core.policy import CompiledPolicy
from nexuml.core.serialization import lower_model
from nexuml.core.types import (
    DataSpec,
    InteractionContract,
    InteractionSpec,
    LayerSpec,
    PipelineSpec,
    PolicySpec,
    RLEvaluationSpec,
    ReinforcementLearningSpec,
    ScenarioSpec,
    SensorProcessingContract,
    TensorFieldContract,
)
from nexuml_library.action_adapters.adapters import DirectActionAdapter
from nexuml_library.environments.gymnasium import GymnasiumEnvironment
from nexuml_library.layers.model.linear_encoder import LinearEncoder
from nexuml_library.layers.processing.sensor_history import SensorHistoryProcessor


def _sensor(**overrides):
    return SensorProcessingContract.model_validate(
        {
            "owner": "package",
            "raw_shape": (2,),
            "raw_dtype": "float32",
            "raw_layout": "vector",
            "units": "rad",
            **overrides,
        }
    )


def _contract(fields=None):
    return InteractionContract(
        observations=fields
        or {
            "state": TensorFieldContract(
                shape=(6,),
                dtype="float32",
                modality="state",
                sensor=_sensor(history_length=3, scale=2.0, offset=-1.0),
            ),
            "rgb": TensorFieldContract(
                shape=(6, 2, 2),
                dtype="float32",
                modality="rgb",
                sensor=_sensor(
                    raw_shape=(2, 2, 3),
                    raw_layout="HWC",
                    raw_dtype="uint8",
                    units="intensity",
                    scale=1 / 255,
                    history_length=2,
                ),
            ),
            "depth": TensorFieldContract(
                shape=(4, 2, 2),
                dtype="float32",
                modality="depth",
                sensor=_sensor(
                    raw_shape=(2, 2),
                    raw_layout="HW",
                    units="m",
                    depth_representation="z_depth",
                    invalid_depth="mask_zero",
                    depth_min=0.0,
                    depth_max=10.0,
                    history_length=2,
                    scale=0.1,
                    offset=1.0,
                ),
            ),
            "segmentation": TensorFieldContract(
                shape=(4, 2, 2),
                dtype="float32",
                modality="segmentation",
                sensor=_sensor(
                    raw_shape=(1, 2, 2),
                    raw_layout="CHW",
                    raw_dtype="int64",
                    units="class_id",
                    labels=((7, "object"), (0, "background")),
                    label_set_id="scene-v1",
                    history_length=2,
                ),
            ),
            "plain": TensorFieldContract(shape=(1,), dtype="float64"),
        },
        actions={"action": TensorFieldContract(shape=(2,), dtype="float32")},
        action_space="continuous",
    )


def _runtime(contract=None):
    contract = contract or _contract()
    return SensorHistoryProcessor().build(
        LayerBuildContext(
            input_sizes={key: field.shape for key, field in contract.observations.items()},
            keys_in=list(contract.observations),
            keys_out=list(contract.observations),
            metadata={"interaction_contract": contract.model_dump(mode="json")},
        )
    )


def _observation(value=1, batch=()):
    leaves = {
        "state": torch.tensor([value, value + 1], dtype=torch.float32),
        "rgb": (torch.arange(12, dtype=torch.uint8) + value).reshape(2, 2, 3),
        "depth": torch.tensor([[float("nan"), 1.0], [2.0, 30.0]]),
        "segmentation": torch.tensor([[[7, 0], [0, 7]]]),
        "plain": torch.tensor([value], dtype=torch.float64),
    }
    return TensorDict(
        {key: leaf.expand(*batch, *leaf.shape).clone() for key, leaf in leaves.items()},
        batch_size=batch,
    )


def _policy(contract=None, *, processor=True):
    contract = contract or _contract()
    spec = ScenarioSpec(
        name="portable-sensor-history",
        data=DataSpec(
            input_shapes={key: list(field.shape) for key, field in contract.observations.items()}
        ),
        interaction=InteractionSpec(
            environment=GymnasiumEnvironment(
                env_name="custom_sensor_task:inert-metadata",
                contract=contract,
            )
        ),
        policy=PolicySpec(
            action_adapter=DirectActionAdapter(source_key="command"),
            processor=SensorHistoryProcessor() if processor else None,
        ),
        pipeline=PipelineSpec(
            stages={
                "actor": [
                    LayerSpec(
                        component=LinearEncoder(output_dim=2),
                        keys_in=[next(iter(contract.observations))],
                        keys_out=["command"],
                    )
                ]
            }
        ),
    )
    return CompiledPolicy.from_pipeline(compile(spec), contract)


def _equal(left, right):
    assert left.batch_size == right.batch_size
    assert set(left.keys(True, True)) == set(right.keys(True, True))
    for key in left.keys(True, True):
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("batch", [(), (2,), (2, 3)])
def test_mixed_transforms_history_and_caller_owned_storage(batch):
    runtime = _runtime()
    raw, state = _observation(batch=batch), runtime.initialize_state(batch)
    before_raw, before_state = raw.clone(), state.clone()
    processed, following = runtime.step(raw, state)
    _equal(raw, before_raw)
    _equal(state, before_state)
    expected_vector = (raw["state"] * 2 - 1).repeat(*([1] * len(batch)), 3)
    assert torch.equal(processed["state"], expected_vector)
    expected_rgb = raw["rgb"].movedim(-1, -3).float() / 255
    torch.testing.assert_close(processed["rgb"], torch.cat((expected_rgb, expected_rgb), -3))
    depth = torch.tensor([[[0, 1.1], [1.2, 0]], [[0, 1], [1, 0]]])
    torch.testing.assert_close(processed["depth"], depth.repeat(2, 1, 1).expand(*batch, 4, 2, 2))
    labels = torch.tensor([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=torch.float32)
    assert torch.equal(processed["segmentation"], labels.repeat(2, 1, 1).expand(*batch, 4, 2, 2))
    assert torch.equal(processed["plain"], raw["plain"])
    assert processed["plain"].dtype == torch.float64
    assert all(torch.isfinite(processed[key]).all() for key in processed.keys())
    processed["state"].zero_()
    assert following["state", "frames"].count_nonzero() > 0
    following["state", "frames"].zero_()
    _equal(state, before_state)
    _equal(raw, before_raw)
    assert runtime.state_dict() == {} and not list(runtime.buffers())
    assert runtime.raw_fields["rgb"].shape == (2, 2, 3)
    assert runtime.raw_fields["plain"] == runtime.fields["plain"]


@pytest.mark.parametrize("initialization", ["repeat_first", "zeros"])
def test_selected_reset_interleaved_streams_and_oldest_first(initialization):
    contract = _contract(
        {
            "state": TensorFieldContract(
                shape=(6,),
                dtype="float32",
                modality="state",
                sensor=_sensor(history_length=3, initialization=initialization),
            )
        }
    )
    runtime = _runtime(contract)
    state, other = runtime.initialize_state((2,)), runtime.initialize_state((2,))
    first = TensorDict({"state": torch.tensor([[1.0, 2.0], [10.0, 20.0]])}, [2])
    processed, state = runtime.step(first, state)
    padding = [1.0, 2.0, 1.0, 2.0] if initialization == "repeat_first" else [0.0, 0.0, 0.0, 0.0]
    assert processed["state"][0].tolist() == padding + [1.0, 2.0]
    _, other = runtime.step(TensorDict({"state": first["state"] + 100}, [2]), other)
    original = state.clone()
    reset = runtime.reset_state(state, torch.tensor([True, False]))
    _equal(original, state)
    assert not reset["state", "initialized"][0]
    assert reset["state", "frames"][0].count_nonzero() == 0
    assert torch.equal(reset["state", "frames"][1], state["state", "frames"][1])
    second = TensorDict({"state": torch.tensor([[3.0, 4.0], [30.0, 40.0]])}, [2])
    processed, following = runtime.step(second, reset)
    oldest = [10.0, 20.0] if initialization == "repeat_first" else [0.0, 0.0]
    assert processed["state"][1].tolist() == oldest + [10.0, 20.0, 30.0, 40.0]
    assert processed["state"][0, -2:].tolist() == [3.0, 4.0]
    assert other["state", "frames"][0, -1].tolist() == [101.0, 102.0]
    reset["state", "frames"].zero_()
    assert following["state", "frames"][1].count_nonzero() == (
        6 if initialization == "repeat_first" else 4
    )


@pytest.mark.parametrize("modality", ["rgb", "depth", "segmentation"])
@pytest.mark.parametrize("layout", ["HWC", "CHW", "HW"])
def test_explicit_image_layouts(modality, layout):
    channels = 3 if modality == "rgb" else 1
    shape = {"HWC": (2, 3, channels), "CHW": (channels, 2, 3), "HW": (2, 3)}[layout]
    metadata = {
        "raw_shape": shape,
        "raw_layout": layout,
        "units": "m",
        "raw_dtype": "int64" if modality == "segmentation" else "float32",
    }
    if modality == "depth":
        metadata.update(depth_representation="range_depth", depth_min=0.0, depth_max=10.0)
    if modality == "segmentation":
        metadata.update(labels=((1, "one"), (0, "zero")))
        channels = 2
    field = TensorFieldContract(
        shape=(channels, 2, 3), dtype="float32", modality=modality, sensor=_sensor(**metadata)
    )
    if modality == "rgb" and layout == "HW":
        with pytest.raises(ValueError, match="RGB"):
            _runtime(_contract({"image": field}))
        return
    runtime = _runtime(_contract({"image": field}))
    raw = TensorDict({"image": torch.ones(shape, dtype=getattr(torch, metadata["raw_dtype"]))}, [])
    processed, _ = runtime.step(raw, runtime.initialize_state())
    expected = torch.ones(channels, 2, 3)
    if modality == "segmentation":
        expected[1].zero_()
    assert torch.equal(processed["image"], expected)


@pytest.mark.parametrize(
    "overrides",
    [
        {"owner": None},
        {"owner": "implicit"},
        {"raw_shape": (True,)},
        {"raw_shape": (1.5,)},
        {"raw_shape": (0,)},
        {"raw_dtype": " "},
        {"units": " "},
        {"scale": float("nan")},
        {"scale": "1"},
        {"offset": float("inf")},
        {"history_length": True},
        {"history_length": 0},
        {"history_order": "newest_first"},
        {"latency_steps": -1},
        {"assets": {"asset": "bad"}},
        {"labels": ((1, "a"), (1, "b"))},
        {"labels": ((1, "a"), (2, "a"))},
        {"labels": ((-1, "a"),)},
        {"labels": ((True, "a"),)},
        {"labels": ((0, " "),)},
        {"label_set_id": ""},
        {"depth_representation": "z_depth"},
        {"depth_representation": "z_depth", "depth_min": 0.0, "depth_max": 5.0, "units": "mm"},
        {"depth_min": -1.0},
        {"depth_min": 2.0, "depth_max": 1.0},
        {"unexpected": True},
    ],
)
def test_strict_sensor_metadata(overrides):
    with pytest.raises(ValidationError):
        _sensor(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"processor_id": "Other"},
        {"processor_version": "2"},
        {"timing": "pre_action"},
        {"latency_steps": 1},
        {"raw_layout": "HW"},
        {"raw_dtype": "complex64"},
        {"raw_shape": (3,)},
    ],
)
def test_unsupported_owned_semantics_reject_before_processing(overrides):
    with pytest.raises(ValueError):
        _runtime(
            _contract(
                {
                    "state": TensorFieldContract(
                        shape=(2,),
                        dtype="float32",
                        modality="state",
                        sensor=_sensor(**overrides),
                    )
                }
            )
        )


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "shape",
        "dtype",
        "state_nan",
        "rgb_inf",
        "plain_nan",
        "unknown_label",
        "batch",
    ],
)
def test_invalid_raw_boundary(failure):
    runtime, raw = _runtime(), _observation(batch=(2,))
    if failure == "missing":
        raw = raw.exclude("state")
    elif failure == "shape":
        raw["rgb"] = torch.zeros(2, 3, 2, 2, dtype=torch.uint8)
    elif failure == "dtype":
        raw["rgb"] = raw["rgb"].float()
    elif failure == "state_nan":
        raw["state"][0, 0] = float("nan")
    elif failure == "rgb_inf":
        # Even a declared floating RGB sensor must not inherit depth sanitization.
        field = runtime.fields["rgb"].model_dump()
        field["sensor"]["raw_dtype"] = "float32"
        fields = dict(runtime.fields)
        fields["rgb"] = TensorFieldContract.model_validate(field)
        runtime = _runtime(_contract(fields))
        raw["rgb"] = raw["rgb"].float()
        raw["rgb"][0, 0, 0, 0] = float("inf")
    elif failure == "plain_nan":
        raw["plain"][0, 0] = float("nan")
    elif failure == "unknown_label":
        raw["segmentation"][0, 0, 0, 0] = 99
    state = runtime.initialize_state((1,) if failure == "batch" else (2,))
    with pytest.raises(ValueError):
        runtime.step(raw, state)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0, 11.0])
def test_depth_rejects_invalid_unless_mask_explicit(invalid):
    field = TensorFieldContract(
        shape=(1, 1, 1),
        dtype="float32",
        modality="depth",
        sensor=_sensor(
            raw_shape=(1, 1),
            raw_layout="HW",
            units="m",
            depth_representation="z_depth",
            depth_min=0.0,
            depth_max=10.0,
        ),
    )
    runtime = _runtime(_contract({"depth": field}))
    with pytest.raises(ValueError):
        runtime.step(
            TensorDict({"depth": torch.tensor([[invalid]])}, []), runtime.initialize_state()
        )


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "extra",
        "dtype",
        "shape",
        "flag_dtype",
        "nan",
        "uninitialized",
        "no_device",
    ],
)
def test_invalid_state_is_rejected_on_both_step_and_reset(failure):
    runtime = _runtime()
    state = runtime.initialize_state((2,))
    if failure == "missing":
        state = state.exclude("state")
    elif failure == "extra":
        state["extra"] = torch.zeros(2)
    elif failure == "dtype":
        state["state", "frames"] = state["state", "frames"].double()
    elif failure == "shape":
        state["state", "frames"] = torch.zeros(2, 3, 3)
    elif failure == "flag_dtype":
        state["state", "initialized"] = torch.zeros(2)
    elif failure == "nan":
        state["state", "frames"][0, 0, 0] = float("nan")
    elif failure == "uninitialized":
        state["state", "frames"][0, 0, 0] = 1.0
    else:
        state = TensorDict(state.to_dict(), batch_size=[2])
    with pytest.raises(ValueError):
        runtime.step(_observation(batch=(2,)), state)
    with pytest.raises(ValueError):
        runtime.reset_state(state, torch.zeros(2, dtype=torch.bool))


@pytest.mark.parametrize(
    "mask",
    [
        torch.ones(2),
        torch.ones(2, 1, dtype=torch.bool),
        torch.tensor(True),
        torch.ones(2, dtype=torch.bool, device="meta"),
    ],
)
def test_reset_mask_shape_dtype_and_device_must_match_exactly(mask):
    runtime = _runtime()
    with pytest.raises(ValueError, match="Reset mask"):
        runtime.reset_state(runtime.initialize_state((2,)), mask)


def test_raw_and_state_devices_must_match_without_gpu():
    runtime = _runtime(
        _contract(
            {
                "state": TensorFieldContract(
                    shape=(2,),
                    dtype="float32",
                    sensor=_sensor(),
                )
            }
        )
    )
    raw = TensorDict({"state": torch.zeros(2, 2, device="meta")}, [2])
    # Meta has no values: reject device mismatch before trying to inspect them.
    with pytest.raises(ValueError, match="same device"):
        runtime.step(raw, runtime.initialize_state((2,)))


def test_scalar_reset_and_frozen_required_metadata():
    runtime = _runtime()
    _, state = runtime.step(_observation(), runtime.initialize_state())
    kept = runtime.reset_state(state, torch.tensor(False))
    _equal(kept, state)
    reset = runtime.reset_state(state, torch.tensor(True))
    _equal(reset, runtime.initialize_state())
    assert state["state", "initialized"]
    with pytest.raises(ValidationError, match="frozen"):
        _sensor().history_length = 2
    with pytest.raises(ValidationError, match="owner"):
        SensorProcessingContract(
            raw_shape=(2,), raw_dtype="float32", raw_layout="vector", units="rad"
        )  # ty: ignore[missing-argument]
    sensor = _sensor(assets={"vocabulary": "a" * 64})
    assert SensorProcessingContract.model_validate_json(sensor.model_dump_json()) == sensor
    for batch in (-1, True, (1.5,), (True,), "2"):
        with pytest.raises(ValueError, match="batch_size"):
            runtime.initialize_state(batch)


def test_external_processing_is_declared_processed_passthrough_and_stateless_is_unchanged():
    field = TensorFieldContract(
        shape=(6,),
        dtype="float32",
        modality="state",
        sensor=_sensor(
            owner="external",
            history_length=3,
            timing="pre_action",
            latency_steps=2,
        ),
    )
    contract = _contract({"state": field})
    runtime = _runtime(contract)
    assert runtime.raw_fields["state"] == field
    state = runtime.initialize_state((2,))
    assert not list(state.keys())
    processed = TensorDict({"state": torch.arange(12.0).reshape(2, 6)}, [2])
    output, following = runtime.step(processed, state)
    _equal(output, processed)
    assert following.batch_size == torch.Size([2])
    with pytest.raises(ValueError, match="shape"):
        runtime.step(TensorDict({"state": torch.zeros(2, 2)}, [2]), state)
    policy = _policy(contract)
    ordinary = policy(processed)
    stepped, _ = policy.step(processed, policy.initialize_state((2,)))
    _equal(ordinary, stepped)
    without = CompiledPolicy(policy.pipeline, policy.action_adapter, contract)
    _equal(ordinary, without(processed))
    assert _policy(contract, processor=False).processor is None
    for call in (
        lambda: without.initialize_state(),
        lambda: without.step(processed, state),
        lambda: without.reset_state(state, torch.tensor([True, False])),
    ):
        with pytest.raises(ValueError, match="explicit processor"):
            call()


def test_package_owned_sensor_assembly_requires_a_processor():
    with pytest.raises(ValueError, match="Package-owned.*processor"):
        _policy(processor=False)


def test_direct_policy_construction_cannot_drop_package_owned_processing():
    policy = _policy()
    with pytest.raises(ValueError, match="Package-owned.*processor"):
        CompiledPolicy(policy.pipeline, policy.action_adapter, policy.contract)


def test_finite_policy_output_and_affine_overflow_are_not_silently_accepted():
    policy = _policy()
    parameter = next(policy.pipeline.parameters())
    with torch.no_grad():
        parameter.fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        policy.step(_observation(), policy.initialize_state())
    runtime = _runtime(
        _contract(
            {
                "state": TensorFieldContract(
                    shape=(2,),
                    dtype="float32",
                    sensor=_sensor(scale=1e300),
                )
            }
        )
    )
    with pytest.raises(ValueError, match="nonfinite"):
        runtime.step(TensorDict({"state": torch.ones(2)}, []), runtime.initialize_state())


def test_sensor_semantics_roundtrip_and_order_are_not_lost():
    contract = _contract()
    restored = InteractionContract.model_validate_json(contract.model_dump_json())
    assert restored == contract
    labels = restored.observations["segmentation"].sensor
    assert labels is not None and labels.labels == ((7, "object"), (0, "background"))
    policy = _policy()
    config = policy.pipeline.resolved_config
    loaded = ResolvedConfig.from_yaml(config.to_yaml())
    assert loaded.policy is not None
    assert isinstance(loaded.policy.processor, SensorHistoryProcessor)
    assert loaded.policy.processor.model_dump() == {}
    for changed in ("units", "labels", "history_length", "processor_version", "timing"):
        value = contract.model_dump(mode="json")
        sensor = value["observations"]["segmentation"]["sensor"]
        sensor[changed] = {
            "units": "other",
            "labels": [[0, "background"], [7, "object"]],
            "history_length": 1,
            "processor_version": "2",
            "timing": "pre_action",
        }[changed]
        assert InteractionContract.model_validate(value) != contract


@pytest.mark.parametrize("change", ["units", "labels", "owner", "processor_version", "missing"])
def test_evaluation_rejects_equal_shape_sensor_semantic_mismatches(change):
    from nexuml.evaluation.policy import _validate_evaluation_contract

    expected = _contract()
    value = expected.model_dump(mode="json")
    sensor = value["observations"]["segmentation"]["sensor"]
    if change == "missing":
        value["observations"]["segmentation"].pop("sensor")
    else:
        sensor[change] = {
            "units": "different_ids",
            "labels": [[0, "background"], [7, "object"]],
            "owner": "external",
            "processor_version": "2",
        }[change]
    with pytest.raises(ValueError, match="Evaluation"):
        _validate_evaluation_contract(expected, InteractionContract.model_validate(value))


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def test_existing_so101_semantics_and_physical_schema_are_unchanged():
    from nexuml_library.environments.isaaclab import IsaacLabEnvironment
    from nexuml_library.scenarios.imitation.so101 import _task_identity

    current_task_identity = _task_identity()
    assert _hash(current_task_identity) == (
        "fee5075eb8da77c2192f1577752a883c58da678d992e9940a85072e047d4d4a1"
    )
    contract = IsaacLabEnvironment(task="NexuML-SO101-Joint-Reach").describe()
    assert "sensor" not in contract.observations["observation"].model_dump()
    assert "processor" not in PolicySpec(action_adapter=DirectActionAdapter()).model_dump()
    assert _hash(lower_model(contract)) == (
        "b623a00a33c0ff6ac84cb4519f945e6954f093cce0dc9a238f2ebfd2150b7bf9"
    )


def test_sdk_and_ppo_source_bytes_are_unchanged():
    root = Path(__file__).resolve().parents[2]
    for path, digest in [
        (
            "src/nexuml/training/local_fit.py",
            "6d4c43e72b4ecc0c30c298bcf467392fdbd3ac4ce8f178a83e62436c8fd1faf2",
        ),
        (
            "library/src/nexuml_library/scenarios/imitation/so101.py",
            "c0d22ef6363ab8f4b56e0878cf17fa24ab2edd89c184d740e2ff665ea871b2f8",
        ),
        (
            "library/src/nexuml_library/reinforcement/algorithms/ppo.py",
            "6c52fcb6bc3b03739fe05925807a298ae5637248dd7654f5497be8508a61c635",
        ),
    ]:
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest


def test_ordinary_ppo_numerical_bodies_and_defaults_match_pre_sensor_baseline():
    import ast

    root = Path(__file__).resolve().parents[2]
    tree = ast.parse(
        (root / "library/src/nexuml_library/reinforcement/algorithms/ppo.py").read_text()
    )
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    ppo = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PPO")
    build = next(
        node for node in ppo.body if isinstance(node, ast.FunctionDef) and node.name == "build"
    )
    # Only the learner dependency-resolution block changed, not construction/update math.
    assert isinstance(build.body[0], ast.Try)
    build.body = build.body[1:]
    assert hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest() == (
        "06f57bdae30232fd4e778345781491e4f486ef34e765a53700bf9b253cbf9f3d"
    )


@pytest.mark.requires_optional("torchrl")
def test_actual_trainer_checkpoint_restores_processor_and_export_reset_parity(tmp_path):
    from lightning import Trainer

    from nexuml.training.lightning import NexuSession
    from nexuml.training.reinforcement import NexuRLLightningModule
    from nexuml_library.action_adapters.adapters import CategoricalActionAdapter
    from nexuml_library.reinforcement.algorithms.ppo import PPO

    scenario = _policy().pipeline.resolved_config.to_scenario()
    assert scenario.interaction is not None and scenario.policy is not None
    scenario.interaction.environment = GymnasiumEnvironment(
        env_name="custom_sensor_task:inert-metadata",
        contract=InteractionContract(
            observations=_contract().observations,
            actions={"action": TensorFieldContract(shape=(), dtype="int64", num_values=2)},
            action_space="discrete",
        ),
    )
    scenario.policy.action_adapter = CategoricalActionAdapter(logits_key="command")
    scenario.training = ReinforcementLearningSpec(
        algorithm=PPO(update_epochs=1),
        total_frames=4,
        frames_per_batch=4,
        evaluation=RLEvaluationSpec(episodes=0),
        accelerator="cpu",
        devices=1,
    )
    reference = CompiledPolicy.from_pipeline(compile(scenario))
    module = NexuRLLightningModule(scenario, policy=reference)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    # Exercise Lightning's real checkpoint schema without collecting from a simulator.
    trainer.fit(module)
    initial = reference.initialize_state((2,))
    _, live = reference.step(_observation(99, (2,)), initial)
    checkpoint = tmp_path / "sensor.ckpt"
    trainer.save_checkpoint(checkpoint)
    session = NexuSession.from_trainer_checkpoint(
        checkpoint, log_dir=tmp_path / "resume", enable_loggers=False, enable_progress_bar=False
    )
    assert isinstance(session.lightning_module, NexuRLLightningModule)
    resumed = session.lightning_module.policy
    assert resumed.processor is not None
    assert resumed.contract == reference.contract
    assert resumed.pipeline.resolved_config.policy is not None
    assert isinstance(resumed.pipeline.resolved_config.policy.processor, SensorHistoryProcessor)
    _equal(resumed.initialize_state((2,)), initial)
    assert live["state", "initialized"].all()
    loaded, _, metadata = load_inference_package(export_package(resumed, tmp_path / "resumed"))
    assert not metadata["training_state_available"]
    policies = (reference, resumed, cast(CompiledPolicy, loaded))
    states = [policy.initialize_state((2,)) for policy in policies]
    for state in states:
        _equal(state, initial)
    for index in range(4):
        outputs = []
        for position, policy in enumerate(policies):
            actions, states[position] = policy.step(_observation(index + 1, (2,)), states[position])
            outputs.append(actions)
            _equal(actions, outputs[0])
            _equal(states[position], states[0])
        if index == 1:
            continuing = states[0][1].clone()
            states = [
                policy.reset_state(state, torch.tensor([True, False]))
                for policy, state in zip(policies, states, strict=True)
            ]
            for state in states:
                _equal(state[0], initial[0])
                _equal(state[1], continuing)


def test_complete_export_reloads_fresh_state_and_selected_reset_without_native_imports(tmp_path):
    torch.manual_seed(4)
    policy = _policy()
    state = policy.initialize_state((2,))
    _, live = policy.step(_observation(1, (2,)), state)
    package = export_package(policy, tmp_path / "policy")
    loaded, _, _ = load_inference_package(package)
    restored = cast(CompiledPolicy, loaded)
    assert restored.processor is not None
    _equal(restored.initialize_state((2,)), state)
    assert not list(restored.processor.buffers())
    expected = []
    for index in range(3):
        raw = _observation(index + 2, (2,))
        actions, state = policy.step(raw, state)
        expected.append((raw, actions, state.clone()))
        if index == 1:
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
            'isaaclab', 'isaacsim', 'torchrl', 'viser', 'custom_sensor_task',
            'nexuml', 'nexuml_library',
        }:
            raise ImportError('blocked: ' + fullname)
sys.meta_path.insert(0, Deny())
sys.path[:] = [p for p in sys.path if '/src' not in p]
payload = PackageImporter(sys.argv[1]).load_pickle('nexuml_export', 'artifact.pkl')
policy = payload['policy']
assert not payload['training_state']
state = policy.initialize_state((2,))
assert not state['state', 'initialized'].any()
expected = torch.load(sys.argv[2], weights_only=False)
for index, (raw, actions, reference) in enumerate(expected):
    actual, state = policy.step(raw, state)
    torch.testing.assert_close(actual['action'], actions['action'], rtol=0, atol=0)
    for key in reference.keys(True, True):
        torch.testing.assert_close(state[key], reference[key], rtol=0, atol=0)
    if index == 1:
        state = policy.reset_state(state, torch.tensor([True, False]))
print('portable sensor sequence/reset parity')
"""
    environment = dict(os.environ, PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(package / "pipeline.package"),
            str(tmp_path / "expected.pt"),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "portable sensor sequence/reset parity" in result.stdout
    assert live["state", "initialized"].all()
