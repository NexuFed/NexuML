"""CPU authoring/interface proof only, not native sensor or learning acceptance."""

import builtins
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from pydantic import ValidationError
from tensordict import TensorDict

from nexuml.core.compiler import compile
from nexuml.core.config import ResolvedConfig
from nexuml.core.discovery import DISCOVERED_ATTR
from nexuml.core.policy import CompiledPolicy
from nexuml.core.serialization import lower_model
from nexuml.core.types import InteractionContract, ReinforcementLearningSpec
from nexuml_library.environments import _isaaclab_sensors as sensors
from nexuml_library.environments.isaaclab import IsaacLabEnvironment
from nexuml_library.environments.isaaclab_config import IsaacBindings
from nexuml_library.reinforcement.policies import PolicyHead
from nexuml_library.scenarios.reinforcement import isaaclab_sensors as recipes
from nexuml_library.scenarios.reinforcement.isaaclab import isaac_cartpole_ppo


FAMILIES = (
    ("rgb", ("rgb",)),
    ("depth", ("depth",)),
    ("segmentation", ("segmentation",)),
    ("mixed", ("rgb", "depth", "segmentation", "state")),
)


@pytest.fixture(autouse=True)
def _block_native_and_cuda(monkeypatch):
    original = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name.startswith(
            ("isaaclab", "isaacsim", "omni", "pxr", "viser", "newton", "warp")
        ):
            raise AssertionError("CPU sensor definition proof imported native module: " + name)
        return original(name, globals, locals, fromlist, level)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU sensor definitions cannot inspect or allocate CUDA")

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


def _recipe(family, owner="package"):
    return getattr(recipes, f"isaac_{family}_cartpole_ppo")(processing_owner=owner)


def _raw_fixture(contract: InteractionContract) -> TensorDict:
    leaves = {
        "rgb": torch.arange(32 * 32 * 3).remainder(256).to(torch.uint8).reshape(32, 32, 3),
        "depth": torch.full((32, 32, 1), 5.0),
        "segmentation": torch.arange(32 * 32, dtype=torch.int32).remainder(3).reshape(32, 32, 1),
        "state": torch.tensor([0.1, 0.2, 0.3, 0.4]),
    }
    leaves["depth"][0, :4, 0] = torch.tensor([0.0, float("nan"), 101.0, 0.05])
    return TensorDict(
        {key: leaves[key].expand(2, *leaves[key].shape).clone() for key in contract.observations},
        batch_size=(2,),
    )


def _processed_fixture(raw: TensorDict) -> TensorDict:
    processed = raw.clone()
    for key in raw.keys():
        value = cast(torch.Tensor, raw[key])
        if key == "state":
            continue
        chw = value.movedim(-1, -3)
        if key == "rgb":
            frame = chw.float() / 255
        elif key == "depth":
            valid = torch.isfinite(chw) & (chw >= 0.1) & (chw <= 100)
            frame = torch.cat((torch.where(valid, chw, 0.0), valid.float()), dim=1)
        else:
            frame = torch.cat([chw == identifier for identifier in (0, 1, 2)], dim=1).float()
        processed[key] = torch.cat((frame, frame), dim=1)
    return processed


def test_static_imports_and_typed_capture_are_native_and_cuda_inert():
    importlib.reload(sensors)
    importlib.reload(recipes)
    for family, _ in FAMILIES:
        function = getattr(recipes, f"isaac_{family}_cartpole_ppo")
        assert getattr(function, DISCOVERED_ATTR)["key"] == f"isaac-{family}-cartpole-ppo"
        recipe = function()
        assert recipe.interaction is not None
        definition = cast(IsaacLabEnvironment, recipe.interaction.environment)
        task_config = definition.task_config
        assert task_config is not None
        assert task_config.factory == (
            "nexuml_library.environments._isaaclab_sensors:configure_sensor_cartpole"
        )
        assert task_config.kwargs == {
            "modalities": [key for key in definition.describe().observations if key != "state"],
            "width": 32,
            "height": 32,
        }
        assert task_config.args == []


@pytest.mark.parametrize("family,keys", FAMILIES)
@pytest.mark.parametrize("owner", ["package", "external"])
def test_recipe_roundtrip_exact_contract_and_complete_policy(family, keys, owner):
    recipe = _recipe(family, owner)
    restored = ResolvedConfig.from_yaml(
        ResolvedConfig.from_scenario(recipe).to_yaml()
    ).to_scenario()
    assert restored == recipe
    assert restored.interaction is not None and restored.policy is not None
    definition = cast(IsaacLabEnvironment, restored.interaction.environment)
    contract = definition.describe()
    assert definition.task == "NexuML-Cartpole-Sensors"
    assert definition.physics == "newton_mjwarp"
    assert definition.presets == ("rgb", "newton_renderer")
    assert tuple(contract.observations) == keys
    assert definition.bindings is not None and definition.bindings.processing == "raw"
    assert definition.bindings.observations == {
        key: ("policy",) if key == "state" else ("images", key) for key in keys
    }
    assert contract.control_period_s == 1 / 60 and contract.num_envs == 1
    action = contract.actions["action"]
    assert action.shape == (1,) and action.dtype == "float32"
    assert (action.low, action.high, action.rate_hz) == (-1, 1, 60)
    assert action.physical is not None
    assert action.physical.units == ("N",) and action.physical.frame == "joint"
    assert action.physical.joints == ("slider_to_cart",)
    assert action.physical.control_mode == "effort"
    assert action.physical.scale == (100.0,) and action.physical.offset == (0.0,)
    reference = isaac_cartpole_ppo(total_frames=256, frames_per_batch=64, num_envs=4)
    assert restored.training == reference.training
    assert isinstance(restored.training, ReinforcementLearningSpec)
    assert restored.training.evaluation.episodes == 0
    assert restored.pipeline.stages["policy"][0].keys_in == list(keys)
    head = restored.pipeline.stages["policy"][0].component
    assert isinstance(head, PolicyHead) and head.hidden_dims == (32,)

    policy = CompiledPolicy.from_pipeline(compile(restored))
    raw = _raw_fixture(contract)
    processed = _processed_fixture(raw)
    if owner == "package":
        assert restored.policy.processor is not None and policy.processor is not None
        initial = policy.initialize_state((2,), device="cpu")
        stepped, following = policy.step(raw, initial)
        actual, _ = cast(Any, policy.processor).step(raw, initial)
        assert set(initial.keys()) == set(keys) - {"state"}
        for key in keys:
            torch.testing.assert_close(actual[key], processed[key])
        for key in initial.keys():
            assert not initial[key, "initialized"].any()
            assert following[key, "initialized"].all()
        reset = policy.reset_state(following, torch.tensor([True, False]))
        for key in following.keys():
            assert not reset[key, "frames"][0].any()
            assert torch.equal(
                cast(torch.Tensor, reset[key, "frames"])[1],
                cast(torch.Tensor, following[key, "frames"])[1],
            )
            assert following[key, "initialized"].all()
        torch.testing.assert_close(stepped["action"], policy(processed)["action"])
    else:
        assert restored.policy.processor is None and policy.processor is None
        with pytest.raises(ValueError, match="explicit processor"):
            policy.initialize_state((2,), device="cpu")
    with pytest.raises(ValueError, match="shape|dtype"):
        policy(raw)

    captured = []
    first_linear = next(
        module for module in policy.modules() if isinstance(module, torch.nn.Linear)
    )
    handle = first_linear.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().clone())
    )
    try:
        actions = policy(processed)
        sampled = policy(processed, deterministic=False)
    finally:
        handle.remove()
    expected_input = torch.cat(
        [cast(torch.Tensor, processed[key]).reshape(2, -1) for key in keys], dim=1
    )
    torch.testing.assert_close(captured[0], expected_input)
    for output in (actions, sampled):
        assert output["action"].shape == (2, 1) and output["action"].dtype == torch.float32
        assert torch.isfinite(output["action"]).all()
        assert ((output["action"] >= -1) & (output["action"] <= 1)).all()


@pytest.mark.parametrize("owner", ["package", "external"])
def test_sensor_descriptors_retain_raw_meaning_and_processed_interfaces(owner):
    recipe = _recipe("mixed", owner)
    assert recipe.interaction is not None
    fields = recipe.interaction.environment.describe().observations
    assert fields["state"] == IsaacLabEnvironment().describe().observations["observation"]
    for key, channels, raw_channels, dtype, units, physical_units, high in (
        ("rgb", 6, 3, "uint8", "uint8_sRGB", "1", 1),
        ("depth", 4, 1, "float32", "m", "m", 100),
        ("segmentation", 6, 1, "int32", "class_id", "1", 1),
    ):
        field = fields[key]
        assert field.shape == (channels, 32, 32) and field.dtype == "float32"
        assert field.modality == key and (field.low, field.high, field.rate_hz) == (0, high, 60)
        sensor = field.sensor
        assert sensor is not None and sensor.owner == owner
        assert sensor.raw_shape == (32, 32, raw_channels)
        assert (sensor.raw_dtype, sensor.raw_layout, sensor.units) == (dtype, "HWC", units)
        assert (sensor.processor_id, sensor.processor_version) == ("SensorHistoryProcessor", "1")
        assert sensor.assets == {"camera_geometry": sensors.camera_geometry_identity(32, 32)}
        assert (sensor.history_length, sensor.history_order) == (2, "oldest_first")
        assert (sensor.initialization, sensor.timing, sensor.latency_steps) == (
            "repeat_first",
            "post_control",
            0,
        )
        assert field.physical is not None
        assert field.physical.units == (physical_units,)
        assert field.physical.frame == "camera_optical"
        assert field.physical.sensor_id == "tiled_camera" and field.physical.joints == ()
    rgb, depth, segmentation = (fields[key].sensor for key in ("rgb", "depth", "segmentation"))
    assert rgb is not None and depth is not None and segmentation is not None
    assert (rgb.scale, rgb.offset) == (1 / 255, 0)
    assert (depth.scale, depth.offset) == (1, 0)
    assert (depth.depth_representation, depth.depth_min, depth.depth_max, depth.invalid_depth) == (
        "z_depth",
        0.1,
        100,
        "mask_zero",
    )
    assert segmentation.labels == ((0, "BACKGROUND"), (1, "UNLABELLED"), (2, "cartpole"))
    assert segmentation.label_set_id == "cartpole-class-v1"


@pytest.mark.parametrize("family,_", FAMILIES)
def test_recipe_rejects_undeclared_processing_owner(family, _):
    with pytest.raises(ValidationError, match="owner"):
        _recipe(family, "implicit")


def _camera_cfg():
    return SimpleNamespace(
        prim_path="/World/envs/env_[^/]+/Camera",
        width=32,
        height=32,
        offset=SimpleNamespace(pos=(-5, 0, 2), rot=(0, 0, 0, 1), convention="world"),
        spawn=SimpleNamespace(
            focal_length=24,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 100),
            projection_type="pinhole",
            distortion=None,
            vertical_aperture=None,
            horizontal_aperture_offset=0.0,
            vertical_aperture_offset=0.0,
            f_stop=0.0,
        ),
    )


def _native_sensor_case(family="mixed"):
    recipe = _recipe(family)
    assert recipe.interaction is not None
    definition = cast(IsaacLabEnvironment, recipe.interaction.environment)
    assert definition.bindings is not None
    contract = definition.describe()
    camera_cfg = _camera_cfg()
    camera_cfg.update_period = 1 / 60
    camera_cfg.renderer_cfg = SimpleNamespace(
        renderer_type="newton_warp",
        depth_clipping_behavior="zero",
        colorize_semantic_segmentation=False,
    )
    camera_cfg.data_types = ["rgb", "depth", "semantic_segmentation"]
    images = SimpleNamespace(concatenate_terms=False, enable_corruption=False, history_length=0)
    for key, field in contract.observations.items():
        if field.sensor is not None:
            setattr(
                images,
                key,
                SimpleNamespace(
                    func=sensors.camera_frame,
                    params={"data_type": "semantic_segmentation" if key == "segmentation" else key},
                    modifiers=None,
                    noise=None,
                    clip=None,
                    scale=None,
                    history_length=0,
                    delay_min_lag=0,
                    delay_max_lag=0,
                ),
            )
    camera = SimpleNamespace(
        cfg=camera_cfg,
        data=SimpleNamespace(
            pos_w=SimpleNamespace(torch=torch.tensor([[-5.0, 0.0, 2.0]])),
            quat_w_world=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0]])),
            intrinsic_matrices=SimpleNamespace(
                torch=torch.tensor(
                    [
                        [
                            [32 * 24 / 20.955, 0.0, 16.0],
                            [0.0, 32 * 24 / 20.955, 16.0],
                            [0.0, 0.0, 1.0],
                        ]
                    ]
                )
            ),
            info={
                "semantic_segmentation": {
                    "idToLabels": {
                        0: {"class": "BACKGROUND"},
                        1: {"class": "UNLABELLED"},
                        2: {"class": "cartpole"},
                    }
                }
            },
        ),
    )

    class Scene(dict):
        env_regex_ns = "/World/envs/env_[^/]+"
        env_origins = torch.zeros(1, 3)

    env = SimpleNamespace(
        cfg=SimpleNamespace(observations=SimpleNamespace(images=images)),
        scene=Scene(tiled_camera=camera),
    )
    return env, contract, definition.bindings


def _change_sensor(contract, key, **updates):
    field = contract.observations[key]
    assert field.sensor is not None
    contract.observations[key] = field.model_copy(
        update={"sensor": field.sensor.model_copy(update=updates)}
    )


@pytest.mark.parametrize("family,_", FAMILIES)
def test_native_sensor_admission_accepts_only_configured_raw_snapshots_without_reading_pixels(
    family, _
):
    env, contract, bindings = _native_sensor_case(family)
    sensors.validate_native_sensors(env, contract, bindings)
    assert not hasattr(env.scene["tiled_camera"].data, "output")


def test_native_sensor_admission_leaves_vector_and_undescribed_images_unchanged():
    contract = IsaacLabEnvironment().describe()
    sensors.validate_native_sensors(object(), contract, IsaacBindings(observations={"x": ("x",)}))
    image = _native_sensor_case("rgb")[1]
    image.observations["rgb"] = image.observations["rgb"].model_copy(update={"sensor": None})
    sensors.validate_native_sensors(object(), image, IsaacBindings(observations={"x": ("x",)}))


@pytest.mark.parametrize("path", [("images",), ("images", "rgb", "extra"), ("missing", "rgb")])
def test_native_sensor_admission_rejects_unqualified_binding_paths(path):
    env, contract, bindings = _native_sensor_case("rgb")
    bindings = bindings.model_copy(update={"observations": {"rgb": path}})
    with pytest.raises(ValueError, match="Native sensor"):
        sensors.validate_native_sensors(env, contract, bindings)


def test_native_sensor_admission_rejects_processed_bindings_and_lookalike_functions():
    env, contract, bindings = _native_sensor_case("rgb")
    with pytest.raises(ValueError, match="Native sensor.*processed image bindings"):
        sensors.validate_native_sensors(
            env, contract, bindings.model_copy(update={"processing": "processed"})
        )
    env.cfg.observations.images.rgb.func = lambda env, data_type: torch.zeros(1, 32, 32, 3)
    with pytest.raises(ValueError, match="Native sensor.*exact camera_frame"):
        sensors.validate_native_sensors(env, contract, bindings)
    env.cfg.observations.images.rgb.func = (
        "nexuml_library.environments._isaaclab_sensors:camera_frame"
    )
    with pytest.raises(ValueError, match="Native sensor.*exact camera_frame"):
        sensors.validate_native_sensors(env, contract, bindings)


@pytest.mark.parametrize(
    "part,attribute,value",
    [
        ("term", "modifiers", [object()]),
        ("term", "noise", object()),
        ("term", "clip", (0, 255)),
        ("term", "scale", 1.0),
        ("term", "history_length", 1),
        ("term", "delay_min_lag", 1),
        ("term", "delay_max_lag", 1),
        ("term", "history_length", False),
        ("group", "enable_corruption", True),
        ("group", "history_length", 1),
        ("group", "history_length", None),
        ("group", "concatenate_terms", True),
    ],
)
def test_native_sensor_admission_rejects_double_processing_history_and_delay(
    part, attribute, value
):
    env, contract, bindings = _native_sensor_case("rgb")
    group = env.cfg.observations.images
    setattr(group.rgb if part == "term" else group, attribute, value)
    with pytest.raises(ValueError, match="Native sensor.*native processing, history or delay"):
        sensors.validate_native_sensors(env, contract, bindings)


@pytest.mark.parametrize("rate", [None, 0, -1, 30, float("nan"), float("inf")])
def test_native_sensor_admission_rejects_missing_or_changed_rates(rate):
    env, contract, bindings = _native_sensor_case("rgb")
    contract.observations["rgb"] = contract.observations["rgb"].model_copy(update={"rate_hz": rate})
    with pytest.raises(ValueError, match="Native sensor.*sensor rate/control period"):
        sensors.validate_native_sensors(env, contract, bindings)


@pytest.mark.parametrize(
    "target,attribute,value",
    [
        ("camera", "update_period", 1 / 30),
        ("camera", "update_period", 0),
        ("contract", "control_period_s", None),
        ("contract", "control_period_s", 1 / 30),
        ("sensor", "timing", "pre_action"),
        ("sensor", "latency_steps", 1),
    ],
)
def test_native_sensor_admission_rejects_mismatched_control_updates(target, attribute, value):
    env, contract, bindings = _native_sensor_case("rgb")
    if target == "camera":
        setattr(env.scene["tiled_camera"].cfg, attribute, value)
    elif target == "contract":
        contract = contract.model_copy(update={attribute: value})
    else:
        _change_sensor(contract, "rgb", **{attribute: value})
    with pytest.raises(ValueError, match="Native sensor.*sensor rate/control period"):
        sensors.validate_native_sensors(env, contract, bindings)


@pytest.mark.parametrize(
    "key,attribute,value",
    [
        ("rgb", "raw_layout", "CHW"),
        ("rgb", "raw_shape", (32, 32, 4)),
        ("rgb", "raw_dtype", "float32"),
        ("rgb", "units", "normalized_sRGB"),
        ("depth", "raw_dtype", "float64"),
        ("depth", "units", "mm"),
        ("segmentation", "raw_dtype", "uint8"),
        ("segmentation", "units", "instance_id"),
    ],
)
def test_native_sensor_admission_rejects_raw_layout_dtype_and_units(key, attribute, value):
    env, contract, bindings = _native_sensor_case()
    _change_sensor(contract, key, **{attribute: value})
    with pytest.raises(
        ValueError, match="Native sensor.*layout/shape/dtype/units|Native sensor.*metre bounds"
    ):
        sensors.validate_native_sensors(env, contract, bindings)


@pytest.mark.parametrize(
    "attribute,value",
    [("sensor_id", "spectator"), ("frame", "world"), ("units", ("mm",))],
)
def test_native_sensor_admission_rejects_wrong_physical_meaning(attribute, value):
    env, contract, bindings = _native_sensor_case("rgb")
    field = contract.observations["rgb"]
    assert field.physical is not None
    contract.observations["rgb"] = field.model_copy(
        update={"physical": field.physical.model_copy(update={attribute: value})}
    )
    with pytest.raises(ValueError, match="Native sensor"):
        sensors.validate_native_sensors(env, contract, bindings)


@pytest.mark.parametrize("renderer", ["isaac_rtx", "default", None])
def test_native_sensor_admission_never_falls_back_to_another_renderer(renderer):
    env, contract, bindings = _native_sensor_case("rgb")
    env.scene["tiled_camera"].cfg.renderer_cfg.renderer_type = renderer
    with pytest.raises(ValueError, match="Native sensor.*explicit renderer_type='newton_warp'"):
        sensors.validate_native_sensors(env, contract, bindings)


def test_native_sensor_admission_compares_actual_geometry_instead_of_trusting_asset_id():
    env, contract, bindings = _native_sensor_case("rgb")
    env.scene["tiled_camera"].cfg.spawn.focal_length = 25
    with pytest.raises(ValueError, match="Native sensor.*Camera geometry differs"):
        sensors.validate_native_sensors(env, contract, bindings)
    env.scene["tiled_camera"].cfg.spawn.focal_length = 24
    _change_sensor(contract, "rgb", assets={"camera_geometry": "0" * 64})
    with pytest.raises(ValueError, match="Native sensor.*Camera geometry differs"):
        sensors.validate_native_sensors(env, contract, bindings)


def test_native_sensor_rejects_a_body_mounted_camera_with_identical_offsets():
    env, contract, bindings = _native_sensor_case("rgb")
    env.scene["tiled_camera"].cfg.prim_path = "/World/envs/env_[^/]+/Robot/body/Camera"
    with pytest.raises(ValueError, match="Native sensor.*mount"):
        sensors.validate_native_sensors(env, contract, bindings)


@pytest.mark.parametrize("field", ["pos_w", "quat_w_world", "intrinsic_matrices"])
def test_native_sensor_rejects_changed_resolved_pose_or_intrinsics(field):
    env, contract, bindings = _native_sensor_case("rgb")
    getattr(env.scene["tiled_camera"].data, field).torch += 1.0
    with pytest.raises(ValueError, match="Native sensor.*actual camera pose/intrinsics"):
        sensors.validate_native_sensors(env, contract, bindings)


def test_native_camera_accepts_derived_vertical_aperture_and_translated_environment_origin():
    env, contract, bindings = _native_sensor_case("rgb")
    camera = env.scene["tiled_camera"]
    camera.cfg.spawn.vertical_aperture = camera.cfg.spawn.horizontal_aperture
    env.scene.env_origins = torch.tensor([[4.0, 8.0, 0.0]])
    camera.data.pos_w.torch += env.scene.env_origins
    sensors.validate_native_sensors(env, contract, bindings)


@pytest.mark.parametrize(
    "modality,representation,data_type,valid",
    [
        ("rgb", None, "rgb", True),
        ("rgb", None, "rgba", False),
        ("depth", "z_depth", "depth", True),
        ("depth", "z_depth", "distance_to_image_plane", True),
        ("depth", "z_depth", "distance_to_camera", False),
        ("depth", "range_depth", "distance_to_camera", True),
        ("depth", "range_depth", "depth", False),
        ("depth", None, "depth", False),
        ("segmentation", None, "semantic_segmentation", True),
        ("segmentation", None, "instance_segmentation", False),
    ],
)
def test_native_sensor_admission_requires_exact_modality_and_depth_data_key(
    modality, representation, data_type, valid
):
    env, contract, bindings = _native_sensor_case(modality)
    _change_sensor(contract, modality, depth_representation=representation)
    getattr(env.cfg.observations.images, modality).params["data_type"] = data_type
    env.scene["tiled_camera"].cfg.data_types = [data_type]
    if valid:
        sensors.validate_native_sensors(env, contract, bindings)
    else:
        with pytest.raises(ValueError, match="Native sensor"):
            sensors.validate_native_sensors(env, contract, bindings)


@pytest.mark.parametrize(
    "part,attribute,value",
    [
        ("sensor", "depth_min", None),
        ("sensor", "depth_max", None),
        ("sensor", "depth_min", 0),
        ("sensor", "depth_min", 0.2),
        ("sensor", "depth_max", 99),
        ("sensor", "invalid_depth", "reject"),
        ("renderer", "depth_clipping_behavior", "max"),
        ("renderer", "depth_clipping_behavior", "none"),
    ],
)
def test_native_sensor_admission_requires_declared_depth_bounds_and_zero_sentinel(
    part, attribute, value
):
    env, contract, bindings = _native_sensor_case("depth")
    if part == "sensor":
        _change_sensor(contract, "depth", **{attribute: value})
    else:
        setattr(env.scene["tiled_camera"].cfg.renderer_cfg, attribute, value)
    with pytest.raises(ValueError, match="Native sensor.*metre bounds and zero-sentinel"):
        sensors.validate_native_sensors(env, contract, bindings)


@pytest.mark.parametrize("string_keys", [False, True])
def test_native_sensor_admission_compares_label_maps_without_reordering_portable_channels(
    string_keys,
):
    env, contract, bindings = _native_sensor_case("segmentation")
    labels = ((2, "cartpole"), (0, "BACKGROUND"), (1, "UNLABELLED"))
    _change_sensor(contract, "segmentation", labels=labels)
    if string_keys:
        info = env.scene["tiled_camera"].data.info["semantic_segmentation"]
        info["idToLabels"] = {str(key): value for key, value in info["idToLabels"].items()}
    sensors.validate_native_sensors(env, contract, bindings)
    assert contract.observations["segmentation"].sensor.labels == labels


@pytest.mark.parametrize(
    "labels",
    [
        {},
        {0: {"class": "BACKGROUND"}, 1: {"class": "UNLABELLED"}},
        {0: {"class": "BACKGROUND"}, 1: {"class": "UNLABELLED"}, 2: {"class": "pole"}},
        {0: {"class": "BACKGROUND"}, 1: {"class": "UNLABELLED"}, 3: {"class": "cartpole"}},
        {0: {"class": "BACKGROUND"}, 1: {"class": "UNLABELLED"}, 2: "cartpole"},
        {
            0: {"class": "BACKGROUND"},
            1: {"class": "UNLABELLED"},
            2: {"class": "cartpole", "x": "y"},
        },
        {
            0: {"class": "BACKGROUND"},
            1: {"class": "UNLABELLED"},
            2: {"class": "cartpole"},
            3: {"class": "extra"},
        },
        None,
        [],
    ],
)
def test_native_sensor_admission_rejects_missing_extra_or_changed_actual_label_metadata(labels):
    env, contract, bindings = _native_sensor_case("segmentation")
    env.scene["tiled_camera"].data.info["semantic_segmentation"]["idToLabels"] = labels
    with pytest.raises(ValueError, match="Native sensor.*idToLabels"):
        sensors.validate_native_sensors(env, contract, bindings)


@pytest.mark.parametrize("identifier", [False, 0.0, -1, "-1", "00", "+0", "0.0", "٠", (0, 0, 0, 0)])
def test_native_sensor_admission_does_not_coerce_bad_segmentation_ids(identifier):
    env, contract, bindings = _native_sensor_case("segmentation")
    info = env.scene["tiled_camera"].data.info["semantic_segmentation"]
    labels = info["idToLabels"]
    background = labels.pop(0)
    labels[identifier] = background
    with pytest.raises(
        ValueError, match="Native sensor.*integer ID|Native sensor.*negative or duplicate ID"
    ):
        sensors.validate_native_sensors(env, contract, bindings)


def test_native_sensor_admission_rejects_duplicate_normalized_ids_and_fabricated_label_identity():
    env, contract, bindings = _native_sensor_case("segmentation")
    info = env.scene["tiled_camera"].data.info["semantic_segmentation"]
    info["idToLabels"]["0"] = {"class": "BACKGROUND"}
    with pytest.raises(ValueError, match="Native sensor.*duplicate ID"):
        sensors.validate_native_sensors(env, contract, bindings)
    del info["idToLabels"]["0"]
    _change_sensor(contract, "segmentation", label_set_id="another-vocabulary")
    with pytest.raises(ValueError, match="Native sensor.*cartpole-class-v1"):
        sensors.validate_native_sensors(env, contract, bindings)
    _change_sensor(
        contract, "segmentation", label_set_id="cartpole-class-v1", labels=((0, "different"),)
    )
    with pytest.raises(ValueError, match="Native sensor.*cartpole-class-v1"):
        sensors.validate_native_sensors(env, contract, bindings)


def test_native_sensor_admission_rejects_colorized_segmentation():
    env, contract, bindings = _native_sensor_case("segmentation")
    env.scene["tiled_camera"].cfg.renderer_cfg.colorize_semantic_segmentation = True
    with pytest.raises(ValueError, match="Native sensor.*uncolorized"):
        sensors.validate_native_sensors(env, contract, bindings)


@pytest.mark.parametrize(
    "missing",
    [
        "camera",
        "renderer",
        "geometry",
        "asset",
        "physical",
        "binding",
        "term",
        "func",
        "scale",
        "delay_min_lag",
        "delay_max_lag",
        "group_history",
        "data_type",
        "enabled_data_type",
        "info",
        "semantic_info",
        "labels",
        "update_period",
    ],
)
def test_native_sensor_admission_wraps_missing_native_metadata_as_value_error(missing):
    env, contract, bindings = _native_sensor_case("segmentation")
    camera = env.scene["tiled_camera"]
    group = env.cfg.observations.images
    term = group.segmentation
    if missing == "camera":
        env.scene.clear()
    elif missing == "renderer":
        del camera.cfg.renderer_cfg
    elif missing == "geometry":
        del camera.cfg.offset
    elif missing == "asset":
        _change_sensor(contract, "segmentation", assets={})
    elif missing == "physical":
        contract.observations["segmentation"] = contract.observations["segmentation"].model_copy(
            update={"physical": None}
        )
    elif missing == "binding":
        bindings.observations.clear()
    elif missing == "term":
        del group.segmentation
    elif missing in {"func", "scale", "delay_min_lag", "delay_max_lag"}:
        delattr(term, missing)
    elif missing == "group_history":
        del group.history_length
    elif missing == "data_type":
        term.params.clear()
    elif missing == "enabled_data_type":
        camera.cfg.data_types.clear()
    elif missing == "info":
        camera.data.info = None
    elif missing == "semantic_info":
        camera.data.info.clear()
    elif missing == "labels":
        camera.data.info["semantic_segmentation"].clear()
    else:
        del camera.cfg.update_period
    with pytest.raises(ValueError, match="Native sensor 'segmentation':"):
        sensors.validate_native_sensors(env, contract, bindings)


def test_camera_geometry_projection_is_stable_and_numeric_types_do_not_change_identity():
    projection = {
        "width": 32,
        "height": 32,
        "position": [-5.0, 0.0, 2.0],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "convention": "world",
        "focal_length": 24.0,
        "horizontal_aperture": 20.955,
        "clipping_range": [0.1, 100.0],
    }
    expected = _hash(projection)
    assert sensors.camera_geometry_identity(32, 32) == expected
    sensors.validate_camera_geometry(_camera_cfg(), expected)
    assert sensors.camera_geometry_identity(64, 32) != expected


@pytest.mark.parametrize(
    "part,attribute,value",
    [
        ("spawn", "focal_length", 25),
        ("spawn", "horizontal_aperture", 21),
        ("spawn", "clipping_range", (0.1, 20)),
        ("offset", "pos", (-4, 0, 2)),
        ("offset", "rot", (0, 0, 1, 0)),
        ("offset", "convention", "ros"),
        (None, "width", 64),
        (None, "height", 64),
    ],
)
def test_actual_changed_geometry_rejects_instead_of_trusting_identity(part, attribute, value):
    cfg = _camera_cfg()
    setattr(getattr(cfg, part) if part else cfg, attribute, value)
    with pytest.raises(ValueError, match="Camera geometry differs"):
        sensors.validate_camera_geometry(cfg, sensors.camera_geometry_identity(32, 32))


def test_missing_geometry_rejects():
    cfg = _camera_cfg()
    del cfg.spawn.focal_length
    with pytest.raises(ValueError, match="Camera geometry is missing or invalid"):
        sensors.validate_camera_geometry(cfg, sensors.camera_geometry_identity(32, 32))


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("vertical_aperture", 10.0),
        ("horizontal_aperture_offset", 1.0),
        ("vertical_aperture_offset", 1.0),
        ("projection_type", "fisheyePolynomial"),
        ("distortion", SimpleNamespace(fx=1.0)),
        ("f_stop", 1.0),
    ],
)
def test_equal_shape_camera_cannot_change_undeclared_projection(attribute, value):
    cfg = _camera_cfg()
    setattr(cfg.spawn, attribute, value)
    with pytest.raises(ValueError, match="Camera geometry"):
        sensors.validate_camera_geometry(cfg, sensors.camera_geometry_identity(32, 32))


@pytest.mark.parametrize("width,height", [(0, 32), (32, -1), (True, 32), (32, False), (32.0, 32)])
def test_invalid_resolution_rejects_before_native_import(width, height):
    with pytest.raises(ValueError, match="positive integers"):
        sensors.camera_geometry_identity(width, height)
    with pytest.raises(ValueError, match="positive integers"):
        sensors.configure_sensor_cartpole(
            SimpleNamespace(), modalities=["rgb"], width=width, height=height
        )


@pytest.mark.parametrize(
    "modalities",
    [[], ["rgb", "rgb"], ["state"], ["semantic_segmentation"], [False], [["rgb"]], ("rgb",)],
)
def test_invalid_modalities_reject_before_native_import(modalities):
    with pytest.raises(ValueError, match="modalities"):
        sensors.configure_sensor_cartpole(
            SimpleNamespace(), modalities=modalities, width=32, height=32
        )


def test_renderer_selection_never_falls_back():
    cfg = SimpleNamespace(
        scene=SimpleNamespace(
            tiled_camera=SimpleNamespace(renderer_cfg=SimpleNamespace(renderer_type="isaac_rtx"))
        )
    )
    with pytest.raises(ValueError, match="explicit renderer_type='newton_warp'"):
        sensors.configure_sensor_cartpole(cfg, modalities=["rgb"], width=32, height=32)
    assert cfg.scene.tiled_camera.renderer_cfg.renderer_type == "isaac_rtx"


@pytest.mark.parametrize("data_type", ["rgb", "depth", "semantic_segmentation"])
def test_camera_frame_snapshot_never_aliases_native_storage(data_type):
    raw = torch.arange(12).reshape(1, 2, 2, 3)
    env = SimpleNamespace(
        scene={
            "tiled_camera": SimpleNamespace(
                data=SimpleNamespace(output={data_type: SimpleNamespace(torch=raw)})
            )
        }
    )
    snapshot = sensors.camera_frame(env, data_type=data_type)
    assert torch.equal(raw, snapshot) and snapshot.data_ptr() != raw.data_ptr()
    snapshot.zero_()
    assert raw.count_nonzero() == 11


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def test_so101_identities_and_existing_ppo_source_are_unchanged():
    from nexuml_library.scenarios.imitation.so101 import _task_identity

    assert _hash(_task_identity()) == (
        "fee5075eb8da77c2192f1577752a883c58da678d992e9940a85072e047d4d4a1"
    )
    contract = IsaacLabEnvironment(task="NexuML-SO101-Joint-Reach").describe()
    assert _hash(lower_model(contract)) == (
        "b623a00a33c0ff6ac84cb4519f945e6954f093cce0dc9a238f2ebfd2150b7bf9"
    )
    root = Path(__file__).resolve().parents[2]
    assert (
        hashlib.sha256(
            (root / "library/src/nexuml_library/reinforcement/algorithms/ppo.py").read_bytes()
        ).hexdigest()
        == "6c52fcb6bc3b03739fe05925807a298ae5637248dd7654f5497be8508a61c635"
    )
