"""CPU proofs for the local SO-101 task; no Isaac installation or GPU is needed."""

import builtins
from copy import deepcopy
import importlib
import importlib.util
import math
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace

import pytest
import torch

from nexuml_library.environments import _isaaclab_so101 as so101


@pytest.fixture(autouse=True)
def forbid_native_and_cuda(monkeypatch):
    original_import = builtins.__import__
    original_import_module = importlib.import_module

    def check(name):
        root = name.split(".")[0]
        if root.startswith(("isaaclab", "isaacsim")) or root in {"omni", "pxr"}:
            raise AssertionError(f"CPU tests must not import {name}")

    def guarded_import(name, *args, **kwargs):
        check(name)
        return original_import(name, *args, **kwargs)

    def guarded_import_module(name, *args, **kwargs):
        check(name)
        return original_import_module(name, *args, **kwargs)

    def forbidden_cuda(*args, **kwargs):
        raise AssertionError("CPU tests must not probe or initialize CUDA")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(importlib, "import_module", guarded_import_module)
    monkeypatch.setattr(so101, "import_module", guarded_import_module)
    for name in ("is_available", "init", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, forbidden_cuda)


class _Robot:
    # Scramble articulation order so positional assumptions cannot accidentally pass.
    joint_names = (
        "gripper",
        "elbow_flex",
        "wrist_roll",
        "shoulder_pan",
        "wrist_flex",
        "shoulder_lift",
    )

    def __init__(self, num_envs):
        self.data = SimpleNamespace(
            joint_pos=SimpleNamespace(torch=torch.full((num_envs, 6), 9.0)),
            joint_vel=SimpleNamespace(torch=torch.full((num_envs, 6), 8.0)),
        )
        self.targets = torch.full((num_envs, 6), 7.0)
        self.calls = []

    def find_joints(self, names, *, preserve_order=False):
        assert preserve_order
        return [self.joint_names.index(name) for name in names], list(names)

    def write_joint_position_to_sim_index(self, *, position, joint_ids, env_ids):
        self.calls.append("position")
        self.data.joint_pos.torch[env_ids[:, None], joint_ids] = position

    def write_joint_velocity_to_sim_index(self, *, velocity, joint_ids, env_ids):
        self.calls.append("velocity")
        self.data.joint_vel.torch[env_ids[:, None], joint_ids] = velocity

    def set_joint_position_target(self, target, *, joint_ids, env_ids):
        self.calls.append("target")
        self.targets[env_ids[:, None], joint_ids] = target


def _env(num_envs=3):
    return SimpleNamespace(
        scene={"robot": _Robot(num_envs)}, num_envs=num_envs, device="cpu", step_dt=4 / 120
    )


def _arm_cfg(env):
    joint_ids, _ = env.scene["robot"].find_joints(so101.JOINT_NAMES, preserve_order=True)
    return SimpleNamespace(name="robot", joint_ids=joint_ids)


def test_module_import_does_not_load_optional_native_apis():
    spec = importlib.util.spec_from_file_location("so101_import_probe", so101.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.JOINT_NAMES == so101.JOINT_NAMES
    assert callable(module.make_so101_config)


def test_canonical_asset_identity_is_part_of_the_current_portable_contract():
    from nexuml_library.environments.isaaclab import IsaacLabEnvironment

    definition = IsaacLabEnvironment(task="NexuML-SO101-Joint-Reach")
    current = definition.describe().actions["action"].description
    assert current is not None and so101.ASSET_SOURCE in current
    assert so101.ASSET_LAYER_SHA256["so101_new_calib.usda"] in current
    assert len(so101.ASSET_LAYER_SHA256) == 15
    assert all(len(value) == 64 for value in so101.ASSET_LAYER_SHA256.values())


def test_task_constants_and_all_targets_fit_declared_and_inspected_limits():
    assert so101.JOINT_NAMES == (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    )
    assert so101.TARGET_SCALE == 0.5
    assert so101.HOME == (0.0, -0.25, 0.25, 0.0, 0.0)
    assert so101.GOALS == (
        so101.HOME,
        (0.2, -0.2, 0.2, -0.15, 0.15),
        (-0.2, -0.3, 0.3, 0.15, -0.15),
    )
    assert so101.TOLERANCE == 0.05
    assert so101.HOLD_SECONDS == 0.25
    # Native PhysX inspection: native-so101-asset-object-config.log, lines 152-176.
    limits = torch.tensor(
        [
            (-1.919859886, 1.919859886),
            (-1.745329857, 1.745329857),
            (-1.689999938, 1.689999938),
            (-1.658059835, 1.658059835),
            (-2.743849754, 2.841209650),
            (-0.174532995, 1.745329857),
        ]
    )
    goals = torch.tensor(so101.GOALS)
    assert goals.shape == (3, 5)
    assert (goals.abs() <= so101.TARGET_SCALE).all()
    assert (-so101.TARGET_SCALE >= limits[:5, 0]).all()
    assert (so101.TARGET_SCALE <= limits[:5, 1]).all()
    assert limits[5, 0] < so101.GRIPPER_POSITION < limits[5, 1]
    assert so101.GRIPPER_POSITION == 0.2555162024919699


def test_normalized_error_is_exact_unweighted_mse_without_control_dt():
    positions = torch.tensor([[0.0] * 5, [0.5] * 5, [1.0, -1.0, 0.0, 0.0, 0.0]])
    target = torch.zeros_like(positions)
    original = positions.clone()
    assert torch.equal(
        so101.normalized_joint_error(positions, target), torch.tensor([0, 0.25, 0.4])
    )
    assert torch.equal(positions, original)
    assert torch.equal(target, torch.zeros_like(target))
    assert so101.normalized_joint_error(positions[1], target[1]).item() == 0.25


def test_initial_observation_shape_probe_does_not_sample_or_allocate_task_state():
    env = _env()
    rng = torch.random.get_rng_state()
    assert torch.equal(so101._target_positions(env), torch.tensor(so101.HOME).repeat(3, 1))
    assert torch.equal(rng, torch.random.get_rng_state())
    assert not hasattr(env, "joint_reach_target")
    assert not hasattr(env, "joint_reach_hold")


@pytest.mark.parametrize("selection", [None, slice(None)])
def test_reset_maps_five_arm_joints_and_explicitly_holds_fixed_gripper(selection):
    env = _env()
    so101._reset_joint_reach(env, selection)
    robot = env.scene["robot"]
    assert robot.calls == ["position", "velocity", "target"]
    arm_ids = _arm_cfg(env).joint_ids
    assert torch.equal(
        robot.data.joint_pos.torch[:, arm_ids], torch.tensor(so101.HOME).repeat(3, 1)
    )
    assert torch.equal(robot.data.joint_vel.torch, torch.zeros(3, 6))
    assert torch.equal(robot.targets, robot.data.joint_pos.torch)
    assert torch.all(robot.targets[:, 0] == so101.GRIPPER_POSITION)
    assert env.joint_reach_target.shape == (3, 5)
    assert torch.equal(env.joint_reach_hold, torch.zeros(3, dtype=torch.long))


def test_reset_sampling_uses_seeded_device_rng_once_per_selected_environment():
    env = _env(20)
    other = _env(20)
    ids = torch.arange(20)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(137)
        expected = torch.tensor(so101.GOALS)[torch.randint(3, (20,))]
        torch.manual_seed(137)
        so101._reset_joint_reach(env, ids)
        first = env.joint_reach_target.clone()
        so101._reset_joint_reach(env, ids)
        second = env.joint_reach_target.clone()
        torch.manual_seed(137)
        so101._reset_joint_reach(other, ids)
        assert torch.equal(first, expected)
        assert torch.equal(first, other.joint_reach_target)
        so101._reset_joint_reach(other, ids)
        assert torch.equal(second, other.joint_reach_target)
        assert not torch.equal(first, second)
        assert len(torch.unique(first, dim=0)) == 3


def test_partial_reset_only_changes_requested_rows_and_does_not_alias_goals_or_hold():
    env = _env()
    other = _env()
    so101._reset_joint_reach(env, None)
    so101._reset_joint_reach(other, None)
    env.joint_reach_hold[:] = torch.tensor([4, 6, 7])
    target = env.joint_reach_target.clone()
    other_target = other.joint_reach_target.clone()
    robot = env.scene["robot"]
    robot.data.joint_pos.torch.fill_(1.0)
    robot.data.joint_vel.torch.fill_(2.0)
    robot.targets.fill_(3.0)
    so101._reset_joint_reach(env, torch.tensor([1]))
    assert torch.equal(env.joint_reach_target[[0, 2]], target[[0, 2]])
    assert torch.equal(env.joint_reach_hold, torch.tensor([4, 0, 7]))
    assert torch.all(robot.data.joint_pos.torch[[0, 2]] == 1)
    assert torch.all(robot.data.joint_vel.torch[[0, 2]] == 2)
    assert torch.all(robot.targets[[0, 2]] == 3)
    env.joint_reach_target[1].fill_(99)
    assert torch.equal(other.joint_reach_target, other_target)
    assert torch.equal(other.joint_reach_hold, torch.zeros(3, dtype=torch.long))
    assert so101.GOALS[0] == so101.HOME
    assert env.joint_reach_hold.data_ptr() != other.joint_reach_hold.data_ptr()
    assert env.joint_reach_target.data_ptr() != other.joint_reach_target.data_ptr()


def test_success_requires_eight_consecutive_steps_and_resets_on_any_joint_error():
    env = _env()
    so101._reset_joint_reach(env, None)
    cfg = _arm_cfg(env)
    env.joint_reach_target.zero_()
    positions = env.scene["robot"].data.joint_pos.torch
    positions[:, cfg.joint_ids] = 0
    positions[1, cfg.joint_ids[0]] = so101.TOLERANCE
    positions[2, cfg.joint_ids[-1]] = so101.TOLERANCE + 1e-4
    assert math.ceil(so101.HOLD_SECONDS / env.step_dt) == 8
    for step in range(1, 8):
        assert not so101._reach_success(env, cfg).any()
        assert torch.equal(env.joint_reach_hold, torch.tensor([step, step, 0]))
    positions[1, cfg.joint_ids[2]] = 0.051
    success = so101._reach_success(env, cfg)
    assert torch.equal(success, torch.tensor([True, False, False]))
    assert torch.equal(env.joint_reach_hold, torch.tensor([8, 0, 0]))
    so101._reset_joint_reach(env, torch.tensor([0]))
    assert not success[1:].any()
    assert success[0], "reset must not mutate the previous terminal result"
    assert env.joint_reach_hold[0] == 0


class _Cfg(SimpleNamespace):
    def replace(self, **kwargs):
        result = deepcopy(self)
        result.__dict__.update(kwargs)
        return result


@pytest.fixture
def task_config(monkeypatch):
    def configclass(cls):
        def init(self, **kwargs):
            self.__dict__.update(
                {
                    name: deepcopy(value)
                    for name, value in vars(cls).items()
                    if not name.startswith("_") and not callable(value)
                }
            )
            self.__dict__.update(kwargs)

        cls.__init__ = init
        return cls

    def scene_entity(name, **kwargs):
        return _Cfg(name=name, **kwargs)

    def joint_pos(env, asset_cfg):
        return env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids]

    def joint_vel(env, asset_cfg):
        return env.scene[asset_cfg.name].data.joint_vel.torch[:, asset_cfg.joint_ids]

    def time_out(env):
        return env.episode_length_buf >= env.max_episode_length

    asset = _Cfg(
        spawn=_Cfg(variants={"Robot": "robot", "Sensor": "sensors", "Physics": "physics"}),
        init_state=_Cfg(joint_pos={"gripper": 0.0}),
        actuators={"usd": _Cfg(joint_names_expr=[".*"], stiffness=None, damping=None)},
    )
    modules = {
        "isaaclab.assets": _Cfg(AssetBaseCfg=_Cfg),
        "isaaclab.envs": _Cfg(ManagerBasedRLEnvCfg=_Cfg),
        "isaaclab.envs.mdp": _Cfg(
            JointPositionActionCfg=_Cfg, joint_pos=joint_pos, joint_vel=joint_vel, time_out=time_out
        ),
        "isaaclab.managers": _Cfg(
            SceneEntityCfg=scene_entity,
            ObservationGroupCfg=_Cfg,
            ObservationTermCfg=_Cfg,
            RewardTermCfg=_Cfg,
            TerminationTermCfg=_Cfg,
            EventTermCfg=_Cfg,
        ),
        "isaaclab.scene": _Cfg(InteractiveSceneCfg=_Cfg),
        "isaaclab.sim": _Cfg(GroundPlaneCfg=_Cfg, DomeLightCfg=_Cfg, SimulationCfg=_Cfg),
        "isaaclab.utils": _Cfg(configclass=configclass),
        "isaaclab_physx.physics": _Cfg(PhysxCfg=_Cfg),
        "isaaclab_tasks.utils": _Cfg(PresetCfg=_Cfg),
        "isaaclab_assets.robots.so101": _Cfg(SO101_CFG=asset),
    }
    monkeypatch.setattr(so101, "import_module", modules.__getitem__)
    cfg = so101.make_so101_config(num_envs=3, device="cpu", seed=17)
    assert asset.spawn.variants["Physics"] == "physics"
    assert asset.init_state.joint_pos == {"gripper": 0.0}
    return cfg


def test_factory_config_preserves_asset_gains_and_exact_control_contract(task_config):
    cfg = task_config
    assert cfg.seed == 17 and cfg.scene.num_envs == 3 and cfg.sim.device == "cpu"
    assert cfg.sim.dt == 1 / 120 and cfg.decimation == cfg.sim.render_interval == 4
    assert cfg.episode_length_s == 4 and cfg.compute_final_obs
    assert set(vars(cfg.scene)) == {"ground", "robot", "light", "num_envs", "env_spacing"}
    assert set(vars(cfg.actions)) == {"arm_action"}
    action = cfg.actions.arm_action
    assert action.asset_name == "robot"
    assert action.joint_names == list(so101.JOINT_NAMES) and action.preserve_order
    assert action.scale == 0.5 and action.offset == 0 and not action.use_default_offset
    assert action.clip is None
    robot = cfg.scene.robot
    assert robot.prim_path == "{ENV_REGEX_NS}/Robot"
    assert robot.spawn.variants == {"Robot": "robot", "Sensor": "sensors", "Physics": "physx"}
    assert robot.init_state.joint_pos == dict(zip(so101.JOINT_NAMES, so101.HOME, strict=True)) | {
        "gripper": so101.GRIPPER_POSITION
    }
    assert robot.init_state.joint_vel == {".*": 0.0}
    assert vars(robot.actuators["usd"]) == {
        "joint_names_expr": [".*"],
        "stiffness": None,
        "damping": None,
    }


def test_configured_observations_reward_and_timeout_use_the_declared_terms(task_config):
    cfg = task_config
    env = _env()
    cfg.events.reset_joint_reach.func(env, None)
    assert cfg.events.reset_joint_reach.mode == "reset"
    policy = cfg.observations.policy
    assert set(vars(cfg.observations)) == {"policy"}
    assert list(vars(policy))[:3] == ["joint_pos", "joint_vel", "target"]
    assert policy.concatenate_terms and not policy.enable_corruption and policy.history_length == 0
    values = []
    for term in (policy.joint_pos, policy.joint_vel):
        entity = term.params["asset_cfg"]
        assert entity.joint_names == list(so101.JOINT_NAMES) and entity.preserve_order
        entity.joint_ids = _arm_cfg(env).joint_ids
        values.append(term.func(env, **term.params))
    values.append(policy.target.func(env))
    observation = torch.cat(values, dim=-1)
    assert observation.shape == (3, 15)
    assert torch.equal(observation[:, :5], torch.tensor(so101.HOME).repeat(3, 1))
    assert torch.equal(observation[:, 5:10], torch.zeros(3, 5))
    assert torch.equal(observation[:, 10:], env.joint_reach_target)
    reward = cfg.rewards.joint_error
    reward.params["asset_cfg"].joint_ids = _arm_cfg(env).joint_ids
    assert reward.weight == -1
    assert torch.equal(
        reward.func(env, **reward.params),
        (observation[:, :5] - observation[:, 10:]).square().mean(dim=-1),
    )
    env.joint_reach_target.fill_(1.0)
    env.scene["robot"].data.joint_pos.torch.zero_()
    assert torch.equal(reward.func(env, **reward.params), torch.ones(3))
    assert torch.allclose(
        reward.func(env, **reward.params) * reward.weight * env.step_dt, torch.full((3,), -1 / 30)
    )
    terminal = cfg.terminations.success
    terminal.params["asset_cfg"].joint_ids = _arm_cfg(env).joint_ids
    assert not getattr(terminal, "time_out", False)
    assert cfg.terminations.time_out.time_out
    env.episode_length_buf = torch.tensor([119, 120, 121])
    env.max_episode_length = 120
    assert torch.equal(cfg.terminations.time_out.func(env), torch.tensor([False, True, True]))
    assert not terminal.func(env, **terminal.params).any(), "timeouts are not task success"


def test_outcomes_keep_pre_reset_goal_error_and_success():
    env = _env()
    so101._reset_joint_reach(env, None)
    cfg = _arm_cfg(env)
    env.joint_reach_target.fill_(0.1)
    env.scene["robot"].data.joint_pos.torch[:, cfg.joint_ids] = 0
    so101._joint_error(env, cfg)
    before = env.joint_reach_error.clone()
    torch.testing.assert_close(before, torch.full((3,), 0.1))
    env.scene["robot"].data.joint_pos.torch[:, cfg.joint_ids] = 0.1
    for _ in range(8):
        so101._reach_success(env, cfg)
    assert env.joint_reach_success.all()
    so101._reset_joint_reach(env, None)
    torch.testing.assert_close(env.joint_reach_error, before)
    assert env.joint_reach_success.all(), "reset must not overwrite the ending transition"
    assert not env.joint_reach_hold.any()


def _asset_env():
    values = {
        "joint_pos_limits": so101.JOINT_LIMITS,
        "joint_stiffness": so101.STIFFNESS,
        "joint_damping": so101.DAMPING,
        "joint_effort_limits": [3.35] * 6,
        "joint_vel_limits": [30.0] * 6,
        "default_joint_pos": (*so101.HOME, so101.GRIPPER_POSITION),
    }
    data = SimpleNamespace()
    for name, expected in values.items():
        value = torch.tensor(expected)
        setattr(data, name, SimpleNamespace(torch=value.expand(3, *value.shape).clone()))
    robot = SimpleNamespace(joint_names=[*so101.JOINT_NAMES, "gripper"], data=data)
    cfg = SimpleNamespace(
        spawn=SimpleNamespace(usd_path=so101.ASSET_SOURCE, variants={"Physics": "physx"}),
        actuators={"usd": SimpleNamespace(stiffness=None, damping=None)},
    )
    return SimpleNamespace(
        device="cpu", scene={"robot": robot}, cfg=SimpleNamespace(scene=SimpleNamespace(robot=cfg))
    )


@pytest.mark.parametrize(
    "change",
    [
        None,
        "source",
        "variant",
        "joints",
        "gain_override",
        "joint_pos_limits",
        "joint_stiffness",
        "joint_damping",
        "joint_effort_limits",
        "joint_vel_limits",
        "default_joint_pos",
    ],
)
def test_native_asset_validation_rejects_control_profile_changes(change, monkeypatch):
    validated = []
    monkeypatch.setattr(so101, "_validate_asset_content", lambda: validated.append(True))
    env = _asset_env()
    cfg = env.cfg.scene.robot
    if change is None:
        so101.validate_asset(env)
        assert validated == [True]
        return
    if change == "source":
        cfg.spawn.usd_path = "file:///replacement.usd"
    elif change == "variant":
        cfg.spawn.variants["Physics"] = "physics"
    elif change == "joints":
        env.scene["robot"].joint_names.reverse()
    elif change == "gain_override":
        cfg.actuators["usd"].stiffness = 1.0
    else:
        getattr(env.scene["robot"].data, change).torch[0].add_(0.1)
    with pytest.raises(ValueError, match="SO-101"):
        so101.validate_asset(env)
    assert not validated


@pytest.mark.parametrize("change", [None, "root", "physics", "missing"])
def test_authored_asset_bytes_are_checked_without_accepting_rewritten_cache_hashes(
    monkeypatch, change
):
    layers = {
        "so101_new_calib.usda": b"authored root",
        "payloads/physics.usda": b"authored physics",
    }
    monkeypatch.setattr(
        so101,
        "ASSET_LAYER_SHA256",
        {name: sha256(value).hexdigest() for name, value in layers.items()},
    )
    root = so101.ASSET_SOURCE.rsplit("/", 1)[0] + "/"
    calls = []

    def read_file(path):
        name = path.removeprefix(root)
        calls.append(name)
        if change == "missing":
            raise FileNotFoundError(path)
        content = layers[name]
        if name == ("so101_new_calib.usda" if change == "root" else "payloads/physics.usda"):
            if change is not None:
                content += b" altered or localized"
        return BytesIO(content)

    monkeypatch.setattr(so101, "import_module", lambda name: SimpleNamespace(read_file=read_file))
    if change == "missing":
        with pytest.raises(FileNotFoundError):
            so101._validate_asset_content()
    elif change:
        with pytest.raises(ValueError, match="pinned layer"):
            so101._validate_asset_content()
    else:
        so101._validate_asset_content()
        assert calls == list(layers)
