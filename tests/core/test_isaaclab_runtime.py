"""CPU-only proofs of Isaac transition normalization and owned lifetime."""

import asyncio
import builtins
import json
import sys
import traceback
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from tensordict import TensorDict

from nexuml.core.components import EnvironmentBuildContext
from nexuml.core.types import PhysicalTensorContract, SensorProcessingContract
from nexuml_library.environments import _isaaclab_runtime
from nexuml_library.environments.isaaclab import (
    IsaacLabEnvironment,
    IsaacBindings,
    isaac_task_config,
)


class _FakeLauncher:
    def __init__(self, events, *, close_error=None):
        self.events = events
        self.close_error = close_error
        self.enter_calls = 0
        self.exit_calls = []

    def __enter__(self):
        self.enter_calls += 1
        self.events.append("launcher.enter")
        return self

    def __exit__(self, *args):
        self.exit_calls.append(args)
        self.events.append("launcher.exit")
        if self.close_error is not None:
            raise self.close_error


class _EffortAction(SimpleNamespace):
    pass


class _PositionAction(SimpleNamespace):
    pass


class _VelocityAction(SimpleNamespace):
    pass


class _PositionActionCfg(SimpleNamespace):
    pass


class _EffortActionCfg(SimpleNamespace):
    pass


class _FakeNative:
    def __init__(self, num_envs=3, *, close_error=None):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.single_observation_space: dict[str, Any] = {"policy": SimpleNamespace(shape=(4,))}
        self.single_action_space = SimpleNamespace(shape=(1,), dtype="float32")
        self.step_dt = 1 / 60
        self.cfg = SimpleNamespace(compute_final_obs=True)
        self.term: Any = _EffortAction(
            cfg=SimpleNamespace(scale=100.0, clip=None, class_type=_EffortAction),
            IO_descriptor=SimpleNamespace(joint_names=["slider_to_cart"], offset=0.0),
        )
        self.action_manager = SimpleNamespace(active_terms=["joint_effort"], get_term=self.get_term)
        self.observation = torch.zeros(num_envs, 4)
        self.final_observation = torch.zeros(num_envs, 4)
        self.reward = torch.zeros(num_envs)
        self.terminated = torch.zeros(num_envs, dtype=torch.bool)
        self.truncated = torch.zeros(num_envs, dtype=torch.bool)
        self.next_terminated = [False] * num_envs
        self.next_truncated = [False] * num_envs
        self.include_final_obs = True
        self.reset_calls = []
        self.step_calls = []
        self.events = []
        self.close_error = close_error
        self.close_calls = 0

    def get_term(self, name):
        assert name == "joint_effort"
        self.events.append("action.validate")
        return self.term

    def reset(self, *, seed=None, env_ids=None):
        assert env_ids is not None
        self.reset_calls.append((seed, env_ids.clone()))
        self.events.append("native.reset")
        values = torch.arange(self.num_envs * 4, dtype=torch.float32).reshape(self.num_envs, 4)
        self.observation[env_ids] = values[env_ids] + 1000 * len(self.reset_calls)
        self.final_observation.fill_(-999)
        self.reward.fill_(-777)
        self.terminated.zero_()
        self.truncated.zero_()
        return {"policy": self.observation}, {}

    def step(self, action):
        assert action.shape == (self.num_envs, 1)
        assert action.dtype == torch.float32 and action.device == self.device
        self.step_calls.append(action.clone())
        self.events.append("native.step")
        index = len(self.step_calls)
        values = torch.arange(self.num_envs * 4, dtype=torch.float32).reshape(self.num_envs, 4)
        self.final_observation.copy_(values + 100 * index)
        self.observation.copy_(self.final_observation)
        self.reward.copy_(torch.arange(self.num_envs, dtype=torch.float32) + index + 0.5)
        self.terminated.copy_(torch.tensor(self.next_terminated))
        self.truncated.copy_(torch.tensor(self.next_truncated))
        done = self.terminated | self.truncated
        self.observation[done] = values[done] - 100 * index
        info = {"final_obs": {"policy": self.final_observation}} if self.include_final_obs else {}
        return {"policy": self.observation}, self.reward, self.terminated, self.truncated, info

    def close(self):
        self.close_calls += 1
        self.events.append("native.close")
        if self.close_error is not None:
            raise self.close_error


class _ManagerConfig(SimpleNamespace):
    pass


class _DirectConfig(SimpleNamespace):
    pass


def _change_owned_setting(cfg: Any, *, setting: str) -> None:
    if setting == "count":
        cfg.scene.num_envs += 1
    elif setting == "device":
        cfg.sim.device = "cuda:1"
    elif setting == "seed":
        cfg.seed += 1
    elif setting == "physics":
        cfg.sim.physics = object()
    else:
        cfg.compute_final_obs = False


@pytest.fixture(autouse=True)
def forbid_native_and_cuda(monkeypatch):
    original_import = builtins.__import__
    original_import_module = _isaaclab_runtime.import_module

    def native_module(name):
        if name == "isaaclab.envs.mdp":
            return SimpleNamespace(
                JointEffortAction=_EffortAction,
                JointPositionAction=_PositionAction,
                JointEffortActionCfg=_EffortActionCfg,
                JointPositionActionCfg=_PositionActionCfg,
            )
        return original_import_module(name)

    def guarded_import(name, *args, **kwargs):
        root = name.split(".")[0]
        if root.startswith(("isaaclab", "isaacsim")) or root in {"omni", "pxr"}:
            raise AssertionError(f"CPU tests must not import {name}")
        return original_import(name, *args, **kwargs)

    def forbidden_cuda(*args, **kwargs):
        raise AssertionError("CPU tests must not probe or initialize CUDA")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(_isaaclab_runtime, "import_module", native_module)
    for name in ("is_available", "init", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, forbidden_cuda)


@pytest.fixture
def make_runtime(monkeypatch):
    runtimes = []

    def build(
        native=None,
        *,
        num_envs=3,
        launcher=None,
        creation_error=None,
        attach_native=True,
        device="cpu",
        definition=None,
    ):
        if native is None:
            native = _FakeNative(num_envs)
        if launcher is None:
            launcher = _FakeLauncher(native.events)
        if definition is None:
            definition = IsaacLabEnvironment()
        context = EnvironmentBuildContext(
            purpose="eval" if num_envs == 1 else "train",
            device=device,
            num_envs=num_envs,
            seed=17,
        )

        def create_environment(owner, received_definition, received_context):
            assert received_definition is definition
            assert received_context is context
            owner._launcher = launcher
            owner._direct = definition.task == "Isaac-Cartpole-Direct"
            launcher.__enter__()
            if attach_native:
                owner.native = native
            if creation_error is not None:
                raise creation_error

        monkeypatch.setattr(_isaaclab_runtime, "_create_environment", create_environment)
        runtime = _isaaclab_runtime._IsaacLabRuntime(definition, context)
        runtimes.append(runtime)
        return runtime

    yield build
    for runtime in reversed(runtimes):
        runtime.close()


def _actions(num_envs=3):
    batch = () if num_envs == 1 else (num_envs,)
    return TensorDict({"action": torch.full((*batch, 1), 0.25)}, batch_size=batch)


def test_constructor_owns_one_validated_native_batch(make_runtime):
    runtime = make_runtime()
    assert runtime.native.events == ["launcher.enter", "action.validate"]
    assert runtime._launcher.enter_calls == 1
    assert runtime.native.reset_calls == []
    assert runtime.contract.num_envs == 3
    assert runtime.contract.observations["observation"].shape == (4,)
    assert runtime.contract.actions["action"].shape == (1,)
    assert runtime.device == torch.device("cpu")
    assert runtime.render_fps == pytest.approx(60)
    reset = runtime.reset(seed=29)
    assert reset.batch_size == (3,)
    assert reset["observation"].shape == (3, 4)
    assert runtime.native.reset_calls[0][0] == 29
    assert torch.equal(runtime.native.reset_calls[0][1], torch.arange(3))


def _history_definition(*, owner="package"):
    reference = IsaacLabEnvironment().describe()
    field = reference.observations["observation"].model_copy(
        update={
            "shape": (8,),
            "sensor": SensorProcessingContract(
                owner=owner,
                raw_shape=(4,),
                raw_dtype="float32",
                raw_layout="vector",
                units="unitless",
                scale=2.0,
                offset=-1.0,
                history_length=2,
            ),
        }
    )
    return IsaacLabEnvironment(
        task="History-State-Reference",
        contract=reference.model_copy(update={"observations": {"observation": field}}),
        bindings=IsaacBindings(observations={"observation": ("policy",)}, processing="raw"),
    )


@pytest.mark.parametrize("owner", ["package", "external"])
def test_package_history_uses_final_frames_then_caches_fresh_reset_without_advancing_peers(
    make_runtime,
    owner,
):
    instance = make_runtime(definition=_history_definition(owner=owner))
    first = instance.reset()["observation"]
    native = instance.native
    initial = native.observation.clone() * 2 - 1
    assert torch.equal(first, torch.cat((initial, initial), dim=-1))
    native.next_truncated = [True, False, False]
    transition = instance.step(_actions())
    ending = native.final_observation.clone() * 2 - 1
    assert torch.equal(transition.observation["observation"], torch.cat((initial, ending), dim=-1))
    state = instance._processing_state.clone()
    cached = instance.reset(mask=torch.tensor([True, False, False]))["observation"]
    reset_frame = native.observation.clone() * 2 - 1
    assert torch.equal(cached[0], torch.cat((reset_frame[0], reset_frame[0])))
    assert torch.equal(cached[1:], transition.observation["observation"][1:])
    assert torch.equal(
        state["observation", "frames"], instance._processing_state["observation", "frames"]
    )
    assert len(native.reset_calls) == 1
    native.next_truncated = [False, False, False]
    following = instance.step(_actions()).observation
    next_frame = native.observation.clone() * 2 - 1
    assert torch.equal(following["observation"][0], torch.cat((reset_frame[0], next_frame[0])))


@pytest.mark.parametrize("owner", ["package", "external"])
def test_explicit_partial_reset_initializes_only_selected_package_history(make_runtime, owner):
    instance = make_runtime(definition=_history_definition(owner=owner))
    instance.reset()
    instance.step(_actions())
    previous = instance._processing_state.clone()
    previous_observation = instance._observation.clone()
    result = instance.reset(mask=torch.tensor([False, True, False]))["observation"]
    frame = instance.native.observation[1] * 2 - 1
    assert torch.equal(result[1], torch.cat((frame, frame)))
    assert torch.equal(result[[0, 2]], previous_observation["observation"][[0, 2]])
    assert torch.equal(
        previous["observation", "frames"][[0, 2]],
        instance._processing_state["observation", "frames"][[0, 2]],
    )
    assert len(instance.native.reset_calls) == 2


def test_processed_bindings_never_guess_raw_sensor_transforms(make_runtime):
    definition = _history_definition().model_copy(
        update={
            "bindings": IsaacBindings(observations={"observation": ("policy",)}),
        }
    )
    native = _FakeNative()
    with pytest.raises(ValueError, match="observation dimensions"):
        make_runtime(native, definition=definition)
    assert native.close_calls == 1 and native.step_calls == []


def test_native_lazy_action_target_is_resolved_before_identity_validation(
    make_runtime, monkeypatch
):
    native = _FakeNative()
    native.term.cfg.class_type = "native.actions:JointEffortAction"
    original_import = _isaaclab_runtime.import_module
    resolved = []

    def native_module(name):
        if name == "isaaclab.utils.string":

            def resolve(target):
                resolved.append(target)
                return _EffortAction

            return SimpleNamespace(string_to_callable=resolve)
        return original_import(name)

    monkeypatch.setattr(_isaaclab_runtime, "import_module", native_module)
    instance = make_runtime(native)
    assert resolved == ["native.actions:JointEffortAction"]
    assert instance.native is native


@pytest.mark.parametrize("scale", [100.0, 1.0])
def test_direct_task_validates_its_public_control_mapping_without_a_manager(make_runtime, scale):
    native = _FakeNative()
    native.cfg.cart_dof_name = "slider_to_cart"
    native.cfg.pole_dof_name = "cart_to_pole"
    native.cfg.action_scale = scale
    setattr(native, "cartpole", SimpleNamespace(find_joints=lambda name: ([0], ["slider_to_cart"])))
    original_reset = native.reset
    setattr(
        native, "reset", lambda *, seed=None: original_reset(seed=seed, env_ids=torch.arange(3))
    )
    definition = IsaacLabEnvironment(task="Isaac-Cartpole-Direct", physics="newton_mjwarp")
    if scale == 100:
        runtime = make_runtime(native, definition=definition)
        assert "action.validate" not in native.events
        assert runtime.reset()["observation"].shape == (3, 4)
        with pytest.raises(ValueError, match="explicit partial native reset"):
            runtime.reset(mask=torch.tensor([True, False, False]))
        native.next_truncated = [True, False, False]
        runtime.step(_actions())
        runtime.reset(mask=torch.tensor([True, False, False]))
        assert len(native.reset_calls) == 1
    else:
        with pytest.raises(ValueError, match="direct Cartpole control mapping"):
            make_runtime(native, definition=definition)
        assert native.close_calls == 1


@pytest.mark.parametrize(
    "task", ["Isaac-Cartpole", "Isaac-Reach-Franka", "NexuML-SO101-Joint-Reach"]
)
@pytest.mark.parametrize("substitution", ["velocity", "forged-config"])
def test_reference_rejects_same_shape_wrong_control_implementation(
    make_runtime, monkeypatch, task, substitution
):
    native = _FakeNative()
    definition = IsaacLabEnvironment(task=task)
    contract = definition.describe()
    native.single_observation_space["policy"].shape = contract.observations["observation"].shape
    native.single_action_space.shape = contract.actions["action"].shape
    assert contract.control_period_s is not None
    native.step_dt = contract.control_period_s
    expected = _EffortAction if task == "Isaac-Cartpole" else _PositionAction
    native.term = (_VelocityAction if substitution == "velocity" else expected)(
        cfg=SimpleNamespace(class_type=_VelocityAction),
        IO_descriptor=native.term.IO_descriptor,
    )
    name = "joint_effort" if task == "Isaac-Cartpole" else "arm_action"
    native.action_manager.active_terms = [name]
    native.action_manager.get_term = lambda key: native.term
    import nexuml_library.environments._isaaclab_so101 as so101

    monkeypatch.setattr(so101, "validate_asset", lambda env: None)
    with pytest.raises(ValueError, match="implementation/control mode"):
        make_runtime(native, definition=definition)
    assert native.reset_calls == native.step_calls == []
    assert native.close_calls == 1


@pytest.mark.parametrize("implementation", [_PositionAction, _VelocityAction])
def test_custom_physical_contract_validates_executed_action_not_only_config(
    make_runtime, implementation
):
    native = _FakeNative()
    reference = IsaacLabEnvironment().describe()
    field = reference.actions["action"].model_copy(
        update={
            "physical": PhysicalTensorContract(
                units=("rad",),
                frame="joint",
                joints=("slider_to_cart",),
                control_mode="absolute_position",
                scale=(0.5,),
                offset=(0.0,),
            ),
        }
    )
    definition = IsaacLabEnvironment(
        task="Custom-Physical",
        contract=reference.model_copy(update={"actions": {"action": field}}),
        bindings=IsaacBindings(observations={"observation": ("policy",)}),
    )
    native.term = implementation(
        cfg=_PositionActionCfg(class_type=implementation, use_default_offset=False, clip=None),
        IO_descriptor=SimpleNamespace(joint_names=["slider_to_cart"], scale=0.5, offset=0.0),
    )
    if implementation is _PositionAction:
        assert make_runtime(native, definition=definition).native is native
    else:
        with pytest.raises(ValueError, match="implementation/control mode"):
            make_runtime(native, definition=definition)
        assert native.close_calls == 1 and native.step_calls == []


@pytest.fixture
def native_launch(monkeypatch, tmp_path):
    """Exercise the real constructor with fake public native APIs and VCS metadata."""
    native = _FakeNative()
    native.device = torch.device("cuda:0")
    native.cfg = _ManagerConfig(
        compute_final_obs=True,
        observations=SimpleNamespace(
            policy=SimpleNamespace(history_length=7, enable_corruption=True)
        ),
        sim=SimpleNamespace(physics=SimpleNamespace(), device="cuda:0"),
        scene=SimpleNamespace(num_envs=3),
        seed=17,
    )
    source = tmp_path / "site-packages/isaaclab/__init__.py"
    state = SimpleNamespace(
        module=SimpleNamespace(__file__=str(source)),
        origin={"vcs_info": {"commit_id": IsaacLabEnvironment().revision}},
        has_kit=False,
        launchers=[],
        args=[],
        parsed=[],
        kit_after_launch=False,
    )

    def launch(cfg, args):
        assert cfg is native.cfg
        state.args.append(args)

        class Launcher(_FakeLauncher):
            def __enter__(self):
                result = super().__enter__()
                state.has_kit = state.kit_after_launch
                return result

        launcher = Launcher(native.events)
        state.launchers.append(launcher)
        return launcher

    def parse(task, **kwargs):
        state.parsed.append((task, kwargs))
        native.cfg.scene.num_envs = kwargs["num_envs"]
        native.cfg.sim.device = kwargs["device"]
        if task.endswith("-Direct"):
            native.cfg = _DirectConfig(**vars(native.cfg))
        return native.cfg

    modules = {
        "isaaclab": state.module,
        "isaaclab_tasks": SimpleNamespace(),
        "isaaclab.app": SimpleNamespace(launch_simulation=launch),
        "isaaclab.utils": SimpleNamespace(
            has_kit=lambda: state.has_kit, to_dict=lambda cfg: vars(cfg)
        ),
        "isaaclab_tasks.utils": SimpleNamespace(parse_env_cfg=parse),
        "isaaclab_tasks.utils.hydra": SimpleNamespace(resolve_presets=lambda cfg, **kwargs: cfg),
        "isaaclab.sim": SimpleNamespace(
            SimulationContext=SimpleNamespace(instance=lambda: None, clear_instance=lambda: None)
        ),
        "isaaclab.envs": SimpleNamespace(
            ManagerBasedRLEnv=_FakeNative,
            DirectRLEnv=type("Direct", (), {}),
            ManagerBasedRLEnvCfg=_ManagerConfig,
            DirectRLEnvCfg=_DirectConfig,
        ),
        "isaaclab.envs.mdp": SimpleNamespace(
            JointEffortAction=_EffortAction,
            JointPositionAction=_PositionAction,
        ),
    }
    monkeypatch.setattr(
        _isaaclab_runtime, "sys", SimpleNamespace(version_info=(3, 12), modules=sys.modules)
    )
    monkeypatch.setattr(_isaaclab_runtime, "import_module", modules.__getitem__)
    monkeypatch.setattr(
        _isaaclab_runtime,
        "distribution",
        lambda name: SimpleNamespace(
            read_text=lambda filename: json.dumps(state.origin),
            locate_file=lambda name: source,
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "gymnasium",
        SimpleNamespace(
            make=lambda *args, **kwargs: SimpleNamespace(unwrapped=native),
            registry={
                "NexuML-SO101-Joint-Reach": SimpleNamespace(
                    entry_point="isaaclab.envs:ManagerBasedRLEnv",
                    kwargs={
                        "env_cfg_entry_point": (
                            "nexuml_library.environments._isaaclab_so101:make_so101_config"
                        )
                    },
                ),
                "NexuML-Custom-Cartpole": SimpleNamespace(
                    entry_point="isaaclab.envs:ManagerBasedRLEnv",
                    kwargs={
                        "env_cfg_entry_point": (
                            "nexuml_library.environments._isaaclab_tasks:make_custom_cartpole_config"
                        )
                    },
                ),
                "NexuML-Cartpole-Categorical": SimpleNamespace(
                    entry_point=(
                        "nexuml_library.environments._isaaclab_categorical:make_categorical_cartpole_environment"
                    ),
                    kwargs={
                        "env_cfg_entry_point": (
                            "nexuml_library.environments._isaaclab_categorical:make_categorical_cartpole_config"
                        )
                    },
                ),
                "NexuML-Cartpole-Sensors": SimpleNamespace(
                    entry_point="isaaclab.envs:ManagerBasedRLEnv",
                    kwargs={
                        "env_cfg_entry_point": (
                            "isaaclab_tasks.core.cartpole.cartpole_manager_camera_env_cfg:CartpoleCameraEnvCfg"
                        )
                    },
                ),
            },
            spec=lambda task: SimpleNamespace(id=task),
        ),
    )
    context = EnvironmentBuildContext(purpose="train", device="cuda:0", num_envs=3, seed=17)
    state.build = lambda context=context, **kwargs: _isaaclab_runtime._IsaacLabRuntime(
        IsaacLabEnvironment(**kwargs), context
    )
    state.native = native
    state.modules = modules
    return state


@pytest.mark.parametrize("physics", ["isaacsim_physx", "newton_mjwarp", "ovphysx", None])
def test_native_startup_disables_telemetry_and_crash_uploads_before_launch(native_launch, physics):
    with _isaaclab_runtime.isaaclab_process():
        native_launch.build(physics=physics)
        assert native_launch.args[0]["kit_args"].split() == [
            "--/crashreporter/enabled=false",
            "--/crashreporter/skipOldDumpUpload=true",
            "--/app/uploadDumpsOnStartup=false",
            "--/telemetry/enableAnonymousData=false",
            "--/telemetry/enableAnonymousAppName=false",
            "--/telemetry/enableSentry=false",
            "--/structuredLog/enable=false",
        ]
        assert native_launch.args[0]["headless"] is True
        assert native_launch.args[0]["visualizer"] == ["none"]


@pytest.mark.parametrize(
    "name,value",
    [("OMNI_CRASHREPORTER_ENABLED", value) for value in ("1", "t", "Y", "true", "")]
    + [("OMNI_CRASHREPORTER_SKIPOLDDUMPUPLOAD", value) for value in ("0", "f", "N", "false", "")],
)
def test_conflicting_crash_upload_environment_rejected_without_mutation(
    native_launch, monkeypatch, name, value
):
    import os

    monkeypatch.setenv(name, value)
    before = dict(os.environ)
    with _isaaclab_runtime.isaaclab_process(), pytest.raises(ValueError, match=name):
        native_launch.build()
    assert dict(os.environ) == before
    assert native_launch.args == []
    assert native_launch.launchers == []


@pytest.mark.parametrize(
    "disabled,skip", [("0", "1"), ("n", "y"), ("N", "Y"), ("f", "t"), ("F", "T")]
)
def test_safe_crash_upload_environment_preserved(native_launch, monkeypatch, disabled, skip):
    import os

    monkeypatch.setenv("OMNI_CRASHREPORTER_ENABLED", disabled)
    monkeypatch.setenv("OMNI_CRASHREPORTER_SKIPOLDDUMPUPLOAD", skip)
    before = dict(os.environ)
    with _isaaclab_runtime.isaaclab_process():
        native_launch.build()
    assert dict(os.environ) == before
    assert len(native_launch.args) == 1


@pytest.mark.parametrize("purpose", ["train", "eval"])
def test_viewer_requires_a_separate_purpose_opt_in_without_importing_viser(native_launch, purpose):
    from nexuml_library.environments.isaaclab_config import ViserSpec

    count = 3 if purpose == "train" else 1
    native_launch.native.num_envs = count
    context = EnvironmentBuildContext(purpose=purpose, device="cuda:0", num_envs=count, seed=17)
    with _isaaclab_runtime.isaaclab_process():
        runtime = native_launch.build(
            context=context,
            viewer=ViserSpec(
                purposes=("eval",) if purpose == "train" else ("train",),
            ),
        )
        assert native_launch.args[0]["visualizer"] == ["none"]
        assert runtime._spectator_type is None


@pytest.mark.parametrize("fail", ["none", "construction"])
def test_exact_spectator_selection_and_partial_native_ownership(native_launch, monkeypatch, fail):
    from nexuml_library.environments.isaaclab_config import ViserSpec

    events = native_launch.native.events

    class Spectator:
        def close(self):
            events.append("spectator.close")

    spectator = Spectator()
    sim = SimpleNamespace(
        visualizers=[spectator] if fail == "none" else [],
        _pending_visualizers=[] if fail == "none" else [spectator],
    )

    def configure(cfg, spec, count):
        assert spec.port == 8080 and count == 3
        cfg.sim.visualizer_cfgs = [SimpleNamespace(class_type=Spectator)]

    native_launch.modules["nexuml_library.environments._isaaclab_viser"] = SimpleNamespace(
        configure_spectator=configure,
    )
    native_launch.modules["isaaclab.sim"] = SimpleNamespace(
        SimulationContext=SimpleNamespace(
            instance=lambda: sim, clear_instance=lambda: events.append("context.close")
        )
    )
    native_launch.native.sim = sim
    if fail == "construction":

        def fail_make(*args, **kwargs):
            raise RuntimeError("native construction failed after viewer startup")

        monkeypatch.setattr(sys.modules["gymnasium"], "make", fail_make)
    with _isaaclab_runtime.isaaclab_process():
        if fail == "construction":
            with pytest.raises(RuntimeError, match="native construction failed"):
                native_launch.build(viewer=ViserSpec())
        else:
            runtime = native_launch.build(viewer=ViserSpec())
            assert runtime._spectator_type is Spectator
            runtime.close()
            assert events.index("spectator.close") < events.index("native.close")
    assert native_launch.args[0]["visualizer"] == ["viser"]
    assert events.count("spectator.close") == 1
    assert events.count("context.close") == 1
    assert native_launch.launchers[0].exit_calls


def test_an_existing_kitless_context_is_not_adopted_or_destroyed(native_launch, monkeypatch):
    foreign = object()
    context = SimpleNamespace(instance=lambda: foreign)
    monkeypatch.setitem(
        sys.modules, "isaaclab.sim.simulation_context", SimpleNamespace(SimulationContext=context)
    )
    with _isaaclab_runtime.isaaclab_process():
        with pytest.raises(RuntimeError, match="already active"):
            native_launch.build()
    assert context.instance() is foreign
    assert native_launch.launchers == []


def test_headless_native_constructor_failure_releases_its_partial_context(
    native_launch, monkeypatch
):
    state = SimpleNamespace(sim=None)
    events = native_launch.native.events

    def clear():
        state.sim = None
        events.append("context.close")

    native_launch.modules["isaaclab.sim"] = SimpleNamespace(
        SimulationContext=SimpleNamespace(instance=lambda: state.sim, clear_instance=clear)
    )

    def fail_make(*args, **kwargs):
        state.sim = object()
        raise RuntimeError("headless native constructor failed")

    monkeypatch.setattr(sys.modules["gymnasium"], "make", fail_make)
    with _isaaclab_runtime.isaaclab_process():
        with pytest.raises(RuntimeError, match="headless native constructor failed"):
            native_launch.build()
    assert state.sim is None and events.count("context.close") == 1
    assert native_launch.launchers[0].exit_calls


@pytest.mark.parametrize("physics", ["newton_mjwarp", "isaacsim_physx", "ovphysx"])
def test_canonical_wheel_launch_and_profile_specific_owned_shutdown(native_launch, physics):
    native_launch.kit_after_launch = physics == "isaacsim_physx"
    with _isaaclab_runtime.isaaclab_process():
        runtime = native_launch.build(physics=physics)
        assert native_launch.parsed == [
            (
                "Isaac-Cartpole",
                {
                    "device": "cuda:0",
                    "num_envs": 3,
                    "overrides": [f"physics={physics}"],
                },
            )
        ]
        assert native_launch.args[0]["fast_shutdown"] is True
        assert native_launch.native.cfg.compute_final_obs
        assert native_launch.native.cfg.observations.policy.history_length == 7
        assert native_launch.native.cfg.observations.policy.enable_corruption is True
        with pytest.raises(RuntimeError, match="already active"):
            native_launch.build(physics=physics)
        assert _isaaclab_runtime._active_runtime is runtime
        runtime.close()
        assert _isaaclab_runtime._active_runtime is None
        assert bool(native_launch.launchers[0].exit_calls) == (physics != "isaacsim_physx")
        native_launch.native.events.append("job.finalized")
    assert len(native_launch.launchers[0].exit_calls) == 1
    if physics == "isaacsim_physx":
        assert native_launch.native.events[-2:] == ["job.finalized", "launcher.exit"]


@pytest.mark.parametrize(
    "invalid", ["revision", "shadowed-import", "missing-origin", "foreign-kit"]
)
def test_invalid_source_or_foreign_runtime_fails_before_launch(native_launch, invalid):
    if invalid == "revision":
        native_launch.origin["vcs_info"]["commit_id"] = "0" * 40
    elif invalid == "shadowed-import":
        native_launch.module.__file__ += ".shadow"
    elif invalid == "missing-origin":
        native_launch.origin.clear()
    else:
        native_launch.has_kit = True
    with _isaaclab_runtime.isaaclab_process():
        with pytest.raises((ValueError, RuntimeError), match="pinned|origin|already active"):
            native_launch.build()
        assert not native_launch.launchers
        assert _isaaclab_runtime._active_runtime is None


def test_unpinned_python_fails_before_launcher_allocation(native_launch, monkeypatch):
    monkeypatch.setattr(
        _isaaclab_runtime, "sys", SimpleNamespace(version_info=(3, 13), modules=sys.modules)
    )
    with _isaaclab_runtime.isaaclab_process():
        with pytest.raises(ValueError, match="source revision/Python.*pinned profile"):
            native_launch.build()
    assert native_launch.launchers == []
    assert native_launch.parsed == []
    assert _isaaclab_runtime._active_runtime is None


def test_failed_launcher_enter_is_not_exited_twice(native_launch, monkeypatch):
    def fail_enter(launcher):
        raise RuntimeError("native launch failed")

    monkeypatch.setattr(_FakeLauncher, "__enter__", fail_enter)
    with _isaaclab_runtime.isaaclab_process():
        with pytest.raises(RuntimeError, match="native launch failed"):
            native_launch.build()
        assert not native_launch.launchers[0].exit_calls
        assert _isaaclab_runtime._active_runtime is None


@pytest.mark.parametrize(
    "task,physics",
    [
        ("Isaac-Cartpole", "newton_mjwarp"),
        ("Isaac-Cartpole", "ovphysx"),
        ("Isaac-Cartpole-Direct", "isaacsim_physx"),
    ],
)
def test_unqualified_renderer_combinations_fail_before_native_launch(task, physics, native_launch):
    context = EnvironmentBuildContext(
        purpose="eval",
        device="cuda:0",
        num_envs=1,
        render_mode="rgb_array",
    )
    with _isaaclab_runtime.isaaclab_process():
        with pytest.raises(ValueError, match="manager task with Kit PhysX"):
            native_launch.build(task=task, physics=physics, context=context)
    assert not native_launch.launchers


def test_actual_source_checkout_revision_is_used_without_wheel_vcs_metadata(
    native_launch, monkeypatch, tmp_path
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").write_text("gitdir: fixture-only\n")
    native_launch.origin.clear()
    native_launch.module.__file__ = str(checkout / "source/isaaclab/isaaclab/__init__.py")
    calls = []

    def revision(command, *, text):
        calls.append(command)
        return IsaacLabEnvironment().revision + "\n"

    monkeypatch.setattr(_isaaclab_runtime.subprocess, "check_output", revision)
    with _isaaclab_runtime.isaaclab_process():
        runtime = native_launch.build(physics="newton_mjwarp")
        runtime.close()
    assert calls == [["git", "-C", str(checkout), "rev-parse", "HEAD"]]


@pytest.mark.parametrize("setting", ["count", "device", "seed", "physics", "final-state"])
def test_customizer_cannot_change_owned_settings_before_launch(native_launch, setting):
    with _isaaclab_runtime.isaaclab_process():
        with pytest.raises(ValueError, match="protected"):
            native_launch.build(
                task_config=isaac_task_config(_change_owned_setting, setting=setting)
            )
    assert not native_launch.launchers
    assert _isaaclab_runtime._active_runtime is None


def test_kit_ownership_is_detected_after_launch_not_inferred_from_physics(native_launch):
    native_launch.kit_after_launch = True
    with _isaaclab_runtime.isaaclab_process():
        instance = native_launch.build(physics="newton_mjwarp")
        instance.close()
        assert not native_launch.launchers[0].exit_calls
    assert len(native_launch.launchers[0].exit_calls) == 1


def test_build_cannot_bypass_a_conflicting_job_definition(native_launch):
    with IsaacLabEnvironment(task="Isaac-Reach-Franka").job_scope():
        with pytest.raises(RuntimeError, match="conflicts with the active"):
            native_launch.build()
    assert not native_launch.parsed and not native_launch.launchers


def test_custom_nested_observations_and_named_action_use_explicit_bindings(
    make_runtime, monkeypatch
):
    native = _FakeNative()
    native.single_observation_space = {"sensors": {"state": SimpleNamespace(shape=(4,))}}
    original_reset, original_step = native.reset, native.step

    def reset(**kwargs):
        observations, info = original_reset(**kwargs)
        return {"sensors": {"state": observations["policy"]}}, info

    def step(actions):
        observations, reward, terminated, truncated, info = original_step(actions)
        info["final_obs"] = {"sensors": {"state": info["final_obs"]["policy"]}}
        return {"sensors": {"state": observations["policy"]}}, reward, terminated, truncated, info

    monkeypatch.setattr(native, "reset", reset)
    monkeypatch.setattr(native, "step", step)
    reference = IsaacLabEnvironment().describe()
    contract = reference.model_copy(
        update={
            "observations": {"state": reference.observations["observation"]},
            "actions": {"command": reference.actions["action"]},
        }
    )
    definition = IsaacLabEnvironment(
        task="Registered-Custom",
        contract=contract,
        bindings=IsaacBindings(observations={"state": ("sensors", "state")}, action="command"),
    )
    instance = make_runtime(native, definition=definition)
    assert instance.reset()["state"].shape == (3, 4)
    native.next_truncated = [True, False, False]
    transition = instance.step(TensorDict({"command": torch.zeros(3, 1)}, batch_size=(3,)))
    assert set(transition.observation.keys()) == {"state"}
    assert torch.equal(transition.observation["state"], native.final_observation)
    assert torch.equal(
        instance.reset(mask=torch.tensor([True, False, False]))["state"], native.observation
    )


def test_unindexed_cuda_alias_matches_native_cuda_zero_without_initializing_cuda(make_runtime):
    native = _FakeNative()
    native.device = torch.device("cuda:0")
    runtime = make_runtime(native, device="cuda")
    assert runtime.device == torch.device("cuda:0")
    assert native.events == ["launcher.enter", "action.validate"]


@pytest.mark.parametrize(
    "terminated,truncated",
    [
        pytest.param([False] * 3, [False] * 3, id="none-done"),
        pytest.param([True, False, False], [False] * 3, id="partial-termination"),
        pytest.param([False] * 3, [False, True, False], id="partial-truncation"),
        pytest.param([True, False, False], [True, False, False], id="partial-both"),
        pytest.param([True] * 3, [False] * 3, id="all-termination"),
        pytest.param([False] * 3, [True] * 3, id="all-truncation"),
        pytest.param([True] * 3, [True] * 3, id="all-both"),
        pytest.param([True, False, True], [False, True, True], id="all-mixed"),
    ],
)
def test_same_step_truth_table_keeps_final_and_reset_states_separate(
    make_runtime, terminated, truncated
):
    runtime = make_runtime()
    runtime.reset()
    native = runtime.native
    native.next_terminated, native.next_truncated = terminated, truncated
    transition = runtime.step(_actions())
    expected_final = torch.arange(12, dtype=torch.float32).reshape(3, 4) + 100
    done = torch.tensor(terminated) | torch.tensor(truncated)
    expected_next = expected_final.clone()
    expected_next[done] -= 200
    assert transition.observation.batch_size == (3,)
    assert torch.equal(transition.observation["observation"], expected_final)
    assert (
        transition.reward.shape
        == transition.terminated.shape
        == transition.truncated.shape
        == (3, 1)
    )
    assert torch.equal(transition.reward, torch.tensor([[1.5], [2.5], [3.5]]))
    assert torch.equal(transition.terminated[:, 0], torch.tensor(terminated))
    assert torch.equal(transition.truncated[:, 0], torch.tensor(truncated))
    assert transition.reward.dtype == torch.float32
    assert transition.terminated.dtype == transition.truncated.dtype == torch.bool
    assert transition.observation.device == transition.reward.device == torch.device("cpu")
    next_observation = runtime.reset(mask=done)["observation"]
    assert torch.equal(next_observation, expected_next)
    assert torch.equal(next_observation[~done], expected_final[~done])
    assert len(native.reset_calls) == 1, "acknowledging auto-reset must not reset physics twice"
    assert torch.equal(native.step_calls[0], _actions()["action"])


def test_pending_rows_can_be_acknowledged_separately_without_disturbing_continuing_rows(
    make_runtime,
):
    runtime = make_runtime()
    runtime.reset()
    native = runtime.native
    native.next_terminated = [True, True, False]
    transition = runtime.step(_actions())
    cached = native.observation.clone()
    for mask in ([True, False, False], [False, True, False]):
        observation = runtime.reset(mask=torch.tensor(mask).reshape(3, 1))
        assert torch.equal(observation["observation"], cached)
    assert len(native.reset_calls) == 1
    assert torch.equal(transition.observation["observation"][2], cached[2])


def test_explicit_masked_reset_only_resets_requested_nonpending_rows(make_runtime):
    runtime = make_runtime()
    runtime.reset()
    native = runtime.native
    native.next_terminated = [True, False, False]
    runtime.step(_actions())
    before = native.observation.clone()
    observation = runtime.reset(seed=91, mask=torch.tensor([True, True, False]))["observation"]
    assert len(native.reset_calls) == 2
    assert native.reset_calls[-1][0] == 91
    assert torch.equal(native.reset_calls[-1][1], torch.tensor([1]))
    assert torch.equal(observation[[0, 2]], before[[0, 2]])
    assert torch.equal(observation[1], torch.tensor([2004.0, 2005.0, 2006.0, 2007.0]))
    # Once acknowledged, another explicit request starts a new episode.
    runtime.reset(mask=torch.tensor([True, False, False]))
    assert torch.equal(native.reset_calls[-1][1], torch.tensor([0]))


@pytest.mark.parametrize("pending", [False, True])
def test_explicit_full_reset_always_calls_native_even_after_auto_reset(make_runtime, pending):
    runtime = make_runtime()
    runtime.reset()
    native = runtime.native
    native.next_truncated = [pending] * 3
    runtime.step(_actions())
    for seed in (101, 102):
        reset = runtime.reset(seed=seed)
        assert native.reset_calls[-1][0] == seed
        assert torch.equal(native.reset_calls[-1][1], torch.arange(3))
        assert torch.equal(reset["observation"], native.observation)
    assert len(native.reset_calls) == 3


@pytest.mark.parametrize(
    "terminated,truncated", [(False, False), (True, False), (False, True), (True, True)]
)
def test_singleton_evaluation_unbatches_outputs_and_rebatches_actions(
    make_runtime, terminated, truncated
):
    runtime = make_runtime(num_envs=1)
    initial = runtime.reset()
    assert initial.batch_size == ()
    assert initial["observation"].shape == (4,)
    native = runtime.native
    native.next_terminated, native.next_truncated = [terminated], [truncated]
    transition = runtime.step(_actions(1))
    assert transition.observation.batch_size == ()
    assert transition.observation["observation"].shape == (4,)
    assert torch.equal(transition.observation["observation"], torch.arange(4) + 100)
    assert (
        transition.reward.shape == transition.terminated.shape == transition.truncated.shape == (1,)
    )
    assert transition.terminated.item() is terminated
    assert transition.truncated.item() is truncated
    assert torch.equal(native.step_calls[0], torch.tensor([[0.25]]))
    next_observation = runtime.reset(mask=torch.tensor(terminated or truncated))
    assert next_observation.batch_size == ()
    assert torch.equal(next_observation["observation"], native.observation[0])
    assert len(native.reset_calls) == 1


def test_returned_buffers_survive_later_native_steps_and_resets(make_runtime):
    runtime = make_runtime()
    initial = runtime.reset()
    expected_initial = initial["observation"].clone()
    native = runtime.native
    native.next_terminated = [True, False, False]
    native.next_truncated = [False, True, False]
    first = runtime.step(_actions())
    expected_observation = first.observation["observation"].clone()
    expected_reward = first.reward.clone()
    expected_terminated, expected_truncated = first.terminated.clone(), first.truncated.clone()
    cached = runtime.reset(mask=torch.tensor([True, True, False]))
    expected_cached = cached["observation"].clone()
    native.next_terminated = native.next_truncated = [False] * 3
    runtime.step(_actions())
    runtime.reset()
    assert torch.equal(initial["observation"], expected_initial)
    assert torch.equal(cached["observation"], expected_cached)
    assert torch.equal(first.observation["observation"], expected_observation)
    assert torch.equal(first.reward, expected_reward)
    assert torch.equal(first.terminated, expected_terminated)
    assert torch.equal(first.truncated, expected_truncated)


def test_mutating_returned_reset_state_cannot_corrupt_runtime_cache(make_runtime):
    runtime = make_runtime()
    initial = runtime.reset()
    expected = initial["observation"].clone()
    initial["observation"].fill_(-123)
    cached = runtime.reset(mask=torch.zeros(3, dtype=torch.bool))
    assert torch.equal(cached["observation"], expected)
    cached["observation"].fill_(-456)
    assert torch.equal(
        runtime.reset(mask=torch.zeros(3, dtype=torch.bool))["observation"], expected
    )
    assert len(runtime.native.reset_calls) == 1


@pytest.mark.requires_optional("torchrl")
@pytest.mark.parametrize(
    "terminated,truncated",
    [
        ([False] * 3, [False] * 3),
        ([True, False, False], [False, True, False]),
        ([True] * 3, [False] * 3),
        ([False] * 3, [True] * 3),
        ([True] * 3, [True] * 3),
    ],
)
def test_generic_adapter_handoff_and_exact_two_step_gae(make_runtime, terminated, truncated):
    from torchrl.objectives.value.functional import generalized_advantage_estimate

    from nexuml.reinforcement.torchrl_env import _TorchRLEnvironmentAdapter

    runtime = make_runtime()
    native = runtime.native
    env = _TorchRLEnvironmentAdapter(runtime)
    try:
        root = env.reset()
        native.next_terminated, native.next_truncated = terminated, truncated
        root["action"] = _actions()["action"]
        first, next_root = env.step_and_maybe_reset(root)
        current_values = [root["observation"][:, :1].clone()]
        current_values.append(next_root["observation"][:, :1].clone())
        assert len(native.reset_calls) == 1, "all-done and partial masks must consume the cache"
        torch.testing.assert_close(next_root["observation"], native.observation)
        final = first["next", "observation"].clone()
        native.next_terminated = native.next_truncated = [False] * 3
        next_root["action"] = _actions()["action"]
        second, _ = env.step_and_maybe_reset(next_root)
        assert torch.equal(first["next", "observation"], final)

        rollout = torch.stack([first, second], dim=1)
        assert isinstance(rollout, TensorDict)
        assert rollout.batch_size == (3, 2) and rollout.numel() == 6
        value = torch.stack(current_values, dim=1)
        next_value = rollout["next", "observation"][..., :1]
        reward = rollout["next", "reward"]
        done, term = rollout["next", "done"], rollout["next", "terminated"]
        assert isinstance(next_value, torch.Tensor)
        assert isinstance(reward, torch.Tensor)
        assert isinstance(done, torch.Tensor)
        assert isinstance(term, torch.Tensor)
        advantage, target = generalized_advantage_estimate(
            gamma=0.9,
            lmbda=0.95,
            state_value=value,
            next_state_value=next_value,
            reward=reward,
            done=done,
            terminated=term,
        )
        delta = reward + 0.9 * (~term) * next_value - value
        expected = delta.clone()
        expected[:, 0] += 0.9 * 0.95 * (~done[:, 0]) * delta[:, 1]
        torch.testing.assert_close(advantage, expected)
        torch.testing.assert_close(target, expected + value)
        timeout = torch.tensor(truncated) & ~torch.tensor(terminated)
        if timeout.any():
            torch.testing.assert_close(
                target[timeout, 0], reward[timeout, 0] + 0.9 * final[timeout, :1]
            )
            wrong = reward[timeout, 0] + 0.9 * current_values[1][timeout]
            assert not torch.allclose(target[timeout, 0], wrong)
    finally:
        env.close()


@pytest.mark.parametrize("done", [False, True])
def test_final_observations_are_required_only_when_a_row_finishes(make_runtime, done):
    runtime = make_runtime()
    runtime.reset()
    runtime.native.include_final_obs = False
    runtime.native.next_truncated = [False, done, False]
    if done:
        with pytest.raises(ValueError, match="pre-reset final_obs"):
            runtime.step(_actions())
    else:
        assert torch.equal(
            runtime.step(_actions()).observation["observation"], runtime.native.observation
        )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("num_envs", 4, "dimensions/device/control period"),
        ("device", "meta", "dimensions/device/control period"),
        ("single_observation_space", {"policy": SimpleNamespace(shape=(5,))}, "dimensions/device"),
        ("single_action_space", SimpleNamespace(shape=(2,)), "dimensions/device"),
        ("step_dt", 1 / 30, "control period"),
        ("cfg", SimpleNamespace(compute_final_obs=False), "final-state settings"),
        ("active_terms", ["other_term"], "action terms"),
        ("joint_names", ["wrong_joint"], "joint order/scale/offset/clipping"),
        ("offset", 0.1, "joint order/scale/offset/clipping"),
        ("scale", 1.0, "joint order/scale/offset/clipping"),
        ("clip", (-1.0, 1.0), "joint order/scale/offset/clipping"),
    ],
)
def test_constructor_rejects_native_contract_mismatches_and_closes_owners(
    make_runtime, field, value, match
):
    native = _FakeNative()
    target = native
    if field == "active_terms":
        target = native.action_manager
    elif field in {"joint_names", "offset"}:
        target = native.term.IO_descriptor
    elif field in {"scale", "clip"}:
        target = native.term.cfg
    setattr(target, field, value)
    launcher = _FakeLauncher(native.events)
    with pytest.raises(ValueError, match=match):
        make_runtime(native, launcher=launcher)
    assert native.close_calls == 1 and len(launcher.exit_calls) == 1
    assert native.reset_calls == native.step_calls == []
    assert native.events[-2:] == ["native.close", "launcher.exit"]


@pytest.mark.parametrize(
    "value,match",
    [
        (torch.zeros(3, 5), "shape/dtype/device"),
        (torch.zeros(3, 4, dtype=torch.float64), "shape/dtype/device"),
        (torch.empty(3, 4, device="meta"), "shape/dtype/device"),
        (torch.full((3, 4), float("nan")), "finite"),
        (torch.full((3, 4), float("inf")), "finite"),
    ],
)
@pytest.mark.parametrize("source", ["reset", "step", "final_obs"])
def test_native_observation_shape_dtype_device_and_finiteness_are_validated(
    make_runtime, monkeypatch, value, match, source
):
    runtime = make_runtime()
    native = runtime.native
    if source == "reset":
        monkeypatch.setattr(native, "reset", lambda **kwargs: ({"policy": value}, {}))
        with pytest.raises(ValueError, match=match):
            runtime.reset()
    else:
        runtime.reset()
        original_step = native.step

        def step(action):
            observations, reward, terminated, truncated, info = original_step(action)
            if source == "step":
                observations = {"policy": value}
            else:
                terminated[0] = True
                info["final_obs"] = {"policy": value}
            return observations, reward, terminated, truncated, info

        monkeypatch.setattr(native, "step", step)
        with pytest.raises(ValueError, match=match):
            runtime.step(_actions())


@pytest.mark.parametrize("field", ["reward", "terminated", "truncated"])
@pytest.mark.parametrize("invalid", ["shape", "dtype", "device"])
def test_native_transition_signal_shape_dtype_and_device_are_validated(
    make_runtime, monkeypatch, field, invalid
):
    runtime = make_runtime()
    runtime.reset()
    native = runtime.native
    original_step = native.step

    def step(action):
        result = list(original_step(action))
        index = {"reward": 1, "terminated": 2, "truncated": 3}[field]
        value = result[index]
        if invalid == "shape":
            value = value.reshape(3, 1)
        elif invalid == "dtype":
            value = value.to(torch.float64 if field == "reward" else torch.int64)
        else:
            value = value.to("meta")
        result[index] = value
        return tuple(result)

    monkeypatch.setattr(native, "step", step)
    with pytest.raises(ValueError, match=rf"{field} shape/dtype/device"):
        runtime.step(_actions())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_native_reward_must_be_finite(make_runtime, monkeypatch, value):
    runtime = make_runtime()
    runtime.reset()
    original_step = runtime.native.step

    def step(action):
        result = original_step(action)
        result[1][0] = value
        return result

    monkeypatch.setattr(runtime.native, "step", step)
    with pytest.raises(ValueError, match="rewards must be finite"):
        runtime.step(_actions())


@pytest.mark.parametrize(
    "value,batch,match",
    [
        (torch.zeros(3, 2), (3,), "must have shape"),
        (torch.zeros(3, 1, dtype=torch.float64), (3,), "must have dtype"),
        (torch.zeros(1), (), "native batch and device"),
        (torch.full((3, 1), float("nan")), (3,), "nonfinite"),
        (torch.full((3, 1), float("inf")), (3,), "upper bound"),
        (torch.full((3, 1), -float("inf")), (3,), "lower bound"),
        (torch.full((3, 1), 1.01), (3,), "upper bound"),
        (torch.full((3, 1), -1.01), (3,), "lower bound"),
        (None, (3,), "missing required field"),
    ],
)
def test_invalid_actions_are_rejected_before_native_step(make_runtime, value, batch, match):
    runtime = make_runtime()
    runtime.reset()
    actions = TensorDict({"action": value} if value is not None else {}, batch_size=batch)
    with pytest.raises(ValueError, match=match):
        runtime.step(actions)
    assert runtime.native.step_calls == []


def test_actions_on_wrong_device_fail_before_tensor_computation(make_runtime):
    runtime = make_runtime()
    runtime.reset()
    actions = TensorDict({"action": torch.empty(3, 1, device="meta")}, batch_size=(3,))
    with pytest.raises(ValueError, match="native batch and device"):
        runtime.step(actions)
    assert runtime.native.step_calls == []


@pytest.mark.parametrize("mask", [torch.ones(3), torch.ones(2, dtype=torch.bool)])
def test_invalid_reset_masks_do_not_call_native_reset(make_runtime, mask):
    runtime = make_runtime()
    runtime.reset()
    with pytest.raises(ValueError, match="one boolean per environment"):
        runtime.reset(mask=mask)
    assert len(runtime.native.reset_calls) == 1


def test_closed_or_uninitialized_runtime_rejects_operations(make_runtime):
    runtime = make_runtime()
    native = runtime.native
    with pytest.raises(RuntimeError, match="Reset an open Isaac runtime"):
        runtime.step(_actions())
    runtime.reset()
    runtime.close()
    with pytest.raises(RuntimeError, match="closed"):
        runtime.reset()
    with pytest.raises(RuntimeError, match="Reset an open Isaac runtime"):
        runtime.step(_actions())
    assert len(native.reset_calls) == 1
    assert native.step_calls == []


def test_owned_close_is_ordered_and_idempotent(make_runtime):
    runtime = make_runtime()
    native, launcher = runtime.native, runtime._launcher
    runtime.reset()
    runtime.step(_actions())
    runtime.close()
    runtime.close()
    assert native.close_calls == 1
    assert launcher.exit_calls == [(None, None, None)]
    assert native.events[-2:] == ["native.close", "launcher.exit"]
    assert runtime.native is None and runtime._launcher is None
    assert runtime._observation is None and runtime._pending_reset is None


def test_task_outcomes_are_cloned_before_native_cache_reuse(make_runtime):
    runtime = make_runtime()
    runtime._so101 = True
    native = runtime.native
    native.joint_reach_success = torch.tensor([False, True, False])
    native.joint_reach_error = torch.tensor([0.3, 0.01, 0.2])
    runtime.reset()
    native.next_terminated = [False, True, False]
    transition = runtime.step(_actions())
    native.joint_reach_success.zero_()
    native.joint_reach_error.fill_(99)
    assert torch.equal(transition.info["task_success"], torch.tensor([[False], [True], [False]]))
    torch.testing.assert_close(
        transition.info["joint_error_rad"], torch.tensor([[0.3], [0.01], [0.2]])
    )


@pytest.mark.parametrize("fail_native,fail_launcher", [(True, False), (False, True), (True, True)])
def test_close_attempts_all_owners_and_preserves_first_cleanup_error(
    make_runtime, fail_native, fail_launcher
):
    native_error, launcher_error = (
        RuntimeError("native close failed"),
        RuntimeError("launcher close failed"),
    )
    native = _FakeNative(close_error=native_error if fail_native else None)
    launcher = _FakeLauncher(native.events, close_error=launcher_error if fail_launcher else None)
    runtime = make_runtime(native, launcher=launcher)
    with pytest.raises(RuntimeError) as caught:
        runtime.close()
    assert caught.value is (native_error if fail_native else launcher_error)
    if fail_native and fail_launcher:
        assert "launcher close failed" in "\n".join(caught.value.__notes__)
    runtime.close()
    assert native.close_calls == len(launcher.exit_calls) == 1
    assert native.events[-2:] == ["native.close", "launcher.exit"]


@pytest.mark.parametrize("attach_native", [False, True])
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError, KeyboardInterrupt])
def test_initialization_failure_or_cancellation_preserves_primary_and_releases_owners(
    make_runtime, attach_native, error_type
):
    primary = error_type("initialization interrupted")
    native = _FakeNative()
    launcher = _FakeLauncher(native.events)
    with pytest.raises(error_type) as caught:
        make_runtime(native, launcher=launcher, creation_error=primary, attach_native=attach_native)
    assert caught.value is primary
    assert launcher.enter_calls == len(launcher.exit_calls) == 1
    assert native.close_calls == int(attach_native)
    assert native.reset_calls == native.step_calls == []


@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError, KeyboardInterrupt])
@pytest.mark.parametrize("fail_native,fail_launcher", [(True, False), (False, True), (True, True)])
def test_initialization_primary_and_all_cleanup_failures_remain_visible(
    make_runtime, error_type, fail_native, fail_launcher
):
    primary = error_type("primary initialization failure")
    native = _FakeNative(close_error=RuntimeError("native close failed") if fail_native else None)
    launcher = _FakeLauncher(
        native.events, close_error=RuntimeError("launcher close failed") if fail_launcher else None
    )
    with pytest.raises(error_type) as caught:
        make_runtime(native, launcher=launcher, creation_error=primary)
    assert caught.value is primary
    assert native.close_calls == len(launcher.exit_calls) == 1
    rendered = "".join(traceback.format_exception(caught.value))
    if fail_native:
        assert "native close failed" in rendered
    if fail_launcher:
        assert "launcher close failed" in rendered
