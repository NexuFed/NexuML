"""Process-owned Isaac application and execution-local transition normalization."""

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from importlib import import_module
from importlib.metadata import distribution
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import current_thread, main_thread
from time import monotonic_ns
import traceback
from typing import Any, cast

import torch
from tensordict import TensorDict, TensorDictBase

from nexuml.core.policy import CompiledPolicy
from nexuml.core.components import LayerBuildContext
from nexuml.core.types import PhysicalTensorContract
from nexuml.interaction.playback_controls import PlaybackControls
from nexuml.interaction.runtime import InteractionStep, InteractionTrace
from nexuml_library.environments.isaaclab_config import IsaacPlaybackOptions


_process_active = False
_process_definition = None
_process_launcher = None
_process_returning_close = False
_active_runtime = None

_KIT_EDITOR_EXTENSIONS = (
    "omni.kit.window.stage",
    "omni.kit.window.property",
    "omni.kit.window.script_editor",
    "omni.kit.manipulator.prim",
    "omni.kit.window.extensions",
    "omni.kit.window.console",
)
_KIT_LOCAL_ARGS = (
    " --enable omni.kit.widget.toolbar"
    " --/exts/omni.renderer.core/present/enabled=true"
    " --/exts/omni.kit.widget.toolbar/Grab/enabled=false"
    " --/exts/omni.kit.widget.toolbar/PlayButton/enabled=false"
    " --/exts/omni.kit.widget.toolbar/SelectionButton/enabled=false"
    " --/exts/omni.kit.widget.toolbar/TransformButton/enabled=false"
)


def _terminate_job(signum, _frame):
    """Unwind cancellation before Kit's non-returning fast close.

    Raises:
        SystemExit: With the conventional signal-derived process status.
    """
    raise SystemExit(128 + signum)


def _viser_selected(viewer, context):
    return (
        viewer is not None
        and context.purpose in viewer.purposes
        and (context.purpose != "play" or context.viewer == "viser")
        and (context.purpose != "train" or context.worker_rank == (viewer.train_rank or 0))
    )


@contextmanager
def isaaclab_process(definition=None, *, returning_close: bool = False):
    """Close an owned Kit app after the complete job or the worker's rollout lifetime.

    Raises:
        RuntimeError: If a nested scope selects a different native job.
    """
    global _process_active, _process_definition, _process_launcher, _process_returning_close
    if _process_active:
        if returning_close != _process_returning_close or (
            definition is not None
            and _process_definition is not None
            and definition != _process_definition
        ):
            raise RuntimeError("Conflicting Isaac job scopes cannot be nested")
        if _process_definition is None:
            _process_definition = definition
        yield
        return
    on_main_thread = current_thread() is main_thread()
    previous_sigterm = signal.signal(signal.SIGTERM, _terminate_job) if on_main_thread else None
    _process_active = True
    _process_definition = definition
    _process_returning_close = returning_close
    try:
        try:
            primary = None
            try:
                yield
            except BaseException as error:
                primary = error
                raise
            finally:
                if _active_runtime is not None:
                    try:
                        _active_runtime.close()
                    except BaseException as error:
                        if primary is None:
                            raise
                        primary.add_note(f"Isaac environment cleanup also failed: {error}")
        except BaseException as primary:
            if _process_launcher is not None:
                try:
                    if returning_close:
                        # The worker reports the Python exception, not a native process exit.
                        _process_launcher.__exit__(None, None, None)
                    else:
                        traceback.print_exception(primary)
                        # Upstream does not map async cancellation to a nonzero status.
                        shutdown = primary
                        if not isinstance(primary, (Exception, SystemExit, KeyboardInterrupt)):
                            shutdown = SystemExit(1)
                        _process_launcher.__exit__(type(shutdown), shutdown, shutdown.__traceback__)
                except BaseException as error:
                    primary.add_note(f"Isaac process shutdown also failed: {error}")
            raise
        else:
            if _process_launcher is not None:
                _process_launcher.__exit__(None, None, None)
    finally:
        _process_launcher = None
        _process_active = False
        _process_definition = None
        _process_returning_close = False
        if on_main_thread and previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)


def _launch_environment(
    owner,
    cfg,
    launch_simulation,
    has_kit,
    *,
    viewing=False,
    local=False,
    streaming="",
    require_kit=False,
):
    global _process_launcher
    # Pinned Lab drops SimulationApp's fast_shutdown argument; Kit honors its own setting.
    close_setting = " --/app/fastShutdown=false" if _process_returning_close else ""
    args = {
        "headless": not local,
        "fast_shutdown": not _process_returning_close,
        "device": str(owner.device),
        "visualizer": ["kit"] if local or streaming else ["viser"] if viewing else ["none"],
        "visualizer_explicit": True,
        "livestream": 2 if streaming else 0,
        "enable_cameras": owner._rendering,
        # Kit's anonymous-data defaults also enable crash-dump upload consent.
        "kit_args": (
            "--/crashreporter/enabled=false --/crashreporter/skipOldDumpUpload=true "
            "--/app/uploadDumpsOnStartup=false --/telemetry/enableAnonymousData=false "
            "--/telemetry/enableAnonymousAppName=false --/telemetry/enableSentry=false "
            "--/structuredLog/enable=false"
            + close_setting
            + (_KIT_LOCAL_ARGS if local or streaming else "")
            + streaming
        ),
    }
    if local:
        # The default non-headless experience loads the authoring editor.
        args["experience"] = "isaaclab.python.headless.rendering.kit"
    elif streaming:
        # Let the pinned launcher resolve its streaming-compatible Lab experience.
        args["enable_cameras"] = True
    if require_kit:
        args["require_kit"] = True
    if current_thread() is not main_thread():
        validate_launch = import_module("isaaclab.app.sim_launcher").validate_launch
        _, kit_sources = validate_launch(cfg, args.copy())
        if kit_sources:
            raise ValueError(f"Off-thread Isaac requires a Kit-less configuration: {kit_sources}")
    launcher = launch_simulation(cfg, args)
    launcher.__enter__()
    owner._launcher = launcher
    if has_kit():
        # The owning scope closes Kit after job finalization or the last worker batch.
        _process_launcher, owner._launcher = owner._launcher, None
    if local or streaming:
        app = import_module("omni.kit.app").get_app()
        if local:
            # This binding registers the native Window type returned by omni.appwindow.
            import_module("carb.windowing")
            window = import_module("omni.appwindow").get_default_app_window()
            if window is None or not window.get_window():
                raise RuntimeError("Isaac local playback could not open an X11 window")
        manager = app.get_extension_manager()
        if streaming and not manager.get_enabled_extension_id("omni.kit.livestream.app"):
            raise RuntimeError("Isaac WebRTC could not start the native Kit streaming extension")
        if any(manager.get_enabled_extension_id(name) for name in _KIT_EDITOR_EXTENSIONS):
            mode = "webrtc" if streaming else "local"
            raise RuntimeError(f"Isaac {mode} playback cannot load Kit authoring extensions")
        owner.playback_is_running = app.is_running
    if current_thread() is main_thread():
        signal.signal(signal.SIGTERM, _terminate_job)


def _register_arena(owner, definition, context, gym):
    arena = import_module("isaaclab_arena.environments.arena_env_builder")
    name = definition.task
    if name in gym.registry:
        raise ValueError(f"Arena task {name!r} would replace an existing Gym registration")
    builder = definition.arena_factory.build(
        device=str(owner.device),
        num_envs=context.num_envs,
        seed=context.seed if context.seed is not None else 0,
    )
    if not isinstance(builder, arena.ArenaEnvBuilder) or builder.arena_env.name != name:
        raise ValueError("Arena factory must return a native ArenaEnvBuilder for the declared task")
    if name in gym.registry:
        raise ValueError(f"Arena factory registered {name!r} before native construction")
    try:
        registered, cfg, _ = builder.build_registered()
    finally:
        if (owned_spec := gym.registry.get(name)) is not None:
            owner._arena_registration = (name, owned_spec)
    if registered != name:
        raise ValueError("Arena builder registered a different native task")
    return cfg


def _create_environment(owner, definition, context):
    global _process_definition, _process_launcher, _active_runtime
    if not _process_active:
        raise RuntimeError(
            "Run Isaac jobs inside environment.job_scope() (normal train/evaluate do this "
            "automatically) so native shutdown follows checkpoint/export/logger finalization"
        )
    if _process_definition is not None and definition != _process_definition:
        raise RuntimeError("Native environment conflicts with the active Isaac job definition")
    if os.environ.get("LIVESTREAM", "0") != "0":
        raise ValueError("Isaac headless and spectator jobs require LIVESTREAM=0 or unset")
    for name, permitted in (
        ("OMNI_CRASHREPORTER_ENABLED", {"0", "n", "N", "f", "F"}),
        ("OMNI_CRASHREPORTER_SKIPOLDDUMPUPLOAD", {"1", "y", "Y", "t", "T"}),
    ):
        value = os.environ.get(name)
        if value is not None and value not in permitted:
            raise ValueError(f"{name} conflicts with Isaac crash-upload protection; unset it")
    _process_definition = definition
    device = owner.device
    viewing = _viser_selected(definition.viewer, context)
    local = context.purpose == "play" and context.viewer == "local"
    streaming = context.purpose == "play" and context.viewer == "webrtc"
    stream_args = ""
    if streaming:
        if _process_returning_close or os.environ.get("XR", "0") != "0":
            raise ValueError("Isaac WebRTC requires a direct playback process without XR")
        from nexuml_library.environments._isaaclab_webrtc import preflight

        stream_args = preflight(definition.webrtc)
    if local:
        if _process_returning_close:
            raise ValueError("Isaac local playback requires a direct job scope")
        if "--no-window" in sys.argv:
            raise ValueError("Isaac local playback cannot use the native --no-window flag")
        if (
            not os.environ.get("DISPLAY")
            or os.environ.get("HEADLESS", "0") != "0"
            or os.environ.get("XR", "0") != "0"
        ):
            raise ValueError(
                "Isaac local playback requires a prepared X11 display with HEADLESS=0 and XR=0"
            )
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Isaac Lab requires a working single-GPU CUDA device")
    if context.purpose == "eval" and context.num_envs != 1:
        raise ValueError("Isaac replay requires exactly one environment")
    if context.render_mode not in (None, "rgb_array") or (
        context.render_mode is not None
        and (context.purpose not in ("eval", "play") or context.num_envs != 1)
    ):
        raise ValueError("Isaac rendering supports only fresh-process N=1 rgb_array replay or play")
    try:
        isaaclab = import_module("isaaclab")
        import_module("isaaclab_tasks")
        launch_simulation = import_module("isaaclab.app").launch_simulation
        has_kit = import_module("isaaclab.utils").has_kit
        parse_env_cfg = import_module("isaaclab_tasks.utils").parse_env_cfg
        resolve_presets = import_module("isaaclab_tasks.utils.hydra").resolve_presets
    except ImportError as error:
        raise ImportError(
            "Isaac replay/training requires nexuml[robots] with the selected native sources"
        ) from error

    source_file = isaaclab.__file__
    if source_file is None:
        raise ValueError("Isaac Lab requires a verifiable pinned source origin")
    installed = distribution("isaaclab")
    origin = json.loads(installed.read_text("direct_url.json") or "{}")
    revision = origin.get("vcs_info", {}).get("commit_id")
    if revision is not None:
        if (
            Path(str(installed.locate_file("isaaclab/__init__.py"))).resolve()
            != Path(source_file).resolve()
        ):
            raise ValueError("Isaac Lab import origin differs from its pinned distribution")
    else:
        root = Path(source_file).resolve().parents[3]
        if not (root / ".git").exists():
            raise ValueError("Isaac Lab requires pinned VCS metadata or a source checkout")
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    if revision != definition.revision or sys.version_info[:2] != (3, 12):
        raise ValueError("Isaac Lab source revision/Python does not match the pinned profile")
    existing = sys.modules.get("isaaclab.sim.simulation_context")
    if (
        _active_runtime is not None
        or _process_launcher is not None
        or has_kit()
        or (existing is not None and existing.SimulationContext.instance() is not None)
    ):
        raise RuntimeError(
            "An Isaac application is already active; use fresh-process replay/training"
        )
    _active_runtime = owner
    from nexuml_library.environments._isaaclab_tasks import register_reference_tasks
    import gymnasium as gym

    composition = definition.arena_factory is not None
    if composition:
        from nexuml_library.evaluation.arena import _runtime_identity

        _runtime_identity()
        physics_type = import_module("isaaclab_physx.physics").PhysxCfg
        _launch_environment(owner, physics_type(), launch_simulation, has_kit, require_kit=True)
        cfg = _register_arena(owner, definition, context, gym)
        cfg.compute_final_obs = True
        resolved_task = definition.task
    else:
        register_reference_tasks()
        # Resolve Gym's supported module:task form before Isaac strips the module prefix.
        module, separator, task = definition.task.partition(":")
        if separator:
            import_module(module)
        resolved_task = gym.spec(task if separator else definition.task).id
        selections = []
        if definition.physics is not None:
            selections.append(f"physics={definition.physics}")
        if definition.presets:
            selections.append("presets=" + ",".join(definition.presets))
        cfg = cast(
            Any,
            parse_env_cfg(
                resolved_task,
                device=str(device),
                num_envs=context.num_envs,
                overrides=selections,
            ),
        )
        cfg.seed = context.seed if context.seed is not None else 0
        cfg.compute_final_obs = True
        physics_type = type(cfg.sim.physics)
        if definition.task_config is not None:
            cfg = definition.task_config.build(cfg)
            cfg = resolve_presets(
                cfg,
                selected=(
                    *definition.presets,
                    *((definition.physics,) if definition.physics else ()),
                ),
            )
    native_envs = import_module("isaaclab.envs")
    marl_config = getattr(native_envs, "DirectMARLEnvCfg", None)
    owner._marl = marl_config is not None and isinstance(cfg, marl_config)
    if not owner._marl and not isinstance(
        cfg, (native_envs.ManagerBasedRLEnvCfg, native_envs.DirectRLEnvCfg)
    ):
        raise TypeError("Native customization must produce a manager/direct RL config")
    if owner._marl != bool(owner.contract.groups):
        raise ValueError("Native DirectMARL type and declared grouped contract must agree")
    if (
        cfg.scene.num_envs != context.num_envs
        or torch.device(cfg.sim.device) != device
        or cfg.seed != (context.seed if context.seed is not None else 0)
        or not cfg.compute_final_obs
        or (definition.physics is not None and type(cfg.sim.physics) is not physics_type)
        or has_kit() != composition
    ):
        raise ValueError(
            "Native customization changed protected count/device/seed/final-state/runtime settings"
        )
    if owner._rendering and (
        definition.physics != "isaacsim_physx"
        or not isinstance(cfg, native_envs.ManagerBasedRLEnvCfg)
    ):
        raise ValueError("Isaac RGB replay currently requires a manager task with Kit PhysX")
    if owner._rendering:
        from nexuml_library.environments._isaaclab_rgb import configure_rgb

        configure_rgb(cfg, owner)
    if viewing:
        if context.purpose == "play" and definition.viewer.playback_controls:
            if owner._marl:
                raise ValueError("Viser playback controls require a single-agent native task")
            factory = owner._bindings.playback_options
            if factory is not None:
                options = factory.build()
                if not isinstance(options, IsaacPlaybackOptions) or not callable(
                    options.configure_reset
                ):
                    raise TypeError(
                        "Native playback option factory must return IsaacPlaybackOptions"
                    )
                owner._playback_options = options
            owner.playback_controls = PlaybackControls(
                context.num_envs,
                selected_reset=isinstance(cfg, native_envs.ManagerBasedRLEnvCfg),
                task_options=None
                if owner._playback_options is None
                else owner._playback_options.choices,
            )
        cfg.sim.default_visualizer_cfg = None
        cfg.sim.visualizer_cfgs = []
    if not composition:
        if local or streaming:
            cfg.ui_window_class_type = None
        _launch_environment(
            owner,
            cfg,
            launch_simulation,
            has_kit,
            viewing=viewing,
            local=local,
            streaming=stream_args,
        )
    if viewing:
        # Viser's Newton viewer imports USD; Kit must select its own USD first when needed.
        import_module("nexuml_library.environments._isaaclab_viser").configure_spectator(
            cfg, definition.viewer, context.num_envs, controls=owner.playback_controls
        )
        owner._spectator_type = cfg.sim.visualizer_cfgs[0].class_type
    owner._context_class = import_module("isaaclab.sim").SimulationContext
    owner.native = gym.make(definition.task, cfg=cfg).unwrapped
    if local and import_module("omni.kit.viewport.utility").get_active_viewport() is None:
        raise RuntimeError("Isaac local playback could not open a native Kit viewport")
    if streaming:
        from nexuml_library.environments._isaaclab_webrtc import report_endpoint

        report_endpoint(definition.webrtc)
    if viewing:
        visualizers = cast(Any, owner.native).sim.visualizers
        if (
            len(visualizers) != 1
            or type(visualizers[0]) is not cfg.sim.visualizer_cfgs[0].class_type
        ):
            raise RuntimeError("Native launcher did not retain the qualified spectator")
    native_envs = import_module("isaaclab.envs")
    marl_environment = getattr(native_envs, "DirectMARLEnv", None)
    if marl_environment is not None and isinstance(owner.native, marl_environment):
        if not owner._marl:
            raise ValueError("Native environment type disagrees with its DirectMARL config")
        owner._direct = True
    elif owner._marl:
        raise ValueError("Native DirectMARL config did not construct a DirectMARLEnv")
    elif isinstance(owner.native, native_envs.ManagerBasedRLEnv):
        owner._direct = False
    elif isinstance(owner.native, native_envs.DirectRLEnv):
        owner._direct = True
    else:
        raise ValueError("Native task is not a supported manager/direct RL environment")
    owner.resolved_task = resolved_task
    owner.resolved_config = import_module("isaaclab.utils").to_dict(cfg)
    if owner._rendering:
        from nexuml_library.environments._isaaclab_rgb import position_camera

        position_camera(owner.native, definition.task)


class _IsaacLabRuntime:
    """One owned native batch; terminal states never alias reset observations."""

    def __init__(self, definition, context):
        self.native: Any = None
        self._launcher = None
        self._arena_registration = None
        self._spectator_type = None
        self.playback_controls: PlaybackControls | None = None
        self.playback_is_running: Callable[[], bool] | None = None
        self._playback_options: IsaacPlaybackOptions | None = None
        self._context_class = None
        self._closed = False
        self._observation = None
        self._pending_reset = None
        self._direct = False
        self._marl = False
        self._marl_boundary = None
        self._bindings = definition.native_bindings()
        self._rendering = context.render_mode == "rgb_array"
        self._terminal_frame = None
        self._in_step = False
        self.render_backend = "isaac_rtx" if self._rendering else None
        self.device = torch.device(context.device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda:0")
        self.contract = definition.describe().model_copy(update={"num_envs": context.num_envs})
        self._processor = None
        self._processing_state = None
        self._observation_fields = self.contract.observations
        if self._bindings.processing == "raw":
            from nexuml_library.layers.processing.sensor_history import SensorHistoryProcessor

            keys = list(self.contract.observations)
            self._processor = SensorHistoryProcessor().build(
                LayerBuildContext(
                    input_sizes={
                        key: field.shape for key, field in self.contract.observations.items()
                    },
                    keys_in=keys,
                    keys_out=keys,
                    metadata={
                        "interaction_contract": self.contract.model_dump(mode="json"),
                        "execution_owner": (
                            "playback_runtime" if context.purpose == "play" else "runtime"
                        ),
                    },
                )
            )
            self._observation_fields = self._processor.raw_fields
        self.render_fps = 1 / self.contract.control_period_s
        try:
            _create_environment(self, definition, context)
            self._validate_native(definition)
        except BaseException as primary:
            try:
                self.close()
            except BaseException as error:
                primary.add_note(f"Isaac initialization cleanup also failed: {error}")
                for note in getattr(error, "__notes__", ()):
                    primary.add_note(note)
            raise

    def _validate_native(self, definition):
        if self._marl:
            from ._isaaclab_marl import IsaacMARLBoundary

            self._marl_boundary = IsaacMARLBoundary(self, definition)
            self._marl_boundary.validate()
            return
        env = self.native
        action_field = self.contract.actions[self._bindings.action]
        if (
            env.num_envs != self.contract.num_envs
            or torch.device(env.device) != self.device
            or tuple(env.single_action_space.shape) != action_field.shape
            or str(env.single_action_space.dtype) != action_field.dtype
            or (
                self.contract.action_space == "discrete"
                and getattr(env.single_action_space, "n", None) != action_field.num_values
            )
            or not math.isclose(env.step_dt, self.contract.control_period_s, rel_tol=1e-9)
            or not env.cfg.compute_final_obs
        ):
            raise ValueError(
                "Native Isaac dimensions/device/control period/final-state settings mismatch"
            )
        for key, path in self._bindings.observations.items():
            space = env.single_observation_space
            for part in path:
                space = space[part]
            if tuple(space.shape) != self._observation_fields[key].shape:
                raise ValueError("Native Isaac observation dimensions/device contract mismatch")
        self._validate_sensors()
        if self._bindings.direct_controls is not None:
            if not self._direct:
                raise ValueError("Procedural control verification requires a native direct task")
            observed = self._bindings.direct_controls.build(env)
            if (
                not isinstance(observed, PhysicalTensorContract)
                or observed != action_field.physical
            ):
                raise ValueError(
                    "Native direct physical controls differ from the declared contract"
                )
            return
        if getattr(self, "resolved_task", definition.task) == "NexuML-Cartpole-Categorical":
            self._validate_categorical_controls(action_field)
            return
        if definition.task == "Isaac-Cartpole-Direct":
            if (
                env.cfg.cart_dof_name != "slider_to_cart"
                or env.cfg.pole_dof_name != "cart_to_pole"
                or env.cfg.action_scale != 100.0
                or env.cartpole.find_joints("slider_to_cart")[1] != ["slider_to_cart"]
            ):
                raise ValueError("Native direct Cartpole control mapping differs from the policy")
            return
        if definition.task == "Isaac-Cartpole":
            name, joints, scale = "joint_effort", ["slider_to_cart"], 100.0
            offset = torch.tensor(0.0)
        elif definition.task == "Isaac-Reach-Franka":
            name, joints, scale = "arm_action", [f"panda_joint{i}" for i in range(1, 8)], 0.5
            offset = torch.tensor([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741])
        else:
            self._validate_bound_controls()
            return
        if env.action_manager.active_terms != [name]:
            raise ValueError("Native Isaac action terms differ from the pinned control mapping")
        term = env.action_manager.get_term(name)
        mdp = import_module("isaaclab.envs.mdp")
        implementation = (
            mdp.JointEffortAction
            if definition.task == "Isaac-Cartpole"
            else mdp.JointPositionAction
        )
        self._validate_action_implementation(term, implementation)
        descriptor = term.IO_descriptor
        native_offset = torch.as_tensor(descriptor.offset)
        offset = offset.to(device=native_offset.device, dtype=native_offset.dtype)
        if (
            descriptor.joint_names != joints
            or term.cfg.scale != scale
            or term.cfg.clip is not None
            or not torch.allclose(native_offset, offset, rtol=0, atol=1e-6)
            or (definition.task == "Isaac-Reach-Franka" and not term.cfg.use_default_offset)
        ):
            raise ValueError(
                "Native Isaac joint order/scale/offset/clipping differs from the policy"
            )

    def _validate_categorical_controls(self, field):
        from nexuml_library.environments._isaaclab_categorical import (
            ACTION_FORCES,
            categorical_cartpole_type,
        )

        physical = field.physical
        env = self.native
        if (
            type(env) is not categorical_cartpole_type()
            or env.action_forces != ACTION_FORCES
            or env.cfg.cart_dof_name != "slider_to_cart"
            or env.cfg.pole_dof_name != "cart_to_pole"
            or env.cartpole.find_joints("slider_to_cart")[1] != ["slider_to_cart"]
            or physical is None
            or physical.control_mode != "effort"
            or physical.units != ("N",)
            or physical.frame != "joint"
            or physical.joints != ("slider_to_cart",)
            or physical.class_controls != tuple((force,) for force in ACTION_FORCES)
            or physical.scale != (1.0,)
            or physical.offset != (0.0,)
        ):
            raise ValueError("Native categorical class-to-effort mapping differs from the policy")

    @staticmethod
    def _validate_action_implementation(term, implementation):
        configured = term.cfg.class_type
        if isinstance(configured, str):
            configured = import_module("isaaclab.utils.string").string_to_callable(configured)
        if type(term) is not implementation or configured is not implementation:
            raise ValueError(
                "Native Isaac action implementation/control mode differs from the policy"
            )

    def _validate_bound_controls(self):
        field = self.contract.actions[self._bindings.action]
        physical = field.physical
        if physical is None:
            return
        manager = getattr(self.native, "action_manager", None)
        if manager is None or len(manager.active_terms) != 1:
            raise ValueError("Declared physical controls require a verifiable native action term")
        term = manager.get_term(manager.active_terms[0])
        descriptor = term.IO_descriptor
        mdp = import_module("isaaclab.envs.mdp")
        position = isinstance(term.cfg, mdp.JointPositionActionCfg)
        if not (
            (physical.control_mode == "effort" and isinstance(term.cfg, mdp.JointEffortActionCfg))
            or (
                physical.control_mode == "absolute_position"
                and position
                and not term.cfg.use_default_offset
            )
            or (
                physical.control_mode == "relative_position"
                and position
                and term.cfg.use_default_offset
            )
        ):
            raise ValueError("Native physical control mode differs from the declared contract")
        self._validate_action_implementation(
            term,
            mdp.JointEffortAction if physical.control_mode == "effort" else mdp.JointPositionAction,
        )
        scale, offset = torch.as_tensor(descriptor.scale), torch.as_tensor(descriptor.offset)
        if (
            tuple(descriptor.joint_names) != physical.joints
            or not torch.allclose(
                scale, torch.tensor(physical.scale, device=scale.device, dtype=scale.dtype)
            )
            or not torch.allclose(
                offset,
                torch.tensor(physical.offset or (0.0,), device=offset.device, dtype=offset.dtype),
            )
            or term.cfg.clip is not None
        ):
            raise ValueError("Native physical control binding differs from the declared contract")

    def _validate_sensors(self):
        if any(
            field.sensor is not None and field.modality in {"rgb", "depth", "segmentation"}
            for field in self.contract.observations.values()
        ):
            validate = import_module(
                "nexuml_library.environments._isaaclab_sensors"
            ).validate_native_sensors
            validate(self.native, self.contract, self._bindings)

    def _state(self, observations):
        self._validate_sensors()
        result = TensorDict({}, batch_size=(self.contract.num_envs,), device=self.device)
        for key, path in self._bindings.observations.items():
            value = observations
            for part in path:
                value = value[part]
            field = self._observation_fields[key]
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != (self.contract.num_envs, *field.shape)
                or str(value.dtype).removeprefix("torch.") != field.dtype
                or value.device != self.device
            ):
                raise ValueError("Native Isaac policy observation shape/dtype/device mismatch")
            result[key] = value.clone()
        CompiledPolicy._validate_fields(
            result,
            self._observation_fields,
            "Native Isaac observation",
            require_finite=self._processor is None,
        )
        return result

    def _process_reset(self, raw, mask):
        if self._processor is None:
            self._observation = raw
        elif self._processing_state is None:
            state = self._processor.initialize_state((self.contract.num_envs,), device=self.device)
            self._observation, self._processing_state = self._processor.step(raw, state)
        else:
            state = self._processor.reset_state(self._processing_state, mask)
            observation, selected = self._processor.step(raw[mask], cast(TensorDict, state[mask]))
            assert self._observation is not None
            self._observation[mask] = observation
            state[mask] = selected
            self._processing_state = state

    def _pack(self, value):
        if self.contract.num_envs == 1:
            return value.squeeze(0)
        return value

    def reset(self, *, seed=None, mask=None) -> TensorDict:
        if self._closed:
            raise RuntimeError("Isaac runtime is closed")
        if self._marl:
            assert self._marl_boundary is not None
            return self._marl_boundary.reset(seed=seed, mask=mask)
        self._terminal_frame = None
        count = self.contract.num_envs
        if mask is None:
            requested = torch.ones(count, dtype=torch.bool, device=self.device)
        else:
            if mask.dtype != torch.bool or mask.numel() != count:
                raise ValueError("Isaac reset mask must contain one boolean per environment")
            requested = mask.reshape(count).to(self.device)
        reset = requested.clone()
        # Collector acknowledgements reuse the native auto-reset state exactly once.
        if mask is not None and self._pending_reset is not None:
            reset &= ~self._pending_reset
        if self._observation is None or reset.any():
            if self._direct:
                if not reset.all():
                    raise ValueError(
                        "Direct Isaac tasks support full reset or cached auto-reset "
                        "acknowledgement, not explicit partial native reset"
                    )
                observations, _ = self.native.reset(seed=seed)
            else:
                observations, _ = self.native.reset(
                    seed=seed, env_ids=reset.nonzero(as_tuple=False).flatten()
                )
            self._process_reset(self._state(observations), reset)
        if self._pending_reset is not None:
            self._pending_reset &= ~requested
        assert self._observation is not None
        return self._pack(self._observation.clone())

    def reset_with_task_option(self, name: str, choice: str) -> TensorDict:
        """Apply an opted-in task option at a full playback reset boundary.

        Returns:
            Fresh observations for the new trajectories.

        Raises:
            ValueError: If the active simulation has not authorized the request.
        """
        if (
            self._closed
            or self._in_step
            or self._playback_options is None
            or self.playback_controls is None
            or not self.playback_controls.accepts_task_option(name, choice)
        ):
            raise ValueError("Native playback did not authorize that task reset")
        self._playback_options.configure_reset(self.native, name, choice)
        return self.reset()

    def step(self, action: TensorDict) -> InteractionStep:
        return self._step(action)

    def step_recorded(self, action: TensorDict, *, source: str) -> InteractionStep:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("A recorded interaction requires an action source")
        if self._marl:
            raise ValueError("Recorded native actions are not qualified for grouped interactions")
        return self._step(action, source=source)

    def _step(self, action: TensorDict, *, source: str | None = None) -> InteractionStep:
        if not isinstance(action, TensorDictBase):
            raise TypeError("Isaac step requires a synchronous TensorDict action, not a goal")
        if self._marl:
            if self._closed:
                raise RuntimeError("Isaac runtime is closed")
            assert self._marl_boundary is not None
            return self._marl_boundary.step(action)
        if self._closed or self._observation is None:
            raise RuntimeError("Reset an open Isaac runtime before stepping")
        expected_batch = () if self.contract.num_envs == 1 else (self.contract.num_envs,)
        if tuple(action.batch_size) != expected_batch or (
            self._bindings.action in action and action[self._bindings.action].device != self.device
        ):
            raise ValueError("Isaac actions must use the native batch and device")
        CompiledPolicy._validate_fields(action, self.contract.actions, "Action")
        native_action = cast(torch.Tensor, action[self._bindings.action]).reshape(
            self.contract.num_envs, *self.contract.actions[self._bindings.action].shape
        )
        trace = None
        if source is not None:
            requested = action.select(*self.contract.actions).detach().clone()
            submitted = native_action.detach().clone()
            frame_before = self.native.common_step_counter
            issued = monotonic_ns()
        self._terminal_frame = None
        self._in_step = True
        try:
            observations, reward, terminated, truncated, info = self.native.step(native_action)
        finally:
            self._in_step = False
        if source is not None:
            frame_after = self.native.common_step_counter
            if frame_after != frame_before + 1:
                raise ValueError("Recorded Isaac actions require one native control step")
            trace = InteractionTrace(
                source=source,
                requested_action=requested,
                native_action=submitted,
                issued_monotonic_ns=issued,
                native_clock="isaaclab.control_step",
                native_frame_before=frame_before,
                native_frame_after=frame_after,
                control_period_s=self.native.step_dt,
            )
        count = self.contract.num_envs
        for value, dtype, name in (
            (reward, torch.float32, "reward"),
            (terminated, torch.bool, "terminated"),
            (truncated, torch.bool, "truncated"),
        ):
            if value.shape != (count,) or value.dtype != dtype or value.device != self.device:
                raise ValueError(f"Native Isaac {name} shape/dtype/device mismatch")
        reward, terminated, truncated = reward.clone(), terminated.clone(), truncated.clone()
        if not torch.isfinite(reward).all():
            raise ValueError("Native Isaac rewards must be finite")
        raw = self._state(observations)
        self._pending_reset = terminated | truncated
        final = raw.clone()
        if self._pending_reset.any():
            if "final_obs" not in info:
                raise ValueError(
                    "Isaac same-step reset requires pre-reset final_obs for PPO/replay"
                )
            terminal = self._state(info["final_obs"])
            final[self._pending_reset] = terminal[self._pending_reset]
        if self._processor is None:
            self._observation = raw
        else:
            assert self._processing_state is not None
            final, self._processing_state = self._processor.step(final, self._processing_state)
            self._observation = final.clone()
            if self._pending_reset.any():
                self._process_reset(raw, self._pending_reset)
        shape = (*expected_batch, 1)
        outcomes = info.get("task_outcomes")
        if outcomes is not None:
            if not isinstance(outcomes, Mapping) or any(
                not isinstance(key, str)
                or not key
                or not isinstance(value, torch.Tensor)
                or value.shape != (count,)
                or value.device != self.device
                for key, value in outcomes.items()
            ):
                raise ValueError("Isaac task_outcomes must map names to native scalar batches")
            outcomes = {key: value.clone().reshape(shape) for key, value in outcomes.items()}
        return InteractionStep(
            observation=self._pack(final),
            reward=reward.reshape(shape),
            terminated=terminated.reshape(shape),
            truncated=truncated.reshape(shape),
            info=outcomes,
            trace=trace,
        )

    def render_frame(self):
        """Return the ending episode's cached frame or the current live replay frame.

        Raises:
            RuntimeError: If the runtime is closed or was built without RGB replay.
        """
        if self._closed or not self._rendering:
            raise RuntimeError("Isaac RGB replay is not enabled on an open runtime")
        if self._terminal_frame is not None:
            return self._terminal_frame.copy()
        from nexuml_library.environments._isaaclab_rgb import read_rgb

        return read_rgb(self.native)

    def close(self) -> None:
        global _active_runtime
        if self._closed:
            return
        self._closed = True
        self.playback_is_running = None
        self._terminal_frame = None
        errors = []
        context_class = self._context_class
        if context_class is not None:
            # gym.make() may fail after opening a viewer, before returning its environment.
            sim = context_class.instance()
            if sim is not None and self._spectator_type is not None:
                for visualizer in (*sim.visualizers, *sim._pending_visualizers):
                    if type(visualizer) is self._spectator_type:
                        try:
                            visualizer.close()
                        except BaseException as error:
                            errors.append(error)
        native, self.native = self.native, None
        self._observation = self._pending_reset = None
        self._processing_state = self._processor = None
        self._marl_boundary = None
        if native is not None:
            try:
                native.close()
            except BaseException as error:
                errors.append(error)
        if context_class is not None and context_class.instance() is not None:
            try:
                context_class.clear_instance()
            except BaseException as error:
                errors.append(error)
        native = None
        if self._arena_registration is not None:
            import gymnasium as gym

            name, spec = self._arena_registration
            if gym.registry.get(name) is spec:
                del gym.registry[name]
            self._arena_registration = None
        launcher, self._launcher = self._launcher, None
        if launcher is not None:
            try:
                launcher.__exit__(None, None, None)
            except BaseException as error:
                errors.append(error)
        if _active_runtime is self:
            _active_runtime = None
        if errors:
            for error in errors[1:]:
                errors[0].add_note(f"Isaac launcher cleanup also failed: {error}")
            raise errors[0]
