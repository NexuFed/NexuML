"""Process-owned Isaac application and execution-local transition normalization."""

from contextlib import contextmanager
from importlib import import_module
from importlib.metadata import distribution
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import traceback
from typing import Any, cast

import torch
from tensordict import TensorDict

from nexuml.core.policy import CompiledPolicy
from nexuml.core.components import LayerBuildContext
from nexuml.interaction.runtime import InteractionStep


_process_active = False
_process_definition = None
_process_launcher = None
_active_runtime = None


def _terminate_job(signum, _frame):
    """Unwind cancellation before Kit's non-returning fast close.

    Raises:
        SystemExit: With the conventional signal-derived process status.
    """
    raise SystemExit(128 + signum)


@contextmanager
def isaaclab_process(definition=None):
    """Fast-close an owned Kit app after the complete Python job finishes.

    Raises:
        RuntimeError: If a nested scope selects a different native job.
    """
    global _process_active, _process_definition, _process_launcher
    if _process_active:
        if (
            definition is not None
            and _process_definition is not None
            and definition != _process_definition
        ):
            raise RuntimeError("Conflicting Isaac job scopes cannot be nested")
        if _process_definition is None:
            _process_definition = definition
        yield
        return
    previous_sigterm = signal.signal(signal.SIGTERM, _terminate_job)
    _process_active = True
    _process_definition = definition
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
                traceback.print_exception(primary)
                # Upstream maps Exception/SystemExit/KeyboardInterrupt, not async cancellation.
                shutdown = primary
                if not isinstance(primary, (Exception, SystemExit, KeyboardInterrupt)):
                    shutdown = SystemExit(1)
                try:
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
        signal.signal(signal.SIGTERM, previous_sigterm)


def _create_environment(owner, definition, context):
    global _process_definition, _process_launcher, _active_runtime
    if not _process_active:
        raise RuntimeError(
            "Run Isaac jobs inside environment.job_scope() (normal train/evaluate do this "
            "automatically) so native shutdown follows checkpoint/export/logger finalization"
        )
    if _process_definition is not None and definition != _process_definition:
        raise RuntimeError("Native environment conflicts with the active Isaac job definition")
    _process_definition = definition
    device = owner.device
    viewing = definition.viewer is not None and context.purpose in definition.viewer.purposes
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Isaac Lab requires a working single-GPU CUDA device")
    if context.purpose == "eval" and context.num_envs != 1:
        raise ValueError("Isaac replay requires exactly one environment")
    if context.render_mode not in (None, "rgb_array") or (
        context.render_mode is not None and context.purpose != "eval"
    ):
        raise ValueError("Isaac rendering supports only fresh-process N=1 rgb_array replay")
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

    register_reference_tasks()
    # Resolve Gym's supported module:task form before Isaac strips the module prefix.
    module, separator, task = definition.task.partition(":")
    if separator:
        import_module(module)
    registered = gym.spec(task if separator else definition.task)
    selections = []
    if definition.physics is not None:
        selections.append(f"physics={definition.physics}")
    if definition.presets:
        selections.append("presets=" + ",".join(definition.presets))
    cfg = cast(
        Any,
        parse_env_cfg(
            registered.id,
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
            selected=(*definition.presets, *((definition.physics,) if definition.physics else ())),
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
        or has_kit()
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
        import_module("nexuml_library.environments._isaaclab_viser").configure_spectator(
            cfg,
            definition.viewer,
            context.num_envs,
        )
        owner._spectator_type = cfg.sim.visualizer_cfgs[0].class_type
    args = {
        "headless": True,
        "fast_shutdown": True,
        "device": str(device),
        "visualizer": ["viser"] if viewing else ["none"],
        "visualizer_explicit": True,
        "enable_cameras": owner._rendering,
    }
    launcher = launch_simulation(cfg, args)
    launcher.__enter__()
    owner._launcher = launcher
    if has_kit():
        # Kit's fast close must follow whole-job checkpoint/export/logger finalization.
        _process_launcher, owner._launcher = owner._launcher, None
    signal.signal(signal.SIGTERM, _terminate_job)
    owner._context_class = import_module("isaaclab.sim").SimulationContext
    owner.native = gym.make(definition.task, cfg=cfg).unwrapped
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
    owner.resolved_task = registered.id
    owner.resolved_config = import_module("isaaclab.utils").to_dict(cfg)
    if owner._rendering:
        from nexuml_library.environments._isaaclab_rgb import position_camera

        position_camera(owner.native, definition.task)


class _IsaacLabRuntime:
    """One owned native batch; terminal states never alias reset observations."""

    def __init__(self, definition, context):
        self.native: Any = None
        self._launcher = None
        self._spectator_type = None
        self._context_class = None
        self._closed = False
        self._observation = None
        self._pending_reset = None
        self._so101 = definition.task == "NexuML-SO101-Joint-Reach"
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
                        "execution_owner": "runtime",
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
        elif definition.task == "NexuML-SO101-Joint-Reach":
            from nexuml_library.environments._isaaclab_so101 import JOINT_NAMES, validate_asset

            validate_asset(env)
            name, joints, scale = "arm_action", list(JOINT_NAMES), 0.5
            offset = torch.tensor(0.0)
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
            or (definition.task == "NexuML-SO101-Joint-Reach" and term.cfg.use_default_offset)
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

    def step(self, action: TensorDict) -> InteractionStep:
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
        native_action = action[self._bindings.action].reshape(
            self.contract.num_envs, *self.contract.actions[self._bindings.action].shape
        )
        self._terminal_frame = None
        self._in_step = True
        try:
            observations, reward, terminated, truncated, info = self.native.step(native_action)
        finally:
            self._in_step = False
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
        outcomes = None
        if self._so101:
            outcomes = {
                "task_success": self.native.joint_reach_success.clone().reshape(shape),
                "joint_error_rad": self.native.joint_reach_error.clone().reshape(shape),
            }
        return InteractionStep(
            observation=self._pack(final),
            reward=reward.reshape(shape),
            terminated=terminated.reshape(shape),
            truncated=truncated.reshape(shape),
            info=outcomes,
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
