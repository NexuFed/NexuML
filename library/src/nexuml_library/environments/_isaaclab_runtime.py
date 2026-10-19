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
from nexuml.interaction.runtime import InteractionStep


_process_active = False
_process_launcher = None
_active_runtime = None


def _terminate_job(signum, _frame):
    """Unwind cancellation before Kit's non-returning fast close.

    Raises:
        SystemExit: With the conventional signal-derived process status.
    """
    raise SystemExit(128 + signum)


@contextmanager
def isaaclab_process():
    """Fast-close an owned Kit app after the complete Python job finishes.

    Raises:
        RuntimeError: If process scopes are nested.
    """
    global _process_active, _process_launcher
    if _process_active:
        raise RuntimeError("Isaac process scopes cannot be nested")
    previous_sigterm = signal.signal(signal.SIGTERM, _terminate_job)
    _process_active = True
    try:
        try:
            yield
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
        signal.signal(signal.SIGTERM, previous_sigterm)


def _create_environment(owner, definition, context):
    global _process_launcher, _active_runtime
    if not _process_active:
        raise RuntimeError(
            "Run Isaac jobs through scripts/isaaclab/run.py so fast shutdown follows "
            "checkpoint/export/logger finalization and preserves the process exit status"
        )
    device = owner.device
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Isaac Lab requires a working single-GPU CUDA device")
    if context.purpose == "eval" and context.num_envs != 1:
        raise ValueError("Isaac replay requires exactly one environment")
    if context.render_mode not in (None, "rgb_array") or (
        context.render_mode is not None and context.purpose != "eval"
    ):
        raise ValueError("Isaac rendering supports only fresh-process N=1 rgb_array replay")
    if owner._rendering and (
        definition.physics != "isaacsim_physx" or definition.task == "Isaac-Cartpole-Direct"
    ):
        raise ValueError("Isaac RGB replay currently requires a manager task with Kit PhysX")
    try:
        isaaclab = import_module("isaaclab")
        import_module("isaaclab_tasks")
        launch_simulation = import_module("isaaclab.app").launch_simulation
        has_kit = import_module("isaaclab.utils").has_kit
        parse_env_cfg = import_module("isaaclab_tasks.utils").parse_env_cfg
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
    if _active_runtime is not None or _process_launcher is not None or has_kit():
        raise RuntimeError(
            "An Isaac application is already active; use fresh-process replay/training"
        )
    cfg: Any
    if definition.task == "NexuML-SO101-Joint-Reach":
        from nexuml_library.environments._isaaclab_so101 import make_so101_config

        cfg = make_so101_config(
            num_envs=context.num_envs, device=str(device), seed=context.seed or 0
        )
    else:
        # Registered task subclasses add fields absent from upstream's base config type.
        cfg = cast(
            Any,
            parse_env_cfg(
                definition.task,
                device=str(device),
                num_envs=context.num_envs,
                overrides=[f"physics={definition.physics}"],
            ),
        )
    cfg.seed = context.seed if context.seed is not None else 0
    cfg.compute_final_obs = True
    if definition.task != "Isaac-Cartpole-Direct":
        cfg.observations.policy.enable_corruption = False
        cfg.observations.policy.history_length = 0
    if owner._rendering:
        from nexuml_library.environments._isaaclab_rgb import configure_rgb

        configure_rgb(cfg, owner)
    args = {
        "headless": True,
        "fast_shutdown": definition.physics == "isaacsim_physx",
        "device": str(device),
        "visualizer": ["none"],
        "visualizer_explicit": True,
        "enable_cameras": owner._rendering,
    }
    _active_runtime = owner
    launcher = launch_simulation(cfg, args)
    launcher.__enter__()
    owner._launcher = launcher
    if definition.physics == "isaacsim_physx":
        # Kit's fast close must follow whole-job checkpoint/export/logger finalization.
        _process_launcher, owner._launcher = owner._launcher, None
    signal.signal(signal.SIGTERM, _terminate_job)
    import gymnasium as gym

    if definition.task == "NexuML-SO101-Joint-Reach":
        owner.native = import_module("isaaclab.envs").ManagerBasedRLEnv(cfg=cfg)
    else:
        owner.native = gym.make(definition.task, cfg=cfg).unwrapped
    if owner._rendering:
        from nexuml_library.environments._isaaclab_rgb import position_camera

        position_camera(owner.native, definition.task)


class _IsaacLabRuntime:
    """One owned native batch; terminal states never alias reset observations."""

    def __init__(self, definition, context):
        self.native: Any = None
        self._launcher = None
        self._closed = False
        self._observation = None
        self._pending_reset = None
        self._so101 = definition.task == "NexuML-SO101-Joint-Reach"
        self._direct = definition.task == "Isaac-Cartpole-Direct"
        self._rendering = context.render_mode == "rgb_array"
        self._terminal_frame = None
        self._in_step = False
        self.render_backend = "isaac_rtx" if self._rendering else None
        self.device = torch.device(context.device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda:0")
        self.contract = definition.describe().model_copy(update={"num_envs": context.num_envs})
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
        env = self.native
        size = self.contract.observations["observation"].shape
        action_size = self.contract.actions["action"].shape
        if (
            env.num_envs != self.contract.num_envs
            or torch.device(env.device) != self.device
            or tuple(env.single_observation_space["policy"].shape) != size
            or tuple(env.single_action_space.shape) != action_size
            or not math.isclose(env.step_dt, self.contract.control_period_s, rel_tol=1e-9)
            or not env.cfg.compute_final_obs
        ):
            raise ValueError(
                "Native Isaac dimensions/device/control period/final-state settings mismatch"
            )
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
            from nexuml_library.environments._isaaclab_so101 import JOINT_NAMES, validate_asset

            validate_asset(env)
            name, joints, scale = "arm_action", list(JOINT_NAMES), 0.5
            offset = torch.tensor(0.0)
        if env.action_manager.active_terms != [name]:
            raise ValueError("Native Isaac action terms differ from the pinned control mapping")
        term = env.action_manager.get_term(name)
        descriptor = term.IO_descriptor
        if (
            descriptor.joint_names != joints
            or term.cfg.scale != scale
            or term.cfg.clip is not None
            or not torch.allclose(torch.as_tensor(descriptor.offset), offset, rtol=0, atol=1e-6)
            or (definition.task == "Isaac-Reach-Franka" and not term.cfg.use_default_offset)
            or (definition.task == "NexuML-SO101-Joint-Reach" and term.cfg.use_default_offset)
        ):
            raise ValueError(
                "Native Isaac joint order/scale/offset/clipping differs from the policy"
            )

    def _state(self, observations):
        value = observations["policy"]
        expected = (self.contract.num_envs, *self.contract.observations["observation"].shape)
        if value.shape != expected or value.dtype != torch.float32 or value.device != self.device:
            raise ValueError("Native Isaac policy observation shape/dtype/device mismatch")
        if not torch.isfinite(value).all():
            raise ValueError("Native Isaac policy observations must be finite")
        return value.clone()

    def _pack(self, value):
        if self.contract.num_envs == 1:
            return TensorDict({"observation": value.squeeze(0)}, batch_size=(), device=self.device)
        return TensorDict(
            {"observation": value}, batch_size=(self.contract.num_envs,), device=self.device
        )

    def reset(self, *, seed=None, mask=None) -> TensorDict:
        if self._closed:
            raise RuntimeError("Isaac runtime is closed")
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
            self._observation = self._state(observations)
        if self._pending_reset is not None:
            self._pending_reset &= ~requested
        return self._pack(self._observation.clone())

    def step(self, action: TensorDict) -> InteractionStep:
        if self._closed or self._observation is None:
            raise RuntimeError("Reset an open Isaac runtime before stepping")
        expected_batch = () if self.contract.num_envs == 1 else (self.contract.num_envs,)
        if tuple(action.batch_size) != expected_batch or (
            "action" in action and action["action"].device != self.device
        ):
            raise ValueError("Isaac actions must use the native batch and device")
        CompiledPolicy._validate_fields(action, self.contract.actions, "Action")
        native_action = action["action"].reshape(self.contract.num_envs, -1)
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
        self._observation = self._state(observations)
        self._pending_reset = terminated | truncated
        final = self._observation.clone()
        if self._pending_reset.any():
            if "final_obs" not in info:
                raise ValueError(
                    "Isaac same-step reset requires pre-reset final_obs for PPO/replay"
                )
            terminal = self._state(info["final_obs"])
            final[self._pending_reset] = terminal[self._pending_reset]
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
        native, self.native = self.native, None
        self._observation = self._pending_reset = None
        if native is not None:
            try:
                native.close()
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
