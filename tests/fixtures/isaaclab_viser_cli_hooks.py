"""Test instrumentation: lazy imports, immutable config delegation, instance-only hooks."""

import atexit
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import sys
from threading import enumerate as enumerate_threads
import time
from typing import Any, cast

MODES = frozenset(
    {"train", "train-viewer", "replay", "replay-viewer", "failure-viewer", "cancel-viewer"}
)


def _destination():
    mode = os.environ["NEXUML_SPECTATOR_PROBE_MODE"]
    if mode not in MODES:
        raise ValueError(f"Unsupported probe mode: {mode}")
    return Path(os.environ["NEXUML_SPECTATOR_PROBE_OUTPUT"]), mode


def configure_probe(cfg: object, *, base_config: dict) -> object:
    from nexuml_library.environments import _isaaclab_runtime
    from nexuml_library.environments.isaaclab_config import IsaacTaskConfigSpec

    owner = _isaaclab_runtime._active_runtime
    if owner is None:
        raise RuntimeError("Probe requires the execution-owned active Isaac runtime")
    transformed = IsaacTaskConfigSpec.model_validate(base_config).build(cfg)
    output, mode = _destination()
    ipc = Path(os.environ["NEXUML_SPECTATOR_PROBE_IPC"]) if mode.endswith("-viewer") else output
    _wrap_runtime(owner, output, mode, ipc)
    return transformed


def record(runtime, transition, action):
    import numpy as np
    import torch

    robot = runtime.native.scene["robot"]
    return {
        "action": action.detach().cpu().clone().to_dict(),
        "observation": transition.observation.detach().cpu().clone().to_dict(),
        "reward": transition.reward.cpu().clone(),
        "terminated": transition.terminated.cpu().clone(),
        "truncated": transition.truncated.cpu().clone(),
        "joints": robot.data.joint_pos.torch.cpu().clone(),
        "velocity": robot.data.joint_vel.torch.cpu().clone(),
        "history": (
            runtime._processing_state.detach().cpu().clone().to_dict()
            if runtime._processing_state is not None
            else None
        ),
        "episode_length": runtime.native.episode_length_buf.cpu().clone(),
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
        "python_rng": random.getstate(),
        "numpy_rng": torch.from_numpy(cast(Any, np.random.get_state())[1].copy()),
    }


def observe(runtime, viewing):
    import torch
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.markers.config import FRAME_MARKER_CFG
    from isaaclab.utils import replace
    from isaaclab.sim import SphereCfg, PreviewSurfaceCfg

    if not viewing:
        assert runtime.native.sim.visualizers == []
        assert "viser" not in sys.modules
        assert "isaaclab_visualizers.viser.viser_visualizer" not in sys.modules
        assert runtime._spectator_type is None
    markers = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/spectator_probe",
            markers={
                "point": SphereCfg(
                    radius=0.15,
                    visual_material=PreviewSurfaceCfg(diffuse_color=(0.9, 0.15, 0.1)),
                )
            },
        )
    )
    markers.visualize(
        translations=runtime.native.scene.env_origins
        + torch.tensor([0.0, 0.0, 1.5], device="cuda:0")
    )
    frame = VisualizationMarkers(
        replace(FRAME_MARKER_CFG, prim_path="/Visuals/spectator_frame_probe")
    )
    frame.visualize(
        translations=runtime.native.scene.env_origins
        + torch.tensor([0.0, 0.0, 1.0], device="cuda:0")
    )
    return [markers, frame]


def progress(runtime, steps, viewing):
    result = {"steps": steps}
    if not viewing:
        return result
    (viewer,) = runtime.native.sim.visualizers
    clients = viewer._viewer._server.get_clients()
    result.update(
        {
            "clients": len(clients),
            "paused": viewer._display_state.paused,
            "selection": viewer._applied_selection,
            "plots": len(viewer._viewer._plot_handles),
            "plot_samples": max(
                (len(values) for values in viewer._viewer._scalar_buffers.values()), default=0
            ),
            "marker_count": len(viewer._marker_names),
            "marker_visible": {
                name: viewer._viewer._scene_handles[name].visible
                for name in viewer._marker_names
                if name in viewer._viewer._scene_handles
            },
            "plot_visible": viewer._viewer._live_plots_folder.visible,
            "scene_visible": {
                name: handle.visible for name, handle in viewer._viewer._scene_handles.items()
            },
            "cameras": [client.camera.position.tolist() for client in clients.values()],
        }
    )
    from isaaclab_newton.physics import NewtonBackendCfg

    physics = runtime.native.sim.get_or_create_backend(
        NewtonBackendCfg(
            physics_cfg=runtime.native.sim.cfg.physics, device=runtime.native.sim.device
        )
    )
    assert viewer.backend.state_0.body_q.ptr != physics.state_0.body_q.ptr
    return result


def _write_trace(output, trajectory, mode):
    import torch

    parts = []
    for index, start in enumerate(range(0, max(1, len(trajectory)), 64)):
        name = "trajectory.pt" if index == 0 else f"trajectory-{index:04d}.pt"
        values = trajectory[start : start + 64]
        target = output / name
        torch.save(values, target)
        size = target.stat().st_size
        assert size <= 32 * 1024**2
        with target.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        parts.append({"path": name, "steps": len(values), "bytes": size, "sha256": digest})
    assert shutil.disk_usage(output).free > 4 * 2**30
    (output / "trajectory-manifest.json").write_text(
        json.dumps({"steps": len(trajectory), "parts": parts})
    )


def _wrap_runtime(owner, output, mode, ipc):
    native_step, native_close = owner.step, owner.close
    before_threads = set(enumerate_threads())
    trajectory, marker_groups = [], []
    viewer_servers = []
    viewing = mode.endswith("-viewer")
    observed = native_closed = trace_written = False
    last_progress = None

    def retain_trace():
        _write_trace(output, trajectory, mode)
        if ipc != output and last_progress is not None:
            (output / "progress.json").write_text(last_progress)

    def finish():
        # Native collection can close before the last update and logger finalization.
        assert native_closed and trace_written
        assert not [t for t in enumerate_threads() if t not in before_threads and t.is_alive()]
        if not mode.startswith("replay"):
            finalizations = json.loads((output / "logger-finalization.json").read_text())
            assert finalizations and all(
                event["completed"] and event["error"] is None and event["job_scope_active"]
                for event in finalizations
            )
        (output / "native-cleanup.json").write_text(
            json.dumps({"status": "PASS", "mode": mode, "steps": len(trajectory)})
        )
        print("CLI_PROBE_NATIVE_CLOSED_AND_OWNED_THREADS_STOPPED", mode, flush=True)

    atexit.register(finish)

    def step(action):
        nonlocal observed, last_progress
        first = not observed
        if first:
            assert shutil.disk_usage(output).free > 4 * 2**30
            marker_groups.extend(observe(owner, viewing))
            observed = True
            if viewing:
                (viewer,) = owner.native.sim.visualizers
                viewer_servers.append(viewer._viewer._server)
        transition = native_step(action)
        trajectory.append(record(owner, transition, action))
        last_progress = json.dumps(progress(owner, len(trajectory), viewing))
        # Distributed artifact storage is not the live browser IPC channel.
        temporary = ipc / "progress.tmp"
        temporary.write_text(last_progress)
        temporary.replace(ipc / "progress.json")
        if first:
            port = owner.native.sim.visualizers[0]._viewer._server.get_port() if viewing else None
            temporary = ipc / "ready.tmp"
            temporary.write_text(json.dumps({"port": port, "mode": mode}))
            temporary.replace(ipc / "ready.json")
            if ipc != output:
                (output / "ready.json").write_text(json.dumps({"port": port, "mode": mode}))
            print("PUBLIC_CLI_TEST_PROBE_READY", mode, port, flush=True)
        if viewing:
            time.sleep(0.3)  # Test-only interaction window, never simulation dt.
        return transition

    def close():
        nonlocal native_closed, trace_written
        primary = sys.exception()
        try:
            result = native_close()
            for server in viewer_servers:
                assert not server._websock_server._server_thread.is_alive()
                assert not any(thread.is_alive() for thread in server._thread_executor._threads)
            native_closed = True
        except BaseException as error:
            try:
                retain_trace()
                trace_written = True
            except BaseException as trace_error:
                error.add_note(f"Probe trace cleanup also failed: {trace_error}")
            raise
        try:
            retain_trace()
            trace_written = True
        except BaseException as trace_error:
            if primary is None:
                raise
            primary.add_note(f"Probe trace cleanup also failed: {trace_error}")
        return result

    owner.step = step
    owner.close = close


class _TrainingObserver:
    def on_train_start(self, trainer, pl_module):
        import torch
        from nexuml_library.environments import _isaaclab_runtime

        self.output, self.mode = _destination()
        self.before = (
            torch.nn.utils.parameters_to_vector(pl_module.policy.parameters()).detach().clone()
        )
        assert torch.isfinite(self.before).all()
        finalizations = []
        assert trainer.loggers
        for logger in trainer.loggers:
            original = logger.finalize

            def finalize(status, original=original):
                completed = False
                try:
                    result = original(status)
                    completed = True
                    return result
                finally:
                    primary = sys.exception()
                    try:
                        finalizations.append(
                            {
                                "status": status,
                                "job_scope_active": _isaaclab_runtime._process_active,
                                "completed": completed,
                                "error": None if completed else "logger finalization failed",
                            }
                        )
                        (self.output / "logger-finalization.json").write_text(
                            json.dumps(finalizations)
                        )
                    except BaseException as trace_error:
                        if primary is None:
                            raise
                        primary.add_note(f"Probe logger trace also failed: {trace_error}")

            logger.finalize = finalize

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 2 and self.mode == "failure-viewer":
            (self.output / "injection.json").write_text(
                json.dumps(
                    {
                        "mode": self.mode,
                        "batch_idx": batch_idx,
                        "identity": "injected spectator training failure",
                    }
                )
            )
            raise RuntimeError("injected spectator training failure")
        if batch_idx == 2 and self.mode == "cancel-viewer":
            (self.output / "injection.json").write_text(
                json.dumps({"mode": self.mode, "batch_idx": batch_idx, "identity": "SIGTERM"})
            )
            os.kill(os.getpid(), signal.SIGTERM)

    def on_train_end(self, trainer, pl_module):
        import torch

        after = torch.nn.utils.parameters_to_vector(pl_module.policy.parameters()).detach()
        assert torch.isfinite(after).all()
        assert not torch.equal(self.before, after)
        assert int(pl_module.frames) == 256 and int(pl_module.batches) == 4
        torch.save(pl_module.state_dict(), self.output / "training-state.pt")
        trainer.save_checkpoint(self.output / "complete.ckpt")


def observer_callback():
    from lightning.pytorch.callbacks import Callback

    class Observe(_TrainingObserver, Callback):
        pass

    return Observe()
