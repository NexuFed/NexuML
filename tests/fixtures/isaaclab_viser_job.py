"""Opt-in native spectator isolation and bounded real learning/browser job."""

from pathlib import Path
import random
import json
import time
import socket
from typing import Any, cast
import shutil
import sys
from threading import enumerate as enumerate_threads
from threading import Event

import torch
import numpy as np
from tensordict import TensorDict

from nexuml.core.components import EnvironmentBuildContext
from nexuml.core.policy import CompiledPolicy
from nexuml_library.environments.isaaclab_config import ViserSpec
from nexuml_library.scenarios.reinforcement.isaaclab_sensors import isaac_mixed_cartpole_ppo


def record(runtime, transition, action):
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


def browser_job(mode, output, port):
    from nexuml_library.environments import _isaaclab_runtime as runtime_module

    output.mkdir(parents=True, exist_ok=False)
    assert shutil.disk_usage(".").free >= 40 * 2**30
    viewing = mode.endswith("-viewer")
    trajectory = []
    marker_groups = []
    before_threads = set(enumerate_threads())

    def observe(runtime):
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
        marker_groups.append(markers)
        markers.visualize(
            translations=runtime.native.scene.env_origins
            + torch.tensor(
                [0.0, 0.0, 1.5],
                device="cuda:0",
            )
        )
        frame = VisualizationMarkers(
            replace(FRAME_MARKER_CFG, prim_path="/Visuals/spectator_frame_probe")
        )
        marker_groups.append(frame)
        frame.visualize(
            translations=runtime.native.scene.env_origins
            + torch.tensor([0.0, 0.0, 1.0], device="cuda:0")
        )
        native_step = runtime.step

        def step(action):
            transition = native_step(action)
            trajectory.append(record(runtime, transition, action))
            progress = {"steps": len(trajectory)}
            if viewing:
                (viewer,) = runtime.native.sim.visualizers
                clients = viewer._viewer._server.get_clients()
                progress.update(
                    {
                        "clients": len(clients),
                        "paused": viewer._display_state.paused,
                        "selection": viewer._applied_selection,
                        "plots": len(viewer._viewer._plot_handles),
                        "plot_samples": max(
                            (len(values) for values in viewer._viewer._scalar_buffers.values()),
                            default=0,
                        ),
                        "marker_count": len(viewer._marker_names),
                        "marker_visible": {
                            name: viewer._viewer._scene_handles[name].visible
                            for name in viewer._marker_names
                            if name in viewer._viewer._scene_handles
                        },
                        "plot_visible": viewer._viewer._live_plots_folder.visible,
                        "scene_visible": {
                            name: handle.visible
                            for name, handle in viewer._viewer._scene_handles.items()
                        },
                        "cameras": [client.camera.position.tolist() for client in clients.values()],
                    }
                )
                from isaaclab_newton.physics import NewtonBackendCfg

                physics = runtime.native.sim.get_or_create_backend(
                    NewtonBackendCfg(
                        physics_cfg=runtime.native.sim.cfg.physics,
                        device=runtime.native.sim.device,
                    )
                )
                assert viewer.backend.state_0.body_q.ptr != physics.state_0.body_q.ptr
            temporary = output / "progress.tmp"
            temporary.write_text(json.dumps(progress))
            temporary.replace(output / "progress.json")
            if viewing:
                time.sleep(
                    0.3
                )  # Test-only wall-clock window for browser interaction, not simulation dt.
            return transition

        runtime.step = step
        (output / "ready.json").write_text(json.dumps({"port": port, "mode": mode}))
        print("NATIVE_BROWSER_JOB_READY", mode, port, flush=True)

    if mode.startswith("replay"):
        from nexuml.core.export import load_package

        policy, _, _ = load_package(output.parent / "train-viewer/policy")
        assert isinstance(policy, CompiledPolicy)
        policy = policy.to("cuda:0").eval()
        assert policy.pipeline.resolved_config.interaction is not None
        definition = policy.pipeline.resolved_config.interaction.environment
        if viewing:
            definition = definition.model_copy(
                update={
                    "viewer": ViserSpec(
                        purposes=("eval",), port=port, eye=(4.0, -4.0, 4.0), lookat=(0.0, 0.0, 2.0)
                    )
                }
            )
        with definition.job_scope():
            runtime = definition.build(
                EnvironmentBuildContext(
                    purpose="eval",
                    device="cuda:0",
                    num_envs=1,
                    seed=11,
                )
            )
            try:
                observation = runtime.reset(seed=11)
                observe(runtime)
                with torch.no_grad():
                    for _ in range(128):
                        transition = runtime.step(policy(observation))
                        observation = (
                            runtime.reset(mask=(transition.terminated | transition.truncated))
                            if (transition.terminated | transition.truncated).any()
                            else transition.observation
                        )
            finally:
                runtime.close()
    else:
        from lightning.pytorch.callbacks import Callback
        from nexuml.training.lightning import NexuSession
        from nexuml.training.reinforcement import NexuRLLightningModule
        from nexuml.core.export import export_package

        scenario = isaac_mixed_cartpole_ppo(total_frames=256, frames_per_batch=64, num_envs=2)
        scenario.training.algorithm = scenario.training.algorithm.model_copy(
            update={"update_epochs": 1, "minibatch_size": 64},
        )
        if viewing:
            scenario.interaction.environment = scenario.interaction.environment.model_copy(
                update={
                    "viewer": ViserSpec(port=port, eye=(4.0, -4.0, 4.0), lookat=(0.0, 0.0, 2.0))
                },
            )

        class Observe(Callback):
            def on_train_start(self, trainer, pl_module):
                observe(runtime_module._active_runtime)

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                if batch_idx == 2 and mode.startswith("failure"):
                    raise RuntimeError("injected spectator training failure")
                if batch_idx == 2 and mode.startswith("cancel"):
                    import os
                    import signal

                    os.kill(os.getpid(), signal.SIGTERM)

        definition = scenario.interaction.environment
        with definition.job_scope():
            session = NexuSession(scenario, log_dir=output, enable_progress_bar=False)
            cast(Any, session.trainer).callbacks.append(Observe())
            finalizations = []
            loggers = session.trainer_loggers
            assert isinstance(loggers, list) and loggers
            for logger in loggers:
                original = logger.finalize

                def finalize(status, original=original):
                    original(status)
                    finalizations.append(
                        {"status": status, "job_scope_active": runtime_module._process_active}
                    )
                    (output / "logger-finalization.json").write_text(json.dumps(finalizations))

                logger.finalize = finalize
            assert isinstance(session.lightning_module, NexuRLLightningModule)
            before = (
                torch.nn.utils.parameters_to_vector(session.lightning_module.policy.parameters())
                .detach()
                .clone()
            )
            result = session.run()
            module = result.lightning_module
            assert isinstance(module, NexuRLLightningModule)
            after = torch.nn.utils.parameters_to_vector(module.policy.parameters()).detach()
            assert not torch.equal(before, after)
            assert int(module.frames) == 256 and int(module.batches) == 4
            torch.save(module.state_dict(), output / "training-state.pt")
            export_package(module.policy, output / "policy")
            result.trainer.save_checkpoint(output / "complete.ckpt")
    torch.save(trajectory, output / "trajectory.pt")
    assert not [t for t in enumerate_threads() if t not in before_threads and t.is_alive()]
    assert shutil.disk_usage(".").free >= 40 * 2**30
    print("NATIVE_BROWSER_JOB_COMPLETE_AND_OWNED_THREADS_STOPPED", mode, flush=True)


def snapshot(output, viewing, port, noisy=False):
    assert shutil.disk_usage(".").free >= 40 * 2**30
    scenario = isaac_mixed_cartpole_ppo()
    definition = scenario.interaction.environment
    if noisy:
        from isaaclab_viser_noisy import configure_noisy
        from nexuml_library.environments.isaaclab import IsaacLabEnvironment, isaac_task_config

        definition = IsaacLabEnvironment(
            physics="newton_mjwarp", task_config=isaac_task_config(configure_noisy)
        )
    if viewing:
        definition = definition.model_copy(
            update={"viewer": ViserSpec(port=port, live_plots_update_interval=1)}
        )
    baseline_threads = set(enumerate_threads())
    with definition.job_scope():
        runtime: Any = definition.build(
            EnvironmentBuildContext(
                purpose="train",
                device="cuda:0",
                num_envs=2,
                seed=11,
            )
        )
        connection = None
        try:
            if noisy and viewing:
                from websockets.sync.client import connect
                from websockets.typing import Subprotocol
                from viser import _messages
                import msgspec

                (spectator,) = runtime.native.sim.visualizers
                connected = Event()
                spectator._viewer._server.on_client_connect(lambda _: connected.set())
                connection = connect(
                    f"ws://127.0.0.1:{port}", subprotocols=[Subprotocol("viser-v1.1.1")]
                )
                camera = _messages.ViewerCameraMessage(
                    wxyz=(1.0, 0.0, 0.0, 0.0),
                    position=(2.0, 3.0, 4.0),
                    fov=1.0,
                    near=0.01,
                    far=100.0,
                    image_height=240,
                    image_width=320,
                    look_at=(0.0, 0.0, 0.0),
                    up_direction=(0.0, 0.0, 1.0),
                )
                connection.send(msgspec.msgpack.encode(camera.as_serializable_dict()))
                assert connected.wait(5)
                manager = runtime.native.observation_manager
                assert manager.cfg.policy.enable_corruption
                manager._obs_buffer = None
                before = torch.cuda.get_rng_state()
                runtime.native.sim.render()
                assert manager._obs_buffer is None
                assert torch.equal(before, torch.cuda.get_rng_state())
                print(
                    "CONNECTED_BEFORE_FIRST_RESET_NOISY_PLOTS_DO_NOT_COMPUTE_OR_CONSUME_RNG",
                    flush=True,
                )
            runtime.reset(seed=11)
            sim = runtime.native.sim
            if viewing:
                import warp as wp
                from isaaclab_newton.physics import NewtonBackendCfg

                physics = sim.get_or_create_backend(
                    NewtonBackendCfg(
                        physics_cfg=sim.cfg.physics,
                        device=sim.device,
                    )
                )
                (spectator,) = sim.visualizers
                assert physics is not spectator.backend
                for name in ("model", "state_0", "state_1", "control"):
                    original, copied = getattr(physics, name), getattr(spectator.backend, name)
                    assert original is not copied
                    for key, value in vars(original).items():
                        other = getattr(copied, key, None)
                        if (
                            isinstance(value, wp.array)
                            and value.size
                            and isinstance(other, wp.array)
                        ):
                            assert value.ptr != other.ptr, (name, key)
                print("NATIVE_SPECTATOR_MODEL_STATE_CONTROL_STORAGE_ISOLATED", flush=True)
            trajectory = []
            for step in range(12):
                action = TensorDict(
                    {"action": torch.tensor([[0.2], [-0.3]], device="cuda:0")}, batch_size=(2,)
                )
                transition = runtime.step(action)
                trajectory.append(record(runtime, transition, action))
                runtime.reset(mask=(transition.terminated | transition.truncated).flatten())
            torch.save(trajectory, output)
        finally:
            if connection is not None:
                connection.close()
            runtime.close()
    assert not [t for t in enumerate_threads() if t not in baseline_threads and t.is_alive()]
    assert shutil.disk_usage(".").free >= 40 * 2**30
    print("NATIVE_SPECTATOR_SNAPSHOT_COMPLETE_AND_OWNED_THREADS_STOPPED", flush=True)


def startup_failure(mode, output, port):
    from isaaclab_viser_noisy import configure_failure
    from nexuml_library.environments.isaaclab import IsaacLabEnvironment, isaac_task_config

    assert shutil.disk_usage(".").free >= 40 * 2**30
    output.mkdir(parents=True, exist_ok=False)
    before_threads = set(enumerate_threads())
    reservation = socket.socket()
    if mode == "port-conflict":
        reservation.bind(("127.0.0.1", port))
        reservation.listen()
    definition = IsaacLabEnvironment(
        physics="newton_mjwarp",
        viewer=ViserSpec(port=port),
        task_config=isaac_task_config(configure_failure, receipt_path=str(output / "started.json")),
    )
    try:
        with definition.job_scope():
            try:
                definition.build(
                    EnvironmentBuildContext(purpose="train", device="cuda:0", num_envs=2, seed=11)
                )
            except (RuntimeError, OSError) as error:
                if mode == "port-conflict":
                    assert isinstance(error, OSError) and error.errno == 98
                    assert not (output / "started.json").exists()
                else:
                    assert "injected native observation construction failure" in str(error)
                    receipt = json.loads((output / "started.json").read_text())
                    assert receipt == {"host": "127.0.0.1", "port": port}
                from isaaclab.sim import SimulationContext

                assert SimulationContext.instance() is None
            else:
                raise AssertionError("Native startup should have failed")
    finally:
        reservation.close()
    with socket.socket() as released:
        released.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        released.bind(("127.0.0.1", port))
    assert not [t for t in enumerate_threads() if t not in before_threads and t.is_alive()]
    print("NATIVE_PARTIAL_STARTUP_CONTEXT_LISTENER_AND_THREADS_RELEASED", mode, flush=True)


if __name__ == "__main__":
    if sys.argv[1] in {
        "train",
        "train-viewer",
        "replay",
        "replay-viewer",
        "failure-viewer",
        "cancel-viewer",
    }:
        browser_job(sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1] == "noise":
        snapshot(Path(sys.argv[2]), sys.argv[3] == "viewer", int(sys.argv[4]), noisy=True)
    elif sys.argv[1] in {"port-conflict", "startup-failure"}:
        startup_failure(sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]))
    else:
        snapshot(Path(sys.argv[1]), sys.argv[2] == "viewer", int(sys.argv[3]))
