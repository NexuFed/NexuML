"""Replay-only public camera/recorder composition; no second media writer."""

from importlib import import_module

import torch


_VIEWS = {
    "Isaac-Cartpole": ((4.0, -6.0, 3.0), (0.0, 0.0, 1.5)),
    "Isaac-Reach-Franka": ((1.6, -1.6, 1.3), (0.0, 0.0, 0.45)),
    "NexuML-SO101-Joint-Reach": ((0.65, -0.65, 0.55), (0.0, 0.0, 0.16)),
}


def configure_rgb(cfg, owner) -> None:
    """Attach one fixed-size RTX camera and a non-exporting pre-reset frame cache."""
    camera = import_module("isaaclab.sensors").CameraCfg
    sim = import_module("isaaclab.sim")
    renderer = import_module("isaaclab_physx.renderers").IsaacRtxRendererCfg
    managers = import_module("isaaclab.managers")
    recorders = import_module("isaaclab.managers.recorder_manager")
    cfg.scene.nexuml_replay_camera = camera(
        prim_path="{ENV_REGEX_NS}/NexuMLReplayCamera",
        width=320,
        height=240,
        update_period=0.0,
        data_types=["rgb"],
        renderer_cfg=renderer(),
        spawn=sim.PinholeCameraCfg(clipping_range=(0.01, 100.0)),
    )
    cfg.sim.render_interval = cfg.decimation
    cfg.num_rerenders_on_reset = 1

    class TerminalFrame(managers.RecorderTerm):
        """Keep only the ending frame, not a dataset or a complete video."""

        def record_pre_reset(self, env_ids):
            """Capture native same-step auto-reset, never an explicit episode reset.

            Returns:
                An empty recorder entry; no frame is stored in a dataset buffer.
            """
            if owner._in_step:
                owner._terminal_frame = read_rgb(self._env)
            return None, None

    cfg.recorders = recorders.RecorderManagerBaseCfg(
        dataset_export_mode=recorders.DatasetExportMode.EXPORT_NONE,
        export_in_record_pre_reset=False,
        export_in_close=False,
    )
    cfg.recorders.terminal_rgb = managers.RecorderTermCfg(class_type=TerminalFrame)


def position_camera(env, task) -> None:
    """Position the supported task's spectator camera without changing observations."""
    eye, target = _VIEWS[task]
    env.scene["nexuml_replay_camera"].set_world_poses_from_view(
        torch.tensor([eye], device=env.device), torch.tensor([target], device=env.device)
    )


def read_rgb(env):
    """Return one copied CPU uint8 HWC frame without stepping physics.

    Raises:
        ValueError: If the selected camera cannot supply the declared RGB buffer.
    """
    env.sim.render()
    camera = env.scene["nexuml_replay_camera"]
    camera.update(dt=0.0, force_recompute=True)
    output = camera.data.output
    if output is None or "rgb" not in output:
        raise ValueError("Isaac RTX replay camera did not supply RGB output")
    image = output["rgb"].torch
    if (
        not isinstance(image, torch.Tensor)
        or image.dtype != torch.uint8
        or image.ndim != 4
        or tuple(image.shape[:3]) != (1, 240, 320)
        or image.shape[-1] not in (3, 4)
    ):
        raise ValueError("Isaac RTX replay requires a single 320x240 uint8 RGB/RGBA camera")
    return image[0, ..., :3].cpu().numpy().copy()
