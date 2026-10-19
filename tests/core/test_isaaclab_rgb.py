"""CPU frame/recorder identity proofs; never construct a simulator or media writer."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nexuml_library.environments import _isaaclab_rgb as rgb
from nexuml_library.environments._isaaclab_runtime import _IsaacLabRuntime


def _env(channels=4):
    events = []
    image = torch.full((1, 240, 320, channels), 35, dtype=torch.uint8)
    camera = SimpleNamespace(
        data=SimpleNamespace(output={"rgb": SimpleNamespace(torch=image)}),
        update=lambda **kwargs: events.append(("camera.update", kwargs)),
        set_world_poses_from_view=lambda *args: events.append(("camera.pose", args)),
    )
    return SimpleNamespace(
        device="cpu",
        scene={"nexuml_replay_camera": camera},
        sim=SimpleNamespace(render=lambda: events.append("render")),
        events=events,
    )


@pytest.fixture
def configured(monkeypatch):
    class RecorderTerm:
        def __init__(self, cfg, env):
            self._env = env

    modules = {
        "isaaclab.sensors": SimpleNamespace(CameraCfg=SimpleNamespace),
        "isaaclab.sim": SimpleNamespace(PinholeCameraCfg=SimpleNamespace),
        "isaaclab_physx.renderers": SimpleNamespace(IsaacRtxRendererCfg=SimpleNamespace),
        "isaaclab.managers": SimpleNamespace(
            RecorderTerm=RecorderTerm, RecorderTermCfg=SimpleNamespace
        ),
        "isaaclab.managers.recorder_manager": SimpleNamespace(
            RecorderManagerBaseCfg=SimpleNamespace,
            DatasetExportMode=SimpleNamespace(EXPORT_NONE=0),
        ),
    }
    monkeypatch.setattr(rgb, "import_module", modules.__getitem__)
    cfg = SimpleNamespace(scene=SimpleNamespace(), sim=SimpleNamespace(), decimation=4)
    owner = SimpleNamespace(_in_step=False, _terminal_frame=None)
    rgb.configure_rgb(cfg, owner)
    return cfg, owner


def test_replay_camera_is_one_explicit_rtx_sensor_without_dataset_writers(configured):
    cfg, _ = configured
    assert set(vars(cfg.scene)) == {"nexuml_replay_camera"}
    camera = cfg.scene.nexuml_replay_camera
    assert (camera.width, camera.height) == (320, 240)
    assert camera.data_types == ["rgb"] and camera.update_period == 0
    assert cfg.sim.render_interval == cfg.decimation == 4
    assert cfg.num_rerenders_on_reset == 1
    assert cfg.recorders.dataset_export_mode == 0
    assert not cfg.recorders.export_in_close and not cfg.recorders.export_in_record_pre_reset
    assert set(vars(cfg.recorders)) == {
        "dataset_export_mode",
        "export_in_record_pre_reset",
        "export_in_close",
        "terminal_rgb",
    }


@pytest.mark.parametrize("channels", [3, 4])
def test_camera_copy_is_rgb_uint8_and_never_steps_physics(channels):
    env = _env(channels)
    frame = rgb.read_rgb(env)
    assert frame.shape == (240, 320, 3) and frame.dtype == np.uint8
    assert env.events == ["render", ("camera.update", {"dt": 0.0, "force_recompute": True})]
    env.scene["nexuml_replay_camera"].data.output["rgb"].torch.fill_(90)
    assert np.all(frame == 35)


@pytest.mark.parametrize(
    "image",
    [
        torch.zeros(1, 240, 320, 3),
        torch.zeros(240, 320, 3, dtype=torch.uint8),
        torch.zeros(2, 240, 320, 3, dtype=torch.uint8),
        torch.zeros(1, 240, 319, 3, dtype=torch.uint8),
        torch.zeros(1, 240, 320, 2, dtype=torch.uint8),
    ],
)
def test_invalid_rgb_is_rejected_without_conversion_or_renderer_fallback(image):
    env = _env()
    env.scene["nexuml_replay_camera"].data.output["rgb"].torch = image
    with pytest.raises(ValueError, match="320x240 uint8"):
        rgb.read_rgb(env)


@pytest.mark.parametrize("output", [None, {}])
def test_missing_rgb_has_an_explicit_diagnostic(output):
    env = _env()
    env.scene["nexuml_replay_camera"].data.output = output
    with pytest.raises(ValueError, match="did not supply RGB"):
        rgb.read_rgb(env)


def test_pre_reset_cache_retains_terminal_scene_not_native_reset_scene(configured):
    cfg, owner = configured
    env = _env()
    term_cfg = cfg.recorders.terminal_rgb
    term = term_cfg.class_type(term_cfg, env)
    assert term.record_pre_reset([0]) == (None, None)
    assert owner._terminal_frame is None, "explicit reset must not capture a previous episode"
    owner._in_step = True
    assert term.record_pre_reset([0]) == (None, None)
    env.scene["nexuml_replay_camera"].data.output["rgb"].torch.fill_(90)
    assert np.all(rgb.read_rgb(env) == 90)
    assert np.all(owner._terminal_frame == 35)

    runtime = object.__new__(_IsaacLabRuntime)
    runtime._closed, runtime._rendering = False, True
    runtime.native = env
    runtime._terminal_frame = owner._terminal_frame
    frame = runtime.render_frame()
    assert np.all(frame == 35)
    frame.fill(1)
    assert np.all(runtime.render_frame() == 35)
    runtime._terminal_frame = None
    assert np.all(runtime.render_frame() == 90)
    runtime._closed = True
    with pytest.raises(RuntimeError, match="not enabled on an open"):
        runtime.render_frame()


@pytest.mark.parametrize("task", list(rgb._VIEWS))
def test_camera_uses_the_declared_task_view(task):
    env = _env()
    rgb.position_camera(env, task)
    name, (eyes, targets) = env.events[0]
    assert name == "camera.pose"
    torch.testing.assert_close(eyes, torch.tensor([rgb._VIEWS[task][0]]))
    torch.testing.assert_close(targets, torch.tensor([rgb._VIEWS[task][1]]))
