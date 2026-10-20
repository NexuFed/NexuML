"""Pinned private spectator construction; importing this module starts no native runtime."""

from contextlib import contextmanager
from hashlib import sha256
from importlib.metadata import distribution
import atexit
import json
import os
from pathlib import Path
import signal
import socket
from threading import current_thread, main_thread
from time import monotonic
from types import SimpleNamespace


_SOURCE_HASHES = {
    "isaaclab": {
        "isaaclab_visualizers/viser/viser_visualizer.py": (
            "01899931ba1b58db548bd460cfa440d4bb3763c7ca0ab9cfeb8ce312c24578ae"
        ),
        "isaaclab_visualizers/viser/viser_visualizer_cfg.py": (
            "0174ce641bf7e03a28e7f097c06b2c47b99461b7d15040d7bbb4c88637cda3ab"
        ),
        "isaaclab_visualizers/newton_adapter.py": (
            "3f50d2c1687dff0ebc3d8e03764b164f2d09ccb6574c6d25b6d343e059d9f3ec"
        ),
    },
    "newton": {
        "newton/_src/viewer/viewer_viser.py": (
            "18f3b4d032f75393112e7a65d9fb0ad063133dbc586ba7fd022187dcd4db0008"
        ),
    },
    "viser": {
        "viser/_viser.py": "377a13103970fadac1022408400e378b3c79c03a6cfcb494f21d2518cc2a7634",
        "viser/infra/_infra.py": "fbb1cd3c9983d9e636ad6574fc3532f7b39015e8968d4d32a94d76d387bab71f",
        "viser/_gui_api.py": "31bfedabb159e23bd9cd878fe1deae7d3fe75dcb10d94634948526a25d9a8853",
        "viser/_client_autobuild.py": (
            "62f4668ca699edb261ad71b6d7cbf9bb82ff6c0971607b268977fc8e28c631ee"
        ),
    },
}


def _qualify_installation():
    for name, expected in (
        ("isaaclab", "3.0.0"),
        ("isaacsim", "6.2.0.0"),
        ("newton", "1.6.1"),
        ("viser", "1.1.1"),
    ):
        installed = distribution(name)
        if installed.version != expected:
            raise RuntimeError(f"Unqualified spectator distribution: {name} {installed.version}")
        for relative, digest in _SOURCE_HASHES.get(name, {}).items():
            if sha256(installed.locate_file(relative).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Unqualified spectator source: {relative}")
    lab = distribution("isaaclab")
    origin = json.loads(lab.read_text("direct_url.json") or "{}")
    if origin.get("vcs_info", {}).get("commit_id") != ("70ab24fec3eabf51222817db0cdb5a4e7133bd7e"):
        raise RuntimeError("Unqualified Isaac Lab spectator revision")
    viser = distribution("viser")
    origin = json.loads(viser.read_text("direct_url.json") or "{}")
    root = Path(viser.locate_file("viser")).resolve()
    index = root / "client/build/index.html"
    if (
        origin.get("dir_info", {}).get("editable")
        or root.parent.name != "site-packages"
        or not index.is_file()
        or index.stat().st_size == 0
    ):
        raise RuntimeError("Spectator requires the installed noneditable bundled Viser frontend")


def _check_signal_ownership():
    if current_thread() is not main_thread() or not hasattr(signal, "setitimer"):
        raise RuntimeError("Spectator construction requires the POSIX job main thread")
    if signal.getsignal(signal.SIGALRM) != signal.SIG_DFL or signal.getitimer(
        signal.ITIMER_REAL
    ) != (0.0, 0.0):
        raise RuntimeError("Spectator construction cannot disturb an existing SIGALRM owner")


@contextmanager
def _constructor_deadline():
    _check_signal_ownership()
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def expired(_signum, _frame):
        raise TimeoutError("Viser spectator constructor exceeded its 10-second deadline")

    try:
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, 10.0)
        yield
    finally:
        # Disable the constructor deadline before partial-resource cleanup starts.
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _validate_configuration(cfg):
    if cfg.bind_address not in {"127.0.0.1", "localhost"} or cfg.share:
        raise ValueError("Spectator requires loopback binding with sharing disabled")
    if type(cfg.port) is not int or not 1 <= cfg.port <= 64535:
        raise ValueError("Spectator requires an explicit TCP port in 1..64535")
    if os.environ.get("_VISER_PORT_OVERRIDE") is not None:
        raise ValueError("Spectator cannot use _VISER_PORT_OVERRIDE")
    if cfg.streaming_view or cfg.record_to_viser is not None:
        raise ValueError("Spectator sensor streaming and native recording are unsupported")
    if (
        cfg.randomly_sample_visible_envs
        and cfg.max_visible_envs is not None
        and cfg.visible_env_indices is None
    ):
        raise ValueError("Spectator selection must not consume task RNG")


def spectator_visualizer_type():
    """Return the qualified native subtype without changing vendor globals.

    Returns:
        Private, display-only native visualizer type, not connected to train/replay.

    Raises:
        RuntimeError: If installed/runtime identities are not qualified.
    """
    _qualify_installation()
    from isaaclab_visualizers.viser.viser_visualizer import NewtonViewerViser, ViserVisualizer
    from isaaclab_visualizers.newton_adapter import apply_viewer_visible_worlds
    from newton.viewer import ViewerViser
    import viser
    from viser import _client_autobuild

    if (
        viser.ViserServer.__module__ != "viser._viser"
        or viser.ViserServer.__name__ != "ViserServer"
        or _client_autobuild.ensure_client_is_built.__name__ != "ensure_client_is_built"
        or Path(viser.__file__).resolve()
        != Path(distribution("viser").locate_file("viser/__init__.py")).resolve()
    ):
        raise RuntimeError("Viser spectator runtime identities lost qualification")
    for function, package, relative in (
        (ViewerViser.__init__, "newton", "newton/_src/viewer/viewer_viser.py"),
        (NewtonViewerViser.__init__, "isaaclab", "isaaclab_visualizers/viser/viser_visualizer.py"),
        (ViserVisualizer.__init__, "isaaclab", "isaaclab_visualizers/viser/viser_visualizer.py"),
        (viser.ViserServer.__init__, "viser", "viser/_viser.py"),
        (_client_autobuild.ensure_client_is_built, "viser", "viser/_client_autobuild.py"),
    ):
        if (
            Path(function.__code__.co_filename).resolve()
            != Path(distribution(package).locate_file(relative)).resolve()
        ):
            raise RuntimeError(f"Unqualified spectator import origin: {relative}")

    class PrivateServer(viser.ViserServer):
        def request_share_url(self, verbose=True):
            return None

        def stop(self):
            deadline = monotonic() + 15.0
            errors = []
            atexit.unregister(self.stop)
            infra = self.__dict__.get("_websock_server")
            executor = self.__dict__.get("_thread_executor")
            thread = infra.__dict__.get("_server_thread") if infra is not None else None
            loop = None
            if infra is not None:
                atexit.unregister(infra.stop)
                # A signal can interrupt start() before the native worker publishes its loop.
                # Reserve the pinned native stop() join's full one-second cost.
                readiness_deadline = deadline - 1.0
                while True:
                    state = infra.__dict__
                    loop = state.get("_background_event_loop")
                    ready = (
                        loop is not None
                        and state.get("_stop_event") is not None
                        and state.get("_broadcast_buffer") is not None
                    )
                    if (
                        ready
                        or thread is None
                        or not thread.is_alive()
                        or monotonic() >= readiness_deadline
                    ):
                        break
                    thread.join(timeout=min(0.01, max(0, readiness_deadline - monotonic())))
                if ready and not loop.is_closed():
                    try:
                        if deadline - monotonic() < 1.0:
                            raise RuntimeError(
                                "Viser shutdown deadline cannot fit the native one-second join"
                            )
                        infra.stop()
                    except BaseException as error:
                        errors.append(error)
            if thread is not None:
                try:
                    thread.join(timeout=max(0, deadline - monotonic()))
                    if thread.is_alive():
                        raise RuntimeError(
                            "Viser server loop thread did not stop within 15 seconds"
                        )
                    if loop is not None and not loop.is_closed() and not loop.is_running():
                        loop.close()
                except BaseException as error:
                    errors.append(error)
            if executor is not None:
                try:
                    executor.shutdown(wait=False)
                    for worker in tuple(executor._threads):
                        worker.join(timeout=max(0, deadline - monotonic()))
                    if any(worker.is_alive() for worker in executor._threads):
                        raise RuntimeError("Viser callback workers did not stop within 15 seconds")
                except BaseException as error:
                    errors.append(error)
            if errors:
                for error in errors[1:]:
                    errors[0].add_note(f"Spectator cleanup also failed: {error!r}")
                raise errors[0]

    class SpectatorViewer(NewtonViewerViser):
        _server_type = PrivateServer

        @classmethod
        def _get_viser(cls):
            raise NotImplementedError("Spectator native recording/static playback is unsupported")

        def __init__(self, port):
            def private_server(*, port, label):
                server = self._server_type.__new__(self._server_type)
                self._pending_server = server
                server.__init__(host="127.0.0.1", port=port, label=label, verbose=False)
                if server.get_host() != "127.0.0.1" or server.get_port() != port:
                    raise RuntimeError("Viser did not bind the exact requested private endpoint")
                return server

            self._get_viser = lambda: SimpleNamespace(ViserServer=private_server)
            try:
                # Bypass Isaac's constructor/global reassignment, not native geometry.
                ViewerViser.__init__(self, port=port, verbose=False, share=False)
            finally:
                del self._get_viser
            self._metadata = {}
            self._isaaclab_plane_grid_cache = {}
            self._per_plot_folders = {}
            self._live_plots_folder = None

        def close(self):
            self._running = False
            server = self.__dict__.get("_pending_server")
            if server is not None:
                server.stop()

        def save_recording(self):
            raise NotImplementedError("Spectator native recording/static playback is unsupported")

        def show_notebook(self, *args, **kwargs):
            raise NotImplementedError("Spectator native recording/static playback is unsupported")

        @staticmethod
        def _serve_viser_recording(*args, **kwargs):
            raise NotImplementedError("Spectator native recording/static playback is unsupported")

    class SpectatorVisualizer(ViserVisualizer):
        _viewer_type = SpectatorViewer

        def __init__(self, cfg):
            _validate_configuration(cfg)
            self._display_state = SimpleNamespace(paused=False)
            super().__init__(cfg)

        @property
        def _paused_rendering(self):
            return self._display_state.paused

        @_paused_rendering.setter
        def _paused_rendering(self, value):
            self._display_state.paused = value

        def _create_viewer(self, record_to_viser, metadata=None):
            _validate_configuration(self.cfg)
            _check_signal_ownership()
            _qualify_installation()
            if record_to_viser is not None:
                raise ValueError("Spectator native recording/static playback is unsupported")
            if self._viewer is not None:
                raise RuntimeError("Spectator already owns a viewer")
            # A bind precheck closes ordinary conflicts; the postcheck still covers the race.
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
                reservation.bind(("127.0.0.1", self.cfg.port))
            viewer = self._viewer_type.__new__(self._viewer_type)
            self._viewer = viewer
            try:
                with _constructor_deadline():
                    viewer.__init__(self.cfg.port)
                viewer.set_model(self.backend.model)
                viewer.show_particles = self.cfg.show_particles
                self._setup_isaaclab_sidebar(viewer._server)
                apply_viewer_visible_worlds(
                    viewer,
                    env_ids=self._env_ids,
                    max_visible_envs=self.cfg.max_visible_envs,
                    num_envs=int((metadata or {}).get("num_envs", 0)),
                )
                viewer.set_world_offsets((0.0, 0.0, 0.0))
                self._set_viser_camera_view(self._resolve_initial_camera_pose())
            except BaseException as primary:
                try:
                    viewer.close()
                except BaseException as error:
                    primary.add_note(f"Spectator construction cleanup also failed: {error!r}")
                    for note in tuple(getattr(error, "__notes__", ())):
                        primary.add_note(note)
                else:
                    self._viewer = None
                raise

        def close(self):
            if self._viewer is not None:
                self._viewer.close()
                self._viewer = None
            self._is_initialized = False
            self._is_closed = True
            self.backend = self._scene_data_provider = None

        def _setup_isaaclab_sidebar(self, server):
            display = self._display_state
            pause = server.gui.add_button("Pause Display")

            @pause.on_click
            def pause_display(_event):
                display.paused = not display.paused

        def is_training_paused(self):
            return False

        def is_reset_requested(self):
            return False

        def consume_reset_request(self):
            return False

    return SpectatorVisualizer
