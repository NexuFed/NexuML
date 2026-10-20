"""Bounded CPU-only native spectator probes; not simulation or browser acceptance."""

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import random
import shutil
import signal
import socket
import struct
import sys
from threading import Event, Thread, enumerate as enumerate_threads
from time import monotonic
from types import SimpleNamespace
from urllib.request import urlopen


def free_port():
    for _ in range(10):
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
            if port < 64535:
                return port
    raise AssertionError("No bounded explicit test port available")


def stopped(server):
    if server is None:
        return
    infra = server.__dict__.get("_websock_server")
    if infra is not None:
        thread = infra.__dict__.get("_server_thread")
        assert thread is None or not thread.is_alive()
        loop = infra.__dict__.get("_background_event_loop")
        assert loop is None or loop.is_closed()
    pool = server.__dict__.get("_thread_executor")
    assert pool is None or all(not worker.is_alive() for worker in pool._threads)


def expect_error(kind, fn, contains=None):
    try:
        fn()
    except kind as error:
        if contains is not None:
            assert contains in str(error), repr(error)
        return error
    raise AssertionError(f"Expected {kind.__name__}")


def main(mode):
    root = Path(__file__).resolve().parents[2]
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert shutil.disk_usage(root).free >= 40 * 2**30
    from nexuml_library.environments import _isaaclab_viser as adapter

    if mode == "static":
        assert not any(name in sys.modules for name in ("viser", "newton", "isaaclab"))
        adapter._qualify_installation()
        assert not any(name in sys.modules for name in ("viser", "newton", "isaaclab"))
        adapter._SOURCE_HASHES["viser"]["viser/_viser.py"] = "0" * 64
        expect_error(
            RuntimeError, adapter.spectator_visualizer_type, "Unqualified spectator source"
        )
        assert "viser" not in sys.modules
        print("STATIC_IMPORT_AND_FAIL_CLOSED_QUALIFICATION_VERIFIED", flush=True)
        return

    adapter._qualify_installation()
    from isaaclab.app import launch_simulation
    from isaaclab_newton.physics import NewtonCfg

    with launch_simulation(NewtonCfg(), {"device": "cpu", "visualizer": ["none"]}):
        import msgspec
        import newton
        import numpy as np
        import psutil
        import viser
        from viser import _client_autobuild, _messages
        import warp as wp
        from websockets.sync.client import connect
        from isaaclab_visualizers.viser.viser_visualizer import NewtonViewerViser, ViserVisualizer
        from isaaclab_visualizers.viser.viser_visualizer_cfg import ViserVisualizerCfg

        original_server = viser.ViserServer
        original_autobuild = _client_autobuild.ensure_client_is_built
        original_newton_constructor = NewtonViewerViser.__init__
        original_infra_start = viser.infra.WebsockServer.start
        visualizer_type = adapter.spectator_visualizer_type()
        viewer_type = visualizer_type._viewer_type
        server_type = viewer_type._server_type
        port = free_port()
        cfg = ViserVisualizerCfg(bind_address="127.0.0.1", port=port, verbose=False)
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            body = builder.add_body()
            builder.add_shape_sphere(body=body, radius=0.1)
            model = builder.finalize(device="cpu")
        spectator = visualizer_type(cfg)
        spectator.backend = SimpleNamespace(model=model)
        server = None
        baseline_threads = set(enumerate_threads())
        original_handler = signal.getsignal(signal.SIGALRM)
        original_timer = signal.getitimer(signal.ITIMER_REAL)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        def construct():
            spectator._create_viewer(None, {"num_envs": 1})

        def listeners():
            return {
                (item.laddr.ip, item.laddr.port)
                for item in psutil.Process().connections(kind="tcp")
                if item.status == "LISTEN"
            }

        try:
            if mode == "prechecks":
                before = listeners()
                for changes in (
                    {"bind_address": "0.0.0.0"},
                    {"share": True},
                    {"port": 0},
                    {"port": True},
                    {"port": 64536},
                    {"streaming_view": True},
                    {"record_to_viser": "unsupported.viser"},
                    {"max_visible_envs": 1, "randomly_sample_visible_envs": True},
                ):
                    expect_error(ValueError, lambda: visualizer_type(cfg.replace(**changes)))
                os.environ["_VISER_PORT_OVERRIDE"] = str(port + 1)
                try:
                    expect_error(ValueError, construct, "_VISER_PORT_OVERRIDE")
                finally:
                    del os.environ["_VISER_PORT_OVERRIDE"]
                with socket.socket() as reservation:
                    reservation.bind(("127.0.0.1", port))
                    reservation.listen()
                    expect_error(OSError, construct)
                    assert spectator._viewer is None
                signal.setitimer(signal.ITIMER_REAL, 60, 60)
                try:
                    expect_error(RuntimeError, construct, "SIGALRM")
                    remaining, interval = signal.getitimer(signal.ITIMER_REAL)
                    assert 55 < remaining <= 60 and interval == 60
                    assert signal.getsignal(signal.SIGALRM) == original_handler
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)

                def handler(*_args):
                    return None

                signal.signal(signal.SIGALRM, handler)
                try:
                    expect_error(RuntimeError, construct, "SIGALRM")
                    assert signal.getsignal(signal.SIGALRM) is handler
                finally:
                    signal.signal(signal.SIGALRM, original_handler)
                errors = []

                def from_thread():
                    errors.append(expect_error(RuntimeError, construct, "main thread"))

                worker = Thread(target=from_thread)
                worker.start()
                worker.join(5)
                assert not worker.is_alive() and len(errors) == 1
                assert spectator._viewer is None and before == listeners()
                print("PREBIND_CONFLICT_CONFIG_AND_SIGNAL_OWNERSHIP_VERIFIED", flush=True)
            elif mode in {"entry-sigterm", "entry-sigint"}:
                delivered_signal = signal.SIGTERM if mode == "entry-sigterm" else signal.SIGINT
                primary = SystemExit(143) if mode == "entry-sigterm" else KeyboardInterrupt()
                previous_job_handler = signal.getsignal(delivered_signal)
                previous_trace = sys.gettrace()
                deadline_code = adapter._constructor_deadline.__wrapped__.__code__
                delivered = False

                def cancel_job(_signum, _frame):
                    raise primary

                def cancel_at_entry(frame, event, _arg):
                    nonlocal delivered
                    if (
                        not delivered
                        and event == "line"
                        and frame.f_code is deadline_code
                        and signal.getsignal(signal.SIGALRM) != original_handler
                    ):
                        delivered = True
                        assert signal.getitimer(signal.ITIMER_REAL) == original_timer
                        os.kill(os.getpid(), delivered_signal)
                    return cancel_at_entry

                class ConstructorMustNotRun(viewer_type):
                    def __init__(self, port):
                        raise AssertionError("Cancellation must occur before native construction")

                spectator._viewer_type = ConstructorMustNotRun
                signal.signal(delivered_signal, cancel_job)
                try:
                    sys.settrace(cancel_at_entry)
                    caught = expect_error(type(primary), construct)
                    assert caught is primary and delivered
                    assert signal.getsignal(signal.SIGALRM) == original_handler
                    assert signal.getitimer(signal.ITIMER_REAL) == original_timer
                    assert spectator._viewer is None
                    print("DELIVERED_ENTRY_CANCELLATION_RESTORES_SIGNAL_OWNER_VERIFIED", flush=True)
                finally:
                    sys.settrace(previous_trace)
                    signal.setitimer(signal.ITIMER_REAL, 0.0)
                    signal.signal(signal.SIGALRM, original_handler)
                    signal.signal(delivered_signal, previous_job_handler)
            elif mode in {"stop-late-ready", "stop-readiness-deadline"}:
                abort_publication, finish, release_teardown = Event(), Event(), Event()
                delayed = mode == "stop-readiness-deadline"

                class DelayedInfra:
                    def __init__(self):
                        self._background_event_loop = None
                        self._stop_event = None
                        self._broadcast_buffer = None
                        self.stop_calls = 0
                        self.join_duration = None
                        self._server_thread = Thread(target=self.publish_loop)

                    def publish_loop(self):
                        if abort_publication.wait(14.25 if delayed else 13.6):
                            return
                        loop = asyncio.new_event_loop()
                        self._background_event_loop = loop
                        self._stop_event = finish
                        self._broadcast_buffer = SimpleNamespace()
                        try:
                            finish.wait(20)
                            release_teardown.wait(1.05)
                        finally:
                            loop.close()

                    def stop(self):
                        self.stop_calls += 1
                        finish.set()
                        began = monotonic()
                        # Same synchronous one-second join cost as the qualified native infra.
                        self._server_thread.join(timeout=1.0)
                        self.join_duration = monotonic() - began

                infra = DelayedInfra()
                pool = ThreadPoolExecutor(max_workers=1)
                pool.submit(lambda: None).result(5)
                server = server_type.__new__(server_type)
                server._websock_server = infra
                server._thread_executor = pool
                infra._server_thread.start()
                began = monotonic()
                try:
                    if delayed:
                        expect_error(RuntimeError, server.stop, "within 15 seconds")
                        assert infra.stop_calls == 0
                    else:
                        server.stop()
                        assert infra.stop_calls == 1
                        assert 0.98 <= infra.join_duration <= 1.3
                        stopped(server)
                    elapsed = monotonic() - began
                    assert elapsed <= 15.3, elapsed
                    print(
                        "REAL_DELAYED_PUBLICATION_TOTAL_STOP_DEADLINE_VERIFIED",
                        mode,
                        f"elapsed={elapsed:.3f}",
                        f"native_calls={infra.stop_calls}",
                        f"native_join_seconds={infra.join_duration}",
                        flush=True,
                    )
                finally:
                    # Explicit deadline failures still reap every test-owned thread and worker.
                    abort_publication.set()
                    finish.set()
                    release_teardown.set()
                    infra._server_thread.join(3)
                    pool.shutdown(wait=False)
                    for worker in tuple(pool._threads):
                        worker.join(3)
                    stopped(server)
            elif mode.startswith("scene-"):
                errors = {
                    "scene-error": RuntimeError("post-bind scene failure"),
                    "scene-interrupt": KeyboardInterrupt("post-bind interrupt"),
                    "scene-sigterm": SystemExit(143),
                    "scene-cleanup-error": RuntimeError("post-bind primary failure"),
                }
                primary = errors[mode]
                owned = []

                def terminate_job(_signum, _frame):
                    raise primary

                if mode == "scene-sigterm":
                    signal.signal(signal.SIGTERM, terminate_job)

                class CleanupErrorServer(server_type):
                    def stop(self):
                        super().stop()
                        if not self.__dict__.get("reported_cleanup_error", False):
                            self.reported_cleanup_error = True
                            raise RuntimeError("retained cleanup failure")

                class FailScene(viewer_type):
                    _server_type = (
                        CleanupErrorServer if mode == "scene-cleanup-error" else server_type
                    )

                    def _setup_scene(self):
                        owned.append(self._pending_server)
                        assert self._pending_server is self._server
                        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
                        super()._setup_scene()
                        if mode == "scene-sigterm":
                            os.kill(os.getpid(), signal.SIGTERM)
                        raise primary

                    def close(self):
                        assert signal.getitimer(signal.ITIMER_REAL) == original_timer
                        assert signal.getsignal(signal.SIGALRM) == original_handler
                        super().close()

                spectator._viewer_type = FailScene
                caught = expect_error(type(primary), construct)
                assert caught is primary
                if mode == "scene-cleanup-error":
                    assert spectator._viewer is not None
                    assert any("retained cleanup failure" in note for note in caught.__notes__)
                else:
                    assert spectator._viewer is None
                server = owned[0]
                stopped(server)
                assert ("127.0.0.1", port) not in listeners()
                print("POSTBIND_SCENE_FAILURE_AND_CANCELLATION_CLEANUP_VERIFIED", flush=True)
            elif mode in {"readiness-timeout", "closed-loop", "port-mismatch", "early-failure"}:
                owned = []
                reservation = socket.socket()
                if mode == "readiness-timeout":
                    reservation.bind(("127.0.0.1", 65535))
                    reservation.listen()

                class FailedServer(server_type):
                    def stop(self):
                        assert signal.getitimer(signal.ITIMER_REAL) == original_timer
                        assert signal.getsignal(signal.SIGALRM) == original_handler
                        super().stop()

                    def __init__(self, **kwargs):
                        owned.append(self)
                        if mode == "early-failure":
                            raise KeyboardInterrupt("server not initialized")
                        if mode == "readiness-timeout":
                            # Native retries past 65535, leaving its real readiness wait stuck.
                            kwargs["port"] = 65535
                        elif mode == "port-mismatch":
                            kwargs["port"] = free_port()
                            assert kwargs["port"] != port
                        super().__init__(**kwargs)
                        if mode == "closed-loop":
                            ran = Event()
                            self._thread_executor.submit(ran.set).result(5)
                            assert ran.is_set() and self._thread_executor._threads
                            self._websock_server.stop()
                            self._websock_server._server_thread.join(5)
                            assert self._websock_server._background_event_loop.is_closed()
                            assert any(t.is_alive() for t in self._thread_executor._threads)
                            raise RuntimeError("post-bind closed-loop failure")

                class FailedViewer(viewer_type):
                    _server_type = FailedServer

                spectator._viewer_type = FailedViewer
                started = monotonic()
                try:
                    kind = (
                        TimeoutError
                        if mode == "readiness-timeout"
                        else KeyboardInterrupt
                        if mode == "early-failure"
                        else RuntimeError
                    )
                    expect_error(kind, construct)
                finally:
                    reservation.close()
                if mode == "readiness-timeout":
                    assert 9 <= monotonic() - started < 20
                assert spectator._viewer is None
                server = owned[0]
                stopped(server)
                assert ("127.0.0.1", port) not in listeners()
                print("PARTIAL_SERVER_FAILURE_OWNERSHIP_AND_CLEANUP_VERIFIED", mode, flush=True)
            else:
                construct()
                viewer = spectator._viewer
                server = viewer._server
                assert isinstance(viewer, NewtonViewerViser)
                assert type(viewer).log_geo is NewtonViewerViser.log_geo
                assert type(viewer).log_mesh is NewtonViewerViser.log_mesh
                assert viewer._meshes
                assert (server.get_host(), server.get_port()) == ("127.0.0.1", port)
                assert ("127.0.0.1", port) in listeners()
                assert all(host == "127.0.0.1" for host, p in listeners() if p == port)
                with urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
                    assert response.status == 200
                handles = list(server.gui._gui_input_handle_from_uuid.values())
                assert [handle.label for handle in handles] == ["Pause Display"]
                callback = handles[0]._impl.update_cb[0]
                assert tuple(cell.cell_contents for cell in callback.__closure__) == (
                    spectator._display_state,
                )
                assert server.request_share_url() is None and server._share_tunnel is None
                expect_error(NotImplementedError, viewer._get_viser)
                expect_error(NotImplementedError, viewer.save_recording)
                expect_error(NotImplementedError, viewer.show_notebook)
                expect_error(NotImplementedError, viewer._serve_viser_recording)
                connected, displayed, shared = Event(), Event(), Event()
                server.on_client_connect(lambda _: connected.set())
                handles[0].on_click(lambda _: displayed.set())
                deny_share = server.request_share_url

                def observe_share(*args, **kwargs):
                    result = deny_share(*args, **kwargs)
                    shared.set()
                    return result

                server.request_share_url = observe_share
                before_body = model.body_q.numpy().copy()
                before_rng = random.getstate()

                def camera():
                    return _messages.ViewerCameraMessage(
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

                if mode == "uncooperative-peer":
                    peer = socket.create_connection(("127.0.0.1", port), timeout=5)
                    try:
                        key = base64.b64encode(b"spectator-probe!").decode()
                        peer.sendall(
                            (
                                f"GET / HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                                "Upgrade: websocket\r\n"
                                f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                                "Sec-WebSocket-Version: 13\r\n"
                                "Sec-WebSocket-Protocol: viser-v1.1.1\r\n\r\n"
                            ).encode()
                        )
                        headers = peer.recv(8192)
                        assert b"101 Switching Protocols" in headers, headers
                        payload = msgspec.msgpack.encode(camera().as_serializable_dict())
                        mask = b"\x01\x02\x03\x04"
                        size = len(payload)
                        header = (
                            bytes((0x82, 0x80 | size))
                            if size < 126
                            else (bytes((0x82, 0x80 | 126)) + struct.pack("!H", size))
                        )
                        peer.sendall(
                            header
                            + mask
                            + bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
                        )
                        assert connected.wait(5)
                        began = monotonic()
                        spectator.close()
                        assert monotonic() - began < 15.5
                        stopped(server)
                        print("UNCOOPERATIVE_PEER_BOUNDED_THREAD_CLEANUP_VERIFIED", flush=True)
                    finally:
                        peer.close()
                elif mode == "dispatch":
                    with connect(
                        f"ws://127.0.0.1:{port}",
                        subprotocols=["viser-v1.1.1"],
                        open_timeout=5,
                        close_timeout=5,
                    ) as connection:

                        def send(message):
                            connection.send(msgspec.msgpack.encode(message.as_serializable_dict()))

                        send(camera())
                        assert connected.wait(5)
                        for unknown in ("Pause Simulation", "Reset Episode", "stale-control-id"):
                            send(_messages.GuiUpdateMessage(unknown, {"value": True}))
                        send(_messages.ShareUrlRequest())
                        assert shared.wait(5)
                        send(_messages.GuiUpdateMessage(handles[0]._impl.uuid, {"value": True}))
                        assert displayed.wait(5)
                        assert spectator.is_rendering_paused()
                        assert not spectator._paused_simulation and not spectator._reset_requested
                        assert server._share_tunnel is None
                        assert tuple(next(iter(server.get_clients().values())).camera.position) == (
                            2,
                            3,
                            4,
                        )

                    async def direct_requests():
                        for unknown in ("Pause Simulation", "Reset Episode", "stale-control-id"):
                            await server._websock_server._handle_incoming_message(
                                0, _messages.GuiUpdateMessage(unknown, {"value": True})
                            )
                        await server._websock_server._handle_incoming_message(
                            0, _messages.ShareUrlRequest()
                        )

                    asyncio.run_coroutine_threadsafe(direct_requests(), server._event_loop).result(
                        5
                    )
                    callback(None)
                    assert not spectator.is_rendering_paused()
                    # Test-only stale vendor registration cannot cross final simulation polls.
                    ViserVisualizer._setup_isaaclab_sidebar(spectator, server)
                    for handle in list(server.gui._gui_input_handle_from_uuid.values()):
                        if handle.label in {"Pause Simulation", "Reset Episode"}:
                            for stale_callback in handle._impl.update_cb:
                                stale_callback(None)
                    assert spectator._paused_simulation and spectator._reset_requested
                    assert not spectator.is_training_paused()
                    assert not spectator.is_reset_requested()
                    assert not spectator.consume_reset_request()
                    assert server._share_tunnel is None
                    print(
                        "NATIVE_GEOMETRY_HTTP_WEBSOCKET_AND_MUTATION_HANDOFF_VERIFIED", flush=True
                    )
                else:
                    raise AssertionError(mode)
                assert np.array_equal(model.body_q.numpy(), before_body)
                assert random.getstate() == before_rng
        finally:
            spectator.close()
            stopped(server)
            signal.signal(signal.SIGTERM, original_sigterm)
            assert signal.getsignal(signal.SIGALRM) == original_handler
            assert signal.getitimer(signal.ITIMER_REAL) == original_timer == (0.0, 0.0)
            assert viser.ViserServer is original_server
            assert _client_autobuild.ensure_client_is_built is original_autobuild
            assert NewtonViewerViser.__init__ is original_newton_constructor
            assert viser.infra.WebsockServer.start is original_infra_start
            assert not set(enumerate_threads()) - baseline_threads
            assert ("127.0.0.1", port) not in listeners()
            assert shutil.disk_usage(root).free >= 40 * 2**30
        print("OWNED_CLEANUP_SIGNAL_RESTORE_AND_VENDOR_IDENTITY_VERIFIED", mode, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
