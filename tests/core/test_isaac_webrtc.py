"""Private WebRTC startup validation without creating a Kit application."""

from types import SimpleNamespace
import socket

import pytest

from nexuml_library.environments import _isaaclab_webrtc
from nexuml_library.environments.isaaclab import IsaacLabEnvironment
from nexuml_library.environments.isaaclab_config import WebRTCSpec


@pytest.mark.parametrize("address", ["8.8.8.8", "0.0.0.0", "example.org", "::1"])
def test_stream_rejects_non_private_or_non_ipv4_address(address):
    with pytest.raises(ValueError):
        WebRTCSpec(advertise_address=address, private_network=True)


def test_web_rtc_opt_in_can_only_change_playback():
    spec = WebRTCSpec(advertise_address="100.64.0.10", private_network=True)
    definition = IsaacLabEnvironment(webrtc=spec)
    assert definition.checkpoint_identity() == IsaacLabEnvironment().checkpoint_identity()
    definition.validate_playback("webrtc")
    with pytest.raises(ValueError, match="private_network"):
        WebRTCSpec.model_validate({"advertise_address": "127.0.0.1"})


def test_web_rtc_requires_requested_local_ports(monkeypatch):
    psutil = pytest.importorskip("psutil")
    spec = WebRTCSpec(advertise_address="127.0.0.1", private_network=True)
    monkeypatch.setattr(
        psutil,
        "net_if_addrs",
        lambda: {"lo": [SimpleNamespace(family=socket.AF_INET, address="127.0.0.1")]},
    )
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as busy:
        busy.bind(("127.0.0.1", 0))
        monkeypatch.setattr(_isaaclab_webrtc, "_MEDIA_PORT", busy.getsockname()[1])
        monkeypatch.setattr(_isaaclab_webrtc, "_SIGNAL_PORT", 0)
        with pytest.raises(ValueError, match="UDP media port"):
            _isaaclab_webrtc.preflight(spec)


def test_web_rtc_reports_listener_separately_from_media_socket(monkeypatch, capsys):
    pytest.importorskip("psutil")
    spec = WebRTCSpec(advertise_address="127.0.0.1", private_network=True)
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as signal,
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as media,
    ):
        signal.bind(("127.0.0.1", 0))
        media.bind(("127.0.0.1", 0))
        signal_port, media_port = signal.getsockname()[1], media.getsockname()[1]
        monkeypatch.setattr(_isaaclab_webrtc, "_SIGNAL_PORT", signal_port)
        monkeypatch.setattr(_isaaclab_webrtc, "_MEDIA_PORT", media_port)
        with pytest.raises(RuntimeError, match="did not open TCP signaling"):
            _isaaclab_webrtc.report_endpoint(spec)

        signal.listen()
        _isaaclab_webrtc.report_endpoint(spec)
        report = capsys.readouterr().out
        assert f"TCP 127.0.0.1:{signal_port}" in report
        assert f"UDP 127.0.0.1:{media_port}" in report

        media.close()
        _isaaclab_webrtc.report_endpoint(spec)
        assert f"UDP awaiting client:{media_port}" in capsys.readouterr().out


def test_stream_uses_minimal_experience_without_x11_presentation():
    from nexuml_library.environments._isaaclab_runtime import _launch_environment

    def check(_cfg, args):
        assert args["headless"] and args["livestream"] == 2
        assert args["experience"] == "isaaclab.python.headless.rendering.kit"
        assert "primaryStream/publicIp=127.0.0.1" in args["kit_args"]
        assert "present/enabled=true" not in args["kit_args"]
        assert "omni.kit.widget.toolbar" not in args["kit_args"]
        raise RuntimeError("stop before native launch")

    with pytest.raises(RuntimeError, match="stop before native launch"):
        _launch_environment(
            SimpleNamespace(device="cuda:0", _rendering=False),
            None,
            check,
            lambda: False,
            streaming=" --/exts/omni.kit.livestream.app/primaryStream/publicIp=127.0.0.1",
        )
