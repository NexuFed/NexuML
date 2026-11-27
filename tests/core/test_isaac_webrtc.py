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
