"""Validate the pinned Kit streaming endpoint without provisioning a network."""

import ctypes
import socket

from .isaaclab_config import WebRTCSpec

_SIGNAL_PORT = 49100
_MEDIA_PORT = 47998


def preflight(spec: WebRTCSpec) -> str:
    """Check the advertised interface, local ports, and graphics/encoder libraries.

    Returns:
        Kit arguments for the approved private endpoint.

    Raises:
        ValueError: If the interface is missing or a native port is busy.
        RuntimeError: If a native graphics or encoder library is missing.
    """
    import psutil

    if not any(
        item.family == socket.AF_INET and item.address == spec.advertise_address
        for addresses in psutil.net_if_addrs().values()
        for item in addresses
    ):
        raise ValueError("Isaac WebRTC advertised IPv4 is not assigned on this worker")
    for name, kind, port in (
        ("TCP signaling", socket.SOCK_STREAM, _SIGNAL_PORT),
        ("UDP media", socket.SOCK_DGRAM, _MEDIA_PORT),
    ):
        with socket.socket(socket.AF_INET, kind) as candidate:
            try:
                # The pinned extension binds wildcard and may otherwise pick a new port.
                candidate.bind(("0.0.0.0", port))
            except OSError as error:
                raise ValueError(f"Isaac WebRTC {name} port {port} is unavailable") from error
    try:
        ctypes.CDLL("libvulkan.so.1")
        ctypes.CDLL("libnvidia-encode.so.1").NvEncodeAPICreateInstance
    except (OSError, AttributeError) as error:
        raise RuntimeError(
            "Isaac WebRTC requires Vulkan graphics and the NVIDIA encoder runtime"
        ) from error
    return (
        " --/exts/omni.kit.livestream.app/primaryStream/publicIp="
        + spec.advertise_address
        + " --/exts/omni.kit.livestream.app/primaryStream/enableEventTracing=false"
    )


def report_endpoint(spec: WebRTCSpec) -> None:
    """Check Kit's actual local signaling listener; UDP delivery needs a client.

    Raises:
        RuntimeError: If Kit did not open the requested native signaling port.
    """
    import psutil

    owner = psutil.Process()
    # Isaac Sim's pinned psutil 5.9.8 still names this method connections().
    inspect = getattr(owner, "net_connections", None) or owner.connections
    owned = inspect(kind="inet")
    signal = next(
        (
            item.laddr.ip
            for item in owned
            if item.type == socket.SOCK_STREAM
            and item.status == psutil.CONN_LISTEN
            and item.laddr.port == _SIGNAL_PORT
        ),
        None,
    )
    if signal is None:
        raise RuntimeError(f"Isaac WebRTC did not open TCP signaling on {_SIGNAL_PORT}")
    media = next(
        (
            item.laddr.ip
            for item in owned
            if item.type == socket.SOCK_DGRAM and item.laddr.port == _MEDIA_PORT
        ),
        None,
    )
    print(
        f"Isaac WebRTC: TCP {signal}:{_SIGNAL_PORT}; "
        f"UDP {media or 'awaiting client'}:{_MEDIA_PORT}; "
        f"advertised {spec.advertise_address} (private network only)",
        flush=True,
    )
