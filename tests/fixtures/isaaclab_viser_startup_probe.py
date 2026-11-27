"""Pinned native deadlock reproducer; run only inside a bounded disposable child."""

from importlib.metadata import version
import socket


def main():
    """Exercise a private occupied endpoint; the parent must bound and reap this job."""
    assert version("viser") == "1.1.1"
    import viser

    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 65535))
        occupied.listen()
        print("OWNED_LOOPBACK_65535_RESERVED", flush=True)
        server = viser.ViserServer(host="127.0.0.1", port=65535, verbose=False)
        try:
            print("CONSTRUCTOR_RETURNED", flush=True)
        finally:
            server.stop()


if __name__ == "__main__":
    main()
