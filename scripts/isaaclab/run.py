"""Run an ordinary Python script/module with a lazy, process-owned Isaac app."""

import runpy
from pathlib import Path
import sys

from nexuml_library.environments._isaaclab_runtime import isaaclab_process


def main():
    """Execute the requested job inside the owned application lifetime.

    Raises:
        SystemExit: If no script or module was supplied.
    """
    args = sys.argv[1:]
    module = bool(args and args[0] == "-m")
    if not args or (module and len(args) < 2):
        raise SystemExit("Usage: run.py SCRIPT [ARGS...] | -m MODULE [ARGS...]")
    target, *remaining = args[1:] if module else args
    sys.argv = [target, *remaining]
    sys.path[0] = str(Path.cwd() if module else Path(target).resolve().parent)
    with isaaclab_process():
        if module:
            runpy.run_module(target, run_name="__main__", alter_sys=True)
        else:
            runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
