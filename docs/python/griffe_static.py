"""Run the locked Griffe CLI with runtime inspection explicitly forbidden."""

import sys

from griffecli import main

if __name__ == "__main__":
    if sys.argv[1:2] != ["dump"]:
        raise SystemExit("Only static Griffe dump is supported")
    raise SystemExit(main(["dump", "--no-inspection", *sys.argv[2:]]))
