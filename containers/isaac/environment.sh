#!/bin/bash
set -euo pipefail

# A normal NexuML CLI job still needs Kit's vendor environment.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
source /isaac-sim/setup_python_env.sh
export PYTHONPATH="/env/lib/python3.12/site-packages:$PYTHONPATH"
exec "$@"
