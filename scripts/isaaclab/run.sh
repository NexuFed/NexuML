#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${NEXUML_ISAAC_PYTHON:-$SCRIPT_DIR/../../.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "Select the prepared worktree interpreter with NEXUML_ISAAC_PYTHON; no installer is run." >&2
    exit 1
fi
exec "$PYTHON" "$SCRIPT_DIR/run.py" "$@"
