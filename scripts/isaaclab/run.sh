#!/bin/bash
set -e

export CARB_APP_PATH=/isaac-sim/kit ISAAC_PATH=/isaac-sim EXP_PATH=/isaac-sim/apps
source /isaac-sim/setup_python_env.sh
export RESOURCE_NAME=IsaacSim
export LD_PRELOAD="/isaac-sim/kit/libcarb.so${LD_PRELOAD:+:$LD_PRELOAD}"
# The vendor python.sh maps every failure to 1; exec preserves Python's actual status.
exec /isaac-sim/kit/python/bin/python3 /workspace/nexuml/scripts/isaaclab/run.py "$@"
