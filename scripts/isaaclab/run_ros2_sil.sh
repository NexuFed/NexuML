#!/usr/bin/env bash
# Run inside the prepared Isaac container on an isolated ROS 2 domain.
set -euo pipefail
script=$(cd "$(dirname "$0")" && pwd)/ros2_sil.py
output=$(realpath -m "${1:?Usage: bash run_ros2_sil.sh NEW_OUTPUT_DIRECTORY}")
case "$script$output" in *[[:space:]\"\\]*) exit 2 ;; esac
mkdir "$output" "$output/materials"
ros=/isaac-sim/exts/isaacsim.ros2.core/jazzy
export ROS_DISTRO=jazzy ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH="$ros/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ros/rclpy:${PYTHONPATH:-}"

timeout --kill-after=5 260 /env/bin/python -u "$script" record "$output" &
recorder=$!
trap 'kill "$recorder" 2>/dev/null || true' EXIT
for ((i=0; i<80; i++)); do
    [[ ! -f "$output/recorder-ready" ]] || break
    kill -0 "$recorder" 2>/dev/null || break
    sleep 1
done
test -f "$output/recorder-ready"
timeout --kill-after=5 200 /isaac-sim/kit/kit /isaac-sim/apps/isaacsim.exp.base.python.kit \
    --no-window --ext-folder /isaac-sim/exts --ext-folder /isaac-sim/apps \
    --enable isaacsim.ros2.nodes --/crashreporter/enabled=false --/app/fastShutdown=true \
    --/exts/omni.kit.material.library/cache_mdl=false \
    "--/exts/omni.kit.material.library/lib_paths=[\"$output/materials\"]" \
    --/exts/isaacsim.ros2.bridge/publish_without_verification=true \
    --exec "$script simulate $output" > "$output/sim.log" 2>&1 &
kit=$!
trap 'kill "$recorder" "$kit" 2>/dev/null || true' EXIT
if ! wait "$recorder"; then
    touch "$output/stop"
    wait "$kit" || true
    exit 1
fi
wait "$kit"
grep -q SIL_OWNED_RESOURCES_CLOSED "$output/sim.log"
test -s "$output/episodes.json"
