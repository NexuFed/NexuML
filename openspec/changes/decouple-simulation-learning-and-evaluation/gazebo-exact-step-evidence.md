# Task 6.2 — Native Gazebo exact-step reference

**Pass: empty-world and two-joint native references on the recorded binary
profile.** This is a headless CPU qualification probe for the
[system-provider audit](system-provider-audit.md#first-deterministic-gazebo-reference)
and [cross-simulator spec](specs/cross-simulator-qualification/spec.md), not a
NexuML adapter or release claim.

## Empty-world substrate observed on 2026-11-26

- Two Harmonic servers ran concurrently in one `--network none` container.
  Each launched paused with a unique world, Gazebo partition and ROS namespace,
  and a distinct ROS domain (181 or 182, localhost discovery). Each used one
  `ros_gz_bridge` for its world `ControlWorld`, WorldStatistics and one
  Gazebo-to-ROS bridge from the world clock to `/clock`. Each domain saw
  only its own ROS control service, namespace and sole `/clock` publisher;
  each Gazebo partition saw only its own world-control service. Both
  deliberately missing peer service lookups timed out at 0.4 s (monotonic).
- With 1 ms physics steps, both worlds in both clean runs reported
  `(iteration, sim time, paused)` of `(0, 0 ms, true)` at startup,
  `(25, 25 ms, true)` after `pause=true, multi_step=25`, and
  `(65, 65 ms, true)` after another 40 steps. A bridge response alone was
  never counted: the probe required fresh ROS `/clock` at the boundary
  and two matching paused statistics observations. No extra ticks were
  observed. Each peer stayed paused at its own prior boundary while the
  other world advanced and reset.
- `pause=true, reset={all: true}` produced a newly observed `(0, 0 ms, true)`
  epoch in both worlds. A separate 25-step batch returned to
  `(25, 25 ms, true)`. Server logs confirm the native launch option
  `--seed 812`; no seed was sent via world control. The repeat tests the
  empty-world clock/iteration sequence, not seeded robot state.
- Both clean runs exited 0 and reaped two servers and two bridges (four
  process groups per run). A separate run sent SIGINT to the probe PID
  after both bridges logged service creation, while it was awaiting its
  baseline. The probe exited `-2` (SIGINT), wrote no success receipt, and
  recorded all four groups reaped. The outer cancellation check exited 0.
  No owned containers or exploratory sandbox network remained; the
  derived local image was removed. No shared world, cluster node, GPU,
  display or hardware was used.

## Empty-world runtime and reproduction

Ubuntu Noble Linux/amd64; Docker 29.9.2-2; ROS 2 Jazzy, Python 3.12.3,
glibc 2.39, Gazebo Sim **8.15.0** (Harmonic, DART), `rclpy` 7.1.13,
Fast DDS RMW 8.4.5, `ros_gz_bridge` and interfaces 1.0.25. The
`gz_ros2_control` 1.2.20 and both joint controllers 4.43.1 packages were
installed, **not exercised**. See `runtime-packages.tsv` for complete
build suffixes and dependencies.

The sandbox used public, signed binary packages on the official
`ros:jazzy-ros-base` image, digest
`sha256:5147d3ae250ce754afd399f6bfec15b33dd4e473ca357e30432a3182031bfad6`.
The temporary Dockerfile used HTTPS Ubuntu archives and the USTC ROS
mirror with the base image's ROS signing key; it installed
`ros-jazzy-ros-gz-bridge`, `ros-jazzy-ros-gz-sim`,
`ros-jazzy-gz-ros2-control`, `ros-jazzy-joint-trajectory-controller`,
and `ros-jazzy-joint-state-broadcaster` with
`apt-get --no-install-recommends`. No source builds or changes to
NexuML dependencies. The derived image ID was
`sha256:abfc64c180d2c37f28fb752aae6a2b3c21eb59627d5fb788aa8faba6ef33c4ac`.

In a fresh private Linux sandbox with sourced Jazzy, native Harmonic,
and `ros_gz_bridge` (no host or shared ROS/Gazebo graph), run:

```bash
source /opt/ros/jazzy/setup.bash
gz sim --versions
timeout --signal=INT -k 10s 60s \
  python3 scripts/gazebo/exact_step_probe.py /path/to/new-receipt-dir
```

Exact local commands used for these receipts, from the repository root:

```bash
d=/tmp/opencode/gz-exact-step.YFtN6i
DOCKER_CONFIG="$d/docker-config" docker pull ros:jazzy-ros-base
DOCKER_CONFIG="$d/docker-config" docker build --progress=plain \
  --tag nexuml-gazebo-exact-step:yftn6i "$d/build"
for run in final-a final-b cancelled; do
  bash "$d/run-final.sh" "$PWD/scripts/gazebo/exact_step_probe.py" "$run"
done
ruff check scripts/gazebo/exact_step_probe.py
ruff format --check scripts/gazebo/exact_step_probe.py
openspec validate decouple-simulation-learning-and-evaluation --strict
```

Local receipts under `/tmp/opencode/gz-exact-step.YFtN6i/`: `build/Dockerfile`,
`runtime-build-binary.log`, `run-final.sh`, `cancel.py`, and `output/`
with `runtime-image.txt`, `runtime-packages.tsv`,
`final-a/receipt.json`, `final-b/receipt.json`,
`cancelled/cancel-receipt.json`, per-world `world.sdf`, `server.log`,
and `bridge.log`, container state/cleanup, and `SHA256SUMS`.
`owned-leftovers.txt` is empty.

## Native two-joint reference observed on 2026-11-26

The standalone `scripts/gazebo/joint_controller_probe.py` generates a paused
SDF world from `two_joint.urdf` and `controllers.yaml` per namespace. It uses
the official Harmonic `gz_ros2_control` plugin and Jazzy controller manager
4.50.0, joint trajectory controller and joint-state broadcaster 4.43.1.
Both controllers were loaded, configured and confirmed active; no mock or
topic-only command path. This fixture does not depend on the earlier
[continuous-time two-joint run](../qualify-physical-ai-workflows-via-ray-jobs/support-handoff.md#job-installed-gazebo-runtime-probe)
or its private RoboStack assets.

The stock ROS joint-state broadcaster did **not** satisfy the exact-stamp
oracle: the first trial saw a newest `JointState` stamp of 619 ms with the
world already paused at tick 620 and `/clock` at 620 ms. This version of
[controller manager](https://github.com/ros-controls/ros2_control/blob/4.50.0/controller_manager/src/controller_manager.cpp)
calls `trigger_update(this->now(), ...)`, and the
[broadcaster](https://github.com/ros-controls/ros2_controllers/blob/4.43.1/joint_state_broadcaster/src/joint_state_broadcaster.cpp)
uses that update time as its stamp. The final reference retains the
controller and broadcaster but uses Gazebo's maintained
[JointStatePublisher](https://github.com/gazebosim/gz-sim/blob/gz-sim8/src/systems/joint_state_publisher/JointStatePublisher.cc)
for the boundary sample. It publishes post-physics joint state stamped from
`UpdateInfo.simTime`; a one-way `ros_gz_bridge` converts `gz.msgs.Model` to
`sensor_msgs/JointState` at the relative `physics_joint_states` topic. It
does not change `/clock`, restamp samples, add an extra tick to let a
publisher catch up, or accept a 1 ms tolerance.

- **Action:** Five counted 120-tick controller-manager windows plus a
  separate 20-tick observation window established a fresh zero position and
  velocity sample at tick 620 / 620 ms. Both the `FollowJointTrajectory`
  accepted response and `EXECUTING` status for the goal UUID were observed
  while the world still reported paused at that boundary. Only then did
  the probe request `pause=true, multi_step=80`. At tick 700 / 700 ms,
  the world had re-paused and the action reported `SUCCEEDED` /
  `SUCCESSFUL`. A fresh joint sample carried exactly that simulation stamp.
- **Signed motion:** In both final runs the left target `[+0.25, -0.25]`
  rad produced `[+0.3356, -0.3356]` rad; the right world's opposite target
  produced `[-0.3356, +0.3356]` rad. Each world reproduced those samples
  on replay after reset (eight successful goals total). The oracle checks
  signed displacement and joint bounds, not setpoint-tracking accuracy;
  the observed final sample exceeds the target magnitude.
- **Reset epoch:** The probe removed the old action client and joint
  reader, stopped its joint bridge, and observed `reset all` return the
  paused world to tick 0 / 0 ms. It checked for an empty new manager,
  reloaded and activated the controllers, and required a different ROS
  publisher GID before installing a fresh volatile DDS reader. The counted
  initialization returned both joints and velocities to zero at 620 ms.
  An actual retained sample stamped 620 ms from the old epoch was fed to
  the sample selector and excluded; the new 620 ms sample was accepted.
  This tests cached stale data, not a claim that delayed old DDS traffic
  was observed.
- **Isolation:** Both worlds overlapped in one `--network none` container
  with distinct Gazebo partitions, ROS domains and namespaces. The peer's
  control, action and state endpoints were absent from the local graph.
  The peer kept both its paused boundary and its measured joint state
  during the other world's action, reset and replay.
- **Cleanup:** Both clean runs reaped eight groups each, including the
  joint bridges retired at reset. SIGINT after both controllers were
  active and one goal was `EXECUTING` exited `-2`, reaped all six live
  groups and wrote no success receipt. A prior SIGINT landed inside an
  `rclpy`/pybind conversion and exited 1; all six groups were still reaped
  and no success receipt was written. The empty-world check also passed
  again after the shared probe changes.

Run in the same private profile as the substrate, with the three controller
packages listed above in addition to `ros_gz_bridge`:

```bash
source /opt/ros/jazzy/setup.bash
timeout --signal=INT -k 10s 95s \
  python3 scripts/gazebo/joint_controller_probe.py /path/to/new-joint-receipt-dir
```

Exact binary-sandbox commands, from the repository root:

```bash
d=/tmp/opencode/gz-exact-step.YFtN6i
DOCKER_CONFIG="$d/docker-config" docker build --progress=plain \
  --tag nexuml-gazebo-exact-step:yftn6i "$d/build"
for run in controller-final-a controller-final-b controller-cancel; do
  bash "$d/run-joint.sh" "$PWD/scripts/gazebo" "$run"
done
bash "$d/run-joint.sh" "$PWD/scripts/gazebo" substrate-repeat exact_step_probe.py
ruff check scripts/gazebo
ruff format --check scripts/gazebo
openspec validate decouple-simulation-learning-and-evaluation --strict
```

Receipts in `$d/joint/`: `controller-first/` is the ROS broadcaster
exact-stamp failure; `controller-final-{a,b}/receipt.json` list UUIDs,
accepted and final boundaries, action result codes, joint samples and
old/new writer GIDs. Each world directory holds the exact generated SDF,
URDF, YAML and `events.jsonl`, `server.log`, control bridge log and
per-epoch joint bridge logs. Container logs include `dpkg-query` build IDs
and SHA-256 of the actual installed simulator, publisher and controller
libraries. The separate `controller-cancel/cancel-receipt.json`,
`controller-cancel-signal-race.log` and `substrate-repeat/receipt.json`
cover cancellation and the unchanged empty-world gate. `SHA256SUMS`
covers the retained proof artifacts. No probe containers or networks
remained; the temporary derived image was removed after verification.

The standalone Gazebo reference does not qualify a NexuML runtime adapter,
a freely running topic-command recipe, general tracking accuracy, CARLA,
or physical ROS 2. No product code, cluster node or hardware was used.
