# Task 6.1 — bounded system-provider audit

No new provider or live support claim. The existing CARLA and simulated ROS
2 recipes keep the same public environment and runtime interfaces. Their fake
tests passed in the focused playback regression suite (443 tests total,
2 live tests deselected). This interpreter has no CARLA client, `rclpy`, or
Gazebo tools. Historical live runs describe a prepared external runtime and do
not qualify the deterministic stepping below.

## Current boundaries

- **CARLA:** Owns one connection's synchronous settings, fixed time delta,
  actor creation and teardown on a dedicated server. It refuses a world
  already in synchronous or no-rendering mode. The camera queue discards
  older frames and refuses frames ahead of the returned tick. The adapter
  also rejects an advance before action submission, a skipped tick, or
  a camera with no matching world/ego snapshot. Ego pose and velocity now
  come from the same world snapshot as the camera, not the last actor
  update. Checks detect observed interference; the CARLA server still
  relies on the operator to provide exactly one tick owner. The sparse
  collision callback has no per-frame negative acknowledgment. A live
  reference must verify collision timing, restoration of world settings
  and actor cleanup on shutdown, timeout and cancellation.
- **ROS 2:** `RclpyTransport` owns its context, node and executor. It validates
  joint order, finite samples, configured bounds and per-command freshness.
  The current simulated joint-reach workflow publishes a `JointTrajectory`,
  waits for a sufficiently recent state, and requests an operator-provided
  `std_srvs/Empty` reset. It is not a physics tick owner. A node namespace does
  not isolate absolute default topic names. Do not map this reset hook to
  hardware or imply that a software timeout is a hardware watchdog.

## First deterministic Gazebo reference

Use one headless Harmonic world and the existing two-joint `ros2_control`
reference on a prepared Jazzy profile. Keep world/SDF, controller, broadcaster
and bridge in the native launch project. The launched server must start paused.
Assign an isolated `ROS_DOMAIN_ID` and `GZ_PARTITION`, unique world and ROS node
namespace, relative command/state topic names, and a single bridge from
`/world/<world>/clock` to ROS `/clock` in that isolated domain. Use
simulation-stamped `JointState` samples, not a bridge wall-time override.

Use `ros_gz_bridge` conversions, not a new protocol:

1. Bridge `/world/<world>/control` as `ros_gz_interfaces/srv/ControlWorld`
   (`gz.msgs.WorldControl` → `gz.msgs.Boolean`), and subscribe to
   `/world/<world>/stats` as `ros_gz_interfaces/msg/WorldStatistics`.
   Require paused status and an observed iteration/time baseline before
   accepting any action. The service response confirms enqueueing, not
   completion. Do not allow a second world-control publisher.
2. The transport must verify controller acceptance before counting an
   action interval. The existing `JointTrajectory` topic alone offers no
   acknowledgment. Qualify a controller-supported acknowledgment or a
   native command latch at the control boundary, not a guessed wall delay.
   The physics step size must divide the declared control period as a
   positive integer `k`; issue `pause=true, multi_step=k`. Wait for exactly
   `k` iterations, the corresponding simulation time advance, re-pause and
   a fresh joint sample at that boundary. Align physics, controller and
   broadcaster update rates so the required sample can actually occur.
   Enforce monotonic host deadlines; reject extra ticks, stale or missing
   data, and ambiguous action application instead of emitting a transition.
3. Reset using `pause=true, reset={all: true}` and wait for the new
   clock/iteration epoch and ready controller. Discard pre-reset messages;
   use a prepared reset barrier/reinitialization if the controller cannot
   distinguish epochs. Observe the ready, fresh joint state after a separately
   counted initialization interval. Do not assume the world-control `seed`
   field works: the Harmonic runner warns that changing it is unsupported.
   Verify supported launch seed and actual reset state by repeated measured
   runs. Never substitute a physical reset.
4. Test two isolated launches to prove namespaces/domains do not cross-talk,
   with signed known controls, exact native counts and bounded failure and
   cleanup. Stop only owned launches and transports. Keep the existing
   public task and fake transport tests; deterministic stepping requires the
   live reference before advertising support.

Upstream references: [Gazebo reset API](https://gazebosim.org/api/sim/8/reset_simulation.html),
[WorldControl schema](https://github.com/gazebosim/gz-msgs/blob/gz-msgs10/proto/gz/msgs/world_control.proto),
[Harmonic world runner](https://github.com/gazebosim/gz-sim/blob/gz-sim8/src/SimulationRunner.cc),
and [Jazzy bridge conversions](https://github.com/gazebosim/ros_gz/tree/jazzy/ros_gz_bridge).

The CARLA guard uses the upstream
[synchronous-mode rules](https://github.com/carla-simulator/carla/blob/0.9.16/Docs/adv_synchrony_timestep.md)
and [snapshot API](https://github.com/carla-simulator/carla/blob/0.9.16/PythonAPI/docs/snapshot.yml).
Focused fake CARLA/ROS and playback tests: **25 passed, 2 live cases
deselected**. The fake CARLA server injected extra ticks, a moved world
snapshot, a missing ego and camera timeout, then verified owned cleanup.
No current live CARLA/Gazebo or physical ROS 2 qualification.
