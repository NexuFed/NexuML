"""Run the two-joint ROS 2 acceptance fixture in an isolated Isaac Sim runtime."""

import argparse
import asyncio
import faulthandler
from hashlib import file_digest
import json
import math
import os
from pathlib import Path
from time import monotonic


_ARM = "/World/Arm"
_NAMESPACE = "nexuml_sil"
_JOINTS = ("joint1", "joint2")
_PERIOD = 0.2
_TIMEOUT = 0.35


def _trajectory_target(names, points):
    if tuple(names) != _JOINTS or len(points) != 1:
        raise ValueError("SIL expects one trajectory point for the two named joints")
    point = json.loads(points[0])
    target = point["positions"]
    interval = point["time_from_start"]
    if interval["sec"] + interval["nanosec"] / 1e9 != _PERIOD:
        raise ValueError("SIL trajectory duration must match the declared control period")
    if len(target) != len(_JOINTS) or not all(-1 <= q <= 1 for q in target):
        raise ValueError("SIL trajectory target is outside configured joint limits")
    return target


def _arm(stage):
    from pxr import Gf, UsdGeom, UsdPhysics

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdPhysics.Scene.Define(stage, "/World/Physics")
    root = UsdGeom.Xform.Define(stage, _ARM).GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(root)
    links = []
    for index in range(3):
        link = UsdGeom.Xform.Define(stage, f"{_ARM}/link{index}")
        link.AddTranslateOp().Set(Gf.Vec3d(index * 0.25, 0, 0.5))
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(link.GetPrim())
        mass.CreateMassAttr(1.0)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.1, 0.1, 0.1))
        links.append(link.GetPath())
    UsdPhysics.FixedJoint.Define(stage, f"{_ARM}/base_fixed").CreateBody1Rel().SetTargets(links[:1])
    for index, name in enumerate(_JOINTS):
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{_ARM}/{name}")
        joint.CreateBody0Rel().SetTargets([links[index]])
        joint.CreateBody1Rel().SetTargets([links[index + 1]])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.25, 0, 0))
        joint.CreateAxisAttr("Z")
        joint.CreateLowerLimitAttr(-math.degrees(1.0))
        joint.CreateUpperLimitAttr(math.degrees(1.0))
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(2000.0)
        drive.CreateDampingAttr(200.0)
        drive.CreateMaxForceAttr(20.0)


def _bridge_graph():
    from omni.graph.core import Controller
    from usdrt import Sdf

    keys = Controller.Keys
    path = "/World/ROSBridge"
    Controller.edit(
        {"graph_path": path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("tick", "omni.graph.action.OnPlaybackTick"),
                ("time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("context", "isaacsim.ros2.bridge.ROS2Context"),
                ("clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ("joints", "isaacsim.sensors.physics.IsaacReadJointState"),
                ("state", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("command", "isaacsim.ros2.bridge.ROS2Subscriber"),
                ("received", "omni.graph.action.Counter"),
                ("drive", "isaacsim.core.nodes.IsaacArticulationController"),
            ],
            keys.SET_VALUES: [
                ("context.inputs:useDomainIDEnvVar", True),
                ("clock.inputs:topicName", "/clock"),
                ("joints.inputs:prim", [Sdf.Path(_ARM)]),
                ("drive.inputs:robotPath", _ARM),
                ("drive.inputs:jointNames", list(_JOINTS)),
                ("drive.inputs:positionCommand", [0.0, 0.0]),
                ("state.inputs:nodeNamespace", _NAMESPACE),
                ("command.inputs:nodeNamespace", _NAMESPACE),
                ("command.inputs:topicName", "command"),
                ("command.inputs:messagePackage", "trajectory_msgs"),
                ("command.inputs:messageName", "JointTrajectory"),
            ],
            keys.CONNECT: [
                ("context.outputs:context", "clock.inputs:context"),
                ("context.outputs:context", "state.inputs:context"),
                ("context.outputs:context", "command.inputs:context"),
                ("tick.outputs:tick", "clock.inputs:execIn"),
                ("time.outputs:simulationTime", "clock.inputs:timeStamp"),
                ("tick.outputs:tick", "joints.inputs:execIn"),
                ("joints.outputs:execOut", "state.inputs:execIn"),
                *[
                    (f"joints.outputs:{key}", f"state.inputs:{key}")
                    for key in (
                        "jointNames",
                        "jointPositions",
                        "jointVelocities",
                        "jointEfforts",
                        "jointDofTypes",
                        "stageMetersPerUnit",
                    )
                ],
                ("time.outputs:simulationTime", "state.inputs:timeStamp"),
                ("tick.outputs:tick", "command.inputs:execIn"),
                ("tick.outputs:tick", "drive.inputs:execIn"),
                ("command.outputs:execOut", "received.inputs:execIn"),
            ],
        },
    )
    return path


async def _simulate(output):
    # Keep all stage operations on Kit's native event loop.
    import omni.kit.app
    import omni.timeline
    import omni.usd
    from omni.kit.viewport.utility import get_active_viewport
    import rclpy
    from omni.graph.core import Controller
    from std_srvs.srv import Empty

    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    viewport = get_active_viewport()
    if viewport is not None:
        viewport.updates_enabled = False
    try:
        context = omni.usd.get_context()
        await context.new_stage_async()
        stage = context.get_stage()
        _arm(stage)
        stage.GetRootLayer().Export(str(output / "arm.usda"))
        graph = _bridge_graph()
        timeline.play()
        for _ in range(12):
            await app.next_update_async()
        print("SIL_PHYSICS_READY", flush=True)
        rclpy.init()
        node = rclpy.create_node("reset_fixture", namespace=_NAMESPACE)
        trajectory = None
        drive = Controller.attribute(f"{graph}/drive.inputs:positionCommand")
        positions = Controller.attribute(f"{graph}/joints.outputs:jointPositions")

        def reset(request, response):
            nonlocal trajectory
            trajectory = None
            drive.set([0.0, 0.0])
            return response

        node.create_service(Empty, "reset", reset)
        count = Controller.attribute(f"{graph}/received.outputs:count")
        received = count.get()
        (output / "bridge-ready").touch()
        try:
            deadline = monotonic() + 180
            next_report = monotonic()
            while not (output / "stop").exists():
                if monotonic() > deadline:
                    raise TimeoutError("SIL watchdog: client did not finish")
                rclpy.spin_once(node, timeout_sec=0)
                await app.next_update_async()
                now = timeline.get_current_time()
                if monotonic() >= next_report:
                    graph_time = Controller.attribute(f"{graph}/time.outputs:simulationTime").get()
                    print("SIL_CLOCK_SOURCE", now, graph_time, flush=True)
                    next_report = monotonic() + 10
                current = count.get()
                if current != received:
                    first_command = received == 0
                    received = current
                    names = Controller.attribute(f"{graph}/command.outputs:joint_names").get()
                    points = Controller.attribute(f"{graph}/command.outputs:points").get()
                    if first_command:
                        print("SIL_FIRST_COMMAND", now, list(names), list(points), flush=True)
                    trajectory = (
                        now,
                        list(positions.get()),
                        _trajectory_target(names, points),
                    )
                    if first_command:
                        print("SIL_FIRST_COMMAND_VALIDATED", flush=True)
                if trajectory is not None:
                    start, initial, target = trajectory
                    elapsed = now - start
                    if elapsed >= _PERIOD + _TIMEOUT:
                        drive.set(list(positions.get()))
                        trajectory = None
                        (output / "watchdog").write_text(str(now))
                        print("SIL_WATCHDOG_HOLD", now, flush=True)
                    else:
                        fraction = min(elapsed / _PERIOD, 1)
                        drive.set([q + fraction * (goal - q) for q, goal in zip(initial, target)])
        finally:
            timeline.pause()
            print("SIL_SIM_TIME", timeline.get_current_time(), flush=True)
            node.destroy_node()
            rclpy.shutdown()
            print("SIL_OWNED_RESOURCES_CLOSED", flush=True)
    finally:
        timeline.stop()


def _record(output, episodes):
    from rclpy.qos import qos_profile_sensor_data
    from rosgraph_msgs.msg import Clock
    from tensordict import TensorDict

    from nexuml.core.components import EnvironmentBuildContext
    from nexuml_library.environments.ros2.joint_reach import Ros2JointReachEnvironment

    definition = Ros2JointReachEnvironment(
        namespace=f"/{_NAMESPACE}",
        joint_names=_JOINTS,
        joint_state_topic="joint_states",
        command_topic="command",
        reset_service="reset",
        control_period_s=_PERIOD,
        timeout_s=20,
        max_episode_steps=20,
    )
    rows = []
    runtime = definition.build(EnvironmentBuildContext(purpose="eval", device="cpu", num_envs=1))
    transport = runtime.transport
    try:
        clock_sample = None

        def receive_clock(message):
            nonlocal clock_sample
            clock_sample = (message.clock.sec, message.clock.nanosec)

        transport.node.create_subscription(Clock, "/clock", receive_clock, qos_profile_sensor_data)
        (output / "recorder-ready").touch()
        deadline = monotonic() + 120
        while (
            transport.node.get_clock().now().nanoseconds <= 0
            or not transport.publisher.get_subscription_count()
        ):
            if monotonic() > deadline:
                raise TimeoutError(
                    "Missing bridge clock or command subscriber: "
                    f"clock_ns={transport.node.get_clock().now().nanoseconds}, "
                    f"clock_sample={clock_sample}, "
                    f"clock_publishers={transport.node.count_publishers('/clock')}, "
                    f"command_subscribers={transport.publisher.get_subscription_count()}, "
                    f"topics={transport.node.get_topic_names_and_types()}"
                )
            transport.executor.spin_once(timeout_sec=0.1)
        for episode in range(episodes):
            obs = runtime.reset()
            # The simulation reset service commands zero; it does not teleport the arm.
            deadline = monotonic() + definition.timeout_s
            while obs["joint_position"].abs().max() >= 0.1:
                if monotonic() > deadline:
                    raise TimeoutError("Simulated arm did not settle at the reset pose")
                obs = runtime._observation()
            last_clock = transport.node.get_clock().now().nanoseconds
            last_state = transport._state_stamp
            for step in range(definition.max_episode_steps):
                fraction = min(1, (step + 1) / (3 + episode % 4))
                command = TensorDict({"joint_command": obs["target_position"] * fraction}, [])
                transition = runtime.step(command)
                stamp = transport.node.get_clock().now().nanoseconds
                state_stamp = transport._state_stamp
                assert stamp > last_clock and state_stamp > last_state
                rows.append(
                    {
                        "episode": episode,
                        "step": step,
                        "clock_ns": stamp,
                        "joint_state_stamp_ns": state_stamp,
                        "observation": {key: value.tolist() for key, value in obs.items()},
                        "action": command["joint_command"].tolist(),
                        "next_observation": {
                            key: value.tolist() for key, value in transition.observation.items()
                        },
                        "terminated": bool(transition.terminated.item()),
                        "truncated": bool(transition.truncated.item()),
                    }
                )
                obs, last_clock, last_state = transition.observation, stamp, state_stamp
                if rows[-1]["terminated"] or rows[-1]["truncated"]:
                    break
            assert obs["joint_position"][0] > 0.2 and obs["joint_position"][1] < -0.2
            print("SIL_EPISODE", episode, len(rows), obs["joint_position"].tolist(), flush=True)
        # Stop sending commands and observe the simulated drive's actual hold.
        deadline = monotonic() + definition.timeout_s
        while (transport.node.get_clock().now().nanoseconds - last_clock) < 1_000_000_000:
            if monotonic() > deadline:
                raise TimeoutError("Simulation clock stopped during watchdog test")
            transport.executor.spin_once(timeout_sec=0.05)
        settled, velocities = transport.receive_joint_state(definition.timeout_s)
        assert all(abs(angle) <= 1 for angle in settled)
        assert max(map(abs, velocities)) < 0.05
        watchdog = float((output / "watchdog").read_text())
        assert watchdog > last_clock / 1e9
        with (output / "arm.usda").open("rb") as asset:
            asset_id = file_digest(asset, "sha256").hexdigest()
        (output / "episodes.json").write_text(
            json.dumps(
                {
                    "provider": "isaac_sim_ros2_bridge",
                    "simulated": True,
                    "asset_sha256": asset_id,
                    "control_mode": "absolute_joint_position",
                    "position_unit": "rad",
                    "velocity_unit": "rad/s",
                    "contract": definition.describe().model_dump(mode="json"),
                    "joint_names": _JOINTS,
                    "control_period_s": _PERIOD,
                    "watchdog_s": _TIMEOUT,
                    "watchdog_clock_s": watchdog,
                    "final_hold": settled,
                    "episodes": rows,
                },
                indent=2,
            )
            + "\n"
        )
    except BaseException as error:
        if isinstance(error, TimeoutError):
            print(
                "SIL_FEEDBACK_DEADLINE",
                f"complete_episodes={sum(row['terminated'] or row['truncated'] for row in rows)}",
                f"clock_ns={transport.node.get_clock().now().nanoseconds}",
                f"sample_ns={transport._state_stamp}",
                f"required_ns={transport._ready_stamp}",
                f"sample_versions={transport._version}/{transport._minimum_version}",
                flush=True,
            )
        if rows:
            (output / "partial-transitions.json").write_text(json.dumps(rows, allow_nan=False))
        raise
    finally:
        runtime.close()
        assert not transport.context.ok()
        (output / "stop").touch()


def main():
    """Run one owned process for simulation or bridge recording."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("simulate", "record"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--episodes", type=int, choices=range(4, 9), default=4)
    args = parser.parse_args()
    if os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE") != "LOCALHOST" or not os.environ.get(
        "ROS_DOMAIN_ID"
    ):
        parser.error("An explicit isolated ROS domain and LOCALHOST discovery are required")
    if os.environ.get("ROS_STATIC_PEERS"):
        parser.error("External ROS peers are not permitted")
    if args.role == "record":
        _record(args.output, args.episodes)
        return

    import omni.kit.app

    async def run():
        import traceback

        app = omni.kit.app.get_app()
        faulthandler.dump_traceback_later(45)
        try:
            await _simulate(args.output)
        except Exception:
            traceback.print_exc()
            app.post_quit(1)
        else:
            app.post_quit()
        finally:
            faulthandler.cancel_dump_traceback_later()

    asyncio.ensure_future(run())


if __name__ == "__main__":
    main()
