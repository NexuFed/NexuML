"""Opt-in two-joint Jazzy/Harmonic reference; requires an isolated Linux sandbox."""

import json
import math
import time
import uuid
from contextlib import ExitStack
from pathlib import Path
from xml.etree import ElementTree as ET

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import (
    ConfigureController,
    ListControllers,
    LoadController,
    SwitchController,
)
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint

from exact_step_probe import _SEED, _STEP_NS, _World, _check, _main, _ns

_FIXTURE = Path(__file__).resolve().parent
_JOINTS = ("joint1", "joint2")
_BROADCASTER = "joint_state_broadcaster"
_CONTROLLER = "joint_trajectory_controller"
_STATE = "physics_joint_states"
_STATE_BRIDGE = "joint_state_bridge"
_ACTION = f"{_CONTROLLER}/follow_joint_trajectory"
_CONTROL_TICKS = 80


def _prepare_model(work, namespace):
    config = work / "controllers.yaml"
    config.write_bytes((_FIXTURE / config.name).read_bytes())
    tree = ET.parse(_FIXTURE / "two_joint.urdf")
    tree.find("./gazebo/plugin/parameters").text = str(config)
    tree.find("./gazebo/plugin/ros/namespace").text = f"/{namespace}"
    urdf = work / "two_joint.urdf"
    tree.write(urdf, encoding="utf-8", xml_declaration=True)
    return f"<include><uri>{urdf.as_uri()}</uri></include>"


def _matching(samples, epoch, stamp):
    return next(
        (sample for sample in samples if sample["epoch"] == epoch and sample["stamp_ns"] == stamp),
        None,
    )


class _JointWorld(_World):
    def __init__(self, owner, root, name, domain):
        super().__init__(owner, root, name, domain, prepare_model=_prepare_model)
        self.iteration = 0
        self.epoch = 0
        self.samples = []
        self.writer = None
        self.sub = None
        self.action = None
        self.work = root / name
        self.state_owner = ExitStack()
        self.state_process = None
        owner.callback(self.disarm)
        self.log = owner.enter_context((self.work / "events.jsonl").open("w", encoding="utf-8"))
        self.description = self.node.create_publisher(
            String,
            "robot_description",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.description.publish(
            String(data=(self.work / "two_joint.urdf").read_text(encoding="utf-8"))
        )

    def event(self, event, **values):
        self.log.write(
            json.dumps({"event": event, "host_ns": time.monotonic_ns(), **values}) + "\n"
        )
        self.log.flush()

    def advance(self, ticks):
        mark = self.mark()
        self.request(steps=ticks)
        boundary = self.witness(self.iteration + ticks, mark)
        self.iteration = boundary[0]
        return boundary

    def rpc(self, interface, service, request, *, initialize=False):
        client = self.node.create_client(interface, f"controller_manager/{service}")
        try:
            self.wait(client.service_is_ready, service)
            future = client.call_async(request)
            # Manager operations can need an update to commit their realtime list.
            if initialize:
                self.advance(120)
            self.wait(future.done, service)
            result = future.result()
            _check(
                result is not None and getattr(result, "ok", True),
                f"{self.name}: rejected {service}: {result}",
            )
            self.event(service, iteration=self.iteration)
            return result
        finally:
            self.node.destroy_client(client)

    def publishers(self):
        return self.node.get_publishers_info_by_topic(f"/{self.name}/{_STATE}")

    def controller_states(self):
        response = self.rpc(ListControllers, "list_controllers", ListControllers.Request())
        return {controller.name: controller.state for controller in response.controller}

    def disarm(self):
        if self.action is not None:
            self.action.destroy()
            self.action = None
        if self.sub is not None:
            self.node.destroy_subscription(self.sub)
            self.sub = None
        self.state_owner.close()
        if self.state_process is not None:
            self.processes.remove(self.state_process)
            self.state_process = None

    def start_state_bridge(self):
        # The ROS-state broadcaster uses the manager's asynchronously received /clock.
        # The native Gazebo sensor stamps the actual PostUpdate physics state.
        gz_topic = f"{self.path}/model/two_joint/joint_state"
        self.state_process = self.start(
            self.state_owner,
            self.work,
            f"state-bridge-{self.epoch}",
            [
                str(self.bridge),
                f"{gz_topic}@sensor_msgs/msg/JointState[gz.msgs.Model",
                "--ros-args",
                "-r",
                f"__node:={_STATE_BRIDGE}",
                "-r",
                f"__ns:=/{self.name}",
                "-r",
                f"{gz_topic}:={_STATE}",
            ],
        )

    def initialize(self, old_writer=None):
        if old_writer is not None:
            self.wait(lambda: not self.publishers(), "old joint-state writer retirement")
        _check(not self.controller_states(), f"{self.name}: manager retained old controllers")
        controllers = [_BROADCASTER, _CONTROLLER]
        for name in controllers:
            self.rpc(
                LoadController,
                "load_controller",
                LoadController.Request(name=name),
                initialize=True,
            )
            self.rpc(
                ConfigureController,
                "configure_controller",
                ConfigureController.Request(name=name),
                initialize=True,
            )
        self.rpc(
            SwitchController,
            "switch_controller",
            SwitchController.Request(
                activate_controllers=controllers,
                strictness=SwitchController.Request.STRICT,
                activate_asap=True,
                timeout=Duration(sec=3),
            ),
            initialize=True,
        )
        _check(
            self.controller_states() == dict.fromkeys(controllers, "active"),
            f"{self.name}: controllers not active",
        )
        self.start_state_bridge()

        def own_writer():
            publishers = self.publishers()
            return (
                len(publishers) == 1
                and publishers[0].node_name == _STATE_BRIDGE
                and publishers[0].node_namespace == f"/{self.name}"
            )

        self.wait(own_writer, "unique named joint-state writer")
        (endpoint,) = self.publishers()
        self.writer = bytes(endpoint.endpoint_gid).hex()
        _check(self.writer != old_writer, f"{self.name}: stale joint-state writer")
        epoch = self.epoch
        # A new volatile DDS reader, matched only after the old writer is gone.
        self.sub = self.node.create_subscription(
            JointState,
            _STATE,
            lambda msg: self.receive(msg, epoch),
            10,
        )
        self.wait(lambda: self.sub.get_publisher_count() == 1, "joint-state reader match")
        self.action = ActionClient(self.node, FollowJointTrajectory, _ACTION)
        self.wait(self.action.server_is_ready, "trajectory action server")
        initial_boundary = self.advance(20)
        sample = self.boundary_sample(initial_boundary)
        _check(
            all(abs(v) < 1e-8 for v in sample["position"] + sample["velocity"]),
            f"{self.name}: nonzero reset state",
        )
        self.event("ready", boundary=initial_boundary, writer=self.writer, sample=sample)
        return {"boundary": initial_boundary, "writer": self.writer, "sample": sample}

    def receive(self, msg, epoch):
        if epoch != self.epoch:
            return
        _check(
            set(msg.name) == set(_JOINTS)
            and len(msg.name) == len(msg.position) == len(msg.velocity) == 2,
            f"{self.name}: unexpected joint sample layout",
        )
        position = [msg.position[msg.name.index(joint)] for joint in _JOINTS]
        velocity = [msg.velocity[msg.name.index(joint)] for joint in _JOINTS]
        _check(
            all(math.isfinite(v) for v in position + velocity),
            f"{self.name}: nonfinite joint sample",
        )
        self.samples.append(
            {
                "epoch": epoch,
                "stamp_ns": _ns(msg.header.stamp),
                "position": position,
                "velocity": velocity,
            }
        )

    def boundary_sample(self, boundary, *, after=0):
        stamp = boundary[1]
        self.wait(
            lambda: _matching(self.samples[after:], self.epoch, stamp) is not None,
            f"fresh joint sample at {stamp} (recent {self.samples[-3:]})",
        )
        _check(
            all(s["stamp_ns"] <= stamp for s in self.samples[after:]),
            f"{self.name}: joint stamp overshot",
        )
        return _matching(self.samples[after:], self.epoch, stamp)

    def act(self, target, initial):
        before = self.mark()
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(_JOINTS)
        goal.trajectory.points = [
            JointTrajectoryPoint(positions=target, time_from_start=Duration(nanosec=25_000_000))
        ]
        future = self.action.send_goal_async(goal)
        self.wait(future.done, "goal response")
        handle = future.result()
        _check(handle.accepted, f"{self.name}: controller did not accept goal {target}")
        self.wait(
            lambda: handle.status == GoalStatus.STATUS_EXECUTING, "accepted goal EXECUTING status"
        )
        paused = self.witness(self.iteration, before)
        _check(
            all(s == paused for s in self.stats[before[0] :]),
            f"{self.name}: advanced before controller acceptance",
        )
        goal_id = bytes(handle.goal_id.uuid).hex()
        self.event("goal_executing_while_paused", goal_id=goal_id, boundary=paused, target=target)
        result = handle.get_result_async()
        cursor = len(self.samples)
        boundary = self.advance(_CONTROL_TICKS)
        self.wait(result.done, "trajectory goal result")
        outcome = result.result()
        _check(
            outcome.status == GoalStatus.STATUS_SUCCEEDED
            and outcome.result.error_code == FollowJointTrajectory.Result.SUCCESSFUL,
            f"{self.name}: trajectory did not succeed: {outcome}",
        )
        action_result = {"status": outcome.status, "error_code": outcome.result.error_code}
        sample = self.boundary_sample(boundary, after=cursor)
        _check(
            all(
                (p - q) * math.copysign(1.0, t) > 0.1 and abs(p) < 1.0
                for p, q, t in zip(sample["position"], initial["position"], target, strict=True)
            ),
            f"{self.name}: incorrect motion for {target}: {sample}",
        )
        self.event(
            "joint_boundary",
            goal_id=goal_id,
            result=action_result,
            boundary=boundary,
            sample=sample,
        )
        return {
            "goal_id": goal_id,
            "result": action_result,
            "accepted_paused": paused,
            "boundary": boundary,
            "sample": sample,
        }

    def reset_joints(self, old_sample):
        old_writer = self.writer
        self.disarm()
        self.epoch += 1
        self.samples.clear()
        reset = self.reset()
        self.iteration = 0
        restarted = self.initialize(old_writer)
        fresh = restarted["sample"]
        _check(
            old_sample["stamp_ns"] == fresh["stamp_ns"],
            f"{self.name}: initialization interval changed",
        )
        _check(
            _matching([old_sample], self.epoch, fresh["stamp_ns"]) is None
            and _matching([old_sample, fresh], self.epoch, fresh["stamp_ns"]) == fresh,
            f"{self.name}: stale joint sample passed the epoch barrier",
        )
        return {"epoch_boundary": reset, "stale_same_stamp_excluded": True, **restarted}

    def joint_isolation(self, peer):
        timeout = self.isolation(peer)
        peer_action = ActionClient(self.node, FollowJointTrajectory, f"/{peer.name}/{_ACTION}")
        try:
            _check(not peer_action.server_is_ready(), f"{self.name}: peer trajectory action leaked")
            _check(
                len(self.publishers()) == 1
                and not self.node.get_publishers_info_by_topic(f"/{peer.name}/{_STATE}")
                and not self.node.get_publishers_info_by_topic("/joint_states"),
                f"{self.name}: joint-state namespace leaked",
            )
        finally:
            peer_action.destroy()
        return timeout


def _run(root):
    token = uuid.uuid4().hex[:10]
    trials = {}
    with ExitStack() as owner:
        worlds = [
            _JointWorld(owner, root, f"{side}_{token}", domain)
            for side, domain in (("left", 181), ("right", 182))
        ]
        for world in worlds:
            world.wait(world.control.service_is_ready, "world-control discovery")
            baseline = world.witness(0, (0, 0))
            trials[world.name] = {
                "domain": world.domain,
                "partition": world.partition,
                "baseline": baseline,
                "initial": world.initialize(),
            }
        for sign, (world, peer) in zip((1, -1), (worlds, worlds[::-1]), strict=True):
            peer_mark = peer.mark()
            peer_state = peer.stats[-1]
            peer_before = peer.boundary_sample(peer_state)
            peer_cursor = len(peer.samples)
            target = [sign * 0.25, -sign * 0.25]
            entry = trials[world.name]
            entry["peer_timeout_s"] = world.joint_isolation(peer)
            initial = entry["initial"]["sample"]
            entry["first"] = world.act(target, initial)
            entry["reset"] = world.reset_joints(initial)
            entry["replay"] = world.act(target, entry["reset"]["sample"])
            entry["target"] = target
            entry["peer_unchanged"] = peer.witness(peer_state[0], peer_mark)
            _check(
                all(s == peer_state for s in peer.stats[peer_mark[0] :]),
                f"{peer.name}: moved during peer action",
            )
            entry["peer_sample"] = peer.boundary_sample(peer_state, after=peer_cursor)
            _check(entry["peer_sample"] == peer_before, f"{peer.name}: peer joint sample changed")
    return {
        "seed": _SEED,
        "step_ns": _STEP_NS,
        "control_ticks": _CONTROL_TICKS,
        "worlds": trials,
        "owned_processes_reaped": True,
    }


if __name__ == "__main__":
    _main(_run, __doc__)
