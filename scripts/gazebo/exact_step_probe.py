"""Probe two empty paused Jazzy/Harmonic worlds in an isolated Linux sandbox."""

import argparse
import json
import os
import signal
import subprocess
import time
import traceback
import uuid
from contextlib import ExitStack
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_prefix
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from ros_gz_interfaces.msg import WorldStatistics
from ros_gz_interfaces.srv import ControlWorld
from rosgraph_msgs.msg import Clock

_STEP_NS = 1_000_000
_SEED = 812


def _check(condition, message):
    if not condition:
        raise RuntimeError(message)


def _ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def _stop(process, log):
    for sig in (signal.SIGINT, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=4)
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            log.write(f"\nprobe: reaped owned process group {process.pid} ({process.returncode})\n")
            return
        except subprocess.TimeoutExpired:
            pass
    raise RuntimeError(f"Owned process group {process.pid} survived cleanup")


class _World:
    def __init__(self, owner, root, name, domain, prepare_model=None):
        self.name = name
        self.domain = domain
        self.partition = f"nexuml_probe_{name}"
        self.path = f"/world/{name}"
        self.stats = []
        self.clock = []
        self.processes = []
        work = root / name
        work.mkdir()
        self.env = {**os.environ, "GZ_PARTITION": self.partition, "ROS_DOMAIN_ID": str(domain)}
        model = prepare_model(work, name) if prepare_model else ""
        sdf = work / "world.sdf"
        sdf.write_text(
            f"""<sdf version="1.9"><world name="{name}">
              <physics name="exact" type="ignored"><max_step_size>0.001</max_step_size>
                <real_time_factor>1</real_time_factor></physics>
              <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
              {model}
            </world></sdf>""",
            encoding="utf-8",
        )
        self.start(
            owner, work, "server", ["gz", "sim", "-v", "4", "-s", "--seed", str(_SEED), str(sdf)]
        )
        self.bridge = (
            Path(get_package_prefix("ros_gz_bridge")) / "lib/ros_gz_bridge/parameter_bridge"
        )
        self.start(
            owner,
            work,
            "bridge",
            [
                str(self.bridge),
                f"{self.path}/control@ros_gz_interfaces/srv/ControlWorld",
                f"{self.path}/stats@ros_gz_interfaces/msg/WorldStatistics[gz.msgs.WorldStatistics",
                f"{self.path}/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
                "--ros-args",
                "-r",
                f"{self.path}/clock:=/clock",
                "-r",
                f"__ns:=/{name}",
            ],
        )
        context = Context()
        rclpy.init(context=context, domain_id=domain)
        owner.callback(context.try_shutdown)
        self.node = Node("exact_step_probe", namespace=f"/{name}", context=context)
        owner.callback(self.node.destroy_node)
        self.executor = SingleThreadedExecutor(context=context)
        self.executor.add_node(self.node)
        owner.callback(self.executor.shutdown)
        self.control = self.node.create_client(ControlWorld, f"{self.path}/control")
        self.node.create_subscription(
            WorldStatistics,
            f"{self.path}/stats",
            lambda msg: self.stats.append((msg.iterations, _ns(msg.sim_time), msg.paused)),
            10,
        )
        self.node.create_subscription(
            Clock,
            "/clock",
            lambda msg: self.clock.append(_ns(msg.clock)),
            qos_profile_sensor_data,
        )

    def start(self, owner, work, label, argv):
        log = owner.enter_context((work / f"{label}.log").open("w", encoding="utf-8"))
        process = subprocess.Popen(
            argv, env=self.env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        owner.callback(_stop, process, log)
        self.processes.append(process)
        return process

    def wait(self, condition, what, timeout=8.0):
        deadline = time.monotonic() + timeout
        while not condition():
            _check(all(p.poll() is None for p in self.processes), f"{self.name}: exited: {what}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"{self.name}: {what}; stats={self.stats[-5:]}, clock={self.clock[-5:]}"
                )
            self.executor.spin_once(timeout_sec=min(0.05, remaining))

    def witness(self, iteration, after):
        boundary = (iteration, iteration * _STEP_NS, True)
        self.wait(
            lambda: (
                self.stats[after[0] :].count(boundary) >= 2
                and boundary[1] in self.clock[after[1] :]
            ),
            f"boundary {boundary}",
        )
        _check(self.stats[-1] == boundary, f"{self.name}: left paused boundary {boundary}")
        _check(
            all(i <= iteration and t == i * _STEP_NS for i, t, _ in self.stats[after[0] :]),
            f"{self.name}: extra or inconsistent ticks at {boundary}",
        )
        _check(max(self.clock[after[1] :]) == boundary[1], f"{self.name}: clock overshot")
        return boundary

    def mark(self):
        return len(self.stats), len(self.clock)

    def request(self, *, steps=0, reset=False):
        request = ControlWorld.Request()
        request.world_control.pause = True
        request.world_control.multi_step = steps
        request.world_control.reset.all = reset
        future = self.control.call_async(request)
        self.wait(future.done, "world-control response")
        _check(future.result().success, f"{self.name}: world-control rejected")

    def reset(self):
        mark = self.mark()
        self.request(reset=True)
        epoch = (0, 0, True)
        self.wait(lambda: epoch in self.stats[mark[0] :], "reset epoch")
        # Wait for post-reset clock and stats, not cached samples from the old epoch.
        return self.witness(0, (self.stats.index(epoch, mark[0]), len(self.clock)))

    def isolation(self, peer):
        services = subprocess.check_output(
            ["gz", "service", "-l"], env=self.env, text=True, timeout=8
        ).splitlines()
        own_control, peer_control = f"{self.path}/control", f"{peer.path}/control"
        _check(own_control in services and peer_control not in services, f"{self.name}: GZ leak")
        ros_services = dict(self.node.get_service_names_and_types())
        nodes = self.node.get_node_names_and_namespaces()
        clocks = self.node.get_publishers_info_by_topic("/clock")
        _check(
            own_control in ros_services and peer_control not in ros_services,
            f"{self.name}: ROS leak",
        )
        _check(not any(ns == f"/{peer.name}" for _, ns in nodes), f"{self.name}: ROS node leak")
        _check(
            len(clocks) == 1 and clocks[0].node_namespace == f"/{self.name}",
            f"{self.name}: wrong clock owner",
        )
        client = self.node.create_client(ControlWorld, peer_control)
        try:
            started = time.monotonic()
            self.wait(client.service_is_ready, "peer endpoint (expected absent)", timeout=0.4)
        except TimeoutError:
            return round(time.monotonic() - started, 3)
        finally:
            self.node.destroy_client(client)
        raise RuntimeError(f"{self.name}: peer world-control endpoint was reachable")


def _run(root):
    token = uuid.uuid4().hex[:10]
    trials = {}
    with ExitStack() as owner:
        worlds = [
            _World(owner, root, f"{side}_{token}", domain)
            for side, domain in (
                ("left", 181),
                ("right", 182),
            )
        ]
        for world, peer in (worlds, worlds[::-1]):
            world.wait(world.control.service_is_ready, "world-control discovery")
            baseline = world.witness(0, (0, 0))
            trials[world.name] = {
                "domain": world.domain,
                "partition": world.partition,
                "baseline": baseline,
                "peer_service_timeout_s": world.isolation(peer),
            }
        for world, peer in (worlds, worlds[::-1]):
            peer_before = peer.stats[-1]
            peer_mark = peer.mark()
            boundaries = {}
            for label, steps, target in (("first", 25, 25), ("second", 40, 65)):
                mark = world.mark()
                world.request(steps=steps)
                boundaries[label] = world.witness(target, mark)
            boundaries["reset"] = world.reset()
            mark = world.mark()
            world.request(steps=25)
            boundaries["replay"] = world.witness(25, mark)
            boundaries["peer_unchanged"] = peer.witness(peer_before[0], peer_mark)
            _check(
                all(s == peer_before for s in peer.stats[peer_mark[0] :]), f"{peer.name}: not idle"
            )
            trials[world.name].update(boundaries)
    return {"seed": _SEED, "step_ns": _STEP_NS, "worlds": trials, "owned_processes_reaped": True}


def _main(run=_run, description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "receipt_dir", type=Path, help="new directory for world/bridge logs and receipt"
    )
    root = parser.parse_args().receipt_dir.resolve()
    root.mkdir()
    os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    try:
        receipt = run(root)
    except (Exception, KeyboardInterrupt):
        (root / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    _main()
