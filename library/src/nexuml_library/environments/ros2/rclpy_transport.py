"""Production ROS transport with an owned context/executor and bounded waits."""

import math
from time import monotonic
from typing import Any
from uuid import uuid4


class RclpyTransport:
    """JointState subscription, JointTrajectory publication and Empty reset service."""

    def __init__(self, definition, build_context):
        try:
            import rclpy  # ty: ignore[unresolved-import]
            from rclpy.context import Context  # ty: ignore[unresolved-import]
            from rclpy.duration import Duration  # ty: ignore[unresolved-import]
            from rclpy.executors import SingleThreadedExecutor  # ty: ignore[unresolved-import]
            from rclpy.parameter import Parameter  # ty: ignore[unresolved-import]
            from rclpy.qos import qos_profile_sensor_data  # ty: ignore[unresolved-import]
            from sensor_msgs.msg import JointState  # ty: ignore[unresolved-import]
            from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # ty: ignore[unresolved-import]
            from std_srvs.srv import Empty  # ty: ignore[unresolved-import]
        except ImportError as error:
            raise ImportError(
                "ROS joint control requires a sourced ROS 2 installation with rclpy, sensor_msgs, "
                "trajectory_msgs and std_srvs, plus a running simulation/controller/reset service"
            ) from error
        self.definition = definition
        self.rclpy, self.Duration = rclpy, Duration
        self.JointTrajectory, self.JointTrajectoryPoint, self.Empty = (
            JointTrajectory,
            JointTrajectoryPoint,
            Empty,
        )
        self.context = Context()
        self.node: Any = None
        self.executor: Any = None
        self.closed = False
        self.state = None
        self._version = self._minimum_version = 0
        self._ready_at = 0.0
        self._ready_stamp = self._state_stamp = 0
        self._state_received_at = 0.0
        try:
            rclpy.init(context=self.context)
            self.node = rclpy.create_node(
                f"nexuml_joint_reach_{build_context.worker_rank}_{uuid4().hex[:8]}",
                namespace=definition.namespace,
                context=self.context,
                parameter_overrides=[Parameter("use_sim_time", value=definition.use_sim_time)],
            )
            self.executor = SingleThreadedExecutor(context=self.context)
            self.executor.add_node(self.node)
            self.publisher = self.node.create_publisher(
                JointTrajectory, definition.command_topic, 10
            )
            self.subscription = self.node.create_subscription(
                JointState,
                definition.joint_state_topic,
                self._on_joint_state,
                qos_profile_sensor_data,
            )
            self.reset_client = self.node.create_client(Empty, definition.reset_service)
        except BaseException:
            self.close()
            raise

    def _on_joint_state(self, message):
        indices = {name: index for index, name in enumerate(message.name)}
        if len(indices) != len(message.name) or any(
            name not in indices for name in self.definition.joint_names
        ):
            return
        positions, velocities = [], []
        for name in self.definition.joint_names:
            index = indices[name]
            if index >= len(message.position) or index >= len(message.velocity):
                return
            positions.append(float(message.position[index]))
            velocities.append(float(message.velocity[index]))
        if not all(math.isfinite(value) for value in (*positions, *velocities)):
            return
        self.state = positions, velocities
        self._state_stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        self._state_received_at = monotonic()
        self._version += 1

    def receive_joint_state(self, timeout_s):
        deadline = monotonic() + timeout_s
        while not self.closed and self.context.ok():
            now = monotonic()
            if now >= deadline:
                raise TimeoutError(f"No fresh joint state on {self.definition.joint_state_topic}")
            if (
                self.state is not None
                and self._version > self._minimum_version
                and self._state_received_at >= self._ready_at
                and self._state_stamp >= self._ready_stamp
            ):
                self._minimum_version = self._version
                return self.state
            self.executor.spin_once(timeout_sec=min(0.01, deadline - now))
        raise RuntimeError("ROS transport is closed or its context was shut down")

    def send_joint_command(self, positions, control_period_s):
        if self.closed:
            raise RuntimeError("ROS transport is closed")
        if len(positions) != len(self.definition.joint_names) or any(
            not math.isfinite(value) or not low <= value <= high
            for value, low, high in zip(
                positions, self.definition.joint_low, self.definition.joint_high
            )
        ):
            raise ValueError("Joint command violates configured shape or limits")
        message = self.JointTrajectory()
        message.joint_names = list(self.definition.joint_names)
        point = self.JointTrajectoryPoint()
        point.positions = list(positions)
        point.time_from_start = self.Duration(seconds=control_period_s).to_msg()
        message.points = [point]
        self._minimum_version = self._version
        self._ready_at = monotonic() + control_period_s
        self._ready_stamp = self.node.get_clock().now().nanoseconds + int(control_period_s * 1e9)
        self.publisher.publish(message)

    def reset(self, timeout_s):
        if self.closed:
            raise RuntimeError("ROS transport is closed")
        deadline = monotonic() + timeout_s
        if not self.reset_client.wait_for_service(timeout_sec=timeout_s):
            raise TimeoutError(
                f"Simulation reset service unavailable: {self.definition.reset_service}"
            )
        future = self.reset_client.call_async(self.Empty.Request())
        self.executor.spin_until_future_complete(
            future, timeout_sec=max(0.0, deadline - monotonic())
        )
        if not future.done():
            future.cancel()
            raise TimeoutError("Simulation reset service timed out")
        future.result()
        self.state = None
        self._minimum_version = self._version
        self._ready_stamp = 0

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.executor is not None:
                self.executor.shutdown(timeout_sec=self.definition.timeout_s)
        finally:
            try:
                if self.node is not None:
                    self.node.destroy_node()
            finally:
                if self.context.ok():
                    self.rclpy.shutdown(context=self.context)
