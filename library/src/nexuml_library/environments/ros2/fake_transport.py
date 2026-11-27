"""Deterministic test transport, never selected as an execution fallback."""


class FakeRos2Transport:
    """Record commands/reset/close and supply joint samples without ROS middleware."""

    def __init__(self, num_joints=2):
        self.position = [0.0] * num_joints
        self.velocity = [0.0] * num_joints
        self.commands = []
        self.resets = 0
        self.closed = False
        self.timeout = False

    def receive_joint_state(self, timeout_s):
        if self.closed:
            raise RuntimeError("Fake ROS transport is closed")
        if self.timeout:
            raise TimeoutError("Fake joint-state timeout")
        return list(self.position), list(self.velocity)

    def send_joint_command(self, positions, control_period_s):
        if self.closed:
            raise RuntimeError("Fake ROS transport is closed")
        self.commands.append((list(positions), control_period_s))
        self.position = list(positions)

    def reset(self, timeout_s):
        if self.timeout:
            raise TimeoutError("Fake reset timeout")
        self.resets += 1
        self.position = [0.0] * len(self.position)
        self.velocity = [0.0] * len(self.velocity)

    def close(self):
        self.closed = True
