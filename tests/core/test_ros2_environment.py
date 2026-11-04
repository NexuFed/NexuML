"""The production task and message boundary without a ROS installation."""

from types import SimpleNamespace
import json
import os
import signal
import subprocess

import pytest
import torch
from pydantic import ValidationError
from tensordict import TensorDict

from nexuml.core.compiler import compile
from nexuml.core.components import EnvironmentBuildContext
from nexuml.core.config import ResolvedConfig
from nexuml_library.environments.ros2.joint_reach import (
    Ros2JointReachEnvironment,
    _JointReachRuntime,
)
from nexuml_library.environments.ros2.fake_transport import FakeRos2Transport
from nexuml_library.environments.ros2.rclpy_transport import RclpyTransport
from nexuml_library.scenarios.reinforcement.ros2 import ros2_joint_reach_ppo


def action(values):
    return TensorDict({"joint_command": torch.tensor(values, dtype=torch.float32)}, batch_size=[])


def test_fake_transport_uses_the_real_task_and_enforces_boundaries():
    definition = Ros2JointReachEnvironment(max_episode_steps=2)
    transport = FakeRos2Transport()
    runtime = _JointReachRuntime(definition, EnvironmentBuildContext(purpose="train"), transport)
    assert runtime.reset()["joint_position"].tolist() == [0.0, 0.0]
    assert transport.resets == 1 and transport.commands[-1][1] == definition.control_period_s
    with pytest.raises(ValueError, match="upper bound"):
        runtime.step(action([10.0, 0.0]))
    transition = runtime.step(action([0.0, 0.0]))
    assert transition.reward.item() < 0 and not transition.terminated.item()
    transition = runtime.step(action([0.0, 0.0]))
    assert transition.truncated.item() and not transition.terminated.item()
    runtime.reset()
    transition = runtime.step(action([0.5, -0.5]))
    assert transition.terminated.item() and not transition.truncated.item()
    assert transition.reward.item() == 0
    runtime.reset()
    transport.timeout = True
    with pytest.raises(TimeoutError):
        runtime.step(action([0.0, 0.0]))
    with pytest.raises(RuntimeError, match="Reset"):
        runtime.step(action([0.0, 0.0]))
    with pytest.raises(TimeoutError):
        runtime.reset()
    runtime.close()
    runtime.close()
    assert transport.closed


def test_joint_configuration_and_simulator_free_round_trip():
    with pytest.raises(ValidationError, match="match joint_names"):
        Ros2JointReachEnvironment(joint_names=("single",))
    with pytest.raises(ValidationError, match="within ordered"):
        Ros2JointReachEnvironment(target_positions=(2.0, 0.0))
    spec = ros2_joint_reach_ppo()
    restored = ResolvedConfig.from_yaml(ResolvedConfig.from_scenario(spec).to_yaml()).to_scenario()
    pipeline = compile(restored)
    assert pipeline.input_sizes["joint_position"] == (2,)
    assert restored.interaction is not None
    with pytest.raises(ValueError, match="one robot"):
        restored.interaction.environment.build(EnvironmentBuildContext(purpose="train", num_envs=2))


def test_production_callback_reorders_and_rejects_invalid_joint_samples():
    transport = RclpyTransport.__new__(RclpyTransport)
    transport.definition = Ros2JointReachEnvironment()
    transport.state, transport._version = None, 0
    transport._on_joint_state(
        SimpleNamespace(
            name=["joint2", "joint1"],
            position=[-0.5, 0.5],
            velocity=[2.0, 1.0],
            header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0)),
        )
    )
    assert transport.state == ([0.5, -0.5], [1.0, 2.0])
    version = transport._version
    for message in (
        SimpleNamespace(name=["joint1"], position=[0.0], velocity=[0.0]),
        SimpleNamespace(
            name=["joint1", "joint2"], position=[float("nan"), 0.0], velocity=[0.0, 0.0]
        ),
        SimpleNamespace(name=["joint1", "joint2"], position=[0.0, 0.0], velocity=[]),
    ):
        transport._on_joint_state(message)
    assert transport._version == version


def test_production_transport_freshness_commands_reset_and_owned_shutdown(monkeypatch):
    from nexuml_library.environments.ros2 import rclpy_transport as module

    now = [10.0]
    monkeypatch.setattr(module, "monotonic", lambda: now[0])
    transport = RclpyTransport.__new__(RclpyTransport)
    transport.definition = Ros2JointReachEnvironment()
    transport.closed = False
    transport.state = None
    transport._version = transport._minimum_version = 0
    transport._ready_at = transport._state_received_at = 0.0
    transport._ready_stamp = transport._state_stamp = 0
    events, commands, samples = [], [], []
    transport.context = SimpleNamespace(ok=lambda: True)
    transport.node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1_000_000_000)),
        destroy_node=lambda: events.append("node"),
    )
    transport.publisher = SimpleNamespace(publish=commands.append)
    transport.JointTrajectory = transport.JointTrajectoryPoint = SimpleNamespace
    transport.Duration = lambda seconds: SimpleNamespace(to_msg=lambda: seconds)
    transport.Empty = SimpleNamespace(Request=lambda: object())
    transport.rclpy = SimpleNamespace(shutdown=lambda context: events.append("context"))

    def spin_once(timeout_sec):
        now[0] += timeout_sec
        if samples:
            transport._on_joint_state(samples.pop(0))

    transport.executor = SimpleNamespace(
        spin_once=spin_once,
        shutdown=lambda timeout_sec: events.append("executor"),
        spin_until_future_complete=lambda future, timeout_sec: None,
    )
    transport.send_joint_command([0.1, -0.1], 0.05)
    assert commands[0].joint_names == ["joint1", "joint2"]
    assert commands[0].points[0].positions == [0.1, -0.1]
    assert commands[0].points[0].time_from_start == 0.05

    def sample(stamp):
        return SimpleNamespace(
            name=["joint1", "joint2"],
            position=[0.1, -0.1],
            velocity=[0.0, 0.0],
            header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=stamp)),
        )

    # A pre-period sample cannot become fresh merely by waiting.
    transport._on_joint_state(sample(60_000_000))
    with pytest.raises(TimeoutError):
        transport.receive_joint_state(0.1)
    # Queued pre-command samples arriving late are rejected by their ROS timestamps.
    samples.append(sample(0))
    with pytest.raises(TimeoutError):
        transport.receive_joint_state(0.1)
    samples.append(sample(60_000_000))
    assert transport.receive_joint_state(0.1) == ([0.1, -0.1], [0.0, 0.0])
    with pytest.raises(ValueError, match="limits"):
        transport.send_joint_command([2.0, 0.0], 0.05)

    future = SimpleNamespace(done=lambda: True, result=lambda: object())
    transport.reset_client = SimpleNamespace(
        wait_for_service=lambda timeout_sec: True,
        call_async=lambda request: future,
    )
    transport.reset(0.1)
    assert transport.state is None
    future.done = lambda: False
    future.cancel = lambda: events.append("cancel")
    with pytest.raises(TimeoutError, match="reset service timed out"):
        transport.reset(0.1)
    assert "cancel" in events
    transport.close()
    transport.close()
    assert events[-3:] == ["executor", "node", "context"]


def test_missing_ros_is_not_replaced_by_fake_transport(monkeypatch):
    import builtins

    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "rclpy" or name.startswith("rclpy."):
            raise ImportError("ROS blocked for test")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match="sourced ROS 2"):
        Ros2JointReachEnvironment().build(EnvironmentBuildContext(purpose="eval"))


@pytest.fixture
def headless_ros_gazebo(tmp_path):
    """Launch an operator-provided simulation argv; never install or target hardware."""
    command = os.environ.get("NEXUML_ROS_GAZEBO_LAUNCH")
    if command is None:
        pytest.fail(
            "Set NEXUML_ROS_GAZEBO_LAUNCH to a JSON argv for the headless simulation launch"
        )
    argv = json.loads(command)
    if not isinstance(argv, list) or not argv or any(not isinstance(value, str) for value in argv):
        pytest.fail("NEXUML_ROS_GAZEBO_LAUNCH must be a nonempty JSON array of strings")
    with (tmp_path / "gazebo.log").open("w") as log:
        process = subprocess.Popen(
            argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            yield process
        finally:
            # Only this fixture's process group is stopped, never a shared ROS/Gazebo runtime.
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)


@pytest.mark.requires_system
@pytest.mark.requires_optional("rclpy")
@pytest.mark.requires_optional("torchrl")
def test_headless_gazebo_production_transport_controls_robot(headless_ros_gazebo, tmp_path):
    from nexuml.core.types import ReinforcementLearningSpec, RLEvaluationSpec
    from nexuml.training.lightning import NexuSession
    from nexuml_library.reinforcement.algorithms.ppo import PPO

    definition = Ros2JointReachEnvironment(timeout_s=30.0)
    runtime = definition.build(EnvironmentBuildContext(purpose="eval"))
    try:
        initial = runtime.reset()["joint_position"].clone()
        for _ in range(20):
            transition = runtime.step(action([0.5, -0.5]))
            if transition.terminated.item():
                break
        assert not torch.allclose(initial, transition.observation["joint_position"], atol=0.1)
        assert transition.terminated.item(), "Simulated joints did not reach the published target"
    finally:
        runtime.close()
    assert headless_ros_gazebo.poll() is None
    spec = ros2_joint_reach_ppo(total_frames=16, frames_per_batch=16)
    assert spec.interaction is not None and isinstance(spec.training, ReinforcementLearningSpec)
    spec.interaction.environment = definition
    spec.training.algorithm = PPO(update_epochs=1, minibatch_size=8)
    spec.training.evaluation = RLEvaluationSpec(episodes=1, max_steps_per_episode=2)
    result = NexuSession(
        spec, log_dir=tmp_path, enable_loggers=False, enable_progress_bar=False
    ).run()
    assert result.policy is not None and result.interaction_results
    assert result.trainer.global_step > 0
