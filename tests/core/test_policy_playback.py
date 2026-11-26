"""Headless playback uses live policy inputs, selected resets and no learner."""

from threading import Event
import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from typer.testing import CliRunner

from nexuml.cli.main import app
from nexuml.core.compiler import compile
from nexuml.core.components import EnvironmentDefinition
from nexuml.core.config import ResolvedConfig
from nexuml.core.export import export_package
from nexuml.core.policy import CompiledPolicy
from nexuml.core.types import RLVideoSpec, SensorProcessingContract
from nexuml.interaction.runtime import InteractionStep
from nexuml.interaction.playback import PlaybackResult, play_policy
from nexuml.interaction.playback_controls import PlaybackControls
from nexuml_library.scenarios.reinforcement.gymnasium import cartpole_ppo
from nexuml_library.environments.gymnasium import _GymnasiumRuntime
from nexuml_library.layers.processing.sensor_history import SensorHistoryProcessor


@pytest.fixture
def playback():
    scenario = cartpole_ppo()
    assert scenario.interaction is not None
    declared = scenario.interaction.environment.describe()

    class ScriptedBatch:
        closed = False
        calls = 0
        masks = []
        raw = False
        contract = declared.model_copy(update={"num_envs": 3})

        def __init__(self):
            self.rows = torch.zeros(3, 4)

        def reset(self, *, seed=None, mask=None):
            if mask is None:
                self.rows.zero_()
                assert seed == (None if self.calls else 23)
            else:
                self.masks.append(mask.tolist())
                self.rows[mask] = 0
            return self.observe()

        def observe(self):
            return TensorDict({"observation": self.rows[:, : 2 if self.raw else 4].clone()}, [3])

        def step(self, action):
            assert action.batch_size == (3,)
            self.calls += 1
            self.rows.add_(1)
            terminal = torch.tensor([[False], [self.calls == 1], [False]])
            truncated = torch.tensor([[False], [False], [self.calls == 2]])
            return InteractionStep(self.observe(), None, terminal, truncated)

        def close(self):
            self.closed = True

    runtime = ScriptedBatch()
    contexts = []

    class Definition(EnvironmentDefinition):
        def describe(self):
            return declared

        def validate_playback(self, viewer):
            assert viewer == "headless"

        def build(self, context):
            contexts.append(context)
            return runtime

    policy = CompiledPolicy.from_pipeline(compile(scenario))
    return policy, Definition(), runtime, contexts


def test_playback_resets_only_completed_rows_and_leaves_weights_unchanged(playback):
    policy, definition, runtime, contexts = playback
    weights = {key: tensor.clone() for key, tensor in policy.state_dict().items()}
    observations = []
    policy.register_forward_pre_hook(lambda _, args: observations.append(args[0].clone()))
    actual = play_policy(policy, definition, num_envs=3, steps=3, seed=23)
    assert actual == PlaybackResult(3, 9, 2, "step_limit")
    assert contexts[0].purpose == "play" and contexts[0].num_envs == 3
    assert runtime.closed and runtime.masks == [[False, True, False], [False, False, True]]
    assert [td["observation"][:, 0].tolist() for td in observations] == [
        [0, 0, 0],
        [1, 0, 1],
        [2, 1, 0],
    ]
    assert policy.training
    torch.testing.assert_close(policy.state_dict(), weights)


def test_native_viewer_close_stops_without_another_action_or_reset(playback):
    policy, definition, runtime, _ = playback
    runtime.playback_is_running = lambda: runtime.calls < 1
    actual = play_policy(policy, definition, num_envs=3, seed=23)
    assert actual == PlaybackResult(1, 3, 1, "stop_requested")
    assert runtime.closed and not runtime.masks


def test_playback_package_history_tracks_only_live_and_reset_rows(playback, monkeypatch):
    policy, definition, runtime, _ = playback
    declared = definition.describe()
    field = declared.observations["observation"]
    field = field.model_copy(
        update={
            "sensor": SensorProcessingContract(
                owner="package",
                raw_shape=(2,),
                raw_dtype="float32",
                raw_layout="vector",
                units="test_units",
                history_length=2,
            )
        }
    )
    declared = declared.model_copy(
        update={
            "observations": {"observation": field},
            "control_period_s": 0.001,
        }
    )
    monkeypatch.setattr(type(definition), "describe", lambda self: declared)
    policy.contract = declared
    config = policy.pipeline.resolved_config.policy
    assert config is not None
    policy.processor = policy._build_processor(
        config.model_copy(update={"processor": SensorHistoryProcessor()}), declared
    )
    runtime.contract = declared.model_copy(update={"num_envs": 3})
    runtime.raw = True
    controls = PlaybackControls(3, selected_reset=True)
    runtime.playback_controls = controls
    processed = []

    def inspect(_, args):
        processed.append(args[0]["observation"][:, ::2].tolist())
        if len(processed) == 3:
            assert controls.submit("reset", 1)
        if len(processed) == 4:
            assert controls.submit("reset")

    policy.register_forward_pre_hook(inspect)
    result = play_policy(policy, definition, num_envs=3, steps=5, seed=23)
    assert processed == [
        [[0, 0], [0, 0], [0, 0]],
        [[0, 1], [0, 0], [0, 1]],
        [[1, 2], [0, 1], [0, 0]],
        [[2, 3], [0, 0], [0, 1]],
        [[0, 0], [0, 0], [0, 0]],
    ]
    assert result == PlaybackResult(5, 15, 2, "step_limit", {"reset": 2})
    assert runtime.closed


def test_playback_keeps_recurrent_state_and_chunk_cursor_with_their_lane(playback):
    policy, definition, runtime, _ = playback
    incoming = []

    class RecurrentChunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.recurrent = torch.nn.GRUCell(4, 1)
            self.chunk = torch.nn.Linear(1, 2)

        def initialize_state(self, batch, *, device):
            return TensorDict(
                {
                    "hidden": torch.zeros(*batch, 1, device=device),
                    "chunk": torch.zeros(*batch, 2, device=device),
                    "cursor": torch.zeros(batch, dtype=torch.long, device=device),
                },
                batch,
                device=device,
            )

        def reset_state(self, state, mask):
            result = state.clone()
            result[mask] = self.initialize_state((int(mask.sum()),), device=state.device)
            return result

        def step(self, observation, state):
            incoming.append(state.clone())
            hidden = self.recurrent(observation["observation"], state["hidden"])
            cursor = state["cursor"].unsqueeze(-1)
            chunk = torch.where(cursor == 0, self.chunk(hidden), state["chunk"])
            command = chunk.gather(-1, cursor)
            return observation.clone().set("observation", command.expand(-1, 4)), TensorDict(
                {"hidden": hidden, "chunk": chunk, "cursor": (cursor.squeeze(-1) + 1) % 2},
                state.batch_size,
                device=state.device,
            )

    policy.processor = RecurrentChunk()
    weights = {key: value.clone() for key, value in policy.state_dict().items()}
    assert play_policy(policy, definition, num_envs=3, steps=3, seed=23).completed_episodes == 2
    assert [state["cursor"].tolist() for state in incoming] == [[0, 0, 0], [1, 0, 1], [0, 1, 0]]
    for state, lane in zip(incoming[1:], (1, 2), strict=True):
        assert state["hidden"][lane].item() == 0 and not state["chunk"][lane].any()
        assert state["hidden"][0].item() != 0
    assert runtime.closed
    torch.testing.assert_close(policy.state_dict(), weights)


def test_playback_stop_event_closes_and_contract_rejects_before_build(playback):
    policy, definition, runtime, contexts = playback
    requested = Event()
    policy.register_forward_pre_hook(lambda _, args: requested.set())
    with pytest.raises(ValueError, match="native control period"):
        play_policy(policy, definition, num_envs=3, real_time=True)
    assert not contexts
    assert play_policy(policy, definition, num_envs=3, seed=23, stop=requested) == PlaybackResult(
        1, 3, 1, "stop_requested"
    )
    assert runtime.closed
    policy.contract = policy.contract.model_copy(update={"control_period_s": 0.1})
    with pytest.raises(ValueError, match="control period does not match"):
        play_policy(policy, definition, num_envs=3, seed=23)
    assert len(contexts) == 1


def test_playback_cli_loads_policy_and_rejects_unqualified_viewer(tmp_path):
    policy = CompiledPolicy.from_pipeline(compile(cartpole_ppo()))
    package = export_package(policy, tmp_path / "policy")
    runner = CliRunner()
    result = runner.invoke(app, ["play", str(package), "--num-envs", "3", "--steps", "5"])
    assert result.exit_code == 0, result.output
    assert "5 steps, 15 transitions" in result.output
    unsupported = runner.invoke(app, ["play", str(package), "--viewer", "webrtc"])
    assert unsupported.exit_code == 1 and "only supports headless" in unsupported.output


def test_playback_viewer_uses_explicit_scenario_and_cli_override_not_artifact(tmp_path):
    scenario = cartpole_ppo()
    scenario.playback_viewer = "webrtc"
    policy = CompiledPolicy.from_pipeline(compile(scenario))
    package = export_package(policy, tmp_path / "policy")
    config = tmp_path / "play.yaml"
    ResolvedConfig.from_scenario(scenario).save(config)
    runner = CliRunner()

    args = ["play", str(package), "--steps", "2"]
    assert "2 transitions" in runner.invoke(app, args).output
    explicit = [*args, "--config", str(config)]
    unsupported = runner.invoke(app, explicit)
    assert unsupported.exit_code == 1 and "only supports headless" in unsupported.output
    override = runner.invoke(app, [*explicit, "--viewer", "headless"])
    assert override.exit_code == 0 and "2 transitions" in override.output


def test_playback_cli_reaps_local_runtime_on_sigterm(tmp_path, monkeypatch):
    import signal

    from nexuml_library.environments.gymnasium import _GymnasiumRuntime

    policy = CompiledPolicy.from_pipeline(compile(cartpole_ppo()))
    package = export_package(policy, tmp_path / "policy")
    close = _GymnasiumRuntime.close
    closed = []
    former = signal.getsignal(signal.SIGTERM)

    def cancel(*_):
        assert signal.getsignal(signal.SIGTERM) != former
        signal.raise_signal(signal.SIGTERM)

    def record_close(runtime):
        close(runtime)
        closed.append(runtime)

    monkeypatch.setattr(_GymnasiumRuntime, "step", cancel)
    monkeypatch.setattr(_GymnasiumRuntime, "close", record_close)
    result = CliRunner().invoke(app, ["play", str(package), "--steps", "2"])
    assert result.exit_code == 143 and "Playback complete:" not in result.output
    assert len(closed) == 1
    assert signal.getsignal(signal.SIGTERM) == former


def test_playback_requires_simulation_opt_in(playback, monkeypatch):
    policy, definition, _, contexts = playback
    monkeypatch.setattr(
        type(definition), "validate_playback", EnvironmentDefinition.validate_playback
    )
    with pytest.raises(ValueError, match="does not support simulated playback"):
        play_policy(policy, definition, steps=1)
    assert not contexts


def test_control_inbox_is_bounded_and_rejects_unadmitted_input():
    controls = PlaybackControls(3, selected_reset=False)
    for value in (True, 1, -1, 3, 1.0):
        assert not controls.submit("reset", value)
    for value in (None, True, float("nan"), 0, 10):
        assert not controls.submit("speed", value)
    assert not controls.submit("pause", 0)
    for _ in range(16):
        assert controls.submit("pause")
    assert not controls.submit("resume")
    assert controls.receive() == ("pause", None)
    assert controls.submit("resume")
    assert not controls.submit("task_reset", ("Goal", "Left"))


def test_task_reset_option_is_checked_and_applied_before_the_next_action(playback, monkeypatch):
    policy, definition, runtime, _ = playback
    declared = definition.describe().model_copy(update={"control_period_s": 0.001})
    monkeypatch.setattr(type(definition), "describe", lambda self: declared)
    policy.contract = declared
    runtime.contract = declared.model_copy(update={"num_envs": 3})
    options = {"Goal": ("Left", "Right")}
    controls = PlaybackControls(3, selected_reset=True, task_options=options)
    options["Goal"] += ("Forged",)
    runtime.playback_controls = controls
    assert not controls.submit("task_reset", ("Goal", "Forged"))
    assert not controls.submit("task_reset", ("Other", "Right"))
    assert controls.submit("pause")
    assert controls.submit("task_reset", ("Goal", "Right"))
    assert controls.submit("step")

    def reset_option(name, choice):
        assert (name, choice) == ("Goal", "Right") and runtime.calls == 0
        runtime.rows.fill_(2)
        return runtime.observe()

    runtime.reset_with_task_option = reset_option
    observations = []
    policy.register_forward_pre_hook(lambda _, args: observations.append(args[0].clone()))
    result = play_policy(policy, definition, num_envs=3, steps=1, seed=23)
    assert observations[0]["observation"][:, 0].tolist() == [2, 2, 2]
    assert result.interventions == {"pause": 1, "task_reset:Goal:Right": 1, "step": 1}
    assert runtime.closed


def test_owner_applies_single_step_and_selected_reset_to_playback_only(playback, monkeypatch):
    policy, definition, runtime, _ = playback
    declared = definition.describe().model_copy(update={"control_period_s": 0.001})
    monkeypatch.setattr(type(definition), "describe", lambda self: declared)
    policy.contract = declared
    runtime.contract = declared.model_copy(update={"num_envs": 3})
    controls = PlaybackControls(3, selected_reset=True)
    runtime.playback_controls = controls
    assert controls.submit("pause")
    assert controls.submit("step")
    assert controls.submit("reset", 2)
    assert controls.submit("speed", 2.0)
    assert controls.submit("resume")
    inputs = []
    policy.register_forward_pre_hook(lambda _, args: inputs.append(args[0].clone()))
    result = play_policy(policy, definition, num_envs=3, steps=2, seed=23)
    assert result.steps == 2 and result.transitions == 6
    assert result.interventions == dict.fromkeys(("pause", "step", "reset", "speed", "resume"), 1)
    assert runtime.masks == [[False, True, False], [False, False, True]]
    assert inputs[1]["observation"][:, 0].tolist() == [1, 0, 0]
    assert runtime.closed


def test_stochastic_playback_is_seeded_and_leaves_caller_rng_unchanged():
    scenario = cartpole_ppo()
    assert scenario.interaction is not None
    definition = scenario.interaction.environment
    policy = CompiledPolicy.from_pipeline(compile(scenario))
    actions = []
    policy.register_forward_hook(lambda _, _args, result: actions.append(result["action"].clone()))
    state = torch.random.get_rng_state()
    for _ in range(2):
        play_policy(policy, definition, num_envs=3, steps=5, seed=73, deterministic=False)
        assert torch.equal(state, torch.random.get_rng_state())
    torch.testing.assert_close(torch.stack(actions[:5]), torch.stack(actions[5:]))


def test_playback_recording_flushes_at_capture_bound_while_playback_continues(monkeypatch):
    scenario = cartpole_ppo()
    assert scenario.interaction is not None
    policy = CompiledPolicy.from_pipeline(compile(scenario))
    frame_reads = []
    published = []
    actions = []
    policy.register_forward_pre_hook(lambda _, args: actions.append(args[0]))

    def capture(self):
        frame_reads.append(len(actions))
        return np.full((4, 8, 3), len(actions), dtype=np.uint8)

    monkeypatch.setattr(_GymnasiumRuntime, "render_frame", capture)
    result = play_policy(
        policy,
        scenario.interaction.environment,
        steps=6,
        video=RLVideoSpec(enabled=True, max_frames=2),
        media_sink=lambda frames, metadata: published.append((len(actions), frames, metadata)),
    )
    assert result.steps == 6 and frame_reads == [1, 2]
    ((at_tick, frames, info),) = published
    assert at_tick == 2 and len(frames) == 2 and info["native_ticks"] == [1, 2]
    assert info["end"] == "capture_limit"
    assert info["monotonic_capture_ns"][0] < info["monotonic_capture_ns"][1]


@pytest.mark.requires_optional("PIL")
def test_play_cli_writes_bounded_recording_to_new_directory(monkeypatch, tmp_path):
    policy = CompiledPolicy.from_pipeline(compile(cartpole_ppo()))
    package = export_package(policy, tmp_path / "policy")
    monkeypatch.setattr(
        _GymnasiumRuntime, "render_frame", lambda self: np.zeros((4, 8, 3), dtype=np.uint8)
    )
    output = tmp_path / "media"
    command = [
        "play",
        str(package),
        "--steps",
        "5",
        "--record",
        str(output),
        "--record-frames",
        "2",
    ]
    runner = CliRunner()
    completed = runner.invoke(app, command)
    assert completed.exit_code == 0 and "5 steps, 5 transitions" in completed.output
    manifest = json.loads((output / "playback.json").read_text())
    assert manifest["source_artifact"] == "policy" and len(manifest["source_sha256"]) == 64
    assert manifest["native_ticks"] == [1, 2] and (output / "playback.gif").is_file()
    repeated = runner.invoke(app, command)
    assert repeated.exit_code == 1 and "requires a new directory" in repeated.output


def test_cancelled_playback_finalizes_only_captured_frames_and_closes(monkeypatch):
    scenario = cartpole_ppo()
    assert scenario.interaction is not None
    policy = CompiledPolicy.from_pipeline(compile(scenario))
    captured, closed = [], []
    step = _GymnasiumRuntime.step
    close = _GymnasiumRuntime.close

    def step_or_cancel(self, action):
        if captured:
            raise KeyboardInterrupt
        captured.append(True)
        return step(self, action)

    def record_close(self):
        closed.append(True)
        close(self)

    monkeypatch.setattr(_GymnasiumRuntime, "step", step_or_cancel)
    monkeypatch.setattr(_GymnasiumRuntime, "close", record_close)
    monkeypatch.setattr(
        _GymnasiumRuntime, "render_frame", lambda self: np.zeros((4, 8, 3), dtype=np.uint8)
    )
    published = []
    with pytest.raises(KeyboardInterrupt):
        play_policy(
            policy,
            scenario.interaction.environment,
            video=RLVideoSpec(enabled=True, max_frames=2),
            media_sink=lambda frames, meta: published.append((len(frames), meta)),
        )
    ((count, info),) = published
    assert count == 1 and info["native_ticks"] == [1] and info["end"] == "cancelled"
    assert closed == [True]


def test_packaged_play_does_not_import_training_or_other_simulators(tmp_path):
    package = export_package(
        CompiledPolicy.from_pipeline(compile(cartpole_ppo())), tmp_path / "policy"
    )
    script = textwrap.dedent(
        """
        import importlib.abc, runpy, sys
        attempts = []
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                top = fullname.split('.')[0]
                if (
                    fullname.startswith('nexuml.training')
                    or top.startswith(('isaaclab', 'isaacsim'))
                    or top in {'lightning', 'ray', 'rclpy', 'torchrl', 'warp', 'newton'}
                ):
                    attempts.append(fullname)
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, Block())
        sys.argv = ['nexuml', 'play', sys.argv[1], '--steps', '4', '--num-envs', '3']
        try:
            runpy.run_module('nexuml', run_name='__main__', alter_sys=True)
        except SystemExit as result:
            assert result.code in (0, None), result.code
        assert not attempts, attempts
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(package)], text=True, capture_output=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "4 steps, 12 transitions" in result.stdout
