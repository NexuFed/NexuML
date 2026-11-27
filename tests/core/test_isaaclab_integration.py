"""Opt-in prepared-profile acceptance; each native app owns a fresh child process."""

import json
import os
from pathlib import Path
import subprocess
import sqlite3
import shutil
import sys

import pytest


pytestmark = [
    pytest.mark.requires_gpu,
    pytest.mark.requires_simulator,
    pytest.mark.requires_optional("isaaclab"),
]
ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = ROOT / "scripts/isaaclab/run.py"
JOB = ROOT / "tests/fixtures/isaaclab_job.py"


@pytest.mark.parametrize("task", ["Isaac-Cartpole", "Isaac-Reach-Franka"])
def test_upstream_native_learning_baseline(tmp_path, task):
    import math

    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from nexuml_library.environments.isaaclab import IsaacLabEnvironment

    checkout_path = os.environ.get("NEXUML_ISAAC_CHECKOUT")
    if checkout_path is None:
        pytest.skip("NEXUML_ISAAC_CHECKOUT must identify the pinned upstream source checkout")
    checkout = Path(checkout_path).resolve()
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True)
    assert revision.strip() == IsaacLabEnvironment().revision
    result = subprocess.run(
        [
            sys.executable,
            str(checkout / "scripts/reinforcement_learning/train.py"),
            "--rl_library",
            "rsl_rl",
            "--task",
            task,
            "--num_envs",
            "8",
            "--max_iterations",
            "2",
            "--device",
            "cuda:0",
            "--seed",
            "0",
            "--visualizer",
            "none",
            "--logger",
            "tensorboard",
            "physics=isaacsim_physx",
            "agent.num_steps_per_env=16",
            "agent.algorithm.num_learning_epochs=1",
            "agent.algorithm.num_mini_batches=2",
            "agent.clip_actions=1.0",
            "env.observations.policy.enable_corruption=false",
            "env.compute_final_obs=true",
        ],
        cwd=tmp_path,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=300,
    )
    (tmp_path / "upstream-native.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SimulationContext cleared" in result.stdout
    (first_path,) = tmp_path.rglob("model_0.pt")
    (last_path,) = tmp_path.rglob("model_1.pt")
    first = torch.load(first_path, map_location="cpu", weights_only=True)
    last = torch.load(last_path, map_location="cpu", weights_only=True)
    assert last["iter"] == 1
    for key in ("actor_state_dict", "critic_state_dict"):
        assert all(torch.isfinite(value).all() for value in last[key].values())
        assert any(not torch.equal(value, first[key][name]) for name, value in last[key].items())
    assert last["optimizer_state_dict"]["state"]
    (events_path,) = tmp_path.rglob("events.out.tfevents.*")
    events = EventAccumulator(str(events_path))
    events.Reload()
    losses = [
        event
        for tag in events.Tags()["scalars"]
        if "loss" in tag.lower()
        for event in events.Scalars(tag)
    ]
    assert {event.step for event in losses} == {0, 1}
    assert all(math.isfinite(event.value) for event in losses)


def _run(args, output, label, *, expected=0, process_scope=True, extra_env=None):
    assert shutil.disk_usage(ROOT).free > 4 * 2**30
    output.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "NEXUML_LOGS_ROOT": str(output / "native-logs"), **(extra_env or {})}
    result = subprocess.run(
        [sys.executable, *([str(ENTRYPOINT)] if process_scope else []), *map(str, args)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=420,
    )
    (output / f"{label}.log").write_text(result.stdout)
    assert shutil.disk_usage(ROOT).free > 4 * 2**30
    assert result.returncode == expected, result.stdout
    return result.stdout


@pytest.mark.parametrize("task", ["Isaac-Cartpole", "Isaac-Cartpole-Direct"])
@pytest.mark.parametrize("physics", ["newton_mjwarp", "isaacsim_physx", "ovphysx"])
def test_native_manager_and_direct_transition_oracle(tmp_path, task, physics):
    log = _run(
        [ROOT / "tests/fixtures/isaaclab_transitions.py", task, physics],
        tmp_path,
        "transitions",
    )
    assert log.count("NATIVE_TRANSITION_GAE") == 7
    assert "NATIVE_ADAPTER_TRANSITIONS_VERIFIED" in log
    assert "NATIVE_EXPLICIT_MASK_RESET_VERIFIED" in log


def test_native_so101_signed_controls_and_held_outcomes(tmp_path):
    log = _run([ROOT / "tests/fixtures/isaaclab_so101_probe.py"], tmp_path, "so101-controls")
    assert "SO101_SIGN_LIMIT_FIXED_GRIPPER_VERIFIED" in log
    assert "SO101_NATIVE_HELD_SUCCESS_VERIFIED" in log
    assert "SO101_NATIVE_TIMEOUT_NOT_SUCCESS_VERIFIED" in log


def test_native_snapshot_evaluator_uses_its_own_gpu_process(tmp_path):
    log = _run(
        [ROOT / "tests/fixtures/isaaclab_snapshot.py"],
        tmp_path,
        "snapshot",
        process_scope=False,
        extra_env={"TMPDIR": str(tmp_path)},
    )
    assert "NATIVE_SNAPSHOT_RECEIPT_VERIFIED" in log


@pytest.mark.requires_optional("ray")
@pytest.mark.skipif(
    os.getenv("NEXUML_TEST_RAY_NATIVE_SNAPSHOT") != "1", reason="opt-in real Ray GPU task"
)
def test_native_snapshot_evaluator_runs_on_a_ray_gpu(tmp_path):
    log = _run(
        [ROOT / "tests/fixtures/isaaclab_snapshot.py", "--ray"],
        tmp_path,
        "ray-snapshot",
        process_scope=False,
        extra_env={"TMPDIR": "/tmp/opencode", "RAY_ADDRESS": "local"},
    )
    assert "NATIVE_SNAPSHOT_RECEIPT_VERIFIED" in log


def test_native_terminal_pixels_match_the_ending_scene_not_reset_scene(tmp_path):
    log = _run([ROOT / "tests/fixtures/isaaclab_terminal_rgb.py"], tmp_path, "terminal-rgb")
    assert "NATIVE_TERMINAL_PIXEL_AND_SCENE_IDENTITY_VERIFIED" in log


@pytest.mark.parametrize(
    "rgb,inference,learner",
    [(True, 0, 0), (False, 0, 0), (False, 0, 1), (False, 1, 1)],
    ids=["rgb-colocated", "physx-colocated", "physx-split-learner", "physx-split-inference"],
)
def test_native_ppo_placement(tmp_path, rgb, inference, learner):
    import torch

    from nexuml.core.config import ResolvedConfig
    from nexuml.core.export import load_inference_package
    from nexuml.core.types import ExportSpec
    from nexuml_library.reinforcement.algorithms.ppo import PPO
    from nexuml_library.scenarios.reinforcement.isaaclab import isaac_cartpole_ppo
    from nexuml_library.scenarios.reinforcement.isaaclab_sensors import isaac_rgb_cartpole_ppo

    # The RGB reference uses Newton; direct split placement is only supported with PhysX.
    reference = isaac_rgb_cartpole_ppo if rgb else isaac_cartpole_ppo
    scenario = reference(total_frames=128, frames_per_batch=64, num_envs=4)
    training = scenario.training
    training.devices = f"{learner},"
    training.collector.policy_device = f"cuda:{inference}"
    training.algorithm = PPO(minibatch_size=32, update_epochs=1)
    assert scenario.interaction is not None
    scenario.interaction.environment.validate_training(training, device=f"cuda:{learner}")
    if torch.cuda.device_count() <= learner:
        pytest.skip("split native PPO needs two compatible visible CUDA devices")
    policy_path = tmp_path / "policy"
    scenario.exports = [ExportSpec(kind="train_package", output=str(policy_path))]
    config_path = tmp_path / "train.yaml"
    ResolvedConfig.from_scenario(scenario).save(config_path)
    log = _run([JOB, "cartpole", "placement", config_path], tmp_path, "native-placement")
    assert "NATIVE_PPO_PLACEMENT_AND_ON_POLICY_PARITY_VERIFIED" in log
    _, _, metadata = load_inference_package(policy_path)
    assert metadata["training_frames"] == 128


@pytest.mark.parametrize("owner", ["package", "external"])
def test_native_multimodal_values_terminal_scene_and_history(tmp_path, owner):
    import numpy as np

    output = tmp_path / "sensor-diagnostics"
    log = _run(
        [ROOT / "tests/fixtures/isaaclab_sensors_probe.py", owner, output], tmp_path, "sensors"
    )
    assert "NATIVE_RGB_METRE_DEPTH_AND_EXACT_CLASS_LABELS_VERIFIED" in log
    assert "NATIVE_FINAL_SCENE_HISTORY_AND_CACHED_SELECTED_RESET_VERIFIED" in log
    metadata = json.loads((output / "sensor-metadata.json").read_text())
    assert metadata["scope"] == (
        "controlled native sensor oracle; not policy behavior/learning/viewer proof"
    )
    assert metadata["owner"] == owner
    assert metadata["task"] == "NexuML-Cartpole-Sensors"
    assert metadata["physics"] == "newton_mjwarp"
    assert metadata["seed"] == 0 and metadata["num_envs"] == 2
    assert metadata["selected_reset_mask"] == [True, False]
    assert metadata["source_size"] == metadata["captured_size"] == [32, 32]
    specs = metadata["observation_specs"]
    assert set(specs) == {"rgb", "depth", "segmentation", "state"}
    assert specs["rgb"]["sensor"]["units"] == "uint8_sRGB"
    assert specs["depth"]["sensor"]["units"] == "m"
    assert specs["depth"]["sensor"]["invalid_depth"] == "mask_zero"
    assert specs["segmentation"]["sensor"]["units"] == "class_id"
    assert specs["segmentation"]["sensor"]["labels"] == [
        [0, "BACKGROUND"],
        [1, "UNLABELLED"],
        [2, "cartpole"],
    ]
    for key in ("rgb", "depth", "segmentation"):
        assert specs[key]["sensor"]["owner"] == owner
        assert specs[key]["sensor"]["history_order"] == "oldest_first"
        assert specs[key]["sensor"]["history_length"] == 2
        assert specs[key]["sensor"]["initialization"] == "repeat_first"
    expected = {
        f"{stage}_{kind}__{key}"
        for stage in ("initial", "final", "reset")
        for kind in ("raw", "processed")
        for key in specs
    }
    with np.load(output / "sensor-snapshots.npz", allow_pickle=False) as arrays:
        assert set(arrays.files) == set(metadata["arrays"]) == expected
        assert sum(arrays[name].nbytes for name in arrays.files) == metadata["array_bytes"] <= 2**20
        for name in arrays.files:
            array = arrays[name]
            kind, key = name.split("__")
            raw = kind.endswith("_raw") and key != "state"
            spec = specs[key]["sensor"] if raw else specs[key]
            assert array.dtype == spec["raw_dtype" if raw else "dtype"]
            assert array.shape == (2, *spec["raw_shape" if raw else "shape"])
            assert metadata["arrays"][name] == {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "nbytes": array.nbytes,
            }
    assert set(metadata["stage_labels"]) == {"initial", "final", "reset"}
    assert "offscreen" in metadata["stage_labels"]["final"]
    for stage in ("initial", "final", "reset"):
        assert (output / f"{stage}-rgb-env0.png").is_file()


def test_native_categorical_class_force_and_signed_motion(tmp_path):
    log = _run(
        [ROOT / "tests/fixtures/isaaclab_categorical_probe.py"], tmp_path, "categorical-control"
    )
    assert "NATIVE_CATEGORICAL_CLASS_FORCE_AND_SIGNED_MOTION_VERIFIED" in log


@pytest.mark.parametrize("noisy", [False, True])
def test_native_spectator_uses_isolated_storage_and_preserves_sensor_trajectory(tmp_path, noisy):
    import torch
    import socket

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    assert port <= 64535
    fixture = ROOT / "tests/fixtures/isaaclab_viser_job.py"
    for mode in ("headless", "viewer"):
        log = _run(
            [fixture, *(["noise"] if noisy else []), tmp_path / f"{mode}.pt", mode, port],
            tmp_path,
            mode,
        )
        assert "NATIVE_SPECTATOR_SNAPSHOT_COMPLETE_AND_OWNED_THREADS_STOPPED" in log
        if mode == "viewer":
            assert "NATIVE_SPECTATOR_MODEL_STATE_CONTROL_STORAGE_ISOLATED" in log
            if noisy:
                assert (
                    "CONNECTED_BEFORE_FIRST_RESET_NOISY_PLOTS_DO_NOT_COMPUTE_OR_CONSUME_RNG" in log
                )
    torch.testing.assert_close(
        torch.load(tmp_path / "headless.pt", weights_only=True),
        torch.load(tmp_path / "viewer.pt", weights_only=True),
        rtol=0,
        atol=0,
    )


@pytest.mark.requires_system
def test_native_spectator_real_browser_train_replay_and_lifecycle(tmp_path):
    import signal
    import socket
    import torch

    for key in ("NEXUML_PLAYWRIGHT_MODULE", "NEXUML_CHROMIUM_EXECUTABLE"):
        if not os.environ.get(key) or not Path(os.environ[key]).exists():
            pytest.skip(f"{key} must select already-installed browser tooling; no installs")
    node = shutil.which("node")
    if node is None:
        pytest.skip("An already-installed Node runtime is required for the browser probe")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    assert port <= 64535
    fixture = ROOT / "tests/fixtures/isaaclab_viser_job.py"
    _run([fixture, "train", tmp_path / "train", port], tmp_path, "headless-train")
    for mode in ("train-viewer", "replay-viewer", "failure-viewer", "cancel-viewer"):
        assert shutil.disk_usage(ROOT).free > 4 * 2**30
        log_path = tmp_path / f"{mode}-browser.log"
        with log_path.open("x") as log:
            process = subprocess.Popen(
                [
                    node,
                    str(ROOT / "tests/fixtures/isaaclab_viser_browser.cjs"),
                    str(tmp_path / mode),
                    mode,
                    str(port),
                ],
                cwd=ROOT,
                env=os.environ.copy(),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                process.wait(timeout=240)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        assert process.returncode == 0, log_path.read_text()
        assert "NATIVE_BROWSER_JOB_STATUS_AND_LISTENER_CLEANUP_VERIFIED" in log_path.read_text()
        if mode == "train-viewer":
            for name in ("trajectory.pt", "training-state.pt"):
                torch.testing.assert_close(
                    torch.load(tmp_path / "train" / name, weights_only=True),
                    torch.load(tmp_path / mode / name, weights_only=True),
                    rtol=0,
                    atol=0,
                )
            _run([fixture, "replay", tmp_path / "replay", port], tmp_path, "headless-replay")
        if mode == "replay-viewer":
            torch.testing.assert_close(
                torch.load(tmp_path / "replay/trajectory.pt", weights_only=True),
                torch.load(tmp_path / "replay-viewer/trajectory.pt", weights_only=True),
                rtol=0,
                atol=0,
            )
        if mode in {"failure-viewer", "cancel-viewer"}:
            finalizations = json.loads((tmp_path / mode / "logger-finalization.json").read_text())
            assert finalizations and finalizations[-1]["status"] == "failed"
            assert all(event["job_scope_active"] for event in finalizations)


@pytest.mark.requires_system
def test_native_spectator_public_cli_browser_train_replay_and_lifecycle(tmp_path):
    import hashlib
    import signal
    import socket
    import torch

    for key in ("NEXUML_PLAYWRIGHT_MODULE", "NEXUML_CHROMIUM_EXECUTABLE"):
        if not os.environ.get(key) or not Path(os.environ[key]).exists():
            pytest.skip(f"{key} must select the prepared private browser runtime")
    node = shutil.which("node")
    if node is None:
        pytest.skip("The prepared private Node runtime is required")
    port = 8081  # KubeRay metrics owns 8080; conflicts fail, never select another port.
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", port))
    fixture = ROOT / "tests/fixtures/isaaclab_viser_cli.py"

    def headless(mode):
        launch = json.loads(
            _run(
                [fixture, "prepare", mode, tmp_path / mode, port],
                tmp_path,
                f"prepare-{mode}",
                process_scope=False,
            )
        )
        assert launch["argv"][:3] == [sys.executable, "-m", "nexuml.cli.main"]
        text = _run(
            launch["argv"][1:],
            tmp_path / mode,
            f"public-cli-{mode}",
            process_scope=False,
            extra_env=launch["env"],
        )
        assert f"CLI_PROBE_NATIVE_CLOSED_AND_OWNED_THREADS_STOPPED {mode}" in text
        assert json.loads((tmp_path / mode / "native-cleanup.json").read_text())["status"] == "PASS"

    def trajectory(directory):
        manifest = json.loads((directory / "trajectory-manifest.json").read_text())
        values = []
        for index, part in enumerate(manifest["parts"]):
            expected = "trajectory.pt" if index == 0 else f"trajectory-{index:04d}.pt"
            assert part["path"] == expected
            target = directory / expected
            assert target.stat().st_size == part["bytes"] <= 32 * 1024**2
            with target.open("rb") as source:
                assert hashlib.file_digest(source, "sha256").hexdigest() == part["sha256"]
            chunk = torch.load(target, weights_only=True)
            assert len(chunk) == part["steps"]
            values.extend(chunk)
        assert len(values) == manifest["steps"]
        return values

    headless("train")
    for mode in ("train-viewer", "replay-viewer", "failure-viewer", "cancel-viewer"):
        assert shutil.disk_usage(ROOT).free > 4 * 2**30
        log_path = tmp_path / f"{mode}-public-cli-browser.log"
        with log_path.open("x") as log:
            process = subprocess.Popen(
                [
                    node,
                    str(ROOT / "tests/fixtures/isaaclab_viser_cli_browser.cjs"),
                    str(tmp_path / mode),
                    mode,
                    str(port),
                ],
                cwd=ROOT,
                env={**os.environ, "NEXUML_SPECTATOR_PYTHON": sys.executable},
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                process.wait(timeout=240)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        text = log_path.read_text()
        assert process.returncode == 0, text
        assert "PUBLIC_CLI_WITH_TEST_OWNED_INSTANCE_PROBE" in text
        assert "NATIVE_BROWSER_JOB_STATUS_AND_LISTENER_CLEANUP_VERIFIED" in text
        if mode == "train-viewer":
            torch.testing.assert_close(
                trajectory(tmp_path / "train"), trajectory(tmp_path / mode), rtol=0, atol=0
            )
            torch.testing.assert_close(
                torch.load(tmp_path / "train/training-state.pt", weights_only=True),
                torch.load(tmp_path / mode / "training-state.pt", weights_only=True),
                rtol=0,
                atol=0,
            )
            headless("replay")
        if mode == "replay-viewer":
            torch.testing.assert_close(
                trajectory(tmp_path / "replay"),
                trajectory(tmp_path / "replay-viewer"),
                rtol=0,
                atol=0,
            )
        if mode in {"failure-viewer", "cancel-viewer"}:
            finalizations = json.loads((tmp_path / mode / "logger-finalization.json").read_text())
            assert finalizations and finalizations[-1]["status"] == "failed"
            assert all(event["job_scope_active"] for event in finalizations)
            assert all(event["completed"] and event["error"] is None for event in finalizations)


@pytest.mark.parametrize("mode", ["port-conflict", "startup-failure"])
def test_native_spectator_partial_startup_releases_context_listener_and_threads(tmp_path, mode):
    import socket

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    assert port <= 64535
    log = _run(
        [ROOT / "tests/fixtures/isaaclab_viser_job.py", mode, tmp_path / mode, port], tmp_path, mode
    )
    assert "NATIVE_PARTIAL_STARTUP_CONTEXT_LISTENER_AND_THREADS_RELEASED" in log


@pytest.mark.parametrize("task", ["Isaac-Cartpole", "Isaac-Cartpole-Direct"])
@pytest.mark.parametrize("mode", ["default", "default-reader", "native", "native-reader"])
def test_native_public_wrapper_does_not_meet_the_complete_reset_contract(tmp_path, task, mode):
    log = _run(
        [ROOT / "tests/fixtures/isaaclab_wrapper_probe.py", task, mode], tmp_path, "wrapper-probe"
    )
    line = next(line for line in log.splitlines() if line.startswith("WRAPPER_ORACLE_RESULT "))
    result = json.loads(line.partition(" ")[2])
    assert result["native_ready"] is True
    assert result["conformant"] is False
    if task == "Isaac-Cartpole" and mode == "default-reader":
        assert result["terminal_truth"] and result["timeout_target_truth"]
        assert "Cannot partially reset" in result["error"]
    elif task == "Isaac-Cartpole-Direct" and mode.startswith("default"):
        assert "Partial resets are not handled" in result["error"]
    else:
        assert not result["terminal_truth"] and not result["timeout_target_truth"]


def test_registered_so101_uses_pinned_native_physics_resolution(tmp_path):
    log = _run([JOB, "so101", "config", tmp_path], tmp_path, "config", process_scope=False)
    assert "PINNED_SO101_FIXED_PHYSICS_SELECTION_VERIFIED" in log


def test_native_interactive_local_fit_restarts_from_two_approved_bases(tmp_path):
    fits, critics = [], []
    for index in ("1", "2"):
        log = _run([JOB, index, "local-fit", tmp_path], tmp_path, f"fit-{index}")
        assert "SimulationContext cleared" in log
        fits.append(json.loads((tmp_path / index / "fit.json").read_text()))
        critics.append(json.loads((tmp_path / index / "oracle.json").read_text())["critic"])
    first, second = [receipt["model_manifest"] for receipt in fits]
    assert first["manifest"] == second["manifest"]
    assert first["base"]["tensor_sha256"] != second["base"]["tensor_sha256"]
    assert first["base"]["global_version"] != second["base"]["global_version"]
    assert fits[0]["updated_tensor_sha256"] != fits[1]["updated_tensor_sha256"]
    assert fits[0]["recipe_sha256"] == fits[1]["recipe_sha256"]
    assert critics[0] == critics[1]


def test_native_scripted_and_policy_actions_share_the_headless_runtime(tmp_path):
    fixture = ROOT / "tests/fixtures/isaaclab_foundation.py"
    rows = {}
    for phase in ("known", "policy", "replay"):
        log = _run([fixture, phase, tmp_path], tmp_path, phase)
        assert "SimulationContext cleared" in log
        rows[phase] = json.loads((tmp_path / f"{phase}.json").read_text())
    policy, replay = rows["policy"], rows["replay"]
    assert policy["purpose"] == "eval" and replay["purpose"] == "play"
    assert policy["source"] != replay["source"] and policy["force_n"] != 0
    for key in ("initial_observation", "requested_action", "submitted_action", "native_frames"):
        assert policy[key] == replay[key]
    for key in ("observation", "reward", "force_n"):
        assert policy[key] == pytest.approx(replay[key])
    for key in ("terminated", "truncated"):
        assert policy[key] == replay[key]


@pytest.mark.requires_optional("mujoco")
def test_native_and_cpu_mujoco_fit_and_reload_one_common_actor(tmp_path):
    import torch

    from nexuml.training.local_fit import _state_hash

    def job(domain, phase):
        _run(
            [JOB, domain, f"paired-{phase}", tmp_path],
            tmp_path,
            f"{domain}-{phase}",
            process_scope=domain == "isaac",
        )
        if phase != "prepare":
            return json.loads((tmp_path / f"{domain}-{phase}.json").read_text())

    job("mujoco", "prepare")
    base = torch.load(tmp_path / "base.pt", weights_only=True)
    domains = ("isaac", "mujoco")
    controls = {domain: job(domain, "controls") for domain in domains}
    first = [job(domain, "fit-1") for domain in domains]
    manifest = first[0]["model_manifest"]
    assert manifest == first[1]["model_manifest"]
    assert manifest["base"]["tensor_sha256"] == _state_hash(base)
    assert first[0]["recipe_sha256"] != first[1]["recipe_sha256"]
    local = [torch.load(tmp_path / f"{domain}-1.pt", weights_only=True) for domain in domains]
    trainable = [
        name
        for name, field in manifest["manifest"]["tensors"].items()
        if field["role"] == "trainable_parameter"
    ]
    for name in base.keys() - set(trainable):
        assert all(torch.equal(base[name], update[name]) for update in local)
    combined = {
        name: local[0][name].lerp(local[1][name], 0.5) if name in trainable else value
        for name, value in base.items()
    }
    combined_digest = _state_hash(combined)
    assert len({combined_digest, *[_state_hash(state) for state in (base, *local)]}) == 4
    torch.save(combined, tmp_path / "combined.pt")
    evaluation = {
        domain: {
            name: job(domain, f"eval-{name}")
            for name in ("base", "isaac-1", "mujoco-1", "combined")
        }
        for domain in domains
    }
    for candidates in evaluation.values():
        for report in candidates.values():
            assert report["completed_episodes"] == 4
            assert [ep["seed"] for ep in report["episodes"]] == [211, 212, 213, 214]

    second = [job(domain, "fit-2") for domain in domains]
    for receipt in second:
        assert receipt["model_manifest"]["manifest"] == manifest["manifest"]
        assert receipt["model_manifest"]["base"] == {
            "global_version": "paired-2",
            "tensor_sha256": combined_digest,
        }
        assert receipt["updated_tensor_sha256"] != combined_digest
    for receipt in (*first, *second):
        assert receipt["frames"] == 1024 and receipt["completed_episodes"] > 0
        assert receipt["minibatch_updates"] == 16 and receipt["optimizer_steps"] == 32
    directions = [
        torch.cat([(update[name] - base[name]).flatten() for name in trainable]) for update in local
    ]
    (tmp_path / "pair.json").write_text(
        json.dumps(
            {
                "controls": controls,
                "base": _state_hash(base),
                "combined": combined_digest,
                "update_norms": [direction.norm().item() for direction in directions],
                "update_cosine": torch.nn.functional.cosine_similarity(*directions, dim=0).item(),
                "return_mean": {
                    domain: {
                        name: result["metrics"]["eval/policy/return_mean"]
                        for name, result in candidates.items()
                    }
                    for domain, candidates in evaluation.items()
                },
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def test_project_owned_direct_task_native_and_adapted_controls(tmp_path):
    import torch

    fixture = ROOT / "tests/fixtures/isaaclab_custom_direct.py"
    for mode in ("native", "adapted", "native-replay"):
        log = _run([fixture, mode, tmp_path], tmp_path, mode, process_scope=False)
        assert f"CUSTOM_DIRECT_VERIFIED {mode}" in log
    native = torch.load(tmp_path / "native-controls.pt", weights_only=True)
    adapted = torch.load(tmp_path / "adapted-controls.pt", weights_only=True)
    assert native.pop("asset") == adapted.pop("asset")
    torch.testing.assert_close(adapted, native)


@pytest.mark.parametrize(
    "reference",
    [
        "cartpole",
        "cartpole-newton",
        "cartpole-ovphysx",
        "cartpole-direct",
        "franka",
        "so101",
        "custom",
        "custom-direct",
        "categorical",
        *[
            f"{family}-{owner}"
            for family in ("rgb", "depth", "segmentation", "mixed")
            for owner in ("package", "external")
        ],
    ],
)
def test_native_checkpoint_remaining_budget_export_and_scalar_replay(tmp_path, reference):
    for stage in ("interrupt", "resume", "complete", "infer"):
        log = _run(
            [JOB, reference, stage, tmp_path],
            tmp_path,
            stage,
            expected=1 if stage == "interrupt" else 0,
            process_scope=stage != "infer",
        )
        if stage == "interrupt":
            assert "INTERRUPTED_CHECKPOINT_AND_FAILED_LOGGER_FINALIZED" in log
            if reference == "custom":
                assert "ACTUAL_CUSTOM_TASK_CONFIG_VERIFIED" in log
        elif stage == "resume":
            assert "ACTOR_CRITIC_OPTIMIZERS_COUNTERS_RESTORED" in log
        elif stage == "complete":
            assert "COMPLETED_BUDGET_NO_SIMULATOR" in log
        else:
            assert "EXACT_PORTABLE_CPU_INFERENCE_VERIFIED" in log
            if reference.endswith(("-package", "-external")):
                assert "PORTABLE_NATIVE_SENSOR_SEQUENCE_SELECTED_RESET_PARITY" in log
    log = _run(
        [
            "-m",
            "nexuml",
            "evaluate",
            tmp_path / reference / "resume/policy",
            "--device",
            "cuda:0",
            "-O",
            "training.evaluation.episodes=2",
            "-O",
            "training.evaluation.max_steps_per_episode=8",
        ],
        tmp_path,
        "scalar-replay",
    )
    assert "Evaluation complete!" in log and "frame step 256" in log
    if reference == "so101":
        assert "task_success_rate" in log and "final_joint_error_rad_mean" in log
    if reference in {"custom-direct", "mixed-package", "mixed-external"}:
        count, ticks = (8, 60) if reference == "custom-direct" else (4, 6)
        log = _run(
            [
                "-m",
                "nexuml",
                "play",
                tmp_path / reference / "resume/policy",
                "--viewer",
                "headless",
                "--device",
                "cuda:0",
                "--num-envs",
                str(count),
                "--steps",
                str(ticks),
                "--seed",
                "23",
            ],
            tmp_path,
            "vector-play",
        )
        assert f"Playback complete: {ticks} steps, {count * ticks} transitions" in log


def test_native_frame_budget_counts_rounded_actual_transitions(tmp_path):
    log = _run([JOB, "cartpole", "rounded", tmp_path], tmp_path, "rounded")
    assert '"frames": 256' in log and '"collected_after_batches": [0, 1]' in log


@pytest.mark.parametrize("reference", ["cartpole", "cartpole-newton"])
def test_native_sigterm_finalizes_failed_logger_before_owned_shutdown(tmp_path, reference):
    log = _run([JOB, reference, "terminate", tmp_path], tmp_path, "sigterm", expected=143)
    assert "INTERRUPTED_CHECKPOINT_AND_FAILED_LOGGER_FINALIZED" in log
    if reference == "cartpole":
        assert "SimulationContext cleared" in log


def test_native_kit_async_cancellation_cannot_exit_successfully(tmp_path):
    log = _run([JOB, "cartpole", "cancel-native", tmp_path], tmp_path, "async-cancel", expected=1)
    assert "NATIVE_RUNTIME_CLOSED_BEFORE_ASYNC_CANCEL" in log and "CancelledError" in log


def test_native_recorded_replay_sigterm_never_marks_mlflow_success(tmp_path):
    _run([JOB, "cartpole", "train", tmp_path], tmp_path, "training")
    log = _run(
        [JOB, "cartpole", "cancel-replay", tmp_path], tmp_path, "cancel-replay", expected=143
    )
    assert "Evaluation complete!" not in log and "SimulationContext cleared" in log
    with sqlite3.connect(tmp_path / "cartpole/cancel-replay.db") as database:
        assert database.execute("SELECT count(*), min(status) FROM runs").fetchone() == (
            1,
            "FAILED",
        )
    assert not list((tmp_path / "cartpole/cancel-artifacts").rglob("*.gif"))


@pytest.mark.parametrize("reference", ["cartpole", "franka", "so101"])
@pytest.mark.parametrize("format", ["gif", "mp4", "both"])
def test_native_recorded_replay_uses_existing_media_pipeline(tmp_path, reference, format):
    ffmpeg = pytest.importorskip("imageio_ffmpeg")
    image = pytest.importorskip("PIL.Image")
    _run([JOB, reference, "train", tmp_path], tmp_path, "training")
    package = tmp_path / reference / "train/policy"
    source_frames = json.loads((package / "metadata.json").read_text())["training_frames"]
    assert source_frames == 256
    log_root = tmp_path / "native-logs"
    prior_events = len(list(log_root.rglob("events.out.tfevents.*")))
    backend = ("tensorboard", "mlflow", "both")[
        (("cartpole", "franka", "so101").index(reference) + ("gif", "mp4", "both").index(format))
        % 3
    ]
    logging = []
    if backend != "tensorboard":
        mlflow = {
            "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
            "artifact_location": str(log_root / "mlflow-artifacts"),
            "auto_nested_runs": False,
        }
        logging = [
            "-O",
            "logging="
            + json.dumps(
                {
                    "tensorboard": None if backend == "mlflow" else {},
                    "mlflow": mlflow,
                }
            ),
        ]
    log = _run(
        [
            "-m",
            "nexuml",
            "evaluate",
            package,
            "--device",
            "cuda:0",
            "-O",
            "training.evaluation.episodes=2",
            "-O",
            "training.evaluation.max_steps_per_episode=8",
            "-O",
            "training.evaluation.video.enabled=true",
            "-O",
            f"training.evaluation.video.format={format}",
            "-O",
            "training.evaluation.video.max_frames=9",
            "-O",
            "training.evaluation.video.max_size=320",
            *logging,
        ],
        tmp_path,
        "recorded-replay",
    )
    assert "Evaluation complete!" in log
    artifacts = list((tmp_path / "native-logs").rglob("*.gif"))
    videos = list((tmp_path / "native-logs").rglob("*.mp4"))
    assert bool(artifacts) == (format in {"gif", "both"})
    assert bool(videos) == (format in {"mp4", "both"})
    metadata = []
    for path in log_root.rglob("*.json"):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "captured_frames" in data:
            metadata.append(data)
    assert metadata and all(m["renderer"] == "isaac_rtx" for m in metadata)
    assert all(
        1 <= m["captured_frames"] <= 9 and m["frame_step"] == source_frames for m in metadata
    )
    expected = metadata[0]["captured_frames"] / metadata[0]["fps"]
    for path in artifacts:
        duration = 0
        with image.open(path) as clip:
            assert clip.size == (320, 240)
            assert 1 <= clip.n_frames <= metadata[0]["captured_frames"]
            for index in range(clip.n_frames):
                clip.seek(index)
                duration += clip.info["duration"] / 1000
        gif_duration = metadata[0]["encoders"]["gif"]["frame_duration_ms"] / 1000
        assert duration == pytest.approx(metadata[0]["captured_frames"] * gif_duration, abs=0.011)
    for path in videos:
        reader = ffmpeg.read_frames(str(path), pix_fmt="rgb24")
        try:
            header = next(reader)
            assert header["size"] == (320, 240) and header["codec"] == "h264"
            assert sum(1 for _ in reader) == metadata[0]["captured_frames"]
            assert header["duration"] == pytest.approx(expected, abs=0.02)
        finally:
            reader.close()
    if backend == "mlflow":
        assert len(list(log_root.rglob("events.out.tfevents.*"))) == prior_events
    if backend != "tensorboard":
        with sqlite3.connect(tmp_path / "mlflow.db") as database:
            assert database.execute("SELECT count(*), min(status) FROM runs").fetchone() == (
                1,
                "FINISHED",
            )
