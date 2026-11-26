"""Asynchronous Ray Jobs submission of the canonical local CLI lifecycle."""

from __future__ import annotations

import json
import math
import shlex
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from nexuml.core.types import ScenarioSpec


def _playback_worker_path(name: str, value: Any) -> str:
    """Validate a worker path lexically without accessing it on the submitter.

    Returns:
        The literal path to pass to the worker.

    Raises:
        ValueError: If the path is not absolute or contains traversal or NUL.
    """
    if type(value) is str or isinstance(value, Path):
        text = str(value)
        path = PurePosixPath(text)
        if path.is_absolute() and ".." not in path.parts and "\x00" not in text:
            return text
    raise ValueError(f"playback {name} must be an absolute worker path without traversal or NUL")


def _playback_argv(playback: dict[str, Any] | None) -> list[str]:
    """Validate and encode only the existing play CLI options.

    Returns:
        Literal arguments for the worker play command.

    Raises:
        ValueError: If the options contain unknown keys or invalid values.
    """
    if playback is None:
        return []
    if type(playback) is not dict:
        raise ValueError("playback must be a plain dict of play options")
    argv = []
    for name, value in playback.items():
        if type(name) is not str:
            raise ValueError("playback option names must be strings")
        if value is None and name in {"steps", "episodes", "seed", "record", "record_frames"}:
            continue
        flag = "--" + name.replace("_", "-")
        if name in {"stochastic", "real_time"}:
            if type(value) is not bool:
                raise ValueError(f"playback {name} must be a bool")
            if value:
                argv.append(flag)
            continue
        if name in {"num_envs", "steps", "episodes", "seed", "record_frames"}:
            minimum = 0 if name == "seed" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"playback {name} must be an integer >= {minimum}")
        elif name == "viewer":
            if type(value) is not str or value not in {"headless", "viser", "webrtc", "local"}:
                raise ValueError("playback viewer must be headless, viser, webrtc or local")
        elif name == "record_format":
            if type(value) is not str or value not in {"gif", "mp4", "both"}:
                raise ValueError("playback record_format must be gif, mp4 or both")
        elif name == "record":
            value = _playback_worker_path(name, value)
        else:
            raise ValueError(f"unknown playback option: {name}")
        argv += [flag, str(value)]
    recording = playback.get("record") is not None
    if recording != (playback.get("record_frames") is not None):
        raise ValueError("playback record and finite record_frames are required together")
    if recording and playback.get("num_envs", 1) != 1:
        raise ValueError("playback recording requires a single environment")
    return argv


def _ray_job_argv(
    config_path: str,
    *,
    operation: str = "train",
    trainer_checkpoint: str | Path | None = None,
    model_package: str | Path | None = None,
    checkpoint: str | Path | None = None,
    output: str | Path | None = None,
    device: str | None = None,
    overrides: Sequence[str] = (),
    export_kind: str = "train_package",
    case: str | None = None,
    attempt_root: str | Path | None = None,
    wall_seconds: float | None = None,
    case_options: dict[str, Any] | None = None,
    submission_id: str | None = None,
    playback: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> list[str]:
    """Validate operation inputs without reading worker paths and build literal CLI argv.

    Returns:
        The allowlisted CLI arguments, with the supplied temporary configuration path.

    Raises:
        ValueError: If an input is missing, malformed or irrelevant to the operation.
    """
    if operation not in {"train", "evaluate", "export", "play", "acceptance"}:
        raise ValueError(
            "ray-jobs-sdk operation must be train, evaluate, export, play or acceptance"
        )
    if playback is not None and operation != "play":
        raise ValueError(f"{operation} does not accept playback")
    if export_kind != "train_package":
        raise ValueError("Ray Jobs SDK checkpoint export supports complete train_package only")
    inputs = {
        "trainer_checkpoint": trainer_checkpoint,
        "model_package": model_package,
        "checkpoint": checkpoint,
        "output": output,
        "device": device,
    }
    allowed = {
        "train": {"trainer_checkpoint"},
        "evaluate": {"model_package", "trainer_checkpoint", "device"},
        "play": {"model_package", "trainer_checkpoint", "device"},
        "export": {"checkpoint", "output"},
        "acceptance": set(),
    }[operation]
    for name, value in inputs.items():
        if value is not None:
            if name not in allowed:
                raise ValueError(f"{operation} does not accept {name}")
            if operation == "play":
                if name != "device":
                    _playback_worker_path(name, value)
                    continue
                if type(value) is not str:
                    raise ValueError("playback device must be a string")
            if not isinstance(value, (str, Path)) or not str(value).strip() or "\x00" in str(value):
                raise ValueError(f"{name} must be a nonempty argument without NUL bytes")
    if isinstance(overrides, str) or not isinstance(overrides, (list, tuple)):
        raise ValueError("overrides must be a sequence of literal -O arguments")
    if overrides and operation != "evaluate":
        raise ValueError(f"{operation} does not accept overrides; use the resolved scenario")
    if any(
        not isinstance(value, str) or not value.strip() or "\x00" in value for value in overrides
    ):
        raise ValueError("overrides must contain nonempty strings without NUL bytes")
    if operation == "acceptance":
        if _kwargs.keys() - {"review"}:
            raise ValueError("acceptance accepts only fixed case inputs")
        if not isinstance(case, str) or case not in {
            "preflight",
            "regression",
            "package",
            "recipe",
            "so101",
            "browser",
            "recovery",
            "learning",
            "logging",
            "logging-failure",
        }:
            raise ValueError("acceptance requires a fixed case name")
        if (
            not isinstance(attempt_root, (str, Path))
            or not str(attempt_root).strip()
            or "\x00" in str(attempt_root)
            or not Path(attempt_root).is_absolute()
            or ".." in Path(attempt_root).parts
            or not Path(attempt_root).is_relative_to("/mnt/data/nexuml-qualification")
            or Path(attempt_root) == Path("/mnt/data/nexuml-qualification")
        ):
            raise ValueError(
                "attempt_root must be a private child of /mnt/data/nexuml-qualification"
            )
        if submission_id is not None:
            try:
                identity = UUID(submission_id) if isinstance(submission_id, str) else None
            except ValueError:
                identity = None
            if identity is None or identity.version != 4 or identity.hex != submission_id:
                raise ValueError("acceptance submission_id must be a UUID4 hex identifier")
        if (
            isinstance(wall_seconds, bool)
            or not isinstance(wall_seconds, (int, float))
            or not math.isfinite(wall_seconds)
            or wall_seconds <= 0
        ):
            raise ValueError("wall_seconds must be positive and finite")
        options = {} if case_options is None else case_options
        if type(options) is not dict:
            raise ValueError("case_options must be a plain JSON mapping with string keys")
        try:
            encoded_options = json.dumps(options, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("case_options must contain only finite JSON values") from error
        if not _is_json_value(options):
            raise ValueError("case_options must be a plain JSON mapping with string keys")
        return ["acceptance", case, str(attempt_root), str(wall_seconds), encoded_options]
    for name, value in {
        "case": case,
        "attempt_root": attempt_root,
        "wall_seconds": wall_seconds,
        "case_options": case_options,
        "submission_id": submission_id,
    }.items():
        if value is not None:
            raise ValueError(f"{operation} does not accept {name}")
    if operation in {"evaluate", "play"}:
        if (model_package is None) == (trainer_checkpoint is None):
            raise ValueError(
                f"{operation} requires exactly one model_package or trainer_checkpoint"
            )
        weights = (
            str(model_package)
            if model_package is not None
            else f"--trainer-checkpoint={trainer_checkpoint}"
        )
        argv = [operation, weights, "--config", config_path]
        if device is not None:
            argv += ["--device", str(device)]
        if operation == "play":
            if _kwargs.keys() - {"review"}:
                raise ValueError("play accepts only fixed inputs and playback options")
            return argv + _playback_argv(playback)
        for value in overrides:
            argv += ["-O", value]
        return argv
    if operation == "export":
        if checkpoint is None or output is None:
            raise ValueError("export requires checkpoint and explicit output")
        return [
            "export",
            "--config",
            config_path,
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
        ]
    argv = ["train", "--config", config_path]
    if trainer_checkpoint is not None:
        argv += ["--trainer-checkpoint", str(trainer_checkpoint)]
    return argv


def _is_json_value(value: Any) -> bool:
    """Check plain JSON types without coercing tuples, objects or mapping keys.

    Returns:
        Whether every nested value has a JSON-compatible type.
    """
    if type(value) is dict:
        return all(type(key) is str and _is_json_value(item) for key, item in value.items())
    if type(value) is list:
        return all(_is_json_value(item) for item in value)
    return value is None or type(value) in {str, int, float, bool}


def submit_ray_job(
    scenario: ScenarioSpec,
    *,
    operation: str = "train",
    trainer_checkpoint: str | Path | None = None,
    model_package: str | Path | None = None,
    checkpoint: str | Path | None = None,
    output: str | Path | None = None,
    device: str | None = None,
    overrides: Sequence[str] = (),
    playback: dict[str, Any] | None = None,
) -> str:
    """Submit through shared execution dispatch without constructing a local runtime.

    Play requires one absolute worker-local package or checkpoint path. The optional
    ``playback`` mapping takes existing play CLI options; recording paths are worker-local.

    Returns:
        Ray's asynchronous submission ID, not a training result.

    Raises:
        ValueError: If execution is not ray-jobs-sdk or launch inputs are invalid.
    """
    from nexuml.execution.definitions import RayJobsSDKExecution
    from nexuml.execution.service import run

    if operation not in {"train", "evaluate", "export", "play"}:
        raise ValueError("submit_ray_job operation must be train, evaluate, export or play")
    if not isinstance(scenario.execution, RayJobsSDKExecution):
        raise ValueError("submit_ray_job requires registered execution type='ray-jobs-sdk'")
    return run(
        scenario,
        operation=operation,
        trainer_checkpoint=trainer_checkpoint,
        model_package=model_package,
        checkpoint=checkpoint,
        output=output,
        device=device,
        overrides=overrides,
        playback=playback,
    )


def _validate_remote_resume(config_path: Path, checkpoint_path: str) -> None:
    """Reject changed learner/task semantics on the worker before native allocation.

    Raises:
        ValueError: If the explicit request differs from the frozen checkpoint semantics.
    """
    from nexuml.core.config import ResolvedConfig
    from nexuml.training.lightning import load_scenario_from_trainer_checkpoint

    requested = ResolvedConfig.load(config_path).to_scenario()
    saved = load_scenario_from_trainer_checkpoint(checkpoint_path, fallback=requested)
    for field in ("pipeline", "training", "interaction", "policy", "policy_evaluation"):
        if getattr(requested, field) != getattr(saved, field):
            raise ValueError(f"Remote resume requires unchanged checkpoint {field} settings")


def _submit_ray_job(scenario: ScenarioSpec, **kwargs: Any) -> str:
    """Launch once after shared preflight; never re-enter the execution service.

    Returns:
        The SDK submission ID.

    Raises:
        ValueError: If execution or command inputs are invalid.
        RuntimeError: If the optional Jobs SDK is unavailable.
    """
    from nexuml.core.config import ResolvedConfig
    from nexuml.execution.definitions import LocalExecution, RayJobsSDKExecution

    if not isinstance(scenario.execution, RayJobsSDKExecution):
        raise ValueError("Ray Jobs transport requires registered execution type='ray-jobs-sdk'")
    argv = _ray_job_argv("<remote-config>", **kwargs)
    execution = RayJobsSDKExecution.model_validate(scenario.execution.model_dump()).model_copy(
        deep=True
    )
    run_id = uuid4().hex
    python = execution.py_executable.replace("{run_id}", run_id)
    runtime_env = execution.runtime_env
    if "env_vars" in runtime_env:
        runtime_env["env_vars"] = {
            key: value.replace("{run_id}", run_id) if isinstance(value, str) else value
            for key, value in runtime_env["env_vars"].items()
        }
    command = shlex.split(python)
    if not command:
        raise ValueError("ray-jobs-sdk py_executable must contain a Python command")

    try:
        from ray.job_submission import JobSubmissionClient
    except ImportError as error:
        raise RuntimeError("Ray Jobs execution requires the nexuml[ray] extra") from error

    # Transport resolved local execution only, never submission credentials or recursive jobs.
    remote = scenario.model_copy(update={"execution": LocalExecution()})
    config = ResolvedConfig.from_scenario(remote).to_yaml()
    acceptance = argv[0] == "acceptance"
    config_index = 3 if argv[0] in {"evaluate", "play"} else 2
    bootstrap = "\n".join(
        [
            "from pathlib import Path",
            "from tempfile import TemporaryDirectory",
            *(
                ["import json", "import runpy"]
                if acceptance
                else [
                    "import signal",
                    "import subprocess",
                    "import sys",
                    "cancel_signal = 0",
                    "def on_stop(signum, frame):",
                    "    global cancel_signal",
                    "    cancel_signal = signum",
                    "signal.signal(signal.SIGTERM, on_stop)",
                    "signal.signal(signal.SIGINT, on_stop)",
                ]
            ),
            "with TemporaryDirectory(prefix='nexuml-ray-job-') as directory:",
            "    config = Path(directory) / 'scenario.yaml'",
            f"    config.write_text({config!r}, encoding='utf-8')",
            f"    argv = {argv!r}",
            *(
                [
                    "    raise SystemExit(runpy.run_path("
                    "'tests/fixtures/physical_ai_acceptance.py')"
                    "['run_case'](argv[1], Path(argv[2]), Path(config), json.loads(argv[4])))",
                ]
                if acceptance
                else [
                    f"    argv[{config_index}] = str(config)",
                    *(
                        [
                            "    from nexuml.execution.ray_job import _validate_remote_resume",
                            "    _validate_remote_resume(config, "
                            f"{str(kwargs['trainer_checkpoint'])!r})",
                        ]
                        if argv[0] == "train" and kwargs.get("trainer_checkpoint") is not None
                        else []
                    ),
                    "    if cancel_signal: raise SystemExit(128 + cancel_signal)",
                    # Kit can segfault when the CLI is embedded in Python -c.
                    '    child = subprocess.Popen([sys.executable, "-u", "-X", '
                    '"faulthandler", "-m", "nexuml.cli.main", *argv])',
                    # Ray signals existing descendants; cover a stop during spawn.
                    "    if cancel_signal and child.poll() is None:",
                    "        try: child.send_signal(cancel_signal)",
                    "        except ProcessLookupError: pass",
                    "    code = child.wait()",
                    "    if cancel_signal: raise SystemExit(128 + cancel_signal)",
                    "    raise SystemExit(128 - code if code < 0 else code)",
                ]
            ),
        ]
    )
    entrypoint = shlex.join([*command, "-c", bootstrap])
    if execution.setup_command is not None:
        entrypoint = f"{shlex.join(execution.setup_command)} && exec {entrypoint}"
    if acceptance:
        entrypoint = shlex.join(
            [
                "python",
                "tests/fixtures/physical_ai_guard.py",
                "--attempt-root",
                argv[2],
                "--receipt",
                str(Path(argv[2]) / "guard.json"),
                "--deadline-seconds",
                argv[3],
                "--disk-floor-bytes",
                str(4 * 1024**3),
                "--disk-path",
                "/tmp/ray",
                "--",
                "sh",
                "-c",
                entrypoint,
            ]
        )
    client = JobSubmissionClient(execution.address, verify=execution.verify)
    return client.submit_job(
        entrypoint=entrypoint,
        runtime_env=runtime_env,
        entrypoint_num_cpus=execution.entrypoint_num_cpus,
        entrypoint_num_gpus=execution.entrypoint_num_gpus,
        entrypoint_memory=execution.entrypoint_memory,
        entrypoint_resources=execution.entrypoint_resources,
        **(
            {"submission_id": kwargs["submission_id"]}
            if acceptance and kwargs.get("submission_id") is not None
            else {}
        ),
    )
