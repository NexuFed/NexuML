"""Policy boundary for Isaac Lab-Arena's maintained rollout runner."""

from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution, version
from importlib.util import find_spec
import json
from math import isclose
import os
from pathlib import Path
import signal
import subprocess
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import unquote, urlsplit

from pydantic import Field
import torch
from tensordict import TensorDict

from nexuml.core.components import BenchmarkEnvironmentDefinition
from nexuml.core.discovery import environment
from nexuml.core.policy import CompiledPolicy
from nexuml.core.types import FactorySpec, InteractionContract, PolicyEvaluationSpec
from nexuml.evaluation.policy import _evaluation_state, _validate_evaluation_contract
from nexuml_library.environments._isaaclab_runtime import _terminate_job
from nexuml_library.environments.isaaclab_config import IsaacBindings


class ArenaPolicy:
    """Adapt one complete NexuML policy to Arena's reset/get_action protocol.

    The suite supplies the contract of its validated native environment, not
    a copy of the policy contract. This adapter never remaps control semantics.
    """

    def __init__(
        self,
        policy: CompiledPolicy,
        contract: InteractionContract,
        bindings: IsaacBindings,
        *,
        deterministic: bool,
    ):
        if (
            contract.groups
            or bindings.groups
            or contract.action_space != "continuous"
            or len(contract.actions) != 1
        ):
            raise ValueError("Arena policy bridge requires one flat continuous action")
        _validate_evaluation_contract(policy.contract, contract.model_copy(update={"num_envs": 1}))
        expected = policy.contract.observations
        if (
            bindings.observations.keys() != expected.keys()
            or bindings.action not in contract.actions
            or any(not path for path in bindings.observations.values())
            or any(
                contract.observations[key].description != field.description
                for key, field in expected.items()
            )
        ):
            raise ValueError("Arena observation/action bindings do not match the policy semantics")
        if bindings.processing != ("raw" if policy.processor is not None else "processed"):
            raise ValueError("Arena observation processing ownership does not match the policy")
        if contract.control_period_s is None:
            raise ValueError("Arena requires a declared control period")
        self.policy = policy
        self.contract = contract
        self.bindings = bindings
        self.deterministic = deterministic
        self.state: TensorDict | None = None
        self.task_description: str | None = None

    def close(self) -> None:
        """Release the evaluation-local processing state."""
        self.reset()

    def set_task_description(self, description: str | None) -> None:
        """Accept Arena's task description; state policies do not consume it."""
        self.task_description = description

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear all state, or only the streams reset by Arena.

        Raises:
            ValueError: If the selected environment indices are invalid.
        """
        if env_ids is None:
            self.state = None
        elif self.state is not None:
            if (
                env_ids.ndim != 1
                or env_ids.dtype not in (torch.int32, torch.int64)
                or torch.any(env_ids < 0)
                or torch.any(env_ids >= self.contract.num_envs)
            ):
                raise ValueError("Arena reset indices must address the current environment batch")
            mask = torch.zeros(self.contract.num_envs, dtype=torch.bool, device=self.state.device)
            mask[env_ids.long()] = True
            self.state = self.policy.reset_state(self.state, mask)

    @torch.inference_mode()
    def get_action(self, env: Any, observation: Mapping[str, Any]) -> torch.Tensor:
        """Bind native tensors to the complete policy.

        Returns:
            The unmodified decoded action batch.

        Raises:
            ValueError: If native observations, timing or actions violate the contract.
        """
        native = env.unwrapped
        field = self.contract.actions[self.bindings.action]
        period = self.contract.control_period_s
        assert period is not None
        if (
            native.num_envs != self.contract.num_envs
            or not isclose(native.step_dt, period, rel_tol=1e-9)
            or tuple(native.single_action_space.shape) != field.shape
            or str(native.single_action_space.dtype) != field.dtype
        ):
            raise ValueError("Arena native action shape/dtype, batch or control period differs")
        batch = (self.contract.num_envs,)
        inputs = TensorDict({}, batch_size=batch, device=native.device)
        for key, path in self.bindings.observations.items():
            value: Any = observation
            for part in path:
                if not isinstance(value, Mapping) or part not in value:
                    raise ValueError(f"Arena observation is missing binding for {key!r}")
                value = value[part]
            declared = self.contract.observations[key]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != (*batch, *declared.shape)
                or value.dtype != getattr(torch, declared.dtype)
                or value.device != torch.device(native.device)
            ):
                raise ValueError(f"Arena observation {key!r} does not match its declared tensor")
            inputs[key] = value
        if self.policy.processor is not None:
            if self.state is None:
                self.state = self.policy.initialize_state(batch, device=native.device)
            actions, self.state = self.policy.step(
                inputs, self.state, deterministic=self.deterministic
            )
        else:
            actions = self.policy(inputs, deterministic=self.deterministic)
        output = actions[self.bindings.action]
        if (
            not isinstance(output, torch.Tensor)
            or output.shape != (*batch, *field.shape)
            or output.dtype != getattr(torch, field.dtype)
            or output.device != torch.device(native.device)
            or not torch.isfinite(output).all()
        ):
            raise ValueError("Arena policy returned an invalid native action")
        return output


def run_arena_condition(
    env: Any, policy: ArenaPolicy, *, condition: str, seed: int, episodes: int, output: Path
) -> list[dict[str, Any]]:
    """Execute a frozen condition with Arena's runner and episode recorder.

    The caller owns the already composed native environment and Kit lifetime.
    One native construction uses one reset seed; individual starts come from
    that seeded environment and Arena's variations, not this adapter.

    Returns:
        Arena's original completed episode records, without synthesized outcomes.

    Raises:
        ValueError: If the run identity, seed or episode records disagree.
        RuntimeError: If the maintained Arena runtime is unavailable.
    """
    if not condition.strip() or episodes < 1 or env.unwrapped.cfg.seed != seed:
        raise ValueError(
            "Arena condition requires a name, matching seed and positive episode budget"
        )
    try:
        runner = import_module("isaaclab_arena.evaluation.policy_runner")
        records = import_module("isaaclab_arena.recording.episode_results")
    except ImportError as error:
        raise RuntimeError(
            "Arena evaluation requires a prepared Isaac Lab-Arena 0.4 runtime"
        ) from error

    # Arena's recorder truncates its destination; claim a new directory first.
    output.mkdir(mode=0o700)
    path = output / "episodes.jsonl"
    recorder = env.unwrapped.episode_recorder
    recorder.set_job_name(condition)
    recorder.set_output_path(path)
    with _evaluation_state(policy.policy, seed):
        try:
            runner.rollout_policy(env, policy, num_steps=None, num_episodes=episodes)
        finally:
            policy.close()
    rows = records.read_episode_records(path)
    keys = set()
    for row in rows:
        env_id, index = row.get("env_id"), row.get("episode_in_env")
        length = row.get("episode_length")
        if (
            row.get("job_name") != condition
            or row.get("seed") != seed
            or type(env_id) is not int
            or not 0 <= env_id < policy.contract.num_envs
            or type(index) is not int
            or index < 0
            or type(length) is not int
            or length < 1
            or type(row.get("success")) is not bool
            or (env_id, index) in keys
        ):
            raise ValueError("Arena returned an incomplete or inconsistent episode record")
        keys.add((env_id, index))
    if len(rows) != episodes:
        raise ValueError(f"Arena completed {len(rows)} episodes; expected {episodes}")
    return rows


_arena_job: ContextVar[ExitStack | None] = ContextVar("_arena_job", default=None)


def _source_revision(package: str) -> str | None:
    origin = json.loads(distribution(package).read_text("direct_url.json") or "{}")
    if "vcs_info" in origin:
        return origin["vcs_info"].get("commit_id")
    source = urlsplit(origin.get("url", ""))
    if source.scheme == "file" and not source.netloc:
        path = Path(unquote(source.path))
        if path.is_absolute() and (path / ".git").exists():
            return subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, timeout=5
            ).strip()
    return None


def _runtime_identity() -> dict[str, dict[str, str | None]]:
    packages = {
        "arena": "isaaclab-arena",
        "isaac_lab": "isaaclab",
        "curobo": "nvidia-curobo",
        "isaac_sim": "isaacsim",
    }
    versions = {name: version(package) for name, package in packages.items()}
    if not versions["isaac_lab"].startswith("3.0.") or not versions["isaac_sim"].startswith("6.2."):
        raise RuntimeError("Arena SO-101 requires Isaac Lab 3.0 and Isaac Sim 6.2")
    if find_spec("curobo") is None:
        raise RuntimeError("Installed nvidia-curobo has no curobo import package")
    return {
        name: {"version": versions[name], "revision": _source_revision(package)}
        for name, package in packages.items()
    }


@environment("ArenaBenchmarkEnvironment")
class ArenaBenchmarkEnvironment(BenchmarkEnvironmentDefinition):
    """A trusted, named Arena suite with explicit worker-side composition and starts."""

    suite: str = Field(min_length=1, pattern=r"\S")
    revision: str = Field(min_length=1, pattern=r"\S")
    contract: InteractionContract
    bindings: IsaacBindings
    factory: FactorySpec
    conditions: dict[str, int] = Field(min_length=1)
    kit_viewer: bool = False

    def describe(self) -> InteractionContract:
        return self.contract

    @contextmanager
    def job_scope(self):
        """Keep the native app alive through ordinary CLI tracking finalization.

        Raises:
            RuntimeError: If the process already owns an Arena job.
        """
        if _arena_job.get() is not None:
            raise RuntimeError("Use a fresh process for each Arena benchmark")
        previous_sigterm = signal.signal(signal.SIGTERM, _terminate_job)
        try:
            with ExitStack() as stack:
                token = _arena_job.set(stack)
                try:
                    yield
                finally:
                    _arena_job.reset(token)
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)

    def run_benchmark(
        self, policy: CompiledPolicy, spec: PolicyEvaluationSpec
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Compose and score each declared condition with the maintained runner.

        The trusted factory receives condition, seed and device and returns
        ``(env, native_contract)``. It owns native semantics and must check the
        actual instantiated task before declaring that contract.

        Returns:
            Aggregate and per-condition metrics, plus native episode records.

        Raises:
            ValueError: If the benchmark settings, declared pair or native data are invalid.
            RuntimeError: If the worker's prepared native/display runtime is missing.
        """
        device = next(policy.parameters(), torch.empty(0)).device
        adapter = ArenaPolicy(
            policy, self.contract, self.bindings, deterministic=spec.deterministic
        )
        count, remainder = divmod(spec.episodes, len(self.conditions))
        if (
            count == 0
            or remainder
            or spec.seed is not None
            or spec.max_steps_per_episode is not None
            or self.contract.num_envs != 1
            or any(not name.strip() or seed < 0 for name, seed in self.conditions.items())
        ):
            raise ValueError(
                "Arena requires equal episode budgets, suite seeds and native horizons"
            )
        job = _arena_job.get()
        if job is None or device.type != "cuda":
            raise RuntimeError(
                "Arena requires the evaluation job scope and an explicit CUDA device"
            )
        if self.kit_viewer and not os.environ.get("DISPLAY"):
            raise RuntimeError("Arena Kit viewing requires a prepared private worker display")
        try:
            runtime = _runtime_identity()
            app = import_module("isaaclab.app")
        except (PackageNotFoundError, ImportError) as error:
            raise RuntimeError(
                "Arena requires a prepared Isaac Lab 3.0 / Arena 0.4 runtime"
            ) from error
        if os.environ.get("LIVESTREAM", "0") != "0":
            raise RuntimeError(
                "Arena must use the prepared private display; implicit streaming is disabled"
            )
        job.enter_context(
            app.launch_simulation(
                None,
                {
                    "device": str(device),
                    "require_kit": True,
                    "headless": not self.kit_viewer,
                    "visualizer": ["kit"] if self.kit_viewer else ["none"],
                    "visualizer_explicit": True,
                    "fast_shutdown": True,
                    "livestream": 0,
                    "kit_args": (
                        "--/crashreporter/enabled=false --/crashreporter/skipOldDumpUpload=true "
                        "--/app/uploadDumpsOnStartup=false --/telemetry/enableAnonymousData=false "
                        "--/telemetry/enableAnonymousAppName=false --/telemetry/enableSentry=false "
                        "--/structuredLog/enable=false"
                    ),
                },
            )
        )
        # Kit installs an immediate-exit handler; unwind through tracking instead.
        signal.signal(signal.SIGTERM, _terminate_job)
        close_environment = import_module(
            "isaaclab_arena.evaluation.resource_cleanup"
        ).close_environment
        rows: list[dict[str, Any]] = []
        selected_failure = None
        conditions = {}
        run_metrics = {"eval/rl/episodes": float(spec.episodes)}
        with TemporaryDirectory(prefix="nexuml-arena-") as directory:
            for index, (condition, seed) in enumerate(self.conditions.items()):
                built: Any = self.factory.build(condition=condition, seed=seed, device=str(device))
                env, actual = built
                try:
                    if self.contract != actual:
                        raise ValueError("Arena native condition contract differs from the suite")
                    trials = run_arena_condition(
                        env,
                        adapter,
                        condition=condition,
                        seed=seed,
                        episodes=count,
                        output=Path(directory) / str(index),
                    )
                except BaseException as error:
                    try:
                        close_environment(env)
                    except BaseException as cleanup:
                        error.add_note(f"Arena condition cleanup also failed: {cleanup}")
                    raise
                else:
                    close_environment(env)
                successes = sum(row["success"] for row in trials)
                conditions[condition] = {
                    "seed": seed,
                    "successes": successes,
                    "episodes": count,
                    "success_rate": successes / count,
                }
                for row in trials:
                    trace = row.pop("failure_trace", None)
                    rows.append({**row, "condition": condition, "episode_id": len(rows)})
                    if not row["success"] and selected_failure is None:
                        selected_failure = {**rows[-1], "trace": trace}
        successes = sum(row["success"] for row in rows)
        run_metrics.update(
            {
                "eval/rl/task_successes": float(successes),
                "eval/rl/task_success_rate": successes / len(rows),
                "eval/rl/length_mean": sum(row["episode_length"] for row in rows) / len(rows),
            }
        )
        return run_metrics, {
            "schema_version": 1,
            "suite": self.suite,
            "suite_revision": self.revision,
            "runtime": runtime,
            "requested_episodes": spec.episodes,
            "completed_episodes": len(rows),
            "deterministic": spec.deterministic,
            "conditions": conditions,
            "episodes": rows,
            "selected_failure": selected_failure,
            "metrics": run_metrics,
        }
