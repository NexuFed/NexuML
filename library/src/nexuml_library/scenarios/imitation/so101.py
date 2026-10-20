"""State-only behavior cloning for the maintained SO-101 three-goal reach task."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import TYPE_CHECKING, cast

import numpy as np

from nexuml.core.discovery import data_source
from nexuml.core.types import (
    DataSpec,
    DatasetSpec,
    InteractionSpec,
    LayerSpec,
    LoaderSpec,
    PipelineSpec,
    PolicySpec,
    RLEvaluationSpec,
    ScenarioSpec,
    TargetSpec,
    TrainingSpec,
)
from nexuml_library.action_adapters.adapters import DirectActionAdapter
from nexuml_library.data.lerobot import LeRobotDataSource
from nexuml_library.environments import isaaclab
from nexuml_library.environments._isaaclab_so101 import (
    GOALS,
    GRIPPER_POSITION,
    HOLD_SECONDS,
    HOME,
    JOINT_LIMITS,
    JOINT_NAMES,
    TARGET_SCALE,
    TOLERANCE,
)
from nexuml_library.environments.isaaclab import IsaacLabEnvironment
from nexuml_library.layers.head.regression_head import LatentRegressionHead
from nexuml_library.layers.loss.regression_loss import RegressionLoss

if TYPE_CHECKING:
    from nexuml_library.data.lerobot import _LeRobotDatasetRuntime


TASK = "NexuML-SO101-Joint-Reach"
REPO_ID = "local/so101-joint-reach-expert"
PROCESSOR = {"name": "identity", "version": "v1"}
EXPERT = {"name": "actor-goal-absolute-position", "version": "v1"}
FPS = 30
MAX_STEPS = 120
_SOURCE_FILES = (
    "isaaclab.py",
    "_isaaclab_so101.py",
    "_isaaclab_runtime.py",
    "_isaaclab_tasks.py",
    "isaaclab_config.py",
)
_PACKAGES = ("nexuml", "nexuml-library", "torch", "lerobot", "isaaclab", "isaacsim")
_QUALIFICATION_CHECKS = {"runtime", "sdk", "native_control", "native_export_replay"}


def _split_episodes(episodes: int) -> dict[str, list[int]]:
    if type(episodes) is not int or episodes < 6 or episodes % 6:
        raise ValueError("Episode count must be a positive multiple of six (at least six)")
    train, val = episodes * 2 // 3, episodes * 5 // 6
    return {
        "train": list(range(train)),
        "val": list(range(train, val)),
        "test": list(range(val, episodes)),
    }


def _task_identity() -> dict:
    environment = IsaacLabEnvironment(task=TASK)
    return {
        "environment": {"task": environment.task, "physics": environment.physics},
        "interaction_contract": environment.describe().model_dump(mode="json"),
        "processor": PROCESSOR,
        "expert": EXPERT,
        "reset": {
            "home_rad": list(HOME),
            "goals_rad": [list(goal) for goal in GOALS],
            "gripper_rad": GRIPPER_POSITION,
        },
        "success": {"tolerance_rad": TOLERANCE, "hold_seconds": HOLD_SECONDS},
    }


def _source_hashes() -> dict[str, str]:
    directory = Path(isaaclab.__file__).parent
    return {name: sha256((directory / name).read_bytes()).hexdigest() for name in _SOURCE_FILES}


def _json_sha256(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _capture_provenance(packages: dict, python: str) -> dict:
    """Build retained capture fields; package records are supplied by the local collector.

    Returns:
        Capture projection and its canonical digest, ready to merge into a receipt.
    """
    capture = {"source_sha256": _source_hashes(), "packages": packages, "python": python}
    return {"capture_provenance": capture, "capture_sha256": _json_sha256(capture)}


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def _receipt(info: dict) -> dict:
    """Check retained semantics/capture/splits, not current-implementation admission.

    Returns:
        The complete original receipt, without rewriting capture identity.

    Raises:
        ValueError: If current-schema semantics, capture integrity or splits are invalid.
    """
    receipt = info.get("provenance", {})
    if (
        not isinstance(receipt, dict)
        or type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != 2
    ):
        raise ValueError("SO-101 demonstrations require provenance schema_version=2")
    for key, expected in _task_identity().items():
        if receipt.get(key) != expected:
            raise ValueError(f"SO-101 demonstration {key} differs from the exact task contract")
    capture = receipt.get("capture_provenance")
    if not isinstance(capture, dict):
        raise ValueError("SO-101 demonstrations require retained capture provenance")
    sources = capture.get("source_sha256")
    if (
        not isinstance(sources, dict)
        or set(sources) != set(_SOURCE_FILES)
        or not all(_is_sha256(value) for value in sources.values())
    ):
        raise ValueError("SO-101 capture requires exact source file names and SHA256 hashes")
    if not _is_sha256(receipt.get("capture_sha256")) or receipt["capture_sha256"] != _json_sha256(
        capture
    ):
        raise ValueError("SO-101 retained capture provenance digest mismatch")
    count, seed = info.get("total_episodes"), receipt.get("seed")
    if type(count) is not int:
        raise ValueError("SO-101 provenance requires an integer episode count")
    splits = _split_episodes(count)
    if receipt.get("split_episodes") != splits:
        raise ValueError("SO-101 splits must be predetermined disjoint episode-ID partitions")
    if (
        type(seed) is not int
        or seed < 0
        or receipt.get("episode_seeds") != [seed + episode for episode in range(count)]
    ):
        raise ValueError("SO-101 episode seeds must be seed + episode ID")
    packages = capture.get("packages", {})
    if not isinstance(packages, dict):
        raise ValueError("SO-101 provenance requires exact runtime/package versions")
    for name in _PACKAGES:
        if (
            not isinstance(packages.get(name), dict)
            or not isinstance(packages[name].get("version"), str)
            or not packages[name]["version"].strip()
        ):
            raise ValueError("SO-101 provenance requires exact runtime/package versions")
    revision = packages["isaaclab"].get("revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
        raise ValueError("SO-101 capture requires the recorded Isaac Lab source revision")
    if not isinstance(capture.get("python"), str) or not capture["python"].strip():
        raise ValueError("SO-101 capture requires the recorded Python version")
    return receipt


def _validate_runtime_qualification(root: Path, receipt: dict, dataset_sha256: str) -> None:
    capture = receipt["capture_provenance"]
    current = _source_hashes()
    revision = IsaacLabEnvironment(task=TASK).revision
    if (
        capture["source_sha256"] == current
        and capture["packages"]["isaaclab"]["revision"] == revision
    ):
        return
    try:
        qualification = json.loads((root / "meta/runtime_qualification.json").read_text())
    except (OSError, ValueError) as error:
        raise ValueError(
            "SO-101 implementation drift requires local runtime qualification"
        ) from error
    expected = {
        "schema_version": 1,
        "dataset_sha256": dataset_sha256,
        "capture_sha256": receipt["capture_sha256"],
        "semantic_sha256": _json_sha256(_task_identity()),
        "implementation_source_sha256": current,
        "isaaclab_revision": revision,
    }
    if (
        not isinstance(qualification, dict)
        or set(qualification) != {*expected, "checks"}
        or type(qualification.get("schema_version")) is not int
        or any(qualification.get(key) != value for key, value in expected.items())
    ):
        raise ValueError(
            "SO-101 runtime qualification does not match data/capture/current identity"
        )
    checks = qualification["checks"]
    if not isinstance(checks, dict) or set(checks) != _QUALIFICATION_CHECKS:
        raise ValueError("SO-101 runtime qualification requires all named affected checks")
    for check in checks.values():
        if (
            not isinstance(check, dict)
            or set(check) != {"passed", "evidence_sha256"}
            or check["passed"] is not True
            or not _is_sha256(check["evidence_sha256"])
        ):
            raise ValueError(
                "SO-101 runtime qualification requires passed checks with evidence SHA256"
            )


@data_source("SO101Demonstrations", version="2")
class SO101DemonstrationSource(LeRobotDataSource):
    """Offline SO-101 episodes admitted by full integrity and runtime qualification checks."""

    def build(self) -> _LeRobotDatasetRuntime:
        dataset = cast("_LeRobotDatasetRuntime", super().build())
        _validate_demonstration_dataset(self, dataset)
        return dataset


def so101_behavior_cloning(root: str) -> ScenarioSpec:
    """Build a linear actor recipe from finalized local episodes, without a simulator.

    The fixed three-goal distribution is shared across episode-disjoint splits;
    these splits do not demonstrate unseen-goal generalization. Identity v1 does
    not fit normalization statistics on any split. Recipe construction alone does
    not qualify implementation drift; every source build/fit checks complete
    demonstrations and requires matching qualification before reuse.

    Returns:
        An ordinary supervised scenario with explicit deployment/evaluation contracts.
    """
    info = json.loads((Path(root).expanduser() / "meta/info.json").read_text())
    receipt = _receipt(info)
    source = SO101DemonstrationSource(
        root=root,
        repo_id=REPO_ID,
        split_episodes=receipt["split_episodes"],
        input_keys=("observation",),
        action_key="action",
    )
    return ScenarioSpec(
        name="so101-behavior-cloning",
        data=DataSpec(
            datasets=[DatasetSpec(source=source, split_type="keep", modality="robot")],
            input_shapes={"observation": [15]},
            input_contracts=IsaacLabEnvironment(task=TASK).describe().observations,
            feature_key="observation",
            targets=[TargetSpec(type="regression", key="action", num_outputs=5)],
            loader=LoaderSpec(batch_size=64, num_workers=0),
        ),
        pipeline=PipelineSpec(
            stages={
                "Actor": [
                    LayerSpec(
                        component=LatentRegressionHead(num_outputs=5, hidden_dims=[]),
                        keys_in=["observation"],
                        keys_out=["predicted_action"],
                    )
                ],
                "Loss": [
                    LayerSpec(
                        component=RegressionLoss(),
                        keys_in=["predicted_action"],
                        keys_out=["imitation_loss"],
                        label_key="action",
                    )
                ],
            }
        ),
        training=TrainingSpec(
            lr=0.01,
            max_epochs=100,
            batch_size=64,
            loss_keys={"imitation_loss": 1.0},
            accelerator="cpu",
            devices=1,
            seed=receipt["seed"],
            deterministic=True,
        ),
        interaction=InteractionSpec(environment=IsaacLabEnvironment(task=TASK)),
        policy=PolicySpec(action_adapter=DirectActionAdapter(source_key="predicted_action")),
        policy_evaluation=RLEvaluationSpec(
            episodes=60, seed=10000, max_steps_per_episode=MAX_STEPS
        ),
    )


def _observation(value) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (15,) or not np.isfinite(value).all():
        raise ValueError("SO-101 observation must be a finite 15-element actor state")
    limits = np.asarray(JOINT_LIMITS[:5])
    if ((value[:5] < limits[:, 0] - 1e-5) | (value[:5] > limits[:, 1] + 1e-5)).any():
        raise ValueError("SO-101 observation exceeds physical joint limits")
    if (np.abs(value[5:10]) > 30.0 + 1e-5).any():
        raise ValueError("SO-101 observation exceeds physical velocity limits")
    if not any(np.allclose(value[10:15], goal, rtol=0, atol=1e-6) for goal in GOALS):
        raise ValueError("SO-101 observation goal differs from the native three-goal distribution")
    return value


def validate_so101_demonstrations(source: LeRobotDataSource) -> dict:
    """Validate complete local expert transitions using the existing offline reader.

    This checks consistency, not authenticity of a claimed native receipt. A
    runtime qualification is trusted operator-owned local evidence, not a
    cryptographic or native-authenticity proof. It is read only after hashing the
    original receipt and frames, and is never part of that dataset identity.
    Returns only counts and content identities, never local filesystem paths.
    Invalid semantics, splits, timing or expert outcomes raise ``ValueError``.

    Returns:
        Dataset/contract hashes and exact eligible frame-window/episode counts.
    """
    dataset = cast("_LeRobotDatasetRuntime", LeRobotDataSource.build(source))
    return _validate_demonstration_dataset(source, dataset)


def _validate_demonstration_dataset(
    source: LeRobotDataSource, dataset: _LeRobotDatasetRuntime
) -> dict:
    if source.input_keys != ("observation",) or source.action_key != "action":
        raise ValueError("SO-101 actor inputs must be only observation; target must be action")
    info = dataset.info
    receipt = _receipt(info)
    splits = {key: list(value) for key, value in source.split_episodes.items()}
    if splits != receipt["split_episodes"]:
        raise ValueError("SO-101 source splits differ from the recorded episode-ID partitions")
    if info.get("fps") != FPS or info.get("total_frames") != len(dataset):
        raise ValueError("SO-101 finalized frame count/FPS mismatch")
    features = {
        "observation": ("float32", [15]),
        "action": ("float32", [5]),
        "next.observation": ("float32", [15]),
        "next.joint_error_rad": ("float32", [1]),
        **{f"next.{key}": ("int64", [1]) for key in ("success", "terminated", "truncated")},
    }
    for key, (dtype, shape) in features.items():
        feature = info.get("features", {}).get(key, {})
        if feature.get("dtype") != dtype or feature.get("shape") != shape:
            raise ValueError(f"SO-101 transition feature schema mismatch: {key}")
    names = [
        f"{kind}.{joint}" for kind in ("position", "velocity", "goal") for joint in JOINT_NAMES
    ]
    for key, expected in (
        ("observation", names),
        ("next.observation", names),
        ("action", list(JOINT_NAMES)),
    ):
        if info["features"][key].get("names") != expected:
            raise ValueError("SO-101 feature joint ordering differs from the exact task contract")
    meta = dataset.meta
    if meta is None:
        raise ValueError("SO-101 frame metadata is unavailable")
    if meta["index"].tolist() != list(range(len(dataset))):
        raise ValueError("SO-101 global frame indices must be complete and ordered")
    digest = sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode())
    counts = {split: {"episodes": len(ids), "windows": 0} for split, ids in splits.items()}
    for episode, episode_meta in dataset.episode_metadata.items():
        rows = meta[meta["episode_index"] == episode]
        length = len(rows)
        if not 1 <= length <= MAX_STEPS or episode_meta["length"] != length:
            raise ValueError("SO-101 episode is incomplete or exceeds the native horizon")
        if rows["frame_index"].tolist() != list(range(length)) or not np.allclose(
            rows["timestamp"].to_numpy(), np.arange(length) / FPS, rtol=0, atol=1e-6
        ):
            raise ValueError("SO-101 episode timestamps/frame indices are not contiguous at 30 Hz")
        previous, goal, held = None, None, 0
        for frame, source_index in enumerate(rows["source_index"]):
            item = dataset.reader.get_item(int(source_index))
            arrays = {key: item[key].detach().cpu().numpy() for key in features}
            obs, following = (
                _observation(arrays["observation"]),
                _observation(arrays["next.observation"]),
            )
            action = arrays["action"]
            if action.shape != (5,) or not np.isfinite(action).all() or (np.abs(action) > 1).any():
                raise ValueError("SO-101 action must be finite and within [-1, 1]")
            if frame == 0:
                goal = obs[10:15].copy()
                if not np.allclose(obs[:10], [*HOME, 0, 0, 0, 0, 0], rtol=0, atol=1e-5):
                    raise ValueError("SO-101 episode must start from the native home reset")
            if previous is not None and not np.array_equal(obs, previous):
                raise ValueError(
                    "SO-101 transition continuity is broken (possible auto-reset state)"
                )
            assert goal is not None  # Every nonempty episode sets its goal in frame zero.
            if not np.array_equal(obs[10:15], goal) or not np.array_equal(following[10:15], goal):
                raise ValueError(
                    "SO-101 goal changed within an episode (possible auto-reset state)"
                )
            if not np.allclose(action, obs[10:15] / TARGET_SCALE, rtol=0, atol=1e-6):
                raise ValueError("SO-101 action differs from the actor-visible scripted expert")
            flags = []
            for key in ("success", "terminated", "truncated"):
                value = arrays[f"next.{key}"]
                if value.size != 1 or value.item() not in (0, 1):
                    raise ValueError("SO-101 outcome flags must be numeric booleans")
                flags.append(bool(value.item()))
            success, terminated, truncated = flags
            error = arrays["next.joint_error_rad"]
            expected_error = np.abs(following[:5] - goal).mean()
            if (
                error.size != 1
                or not np.isfinite(error).all()
                or (error < 0).any()
                or not np.isclose(error.item(), expected_error, rtol=0, atol=1e-5)
            ):
                raise ValueError("SO-101 native joint error is invalid or inconsistent")
            held = held + 1 if np.max(np.abs(following[:5] - goal)) <= TOLERANCE else 0
            last = frame == length - 1
            if success != (held >= math.ceil(HOLD_SECONDS * FPS)) or (
                (success, terminated, truncated)
                != ((True, True, False) if last else (False, False, False))
            ):
                raise ValueError(
                    "SO-101 requires native held success/termination, "
                    "never timeout or incomplete episodes"
                )
            for key in sorted(arrays):
                digest.update(key.encode())
                digest.update(np.ascontiguousarray(arrays[key]).tobytes())
            previous = following.copy()
        split = rows.iloc[0]["split"]
        counts[split]["windows"] += length
    dataset_sha256 = digest.hexdigest()
    _validate_runtime_qualification(Path(source.root).expanduser(), receipt, dataset_sha256)
    return {
        "schema_version": 1,
        "episodes": info["total_episodes"],
        "frames": len(dataset),
        "train_episodes": counts["train"]["episodes"],
        "train_windows": counts["train"]["windows"],
        "splits": counts,
        "dataset_sha256": dataset_sha256,
        "contract_sha256": sha256(
            json.dumps(receipt["interaction_contract"], sort_keys=True).encode()
        ).hexdigest(),
        "processor": dict(PROCESSOR),
    }


__all__ = [
    "SO101DemonstrationSource",
    "so101_behavior_cloning",
    "validate_so101_demonstrations",
]
