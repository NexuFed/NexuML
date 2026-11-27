"""Hermetic replay-boundary checks: no learner, simulator, subprocess or native renderer."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "fixtures/physical_ai_mujoco_recording.py"
SPEC = importlib.util.spec_from_file_location("physical_ai_mujoco_recording", SOURCE)
assert SPEC is not None and SPEC.loader is not None
recording = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recording)
SHARED = runpy.run_path(str(SOURCE.with_name("physical_ai_logging.py")))


class Model(SimpleNamespace):
    @classmethod
    def model_validate(cls, value):
        return cls(
            **{
                key: Model.model_validate(item) if isinstance(item, dict) else item
                for key, item in value.items()
            }
        )

    def model_dump(self, **kwargs):
        return {
            key: value.model_dump() if isinstance(value, Model) else value
            for key, value in vars(self).items()
        }


class Video(Model):
    def __init__(self, **kwargs):
        super().__init__(fps=None, **kwargs)


class Logging(Model):
    @classmethod
    def model_validate(cls, value):
        return super().model_validate({"mlflow": None, "dvclive": None, **value})


class Config(Model):
    @classmethod
    def load(cls, path):
        return cls.model_validate(json.loads(path.read_text()))

    def save(self, path):
        path.write_text(json.dumps(self.model_dump()))

    def model_copy(self, **kwargs):
        return copy.deepcopy(self)


def metrics():
    result = {
        f"eval/rl/{key}": value
        for key, value in {
            "episodes": 2.0,
            "length_mean": 50.0,
            "return_mean": -12.5,
            "return_std": 0.5,
            "terminated": 0.0,
            "truncated": 2.0,
            "native_truncated": 2.0,
            "evaluation_horizon_limits": 0.0,
        }.items()
    }
    return {**result, **{key.replace("/rl/", "/policy/"): value for key, value in result.items()}}


class RecordingTests(unittest.TestCase):
    def setUp(self):
        import numpy as np

        temporary = tempfile.TemporaryDirectory(prefix="m1-unit-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.record: dict = {
            "recipe": "mujoco-reacher-ppo",
            "assertions": {"training_frames": 1024},
            "metrics": metrics(),
            "checks": {},
            "gates": {key: "UNVERIFIED" for key in "CVLRD"},
        }
        self.env = {"MUJOCO_GL": "egl", "CUDA_VISIBLE_DEVICES": ""}
        self.protocol = {
            "episodes": 2,
            "seed": 10000,
            "deterministic": True,
            "max_steps_per_episode": None,
            "at_start": False,
            "every_n_frames": None,
            "video": {
                "enabled": False,
                "format": "gif",
                "max_frames": 500,
                "max_size": 256,
                "fps": None,
            },
        }
        self.config: dict = {
            "name": self.record["recipe"],
            "policy_evaluation": self.protocol,
            "logging": {"mlflow": {"tracking_uri": "must-not-be-used"}},
            "training": {"total_frames": 1024, "seed": 0, "evaluation": {"horizon": None}},
            "interaction": {
                "environment": {"name": "Reacher-v5"},
                "contract": {"observation": [11], "action": [2]},
            },
            "policy": {"state_only": True, "action_adapter": {"scale": [1.0, 1.0]}},
        }
        (self.root / "eval-case.yaml").write_text(json.dumps(self.config))
        policy = self.root / "checkpoint-policy"
        policy.mkdir()
        for name in ("state_dict.pt", "action_adapter_state.pt"):
            (policy / name).write_bytes(name.encode())
        self.policy_metadata = {
            "artifact_kind": "policy",
            "training_frames": 1024,
            "policy_files_sha256": {
                name: recording._sha256(policy / name)
                for name in ("state_dict.pt", "action_adapter_state.pt")
            },
            "checkpoint": {"sha256": "actual-selected-checkpoint"},
            "config_hash": "source-config",
        }
        (policy / "metadata.json").write_text(json.dumps(self.policy_metadata))
        (policy / "resolved_config.yaml").write_text(json.dumps(self.config))
        self.modules = {}
        for name, contents in {
            "nexuml.core.config": {"ResolvedConfig": Config},
            "nexuml.core.types": {"LoggingSpec": Logging, "PolicyVideoSpec": Video},
            "nexuml.core.serialization": {"lower_model": lambda value: value.model_dump()},
        }.items():
            module = ModuleType(name)
            module.__dict__.update(contents)
            self.modules[name] = module
        imageio = ModuleType("imageio")
        v2 = ModuleType("imageio.v2")
        self.reader = MagicMock()
        self.reader.__enter__.return_value = self.reader
        self.reader.count_frames.return_value = 16
        self.reader.get_meta_data.return_value = {"codec": "h264"}
        self.frames = [np.zeros((86, 128, 3), dtype=np.uint8) for _ in range(16)]
        for index, frame in enumerate(self.frames):
            frame[0, 0, 0] = index + 1
        self.reader.iter_data.side_effect = lambda: iter(self.frames)
        setattr(v2, "get_reader", Mock(return_value=self.reader))
        setattr(imageio, "v2", v2)
        self.modules.update({"imageio": imageio, "imageio.v2": v2})
        self.gif_info = Mock(
            return_value={"frames": 16, "size": [128, 85], "changing_nonuniform_frames": True}
        )
        self.shared = {**SHARED, "_gif_info": self.gif_info}
        self.stage = Mock(side_effect=self.produce)
        self.parse = Mock(side_effect=lambda *args, **kwargs: metrics())
        self.edit_metadata = lambda data: None
        self.edit_stage = lambda: None

    def produce(self, record, name, argv, env):
        self.assertIs(record, self.record)
        self.assertEqual(name, "mujoco-recorded-evaluate")
        self.assertEqual(
            argv,
            [
                sys.executable,
                "-m",
                "nexuml.cli.main",
                "evaluate",
                str(self.root / "checkpoint-policy"),
                "--config",
                str(self.root / "recorded-eval-case.yaml"),
                "--device",
                "cpu",
            ],
        )
        self.assertIs(env, self.env)
        saved = Config.load(self.root / "recorded-eval-case.yaml").model_dump()
        self.effective_config = copy.deepcopy(saved)
        self.effective_config["interaction"]["evaluation_environment"] = copy.deepcopy(
            saved["interaction"]["environment"]
        )
        self.directory = self.root / "logs/recorded-tb/run/version_0/artifacts/evaluation"
        self.directory.mkdir(parents=True)
        identity = SHARED["_policy_identity"](self.root / "checkpoint-policy", 1024)
        self.data: dict = {
            "frame_step": 1024,
            "evaluation": saved["policy_evaluation"],
            "metrics": metrics(),
            "seeds": [10000, 10001],
            "formats": ["gif", "mp4"],
            "fps": 60.0,
            "captured_frames": 16,
            "capture_capped": True,
            "source_size": [600, 400],
            "captured_size": [128, 85],
            "provenance": {
                "source_package": identity["path"],
                "source_training_frames": 1024,
                "source_policy_files_sha256": identity["policy_files_sha256"],
                "source_checkpoint": identity["checkpoint"],
                "source_config_hash": identity["config_hash"],
                "resolved_config": self.effective_config,
            },
        }
        self.edit_metadata(self.data)
        for suffix in ("json", "gif", "mp4"):
            (self.directory / f"frame_000000001024.{suffix}").write_text(
                json.dumps(self.data) if suffix == "json" else "decoder stub"
            )
        self.edit_stage()
        return {"log_path": str(self.root / "mujoco-recorded-evaluate.log")}

    def run_replay(self):
        with (
            patch.dict(sys.modules, self.modules),
            patch.object(recording.runpy, "run_path", return_value=self.shared),
        ):
            return recording.run_recorded_mujoco(
                self.record, self.root, self.env, self.stage, self.parse
            )

    def assert_not_passed(self):
        self.assertNotIn("mujoco_recorded_replay", self.record["checks"])
        self.assertEqual(self.record["gates"], {key: "UNVERIFIED" for key in "CVLRD"})

    def test_import_is_stdlib_only(self):
        for node in ast.parse(SOURCE.read_text()).body:
            if isinstance(node, ast.Import):
                self.assertTrue(
                    all(item.name.split(".")[0] in sys.stdlib_module_names for item in node.names)
                )
            if isinstance(node, ast.ImportFrom):
                assert node.module is not None
                self.assertIn(node.module.split(".")[0], sys.stdlib_module_names)

    def test_exact_cli_config_immutability_and_scoped_receipt(self):
        self.run_replay()
        saved = Config.load(self.root / "recorded-eval-case.yaml").model_dump()
        expected = copy.deepcopy(self.config)
        expected["logging"] = {
            "experiment_name": "physical-ai-m1-recording",
            "mlflow": None,
            "dvclive": None,
            "diagram": None,
            "tensorboard": {"log_dir": str(self.root / "logs/recorded-tb")},
            "temp_artifact_dir": str(self.root / "logs/recorded-artifacts"),
        }
        expected["policy_evaluation"]["video"] = {
            "enabled": True,
            "format": "both",
            "max_frames": 16,
            "max_size": 128,
            "fps": None,
        }
        self.assertEqual(saved, expected)
        self.assertEqual(Config.load(self.root / "eval-case.yaml").model_dump(), self.config)
        self.parse.assert_called_once_with(
            self.root / "mujoco-recorded-evaluate.log", expected_frames=1024
        )
        receipt = self.record["mujoco_recording"]
        self.assertEqual(receipt["recorded_metrics"], self.record["metrics"])
        self.assertTrue(receipt["full_metric_parity"])
        self.assertIn("first episode", receipt["media"]["clip_scope"])
        for item in receipt["media"]["files"].values():
            self.assertEqual(
                item["sha256"], hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
            )
        self.assertEqual(receipt["native_gl_selector"]["MUJOCO_GL"], "egl")
        self.assertEqual(self.record["checks"], {"mujoco_recorded_replay": "PASS"})
        self.assertEqual(self.record["gates"], {key: "UNVERIFIED" for key in "CVLRD"})
        self.gif_info.assert_called_once_with(self.directory / "frame_000000001024.gif")

    def test_explicit_native_capture_bounds_do_not_weaken_existing_m1_defaults(self):
        self.run_replay()
        data = copy.deepcopy(self.data)
        data["captured_frames"] = 120
        data["captured_size"] = [320, 213]
        data["metrics"]["eval/rl/length_mean"] = 100.0
        data["evaluation"]["video"].update(max_frames=120, max_size=320)
        identity = SHARED["_policy_identity"](self.root / "checkpoint-policy", 1024)
        args = (
            data,
            1024,
            identity,
            data["evaluation"],
            data["metrics"],
            data["provenance"]["resolved_config"],
        )
        recording._check_metadata(*args, capture_frames=120, capture_size=320)
        with self.assertRaisesRegex(AssertionError, "capture count"):
            recording._check_metadata(*args)

    def test_admission_precedes_dynamic_imports_and_writes(self):
        for recipe, gl in (
            ("cartpole-ppo", "egl"),
            ("mujoco-reacher-ppo", None),
            ("mujoco-humanoid-ppo", "osmesa"),
        ):
            with (
                self.subTest(recipe=recipe, gl=gl),
                patch.object(recording.runpy, "run_path", side_effect=AssertionError("imported")),
                self.assertRaises(ValueError),
            ):
                recording.run_recorded_mujoco(
                    {**self.record, "recipe": recipe},
                    self.root,
                    {} if gl is None else {"MUJOCO_GL": gl},
                    self.stage,
                    self.parse,
                )
        self.stage.assert_not_called()
        self.assertFalse((self.root / "recorded-eval-case.yaml").exists())

    def test_humanoid_is_admitted_without_task_changes(self):
        self.record["recipe"] = self.config["name"] = "mujoco-humanoid-ppo"
        (self.root / "eval-case.yaml").write_text(json.dumps(self.config))
        (self.root / "checkpoint-policy/resolved_config.yaml").write_text(json.dumps(self.config))
        self.run_replay()
        self.assertEqual(self.record["checks"]["mujoco_recorded_replay"], "PASS")

    def test_native_horizon_change_is_rejected_before_stage(self):
        self.config["policy_evaluation"]["max_steps_per_episode"] = 16
        (self.root / "eval-case.yaml").write_text(json.dumps(self.config))
        with self.assertRaisesRegex(AssertionError, "full-native"):
            self.run_replay()
        self.stage.assert_not_called()
        self.assert_not_passed()

    def test_owned_config_is_never_overwritten(self):
        path = self.root / "recorded-eval-case.yaml"
        path.write_text("existing evidence")
        with self.assertRaises(FileExistsError):
            self.run_replay()
        self.assertEqual(path.read_text(), "existing evidence")
        self.stage.assert_not_called()

    def test_task_contract_mutations_in_clone_or_saved_config_are_rejected(self):
        with patch.object(Config, "model_copy", lambda config, **kwargs: config):
            with self.assertRaisesRegex(AssertionError, "task/state/action/shape"):
                self.run_replay()

        def changed_save(config, path):
            data = config.model_dump()
            data["interaction"]["contract"]["action"] = [3]
            path.write_text(json.dumps(data))

        with patch.object(Config, "save", changed_save):
            with self.assertRaisesRegex(AssertionError, "immutable scalar contracts"):
                self.run_replay()
        self.stage.assert_not_called()
        self.assert_not_passed()

    def test_source_file_hash_mismatch_is_rejected_before_stage(self):
        (self.root / "checkpoint-policy/state_dict.pt").write_text("tampered actor")
        with self.assertRaisesRegex(AssertionError, "file hashes"):
            self.run_replay()
        self.stage.assert_not_called()
        self.assert_not_passed()

    def test_scalar_parity_is_exact_and_failure_evidence_is_retained(self):
        changed = metrics()
        changed["eval/rl/return_mean"] += 1e-12
        self.parse.side_effect = None
        self.parse.return_value = changed
        with self.assertRaisesRegex(AssertionError, "original scalar"):
            self.run_replay()
        self.assertEqual(self.record["mujoco_recording"]["recorded_metrics"], changed)
        self.assert_not_passed()

    def test_nonempty_wrong_effective_task_configuration_is_rejected(self):
        self.edit_metadata = lambda data: data["provenance"].update(
            resolved_config={"name": "cartpole-ppo"}
        )
        with self.assertRaisesRegex(AssertionError, "effective configuration"):
            self.run_replay()
        self.assert_not_passed()

    def test_original_stage_error_is_not_retried_or_replaced(self):
        error = RuntimeError("original native failure")
        self.stage.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            self.run_replay()
        self.assertIs(caught.exception, error)
        self.stage.assert_called_once()
        self.assert_not_passed()

    def test_wrong_frame_metadata_provenance_and_metrics_fail(self):
        self.run_replay()
        policy = self.record["mujoco_recording"]["source_policy"]
        evaluation = self.data["evaluation"]
        invalid = [
            ("frame_step", 1024.0),
            ("frame_step", 160),
            ("seeds", [0, 1]),
            ("seeds", [10000.0, 10001.0]),
            ("formats", ["gif", "mp4", "png"]),
            ("captured_frames", 15),
            ("captured_frames", True),
            ("capture_capped", "true"),
            ("captured_size", [128, 128]),
            ("source_size", [0, 600]),
            ("fps", 0),
        ]
        for key, value in invalid:
            data = copy.deepcopy(self.data)
            data[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(AssertionError):
                recording._check_metadata(
                    data, 1024, policy, evaluation, metrics(), self.effective_config
                )
        for key in self.data["provenance"]:
            data = copy.deepcopy(self.data)
            data["provenance"][key] = None
            with self.subTest(provenance=key), self.assertRaises(AssertionError):
                recording._check_metadata(
                    data, 1024, policy, evaluation, metrics(), self.effective_config
                )
        for key, value in (
            ("max_steps_per_episode", 16),
            ("deterministic", False),
            ("deterministic", 1),
            ("episodes", 2.0),
        ):
            data = copy.deepcopy(self.data)
            data["evaluation"][key] = value
            with self.subTest(protocol=key), self.assertRaises(AssertionError):
                recording._check_metadata(
                    data, 1024, policy, evaluation, metrics(), self.effective_config
                )
        data = copy.deepcopy(self.data)
        data["metrics"]["eval/policy/return_mean"] += 1e-12
        with self.assertRaisesRegex(AssertionError, "scalar evaluation"):
            recording._check_metadata(
                data, 1024, policy, evaluation, metrics(), self.effective_config
            )

    def test_missing_nonfinite_and_nonnative_metrics_fail(self):
        for key, value in (
            ("episodes", 1),
            ("native_truncated", 0),
            ("evaluation_horizon_limits", 1),
            ("return_mean", float("nan")),
            ("terminated", 0.5),
            ("return_std", -1),
        ):
            data = metrics()
            data[f"eval/rl/{key}"] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                recording._check_metrics(data)
        data = metrics()
        del data["eval/rl/return_std"]
        with self.assertRaises(AssertionError):
            recording._check_metrics(data)

    def test_wrong_native_artifact_set_fails(self):
        def rename():
            source = self.directory / "frame_000000001024.json"
            source.rename(self.directory / "frame_000000000160.json")

        self.edit_stage = rename
        with self.assertRaisesRegex(AssertionError, "actual-frame"):
            self.run_replay()
        self.assert_not_passed()

    def test_multiple_native_writers_fail(self):
        self.edit_stage = lambda: (self.root / "logs/recorded-tb/other/artifacts/evaluation").mkdir(
            parents=True
        )
        with self.assertRaisesRegex(AssertionError, "exactly one owned"):
            self.run_replay()
        self.assert_not_passed()

    def test_mp4_codec_count_padding_and_decoded_content_fail(self):
        self.run_replay()
        args = (
            self.directory,
            1024,
            self.record["mujoco_recording"]["source_policy"],
            self.data["evaluation"],
            metrics(),
            self.gif_info,
            self.effective_config,
        )
        with patch.dict(sys.modules, self.modules):
            for codec in ("mpeg4", "libx264", None):
                self.reader.get_meta_data.return_value = {"codec": codec}
                with self.subTest(codec=codec), self.assertRaisesRegex(AssertionError, "codec"):
                    recording._media(*args)
            self.reader.get_meta_data.return_value = {"codec": "h264"}
            self.reader.count_frames.return_value = 15
            with self.assertRaisesRegex(AssertionError, "frame count"):
                recording._media(*args)
            self.reader.count_frames.return_value = 16
            original = self.frames
            for frames in (
                [original[0]] * 16,
                [original[0] * 0] * 16,
                original[:-1],
                [frame[:85] for frame in original],
            ):
                self.frames = frames
                with self.subTest(count=len(frames)), self.assertRaises(AssertionError):
                    recording._media(*args)

    def test_strict_json_rejects_duplicate_and_nonfinite_values(self):
        path = self.root / "invalid.json"
        for payload in (
            '{"frame_step": 0, "frame_step": 1024}',
            '{"fps": NaN}',
            '{"fps": Infinity}',
            '{"fps": 1e999}',
        ):
            path.write_text(payload)
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                recording._strict_json(path)

    def test_source_policy_cannot_mutate_during_replay(self):
        self.edit_stage = lambda: (self.root / "checkpoint-policy/state_dict.pt").write_text("new")
        with self.assertRaisesRegex(AssertionError, "file hashes"):
            self.run_replay()
        self.assert_not_passed()


if __name__ == "__main__":
    unittest.main()
