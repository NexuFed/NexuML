"""Hermetic boundary/orchestration tests; no application, native or browser runs."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "fixtures" / "physical_ai_logging.py"
SPEC = importlib.util.spec_from_file_location("physical_ai_logging", SOURCE)
assert SPEC is not None and SPEC.loader is not None
logging = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(logging)


def metrics():
    return {
        f"eval/rl/{key}": value
        for key, value in {
            "episodes": 2.0,
            "length_mean": 23.0,
            "return_mean": 23.0,
            "terminated": 2.0,
            "truncated": 0.0,
            "native_truncated": 0.0,
            "evaluation_horizon_limits": 0.0,
        }.items()
    }


def metadata(frame=32, *, fresh=False):
    return {
        "frame_step": frame,
        "metrics": metrics(),
        "evaluation": {
            "episodes": 2,
            "seed": 10000,
            "deterministic": True,
            "max_steps_per_episode": None,
            "at_start": not fresh,
            "every_n_frames": None if fresh else 32,
            "video": {"enabled": True, "format": "both", "max_frames": 16, "max_size": 128},
        },
        "seeds": [10000, 10001],
        "fps": 50.0,
        "captured_frames": 16,
        "capture_capped": True,
        "source_size": [600, 400],
        "captured_size": [128, 85],
        "formats": ["gif", "mp4"],
        "provenance": {},
    }


class Model(SimpleNamespace):
    @classmethod
    def model_validate(cls, data):
        return cls(
            **{
                key: cls.model_validate(value) if isinstance(value, dict) else value
                for key, value in data.items()
            }
        )

    def model_dump(self, **kwargs):
        return {
            key: value.model_dump() if isinstance(value, Model) else value
            for key, value in vars(self).items()
        }


class Config(Model):
    snapshots: dict

    def save(self, path):
        self.snapshots[path.name] = copy.deepcopy(self)
        path.write_text(path.name)

    def model_copy(self, **kwargs):
        result = copy.deepcopy(self)
        result.snapshots = self.snapshots
        return result


class LoggingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="logger-unit-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_import_is_stdlib_only(self):
        self.assertEqual(self.root.parent, Path(tempfile.gettempdir()))
        tree = ast.parse(SOURCE.read_text())
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertTrue(
                    all(name.name.split(".")[0] in sys.stdlib_module_names for name in node.names)
                )
            elif isinstance(node, ast.ImportFrom):
                assert node.module is not None
                self.assertIn(node.module.split(".")[0], sys.stdlib_module_names)

    def test_fixed_settings_and_after_prepare_override(self):
        factory_calls = []
        scenarios = []

        def factory(**kwargs):
            factory_calls.append(kwargs)
            config = Config(name="cartpole-ppo", training=Model(), logging="original", snapshots={})
            scenarios.append(config)
            return config

        def prepare(path, root, options):
            self.assertTrue(path.is_file())
            self.assertEqual(options, {"recipe": "cartpole-ppo"})
            config = scenarios[-1]
            config.logging = "prepare rewrote URI"
            return config

        modules = {
            "nexuml.core.config": {
                "ResolvedConfig": SimpleNamespace(from_scenario=lambda value: value)
            },
            "nexuml.core.scenario_registry": {
                "get_scenario_registry": lambda: SimpleNamespace(get=lambda name: factory)
            },
            "nexuml.core.types": {
                "LoggingSpec": Model,
                "PolicyEvaluationSpec": Model,
                "ReinforcementLearningSpec": Model,
            },
            "nexuml_library.reinforcement.algorithms.ppo": {"PPO": Model},
        }
        stubs = {}
        for name, contents in modules.items():
            module = ModuleType(name)
            module.__dict__.update(contents)
            stubs[name] = module
        with patch.dict(sys.modules, stubs):
            for mode in logging.MODES:
                owned = self.root / mode
                owned.mkdir()
                result = logging._prepare_mode(mode, owned, prepare)
                config = scenarios[-1]
                training = config.training
                self.assertEqual(factory_calls[-1], {})
                self.assertEqual((training.total_frames, training.frames_per_batch), (64, 32))
                self.assertEqual(
                    (training.seed, training.accelerator, training.devices), (0, "cpu", 1)
                )
                self.assertEqual(
                    training.algorithm.model_dump(), {"update_epochs": 1, "minibatch_size": 16}
                )
                self.assertEqual(
                    (training.evaluation.episodes, training.evaluation.seed), (2, 10000)
                )
                self.assertIsNone(training.evaluation.max_steps_per_episode)
                self.assertTrue(training.evaluation.at_start)
                self.assertEqual(training.evaluation.every_n_frames, 32)
                replay = config.snapshots["eval-case.yaml"].policy_evaluation
                self.assertFalse(replay.at_start)
                self.assertIsNone(replay.every_n_frames)
                self.assertEqual(replay.video.enabled, mode != "disabled")
                self.assertFalse(
                    config.snapshots["scalar-case.yaml"].policy_evaluation.video.enabled
                )
                if mode == "disabled":
                    self.assertIsNone(result["logging"])
                else:
                    self.assertIsNone(config.logging.diagram)
                    self.assertEqual(config.logging.temp_artifact_dir, str(owned / "staging"))
                    if mode in {"mlflow-only", "both"}:
                        self.assertEqual(
                            config.logging.mlflow.tracking_uri,
                            f"sqlite:///{owned / 'logs/mlflow.db'}",
                        )
                        self.assertFalse(config.logging.mlflow.auto_nested_runs)
                        self.assertEqual(
                            config.logging.mlflow.artifact_location,
                            (owned / "logs/mlflow-artifacts").as_uri(),
                        )
        with self.assertRaises(ValueError):
            logging._logging_settings("caller-command", self.root)

    def test_metadata_bounds_and_first_episode_scope(self):
        result = logging._check_metadata("frame_000000000032.json", metadata(), 32)
        self.assertEqual(result["captured_frames"], 16)
        self.assertIn("first episode", result["clip_scope"])
        self.assertEqual(result["source_size"], [600, 400])
        self.assertIn("UNVERIFIED", result["native_reset_terminal_scene_identity"])
        small = metadata()
        small.update(
            captured_frames=4, capture_capped=False, source_size=[64, 32], captured_size=[64, 32]
        )
        logging._check_metadata("frame_000000000032.json", small, 32)
        invalid = (
            ("frame_step", 64),
            ("frame_step", 32.0),
            ("captured_frames", 17),
            ("captured_frames", True),
            ("captured_frames", 0),
            ("captured_frames", 15),
            ("capture_capped", "true"),
            ("source_size", [0, 400]),
            ("source_size", [600]),
            ("captured_size", [129, 86]),
            ("captured_size", [128, 128]),
            ("captured_size", [True, 85]),
            ("seeds", [10000]),
            ("fps", float("nan")),
            ("formats", ["gif"]),
        )
        for key, value in invalid:
            with self.subTest(key=key, value=value), self.assertRaises(AssertionError):
                data = metadata()
                data[key] = value
                logging._check_metadata("frame_000000000032.json", data, 32)
        with self.assertRaises(AssertionError):
            logging._check_metadata("frame_32.json", metadata(), 32)
        missing = metadata()
        del missing["evaluation"]["max_steps_per_episode"]
        with self.assertRaisesRegex(AssertionError, "explicitly recorded"):
            logging._check_metadata("frame_000000000032.json", missing, 32)
        for key, value in (
            ("episodes", 1),
            ("seed", 0),
            ("deterministic", False),
            ("max_steps_per_episode", 16),
            ("at_start", False),
            ("every_n_frames", 16),
        ):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                data = metadata()
                data["evaluation"][key] = value
                logging._check_metadata("frame_000000000032.json", data, 32)

    def test_metrics_not_limited_by_video(self):
        for key, value in (
            ("episodes", 1),
            ("evaluation_horizon_limits", 1),
            ("return_mean", float("inf")),
            ("terminated", 0),
            ("native_truncated", 0.5),
        ):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                data = metrics()
                data[f"eval/rl/{key}"] = value
                logging._check_metrics(data)

    def test_fresh_provenance_must_match_actual_files_frames_and_checkpoint(self):
        identity = {
            "path": str(self.root / "policy"),
            "source_frames": 32,
            "policy_files_sha256": {"state_dict.pt": "actor", "action_adapter_state.pt": "adapter"},
            "checkpoint": {"sha256": "checkpoint"},
            "config_hash": "config",
        }
        data = metadata(fresh=True)
        data["provenance"] = {
            "source_package": identity["path"],
            "source_training_frames": 32,
            "source_policy_files_sha256": identity["policy_files_sha256"],
            "source_checkpoint": identity["checkpoint"],
            "source_config_hash": "config",
            "resolved_config": {"name": "cartpole-ppo"},
        }
        logging._check_metadata("frame_000000000032.json", data, 32, identity)
        for key in data["provenance"]:
            changed = copy.deepcopy(data)
            changed["provenance"][key] = None
            with self.subTest(key=key), self.assertRaises(AssertionError):
                logging._check_metadata("frame_000000000032.json", changed, 32, identity)

    def test_saved_counters_not_filenames_and_no_overwrite(self):
        directory = self.root / "checkpoints"
        directory.mkdir()
        (directory / "final-misleading.ckpt").write_text("[32, 1]")
        (directory / "early-misleading.ckpt").write_text("[64, 2]")

        def checkpoint(path):
            return (None, *json.loads(path.read_text()))

        result = logging._copy_checkpoints(self.root, checkpoint)
        self.assertEqual(result["intermediate"]["frames"], 32)
        self.assertTrue(result["final"]["source"].endswith("early-misleading.ckpt"))
        self.assertEqual(Path(result["final"]["path"]).stat().st_mode & 0o777, 0o400)
        with self.assertRaises(FileExistsError):
            logging._copy_checkpoints(self.root, checkpoint)

    def test_nonfinite_policy_metadata_is_rejected_before_receipt_admission(self):
        (self.root / "metadata.json").write_text('{"checkpoint": {"kth_value": Infinity}}')
        with self.assertRaisesRegex(ValueError, "JSON compliant"):
            logging._policy_identity(self.root, 32)

    def test_checkpoint_step_must_advance(self):
        directory = self.root / "checkpoints"
        directory.mkdir()
        for frame in (32, 64):
            (directory / f"{frame}.ckpt").write_text(json.dumps([frame, 1]))
        with self.assertRaisesRegex(AssertionError, "advance"):
            logging._copy_checkpoints(self.root, lambda path: (None, *json.loads(path.read_text())))

    def test_disabled_never_calls_native_collectors(self):
        with (
            patch.object(logging, "_tensorboard_runs", side_effect=AssertionError("TB imported")),
            patch.object(logging, "_mlflow_runs", side_effect=AssertionError("MLflow imported")),
        ):
            result = logging._collect_mode("disabled", self.root, None, {}, {})
            self.assertEqual(result["writers"], "none")
            for name in (
                "events.out.tfevents.bad",
                "mlflow.db",
                "frame_000000000000.gif",
                "frame_000000000000.mp4",
            ):
                file = self.root / name
                file.touch()
                with self.subTest(name=name), self.assertRaises(AssertionError):
                    logging._collect_mode("disabled", self.root, None, {}, {})
                file.unlink()

    def test_media_checks_use_decoder_frame_count_and_h264(self):
        directory = self.root / "evaluation"
        directory.mkdir()
        for suffix in ("gif", "mp4", "json"):
            path = directory / f"frame_000000000032.{suffix}"
            path.write_text(json.dumps(metadata()) if suffix == "json" else "decoder stub")
        reader = MagicMock()
        reader.__enter__.return_value = reader
        reader.count_frames.return_value = 16
        reader.get_meta_data.return_value = {"codec": "h264"}
        reader.get_data.return_value = SimpleNamespace(shape=(86, 128, 3))
        module = ModuleType("imageio.v2")
        setattr(module, "get_reader", Mock(return_value=reader))
        package = ModuleType("imageio")
        setattr(package, "v2", module)
        with (
            patch.dict(sys.modules, {"imageio": package, "imageio.v2": module}),
            patch.object(logging, "_gif_info", return_value={"frames": 16, "size": [128, 85]}),
        ):
            result = logging._check_media(directory, [32])
            self.assertEqual(result["32"]["mp4"]["frames"], 16)
            reader.count_frames.return_value = 15
            with self.assertRaises(AssertionError):
                logging._check_media(directory, [32])
            reader.count_frames.return_value = 16
            reader.get_meta_data.return_value = {"codec": "mpeg4"}
            with self.assertRaises(AssertionError):
                logging._check_media(directory, [32])
            (directory / "duplicate.json").touch()
            with self.assertRaisesRegex(AssertionError, "exactly one"):
                logging._check_media(directory, [32])

    def test_actual_artifact_sets_identify_main_not_export_runs(self):
        def run(name, frames):
            directory = self.root / name
            directory.mkdir()
            for frame in frames:
                (directory / f"frame_{frame:012d}.json").touch()
            return {
                "run_id": name,
                "evaluation_dir": str(directory),
                "status": "FINISHED",
                "scalar_history": {
                    name: [{"step": frame, "value": value} for frame in frames]
                    for name, value in metrics().items()
                },
            }

        main = run("main", [0, 32, 64])
        runs = [main, *[run(label, [frame]) for label, frame in logging.FRAMES.items()]]
        runs.insert(0, {"run_id": "export", "evaluation_dir": None, "status": "FINISHED"})

        def media(directory, frames, policy=None):
            return {
                str(frame): {"metrics": metrics(), "files": {"gif": {"sha256": str(frame)}}}
                for frame in frames
            }

        with patch.object(logging, "_check_media", side_effect=media):
            classified = logging._classify_runs(
                runs,
                {label: {} for label in logging.FRAMES},
                {label: metrics() for label in logging.FRAMES},
            )
            self.assertIs(classified["main"], main)
            self.assertEqual(len(classified["all_runs"]), 5)
            role = runs[0]["role"]
            assert isinstance(role, str)
            self.assertIn("additional configured run", role)
            main["status"] = "RUNNING"
            with self.assertRaisesRegex(AssertionError, "FINISHED"):
                logging._classify_runs(
                    runs,
                    {label: {} for label in logging.FRAMES},
                    {label: metrics() for label in logging.FRAMES},
                )
            main["status"] = "FINISHED"
            with self.assertRaisesRegex(AssertionError, "one main"):
                logging._classify_runs(
                    [*runs, copy.deepcopy(main)],
                    {label: {} for label in logging.FRAMES},
                    {label: metrics() for label in logging.FRAMES},
                )

    def test_both_compares_all_native_bytes_including_json(self):
        def payload(frame):
            return {
                "files": {
                    suffix: {"sha256": f"{frame}-{suffix}"} for suffix in ("gif", "mp4", "json")
                }
            }

        classified = {
            "main": {"media": {str(frame): payload(frame) for frame in logging.FRAMES.values()}},
            "fresh": {
                label: {"media": {str(frame): payload(frame)}}
                for label, frame in logging.FRAMES.items()
            },
        }
        different = copy.deepcopy(classified)
        different["main"]["media"]["32"]["files"]["json"]["sha256"] = "different provenance"
        with (
            patch.object(logging, "_tensorboard_runs", return_value=[]),
            patch.object(logging, "_mlflow_runs", return_value=[]),
            patch.object(
                logging,
                "_classify_runs",
                side_effect=[classified, copy.deepcopy(classified), classified, different],
            ),
        ):
            result = logging._collect_mode("both", self.root, {"mlflow": {}}, {}, {})
            self.assertEqual(result["same_rollout_payload_bytes"], "PASS")
            with self.assertRaisesRegex(AssertionError, "bytes differ"):
                logging._collect_mode("both", self.root, {"mlflow": {}}, {}, {})

    def test_scalars_require_actual_evaluation_metrics_at_each_physical_frame(self):
        run: dict = {
            "media": {str(frame): {"metrics": metrics()} for frame in logging.FRAMES.values()},
            "scalar_history": {
                name: [{"step": frame, "value": value} for frame in logging.FRAMES.values()]
                for name, value in metrics().items()
            },
        }
        logging._check_run_scalars(run, tensorboard=False)
        rounded = copy.deepcopy(run)
        rounded["scalar_history"]["eval/rl/return_mean"][1]["value"] += 1e-7
        logging._check_run_scalars(rounded, tensorboard=True)
        with self.assertRaises(AssertionError):
            logging._check_run_scalars(rounded, tensorboard=False)
        for change in ("missing", "optimizer-step", "duplicate", "wrong-value"):
            changed = copy.deepcopy(run)
            events = changed["scalar_history"]["eval/rl/return_mean"]
            if change == "missing":
                del changed["scalar_history"]["eval/rl/return_mean"]
            elif change == "optimizer-step":
                events[1]["step"] = 1
            elif change == "duplicate":
                events.append(events[1])
            else:
                events[1]["value"] = 1234
            with self.subTest(change=change), self.assertRaises(AssertionError):
                logging._check_run_scalars(changed, tensorboard=True)

    def test_fresh_previews_must_exist_once_at_the_source_frame(self):
        directory = self.root / "fresh"
        directory.mkdir()
        (directory / "frame_000000000032.json").touch()
        run = {
            "evaluation_dir": str(directory),
            "scalar_history": {},
            "previews": [],
        }
        payload = {"32": {"metrics": metrics(), "files": {"gif": {"sha256": "gif"}}}}
        preview = {"step": 32, "sha256": "gif"}
        with (
            patch.object(logging, "_check_media", return_value=payload),
            patch.object(logging, "_check_run_scalars"),
        ):
            for previews in ([], [preview, preview], [{**preview, "step": 1}]):
                run["previews"] = previews
                with (
                    self.subTest(previews=previews),
                    self.assertRaisesRegex(AssertionError, "image-summary"),
                ):
                    logging._classify_runs(
                        [run], {"intermediate": {}}, {"intermediate": metrics()}, tensorboard=True
                    )

    def _orchestrate(self, *, fail=None, scalar_mismatch=False):
        calls = []
        record = {
            "attempt_root": str(self.root),
            "stages": [],
            "checks": {},
            "gates": {key: "UNVERIFIED" for key in "CVLRD"},
        }

        def prepare(mode, owned, callback):
            for name in ("case.yaml", "eval-case.yaml", "scalar-case.yaml"):
                (owned / name).touch()
            return {"logging": logging._logging_settings(mode, owned), "configs": {}}

        def checkpoint(path):
            return (None, *json.loads(path.read_text()))

        def stage(receipt, name, argv, env):
            calls.append(name)
            if fail is not None and name.endswith(fail):
                raise RuntimeError("original stage failure")
            owned = Path(argv[argv.index("--config") + 1]).parent
            self.assertEqual(env["NEXUML_LOGS_ROOT"], str(owned / "logs"))
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
            operation = argv[3]
            if operation == "train":
                (owned / "checkpoints").mkdir()
                for frame, step in ((32, 1), (64, 2)):
                    (owned / "checkpoints" / f"saved-{step}.ckpt").write_text(
                        json.dumps([frame, step])
                    )
            elif operation == "export":
                policy = Path(argv[argv.index("--output") + 1])
                policy.mkdir()
                saved = (
                    Path(argv[argv.index("--checkpoint") + 1]) if "--checkpoint" in argv else None
                )
                for filename in ("state_dict.pt", "action_adapter_state.pt"):
                    (policy / filename).write_bytes(b"hermetic actor bytes")
                data = {
                    "artifact_kind": "policy",
                    "training_frames": checkpoint(saved)[1] if saved else 0,
                    "policy_files_sha256": {
                        filename: logging._sha256(policy / filename)
                        for filename in ("state_dict.pt", "action_adapter_state.pt")
                    },
                    "config_hash": "hermetic config",
                }
                if saved is not None:
                    data["checkpoint"] = {"sha256": logging._sha256(saved)}
                (policy / "metadata.json").write_text(json.dumps(data))
            log_path = self.root / f"{name}.log"
            log_path.touch()
            result = {
                "name": name,
                "argv": argv,
                "log_path": str(log_path),
                "status": "PASS",
                "exit_code": 0,
            }
            receipt["stages"].append(result)
            return result

        def evaluation(path, expected_frames):
            self.assertIn(expected_frames, logging.FRAMES.values())
            result = metrics()
            if scalar_mismatch and "scalar" in path.name:
                result["eval/rl/return_mean"] += 1
            return result

        collect = logging._collect_mode

        def collector(mode, owned, settings, policies, values):
            if mode == "disabled":
                return collect(mode, owned, settings, policies, values)
            return {"hermetic_stub": True}

        with (
            patch.object(logging, "_prepare_mode", side_effect=prepare),
            patch.object(logging, "_collect_mode", side_effect=collector),
        ):
            try:
                logging.run_logging(
                    record,
                    self.root,
                    {},
                    None,
                    stage,
                    checkpoint,
                    lambda root, path, total: {"actor_changed": True},
                    evaluation,
                )
            except BaseException:
                self.assertNotIn("logger_payloads", record["checks"])
                self.assertFalse((self.root / "logging-manifest.json").exists())
                self.assertEqual(set(record["gates"].values()), {"UNVERIFIED"})
                self.calls = calls
                raise
        return record, calls

    def test_all_modes_serial_cli_manifest_without_gate_claims(self):
        record, calls = self._orchestrate()
        self.assertEqual(len(calls), 32)
        self.assertEqual(calls[:2], ["logging-tb-only-untrained-export", "logging-tb-only-train"])
        manifest = self.root / "logging-manifest.json"
        self.assertEqual(record["logging"]["manifest_sha256"], logging._sha256(manifest))
        self.assertEqual(record["checks"], {"logger_payloads": "PASS"})
        self.assertEqual(set(record["gates"].values()), {"UNVERIFIED"})
        data = json.loads(manifest.read_text())
        self.assertEqual(list(data["modes"]), list(logging.MODES))
        self.assertEqual(data["modes"]["disabled"]["native"]["writers"], "none")
        for evidence in data["modes"].values():
            self.assertEqual(len(evidence["commands"]), 8)
            self.assertTrue(evidence["capture_metric_parity"])
            self.assertEqual(evidence["checkpoints"]["final"]["frames"], 64)

    def test_original_training_failure_stops_before_exports_and_later_modes(self):
        with self.assertRaisesRegex(RuntimeError, "original stage failure"):
            self._orchestrate(fail="-train")
        self.assertEqual(self.calls, ["logging-tb-only-untrained-export", "logging-tb-only-train"])

    def test_metric_parity_failure_stops_without_pass_or_later_modes(self):
        with self.assertRaisesRegex(AssertionError, "recording changed"):
            self._orchestrate(scalar_mismatch=True)
        self.assertEqual(len(self.calls), 8)


if __name__ == "__main__":
    unittest.main()
