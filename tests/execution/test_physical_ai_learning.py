"""Hermetic L1 fixture checks: no application, learner, native runtime or Ray imports."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import runpy
import stat
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "fixtures" / "physical_ai_learning.py"
NAMESPACE = runpy.run_path(str(SCRIPT))
FRAMES = NAMESPACE["FRAMES"]


def metrics(return_mean=0.5, length_mean=30, terminated=32):
    return {
        "eval/rl/episodes": 32.0,
        "eval/rl/length_mean": length_mean,
        "eval/rl/return_mean": return_mean,
        "eval/rl/terminated": float(terminated),
        "eval/rl/truncated": float(32 - terminated),
        "eval/rl/native_truncated": float(32 - terminated),
        "eval/rl/evaluation_horizon_limits": 0.0,
    }


def stage_metrics():
    result = {label: metrics() for label in FRAMES}
    result["final"] = metrics(return_mean=4.0, length_mean=285, terminated=1)
    return result


class LearningMetricsTest(unittest.TestCase):
    def setUp(self):
        self.check = NAMESPACE["check_learning_metrics"]

    def test_exact_threshold_boundaries_pass_without_thresholding_earlier_stages(self):
        self.assertTrue(all(self.check(stage_metrics()).values()))

    def test_native_horizon_upper_boundary_and_simultaneous_end_flags_pass(self):
        values = stage_metrics()
        values["final"]["eval/rl/length_mean"] = 300
        values["final"]["eval/rl/truncated"] = 32
        values["final"]["eval/rl/native_truncated"] = 32
        self.assertTrue(all(self.check(values).values()))

    def test_each_final_threshold_and_strict_improvement(self):
        for key, value in (("return_mean", 3.999), ("length_mean", 284.999), ("terminated", 2)):
            with self.subTest(key=key):
                values = stage_metrics()
                values["final"][f"eval/rl/{key}"] = value
                if key == "terminated":
                    values["final"]["eval/rl/truncated"] = 30
                    values["final"]["eval/rl/native_truncated"] = 30
                with self.assertRaisesRegex(AssertionError, "frozen L1 quality failed"):
                    self.check(values)
        for baseline in (4.0, 4.1):
            with self.subTest(baseline=baseline):
                values = stage_metrics()
                values["untrained"]["eval/rl/return_mean"] = baseline
                with self.assertRaisesRegex(AssertionError, "improves_over_matched_untrained"):
                    self.check(values)

    def test_missing_keys_nonfinite_values_and_episode_counts_in_every_stage(self):
        for label in FRAMES:
            for key in NAMESPACE["_METRICS"]:
                with self.subTest(stage=label, missing=key):
                    values = stage_metrics()
                    del values[label][f"eval/rl/{key}"]
                    with self.assertRaises(AssertionError):
                        self.check(values)
                for bad in (float("nan"), float("inf"), -float("inf"), "32", True):
                    with self.subTest(stage=label, key=key, bad=bad):
                        values = stage_metrics()
                        values[label][f"eval/rl/{key}"] = bad
                        with self.assertRaises(AssertionError):
                            self.check(values)
            for count in (0, 31, 33, 31.5):
                with self.subTest(stage=label, episodes=count):
                    values = stage_metrics()
                    values[label]["eval/rl/episodes"] = count
                    with self.assertRaises(AssertionError):
                        self.check(values)

    def test_invalid_counts_lengths_and_evaluator_capture_limit_rejected(self):
        for key in ("terminated", "truncated", "native_truncated", "evaluation_horizon_limits"):
            for bad in (-1, 33, 0.5):
                with self.subTest(key=key, bad=bad):
                    values = stage_metrics()
                    values["early"][f"eval/rl/{key}"] = bad
                    with self.assertRaises(AssertionError):
                        self.check(values)
        for changes in (
            {"length_mean": 301},
            {"length_mean": 0},
            {"evaluation_horizon_limits": 1},
            {"native_truncated": 1},
            {"terminated": 0, "truncated": 0, "native_truncated": 0},
        ):
            with self.subTest(changes=changes):
                values = stage_metrics()
                values["intermediate"].update(
                    {f"eval/rl/{key}": value for key, value in changes.items()}
                )
                with self.assertRaises(AssertionError):
                    self.check(values)
        values = stage_metrics()
        del values["early"]
        with self.assertRaises(AssertionError):
            self.check(values)


class FakeConfig(SimpleNamespace):
    def model_copy(self, *, deep):
        assert deep is True
        return copy.deepcopy(self)

    def save(self, path):
        # Only the fixture's serialization calls matter; never import a YAML/application library.
        Path(path).write_text("mock canonical/prepared configuration\n")


class LearningWorkerTest(unittest.TestCase):
    def setUp(self):
        self.run_learning = NAMESPACE["run_learning"]
        temporary = tempfile.TemporaryDirectory(prefix="l1-hermetic-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "attempt with spaces"
        self.root.mkdir()
        self.record: dict = {
            "gates": {key: "UNVERIFIED" for key in "CVLRD"},
            "gate_scope": {key: f"unchanged-{key}" for key in "CVLRD"},
        }
        self.env = {"CUDA_VISIBLE_DEVICES": "", "TMPDIR": str(self.root)}
        self.config = FakeConfig(
            name="isaac-cartpole-learning-ppo",
            policy_evaluation=None,
            pipeline={"canonical": "unchanged"},
            interaction=SimpleNamespace(
                environment=SimpleNamespace(
                    physics="newton_mjwarp",
                    viewer=None,
                    native_horizon="unchanged",
                )
            ),
            training=SimpleNamespace(
                seed=0,
                total_frames=2_097_152,
                frames_per_batch=8192,
                algorithm={"canonical": "unchanged"},
                collector=SimpleNamespace(num_collectors=1, num_envs_per_collector=512),
                evaluation=SimpleNamespace(
                    episodes=0,
                    at_start=False,
                    every_n_frames=None,
                    video=SimpleNamespace(enabled=False),
                ),
            ),
        )
        self.original = copy.deepcopy(vars(self.config))
        self.factory = mock.Mock(return_value=self.config)
        self.registry = mock.Mock()
        self.registry.get.return_value = self.factory
        self.from_scenario = mock.Mock(side_effect=lambda scenario: scenario)
        self.evaluation_spec = mock.Mock(side_effect=lambda data: SimpleNamespace(**data))
        modules = {
            name: ModuleType(name)
            for name in (
                "nexuml",
                "nexuml.core",
                "nexuml.core.config",
                "nexuml.core.scenario_registry",
                "nexuml.core.types",
            )
        }
        setattr(
            modules["nexuml.core.config"],
            "ResolvedConfig",
            SimpleNamespace(from_scenario=self.from_scenario),
        )
        setattr(
            modules["nexuml.core.scenario_registry"], "get_scenario_registry", lambda: self.registry
        )
        setattr(
            modules["nexuml.core.types"],
            "PolicyEvaluationSpec",
            SimpleNamespace(model_validate=self.evaluation_spec),
        )
        patch = mock.patch.dict(sys.modules, modules)
        patch.start()
        self.addCleanup(patch.stop)
        self.prepare = mock.Mock(side_effect=self.prepare_config)
        self.stage = mock.Mock(side_effect=self.run_stage)
        self.reader = mock.Mock(
            side_effect=lambda path: (None, *json.loads(Path(path).read_text()))
        )
        self.validate = mock.Mock(
            return_value={"actor_changed": True, "finite_complete_actor": True}
        )
        self.parse = mock.Mock(side_effect=self.parse_metrics)
        self.calls = []
        self.train_exited = False
        self.evaluations = stage_metrics()
        self.failed_stage = None
        self.saved_counters = [
            ("z-first.ckpt", 8192, 1),
            ("a-duplicate.ckpt", 8192, 1),
            ("middle.ckpt", 1_048_576, 128),
            ("last.ckpt", 2_097_152, 256),
        ]

    def prepare_config(self, path, root, options):
        self.assertTrue(path.is_file())
        self.assertEqual(
            (path, root, options),
            (
                self.root / "learning-input.yaml",
                self.root,
                {"recipe": "isaac-cartpole-learning-ppo"},
            ),
        )
        self.config.save(root / "case.yaml")
        return self.config

    def run_stage(self, record, name, argv, env):
        protocol_path = self.root / "learning-protocol.json"
        protocol = json.loads(protocol_path.read_text())
        self.assertEqual(
            protocol["checkpoint_frames"], {key: value for key, value in FRAMES.items() if value}
        )
        self.assertEqual(stat.S_IMODE(protocol_path.stat().st_mode), 0o400)
        self.assertIs(record, self.record)
        self.assertIs(env, self.env)
        self.assertEqual(argv[:3], [sys.executable, "-m", "nexuml.cli.main"])
        self.assertNotEqual(record["gates"]["L"], "PASS")
        self.calls.append((name, argv))
        if name == self.failed_stage:
            raise AssertionError("mock CLI stage failed")
        if name == "learning-train":
            self.assertEqual(argv[3:], ["train", "--config", str(self.root / "case.yaml")])
            directory = self.root / "checkpoints"
            directory.mkdir()
            for filename, frames, step in self.saved_counters:
                (directory / filename).write_text(json.dumps([frames, step]))
            self.train_exited = True
        if argv[3] == "evaluate":
            self.assertTrue(self.train_exited, "evaluation overlapped training native lifetime")
            self.assertEqual(
                argv[-4:],
                ["--config", str(self.root / "heldout-config.yaml"), "--device", "cuda:0"],
            )
        return {"status": "PASS", "exit_code": 0, "cleanup": True}

    def parse_metrics(self, path, *, expected_frames):
        self.assertTrue(self.train_exited)
        label = path.name.removeprefix("learning-").removesuffix("-evaluate.log")
        self.assertEqual(expected_frames, FRAMES[label])
        self.assertIn(f"learning-{label}-evaluate", [name for name, _ in self.calls])
        return self.evaluations[label]

    def execute_case(self):
        self.run_learning(
            self.record,
            self.root,
            self.env,
            self.prepare,
            self.stage,
            self.reader,
            self.validate,
            self.parse,
        )

    def test_import_is_stdlib_only(self):
        imported = []
        original = __import__

        def capture(name, *args, **kwargs):
            imported.append(name)
            return original(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=capture):
            runpy.run_path(str(SCRIPT))
        self.assertFalse(set(name.split(".")[0] for name in imported) - sys.stdlib_module_names)

    def test_frozen_protocol_serial_cli_actual_checkpoints_and_metric_source_frames(self):
        self.execute_case()
        self.registry.get.assert_called_once_with("isaac-cartpole-learning-ppo")
        self.factory.assert_called_once_with()
        self.from_scenario.assert_called_once_with(self.config)
        self.assertEqual(vars(self.config), self.original)
        self.evaluation_spec.assert_called_once_with(
            {
                "episodes": 32,
                "seed": 10000,
                "deterministic": True,
                "max_steps_per_episode": None,
                "video": {"enabled": False},
                "at_start": False,
                "every_n_frames": None,
            }
        )
        evidence = self.record["learning"]
        self.assertEqual(evidence["protocol"]["native_horizon_steps"], 300)
        self.assertEqual(evidence["protocol"]["native_horizon_seconds"], 5)
        self.assertEqual(evidence["protocol"]["training_seed_count"], 1)
        self.assertEqual(evidence["protocol"]["heldout_last_seed"], 10031)
        self.assertEqual(
            evidence["protocol_sha256"],
            hashlib.sha256((self.root / "learning-protocol.json").read_bytes()).hexdigest(),
        )
        for label, frames in list(FRAMES.items())[1:]:
            saved = evidence["checkpoints"][label]
            destination = self.root / "stages" / f"{label}.ckpt"
            self.assertEqual(saved["path"], str(destination))
            self.assertEqual(saved["frames"], frames)
            self.assertEqual(saved["sha256"], hashlib.sha256(destination.read_bytes()).hexdigest())
            self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o400)
            self.assertEqual(destination.read_bytes(), Path(saved["source"]).read_bytes())
            argv = dict(self.calls)[f"learning-{label}-export"]
            self.assertEqual(argv[argv.index("--checkpoint") + 1], str(destination))
        self.assertTrue(evidence["checkpoints"]["early"]["source"].endswith("a-duplicate.ckpt"))
        self.validate.assert_called_once_with(
            self.root, self.root / "stages/final.ckpt", FRAMES["final"]
        )
        self.assertEqual(
            {key: value["source_frames"] for key, value in evidence["metrics"].items()}, FRAMES
        )
        self.assertEqual(self.parse.call_count, 4)
        self.assertEqual(self.calls[0][0], "learning-untrained-export")
        self.assertEqual(self.calls[1][0], "learning-train")
        self.assertEqual(self.record["gates"]["L"], "PASS")
        self.assertIn("single-training-seed scalar quality", self.record["gate_scope"]["L"])
        for key in "CVRD":
            self.assertEqual(self.record["gates"][key], "UNVERIFIED")
            self.assertEqual(self.record["gate_scope"][key], f"unchanged-{key}")

    def test_stage_failure_never_marks_pass_or_continues(self):
        self.failed_stage = "learning-train"
        with self.assertRaisesRegex(AssertionError, "mock CLI stage failed"):
            self.execute_case()
        self.assertEqual(
            [name for name, _ in self.calls], ["learning-untrained-export", "learning-train"]
        )
        self.parse.assert_not_called()
        self.validate.assert_not_called()
        self.assertNotEqual(self.record["gates"]["L"], "PASS")

    def test_late_evaluation_failure_never_marks_pass(self):
        self.failed_stage = "learning-final-evaluate"
        with self.assertRaisesRegex(AssertionError, "mock CLI stage failed"):
            self.execute_case()
        self.assertEqual(self.parse.call_count, 3)
        self.assertNotEqual(self.record["gates"]["L"], "PASS")

    def test_existing_protocol_is_not_overwritten_and_no_child_runs(self):
        path = self.root / "learning-protocol.json"
        path.write_text("existing frozen protocol")
        with self.assertRaises(FileExistsError):
            self.execute_case()
        self.assertEqual(path.read_text(), "existing frozen protocol")
        self.stage.assert_not_called()
        self.assertNotEqual(self.record["gates"]["L"], "PASS")

    def test_existing_checkpoint_is_not_overwritten(self):
        path = self.root / "stages/early.ckpt"
        path.parent.mkdir()
        path.write_text("existing immutable checkpoint")
        with self.assertRaises(FileExistsError):
            self.execute_case()
        self.assertEqual(path.read_text(), "existing immutable checkpoint")
        self.parse.assert_not_called()
        self.assertNotEqual(self.record["gates"]["L"], "PASS")

    def test_metric_failure_never_marks_pass(self):
        self.evaluations["final"]["eval/rl/return_mean"] = 3
        with self.assertRaisesRegex(AssertionError, "frozen L1 quality failed"):
            self.execute_case()
        self.assertNotEqual(self.record["gates"]["L"], "PASS")

    def test_missing_checkpoint_or_nonadvancing_steps_fail_before_evaluation(self):
        for counters in (
            [("early.ckpt", 8192, 1), ("last.ckpt", 2_097_152, 256)],
            [
                ("early.ckpt", 8192, 0),
                ("middle.ckpt", 1_048_576, 128),
                ("last.ckpt", 2_097_152, 256),
            ],
            [("early.ckpt", 8192, 1), ("middle.ckpt", 1_048_576, 1), ("last.ckpt", 2_097_152, 256)],
        ):
            with self.subTest(counters=counters):
                self.root = self.root.parent / f"counters-{len(list(self.root.parent.iterdir()))}"
                self.root.mkdir()
                self.saved_counters = counters
                with self.assertRaises(AssertionError):
                    self.execute_case()
                self.assertNotEqual(self.record["gates"]["L"], "PASS")
                self.parse.assert_not_called()

    def test_drifted_defaults_rejected_before_any_child(self):
        self.config.training.total_frames = 8192
        with self.assertRaisesRegex(AssertionError, "canonical L1 defaults changed"):
            self.execute_case()
        self.stage.assert_not_called()


if __name__ == "__main__":
    unittest.main()
