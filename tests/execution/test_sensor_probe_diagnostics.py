"""Hermetic diagnostics tests: AST-load only the two helpers, never the native probe."""

import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image


SOURCE = Path(__file__).resolve().parents[1] / "fixtures" / "isaaclab_sensors_probe.py"
HELPERS = {"_snapshot", "_save_diagnostics"}
tree = ast.parse(SOURCE.read_text())
helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in HELPERS]
assert {node.name for node in helpers} == HELPERS
tree.body = list(helpers)
namespace = {}
exec(compile(tree, str(SOURCE), "exec"), namespace)
snapshot = namespace["_snapshot"]
save_diagnostics = namespace["_save_diagnostics"]


class Tensor:
    def __init__(self, array):
        self.array = array
        self.calls = []

    def detach(self):
        self.calls.append("detach")
        return self

    def cpu(self):
        self.calls.append("cpu")
        return self

    def numpy(self):
        self.calls.append("numpy")
        return self.array


class Spec:
    def __init__(self, data):
        self.data = data

    def model_dump(self, *, mode):
        assert mode == "json"
        return copy.deepcopy(self.data)


def inputs(owner):
    specs = {}
    raw = {}
    processed = {}
    for key, dtype, raw_shape, shape, units in (
        ("rgb", "uint8", [32, 32, 3], [6, 32, 32], "uint8_sRGB"),
        ("depth", "float32", [32, 32, 1], [4, 32, 32], "m"),
        ("segmentation", "int32", [32, 32, 1], [6, 32, 32], "class_id"),
    ):
        sensor = {
            "owner": owner,
            "raw_dtype": dtype,
            "raw_shape": raw_shape,
            "raw_layout": "HWC",
            "units": units,
            "history_length": 2,
            "history_order": "oldest_first",
            "initialization": "repeat_first",
            "timing": "post_control",
        }
        if key == "depth":
            sensor.update(invalid_depth="mask_zero", depth_representation="z_depth")
        if key == "segmentation":
            sensor["labels"] = [[0, "BACKGROUND"], [1, "UNLABELLED"], [2, "cartpole"]]
        specs[key] = Spec({"dtype": "float32", "shape": shape, "sensor": sensor})
        raw[key] = np.arange(2 * np.prod(raw_shape)).reshape(2, *raw_shape).astype(dtype)
        processed[key] = np.zeros((2, *shape), dtype="float32")
    specs["state"] = Spec({"dtype": "float32", "shape": [4], "sensor": None})
    raw["state"] = np.zeros((2, 4), dtype="float32")
    processed["state"] = raw["state"].copy()
    definition = SimpleNamespace(
        task="NexuML-Cartpole-Sensors",
        physics="newton_mjwarp",
        describe=lambda: SimpleNamespace(observations=specs),
    )
    samples = {
        f"{stage}_{kind}": {key: value.copy() for key, value in batch.items()}
        for stage in ("initial", "final", "reset")
        for kind, batch in (("raw", raw), ("processed", processed))
    }
    return definition, samples


class SensorProbeDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="sensor-diagnostics-")
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "owned"
        self.definition, self.samples = inputs("package")

    def test_snapshot_freezes_independent_cpu_arrays(self):
        batch = {key: Tensor(value) for key, value in self.samples["initial_raw"].items()}
        frozen = snapshot(batch)
        for key, tensor in batch.items():
            self.assertEqual(tensor.calls, ["detach", "cpu", "numpy"])
            expected = tensor.array.copy()
            self.assertFalse(np.shares_memory(frozen[key], tensor.array))
            tensor.array.fill(99)
            np.testing.assert_array_equal(frozen[key], expected)
        batch.clear()
        self.assertEqual(set(frozen), {"rgb", "depth", "segmentation", "state"})

    def test_bounded_json_npz_and_unaltered_png_roundtrip(self):
        for owner in ("package", "external"):
            with self.subTest(owner=owner):
                definition, samples = inputs(owner)
                output = self.output.with_name(owner)
                save_diagnostics(output, owner, definition, samples)
                metadata = json.loads((output / "sensor-metadata.json").read_text())
                self.assertEqual(
                    metadata["scope"],
                    ("controlled native sensor oracle; not policy behavior/learning/viewer proof"),
                )
                self.assertEqual(metadata["owner"], owner)
                self.assertEqual(metadata["task"], definition.task)
                self.assertEqual(metadata["physics"], definition.physics)
                self.assertEqual((metadata["seed"], metadata["num_envs"]), (0, 2))
                self.assertEqual(metadata["selected_reset_mask"], [True, False])
                self.assertEqual(metadata["source_size"], [32, 32])
                self.assertEqual(metadata["captured_size"], [32, 32])
                self.assertEqual(set(metadata["stage_labels"]), {"initial", "final", "reset"})
                self.assertIn("offscreen", metadata["stage_labels"]["final"])
                self.assertIn("not renderer-pass evidence", metadata["stage_labels"]["final"])
                self.assertEqual(
                    metadata["observation_specs"],
                    {key: spec.data for key, spec in definition.describe().observations.items()},
                )
                expected = {
                    f"{stage}__{key}": value
                    for stage, batch in samples.items()
                    for key, value in batch.items()
                }
                with np.load(output / "sensor-snapshots.npz", allow_pickle=False) as arrays:
                    self.assertEqual(set(arrays.files), set(expected))
                    self.assertEqual(set(metadata["arrays"]), set(expected))
                    for name, value in expected.items():
                        self.assertFalse(arrays[name].dtype.hasobject)
                        self.assertEqual(arrays[name].dtype, value.dtype)
                        np.testing.assert_array_equal(arrays[name], value)
                        self.assertEqual(
                            metadata["arrays"][name],
                            {
                                "dtype": str(value.dtype),
                                "shape": list(value.shape),
                                "nbytes": value.nbytes,
                            },
                        )
                self.assertEqual(metadata["array_bytes"], sum(v.nbytes for v in expected.values()))
                self.assertLessEqual(metadata["array_bytes"], 2**20)
                for stage in ("initial", "final", "reset"):
                    with Image.open(output / f"{stage}-rgb-env0.png") as image:
                        self.assertEqual(image.format, "PNG")
                        self.assertEqual(image.size, (32, 32))
                        np.testing.assert_array_equal(
                            np.asarray(image), samples[f"{stage}_raw"]["rgb"][0]
                        )

    def test_byte_cap_rejected_before_output_creation(self):
        self.samples["final_processed"]["state"] = np.zeros(2**20 + 1, dtype="uint8")
        with self.assertRaisesRegex(AssertionError, "1 MiB"):
            save_diagnostics(self.output, "package", self.definition, self.samples)
        self.assertFalse(self.output.exists())

    def test_object_arrays_rejected_before_output_creation(self):
        self.samples["initial_raw"]["state"] = np.array([object()], dtype=object)
        with self.assertRaises(AssertionError):
            save_diagnostics(self.output, "package", self.definition, self.samples)
        self.assertFalse(self.output.exists())

    def test_non_json_contract_rejected_before_output_creation(self):
        self.definition.describe().observations["depth"].data["low"] = float("nan")
        with self.assertRaises(ValueError):
            save_diagnostics(self.output, "package", self.definition, self.samples)
        self.assertFalse(self.output.exists())

    def test_duplicate_output_does_not_overwrite(self):
        save_diagnostics(self.output, "package", self.definition, self.samples)
        before = {path.name: path.read_bytes() for path in self.output.iterdir()}
        self.samples["initial_raw"]["rgb"].fill(0)
        with self.assertRaises(FileExistsError):
            save_diagnostics(self.output, "external", self.definition, self.samples)
        self.assertEqual({path.name: path.read_bytes() for path in self.output.iterdir()}, before)
