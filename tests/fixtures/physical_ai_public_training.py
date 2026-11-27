"""Fixed public-data CLI lifecycle; no manual fitting or invented robot rollout."""

import ast
import hashlib
import math
from pathlib import Path
import re
import runpy
import sys


def model_metrics(path):
    text = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text())
    match = re.search(r"Model evaluation \(test split\):\s*(\{.*?\})", text, re.S)
    if match is None:
        raise AssertionError("CLI did not report held-out model evaluation")
    metrics = ast.literal_eval(match[1])
    if not metrics or not all(
        type(v) in (int, float) and math.isfinite(v) for v in metrics.values()
    ):
        raise AssertionError("model evaluation metrics absent or nonfinite")
    return metrics


def run_workflow(record, root, env, stage):
    import torch

    staging = runpy.run_path(str(Path(__file__).with_name("physical_ai_public_data.py")))
    cli = [sys.executable, "-m", "nexuml.cli.main"]
    scenario = ["--scenario-file", "scripts/so101_public_bc.py"]
    with staging["staged_dataset"]() as (data, manifest):
        private = {
            **env,
            "NEXUML_SO101_DATA_ROOT": str(data),
            "NEXUML_SO101_OUTPUT_ROOT": str(root),
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
        record["public_training"] = {
            "source_uri": staging["SOURCE_URI"],
            "manifest_sha256": staging["MANIFEST_SHA256"],
            "verified_bytes": manifest["total_bytes"],
            "scope": (
                "64-step offline block-stacking CLI smoke; "
                "not native success, convergence or frozen L2"
            ),
        }
        stage(
            record,
            "public-untrained-export",
            [*cli, "export", *scenario, "--output", str(root / "untrained-model")],
            private,
        )
        stage(
            record,
            "public-untrained-evaluate",
            [*cli, "evaluate", str(root / "untrained-model"), *scenario, "--device", "cuda:0"],
            private,
        )
        before = model_metrics(root / "public-untrained-evaluate.log")
        stage(record, "public-train", [*cli, "train", *scenario], private)
        checkpoint = root / "checkpoints/last.ckpt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if saved["global_step"] != 64:
            raise AssertionError("public CLI fit did not complete its real optimizer-step budget")
        stage(
            record,
            "public-checkpoint-export",
            [
                *cli,
                "export",
                *scenario,
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(root / "checkpoint-model"),
            ],
            private,
        )
        stage(
            record,
            "public-trained-evaluate",
            [*cli, "evaluate", str(root / "checkpoint-model"), *scenario, "--device", "cuda:0"],
            private,
        )
        after = model_metrics(root / "public-trained-evaluate.log")
        untrained = torch.load(
            root / "untrained-model/state_dict.pt", map_location="cpu", weights_only=True
        )
        trained = torch.load(
            root / "checkpoint-model/state_dict.pt", map_location="cpu", weights_only=True
        )
        expected = {
            key.removeprefix("pipeline."): value
            for key, value in saved["state_dict"].items()
            if key.startswith("pipeline.")
        }
        if (
            not expected
            or set(expected) != set(trained)
            or any(not torch.equal(expected[key], trained[key]) for key in expected)
        ):
            raise AssertionError(
                "public checkpoint export differs from the complete saved pipeline"
            )
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_sha256:
            raise AssertionError("public checkpoint export mutated the source checkpoint")
        if set(untrained) != set(trained) or not any(
            not torch.equal(untrained[k], trained[k]) for k in trained
        ):
            raise AssertionError("public-data model parameters did not update")
        if not all(torch.isfinite(v).all() for v in trained.values()):
            raise AssertionError("public-data model contains nonfinite weights")
        record["public_training"].update(
            global_step=64,
            parameters_changed=True,
            finite_weights=True,
            untrained_metrics=before,
            trained_metrics=after,
            checkpoint_sha256=checkpoint_sha256,
            exported_state_sha256=hashlib.sha256(
                (root / "checkpoint-model/state_dict.pt").read_bytes()
            ).hexdigest(),
            complete_checkpoint_state_match=True,
        )
    record["checks"]["public_so101_cli"] = "PASS"
