"""Native NexuML held-out FD001 per-engine publication-fidelity diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor

from nexuml.core.components import EvalAlgorithmDefinition, EvalBuildContext
from nexuml.core.discovery import eval_algorithm
from nexuml.evaluation.algorithm import EvalAlgorithm
from nexuml_library.data.maintenance.rul import read_rul_artifact


def _score(error: float) -> float:
    """Compute one engine's published asymmetric NASA penalty.

    Returns:
        NASA asymmetric score component.
    """
    return math.expm1(error / 10.0 if error >= 0 else -error / 13.0)


@eval_algorithm("RULPerEngineAudit")
class RULPerEngineAudit(EvalAlgorithmDefinition):
    """Export per-engine capped/raw residuals using the native test lifecycle.

    Identifiers are recovered from the immutable prepared dataset, using
    `sample_index` propagated by NexuML from each held-out dataset view.
    No test label or test feature participates in model fitting.
    """

    artifact_path: str = Field(min_length=1)
    output_file: str = Field(min_length=1)
    target_scale: float = Field(default=1.0, gt=0)

    def build(self, context: EvalBuildContext) -> EvalAlgorithm:
        """Create the original-test-engine evaluation audit collector.

        Returns:
            Streaming native NexuML evaluator for held-out predictions.
        """
        return _RULPerEngineAuditRuntime(
            artifact_path=self.artifact_path,
            output_file=self.output_file,
            prediction_key=context.feature_key or "predicted_rul",
            target_scale=self.target_scale,
        )


class _RULPerEngineAuditRuntime(EvalAlgorithm):
    """Match every output sample to official FD001 engine and both target laws."""

    def __init__(
        self,
        *,
        artifact_path: str,
        output_file: str,
        prediction_key: str,
        target_scale: float = 1.0,
    ) -> None:
        self.artifact_path = artifact_path
        self.output_file = Path(output_file).expanduser().resolve()
        self.prediction_key = prediction_key
        self.target_scale = target_scale
        self._test_rows: list[dict[str, Any]] | None = None
        self._payload: dict[str, Any] | None = None
        self._records: dict[int, dict[str, Any]] = {}

    def _load(self) -> None:
        if self._test_rows is not None:
            return
        payload = read_rul_artifact(self.artifact_path)
        if "raw_targets" not in payload:
            raise ValueError("RUL audit needs original uncapped NASA labels")
        rows = [dict(row, artifact_index=index) for index, row in enumerate(payload["rows"])
                if row["split"] == "test"]
        if not rows or len({row["asset_id"] for row in rows}) != len(rows):
            raise ValueError("RUL per-engine audit requires one final window per test asset")
        self._payload = payload
        self._test_rows = rows

    def eval_batch(self, x: TensorDict, y: TensorDict | None) -> None:
        """Record an exact per-engine residual for one native test batch.

        Raises:
            ValueError: If sample provenance, labels or predictions disagree.
        """
        self._load()
        if self._test_rows is None or self._payload is None:
            raise RuntimeError("Missing prepared FD001 audit data")
        if y is None or any(key not in y.keys() for key in ("rul", "rul_raw")):
            raise ValueError("RUL audit requires both capped and original labels")
        if "sample_index" not in x.keys() or self.prediction_key not in x.keys():
            raise ValueError("RUL audit requires native test sample_index and prediction")
        indices = cast(Tensor, x["sample_index"]).detach().cpu().long().view(-1).tolist()
        predictions = cast(Tensor, x[self.prediction_key]).detach().cpu().float()
        capped = cast(Tensor, y["rul"]).detach().cpu().float()
        raw = cast(Tensor, y["rul_raw"]).detach().cpu().float()
        n = len(indices)
        if (
            predictions.shape != (n, 1)
            or capped.shape != (n, 1)
            or raw.shape != (n, 1)
            or not bool(torch.isfinite(predictions).all())
        ):
            raise ValueError("RUL per-engine audit received invalid batch shapes/values")
        for position, local_index in enumerate(indices):
            if not 0 <= local_index < len(self._test_rows):
                raise ValueError("Test sample index does not match prepared artifact")
            if local_index in self._records:
                raise ValueError("Duplicate engine encountered in the native test loader")
            row = self._test_rows[local_index]
            artifact_index = int(row["artifact_index"])
            expected_capped = float(self._payload["targets"][artifact_index, 0])
            expected_raw = float(self._payload["raw_targets"][artifact_index, 0])
            observed_capped = float(capped[position, 0])
            observed_raw = float(raw[position, 0])
            if (
                not math.isclose(expected_capped, observed_capped, abs_tol=1e-4)
                or not math.isclose(expected_raw, observed_raw, abs_tol=1e-4)
            ):
                raise ValueError("Native test labels disagree with immutable asset provenance")
            estimate = float(predictions[position, 0]) * self.target_scale
            capped_cycles = observed_capped * self.target_scale
            error_capped = estimate - capped_cycles
            error_raw = estimate - observed_raw
            self._records[local_index] = {
                "asset_id": row["asset_id"],
                "step": row["step"],
                "predicted_rul": estimate,
                "true_rul_capped": capped_cycles,
                "true_rul_raw": observed_raw,
                "error_capped": error_capped,
                "error_raw": error_raw,
                "nasa_capped": _score(error_capped),
                "nasa_raw": _score(error_raw),
                "target_was_capped": int(observed_raw > capped_cycles),
            }

    def eval_end(self) -> None:
        """Write a deterministic CSV and protocol manifest after native test."""
        self._load()
        if self._test_rows is None or self._payload is None:
            raise RuntimeError("Missing prepared audit data")
        if len(self._records) != len(self._test_rows):
            raise ValueError("Audit saw fewer engines than the official test set")
        fields = (
            "asset_id", "step", "predicted_rul", "true_rul_capped",
            "true_rul_raw", "error_capped", "error_raw", "nasa_capped",
            "nasa_raw", "target_was_capped",
        )
        destination = self.output_file
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(".csv.part")
        with partial.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                sorted(self._records.values(), key=lambda row: int(row["asset_id"].split(":")[-1]))
            )
        partial.replace(destination)

        settings = self._payload["settings"]
        splits = settings["split_assets"]
        if set(splits["train"]) & set(splits["val"]):
            raise ValueError("Engine-level training/validation leakage detected")
        scaler = json.dumps(settings["scaler"], sort_keys=True, separators=(",", ":"))
        manifest = {
            "format": "nexuml-rul-engine-audit-v1",
            "paper": "Goel, Galchar & Kanu (2026), arXiv:2604.27234v1",
            "paper_test_target_convention": "not explicitly specified",
            "test_labels": ["capped", "official_raw"],
            "source_artifact": str(Path(self.artifact_path).resolve()),
            "dataset": settings["dataset"],
            "subset": settings["subset"],
            "seed": settings["seed"],
            "window": settings["window"],
            "rul_cap": settings["rul_cap"],
            "padding_mode": settings.get("padding_mode"),
            "normalization": settings.get("normalization", "zscore"),
            "scaler_fit_scope": "training_assets_only",
            "scaler_sha256": hashlib.sha256(scaler.encode("utf-8")).hexdigest(),
            "split_method": settings.get("split_method", "numpy.default_rng.permutation"),
            "split_assets": splits,
            "test_engines": len(self._test_rows),
            "test_engines_above_cap": sum(
                record["target_was_capped"] for record in self._records.values()
            ),
            "best_checkpoint_selection": "minimum validation loss",
            "csv": str(destination),
        }
        manifest_path = destination.with_suffix(".json")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def results(self) -> dict[str, float]:
        """Report audit completeness without flooding native scalar metric output.

        Returns:
            Number of evaluated held-out engines and of capped terminal offsets.

        Raises:
            ValueError: If evaluation is incomplete.
        """
        if self._test_rows is None or len(self._records) != len(self._test_rows):
            raise ValueError("Incomplete FD001 test-engine audit")
        return {
            "rul_audit_test_count": float(len(self._records)),
            "rul_audit_capped_engine_count": float(
                sum(record["target_was_capped"] for record in self._records.values())
            ),
        }
