"""Post-training, test-sealed diagnostics for NEX-233 FD003 CfC cluster gates.

Uses the validation-selected Lightning checkpoint and deterministic, engine-
balanced train/validation sensor windows. No fitting, test inference, checkpoint
modification or dependence on local .experiments symlinks occurs.

Run: python -m nexuml_library.evaluation.maintenance.rul_cluster_gate_audit
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

from nexuml_library.data.maintenance.rul import read_rul_artifact
from nexuml_library.data.maintenance.rul_cluster_conditions import (
    read_cluster_artifact,
)
from nexuml_library.layers.model.rul_advanced import _PatchFormerModel
from nexuml_library.layers.model.rul_enhanced_cfc import _SensorGraphAttention

Arms = Literal["reference", "working", "failure", "both"]
GateKind = Literal["working", "failure"]
ARMS: tuple[Arms, ...] = ("reference", "working", "failure", "both")
SEEDS = (42, 43, 44)
PREPARED_NAME = "cmapss-FD003-seed42-controlled-w30-s42.pt"
CLUSTER_NAME = "cmapss-FD003-seed42-cfc-conditional-clusters-v1.pt"
OUTPUT_DIRECTORY = "rul/enhanced-cfc/cluster-conditioned/fd003/gate-audit-v1"


def _sha256(path: Path) -> str:
    """Hash an immutable input for reproducible source attribution.

    Returns:
        Hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_checkpoint(
    root: Path, arm: Arms, seed: int
) -> tuple[Path, int, dict[str, float]]:
    """Resolve a validation-selected checkpoint through its native audit record.

    Kubernetes records checkpoint paths from inside Pods; the selected filename
    is remapped into the same run directory on the shared Prisma volume.

    Returns:
        Selected checkpoint, its zero-based epoch and matching evaluation curve.

    Raises:
        ValueError: If the audit, checkpoint or selected curve is inconsistent.
        FileNotFoundError: If a required immutable run artifact is missing.
    """
    run = f"rul-FD003-enhanced-cfc-cluster-{arm}-s{seed}"
    audit = root / "rul/audits" / f"{run}.checkpoint.json"
    if not audit.is_file():
        raise FileNotFoundError(f"Missing validation-selection audit: {audit}")
    payload = json.loads(audit.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("selection") != "best minimum val/loss":
        raise ValueError(f"Unexpected checkpoint selection policy: {audit}")
    best = payload.get("best_model_path")
    epoch = payload.get("checkpoint_epoch")
    if (
        not isinstance(best, str)
        or Path(best).parent.name != run
        or Path(best).suffix != ".ckpt"
        or not isinstance(epoch, int)
        or isinstance(epoch, bool)
        or epoch < 0
    ):
        raise ValueError(f"Invalid validation-selected checkpoint audit: {audit}")
    checkpoint = root / "rul/checkpoints" / run / Path(best).name
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing selected checkpoint on Prisma: {checkpoint}")
    curve = (
        root
        / "rul/enhanced-cfc/cluster-conditioned/fd003"
        / f"{run}.curves.csv"
    )
    with curve.open(newline="", encoding="utf-8") as handle:
        matches = [
            row for row in csv.DictReader(handle) if int(row["epoch"]) == epoch
        ]
    if len(matches) != 1:
        raise ValueError(f"Selected epoch missing or duplicated in curve: {curve}")
    fields = ("train_eval_mse", "val_eval_mse", "mse_gap", "val_group_max_mse")
    values = {key: float(matches[0][key]) for key in fields}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"Nonfinite selected-checkpoint diagnostics: {curve}")
    return checkpoint, epoch, values


def _model_state(checkpoint: Path, arm: Arms, epoch: int) -> dict[str, Tensor]:
    """Read the trusted local Lightning checkpoint and isolate the CfC head.

    The stage key is discovered rather than assuming a particular NexuML
    ModuleDict naming convention. A missing or ambiguous head is fatal.

    Returns:
        Model-layer state with the leading compiled-pipeline prefix stripped.

    Raises:
        ValueError: If the checkpoint cannot be unambiguously identified.
    """
    # Only load locally trusted checkpoints produced by our own NexuML Jobs.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        not isinstance(saved, dict)
        or saved.get("epoch") != epoch
        or not isinstance(saved.get("state_dict"), dict)
    ):
        raise ValueError(f"Invalid selected Lightning checkpoint: {checkpoint}")
    state = cast(dict[str, Tensor], saved["state_dict"])
    anchor = "encoder.embedding.0.weight"
    prefixes = [key[: -len(anchor)] for key in state if key.endswith(anchor)]
    if len(prefixes) != 1:
        raise ValueError("Expected exactly one PatchFormer/CfC model layer in checkpoint")
    prefix = prefixes[0]
    if "RULModel" not in prefix:
        raise ValueError(f"Unexpected checkpoint model-stage prefix: {prefix}")
    model = {
        key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)
    }
    if "reference_mean" not in model or "reference_std" not in model:
        raise ValueError("Missing trained PatchFormer normalization buffers")
    enabled = {
        "reference": (),
        "working": ("working",),
        "failure": ("failure",),
        "both": ("working", "failure"),
    }[arm]
    for kind in ("working", "failure"):
        present = any(key.startswith(f"{kind}_gate.") for key in model)
        if present != (kind in enabled):
            raise ValueError(f"Checkpoint gates do not match {arm} configuration")
    return model


def _balanced_windows(
    rows: list[dict[str, Any]], per_engine: int
) -> dict[str, dict[str, list[int]]]:
    """Select evenly spaced historical windows without visiting test engines.

    Returns:
        Per-split, per-engine prepared tensor indices in time order.

    Raises:
        ValueError: If train/validation splits violate the 80/20 engine protocol.
    """
    if per_engine < 2:
        raise ValueError("Need at least two windows per engine")
    all_engines: dict[str, dict[str, list[tuple[int, int]]]] = {
        "train": defaultdict(list),
        "val": defaultdict(list),
    }
    for index, row in enumerate(rows):
        split = row["split"]
        if split in all_engines:
            all_engines[split][str(row["asset_id"])].append((int(row["step"]), index))
    if len(all_engines["train"]) != 80 or len(all_engines["val"]) != 20:
        raise ValueError("Expected FD003 80 train and 20 validation engines")
    if set(all_engines["train"]) & set(all_engines["val"]):
        raise ValueError("Train/validation engine identities overlap")
    result: dict[str, dict[str, list[int]]] = {"train": {}, "val": {}}
    for split, engines in all_engines.items():
        for engine, positions in sorted(engines.items()):
            ordered = [index for _, index in sorted(positions)]
            if len(ordered) < 2:
                raise ValueError(f"Too few history windows for {engine}")
            count = min(per_engine, len(ordered))
            sampled = torch.linspace(0, len(ordered) - 1, count).round().long().tolist()
            indices = [ordered[position] for position in sampled]
            if len(indices) != len(set(indices)):
                raise ValueError("Repeated index in balanced window sample")
            result[split][engine] = indices
    return result


@dataclass
class _EngineGateStats:
    """Per-engine aggregates, preventing long-lived engines from dominating."""

    probabilities: Tensor
    hard_counts: Tensor
    entropy: float = 0.0
    confidence: float = 0.0
    projection_norm: float = 0.0
    projection_to_base_ratio: float = 0.0
    count: int = 0

    @classmethod
    def empty(cls, clusters: int) -> _EngineGateStats:
        """Construct stable CPU accumulators.

        Returns:
            Empty per-engine sufficient statistics.
        """
        return cls(torch.zeros(clusters), torch.zeros(clusters, dtype=torch.long))

    def append(self, probabilities: Tensor, contribution: Tensor, base_norm: Tensor) -> None:
        """Accumulate finite per-window probabilities and representation magnitudes.

        Raises:
            ValueError: If the gate emits invalid probabilities or magnitudes.
        """
        probs = probabilities.detach().to(device="cpu", dtype=torch.float32)
        size = float(contribution.detach().float().norm().cpu())
        baseline = float(base_norm.detach().float().cpu())
        if (
            probs.ndim != 1
            or probs.numel() != self.probabilities.numel()
            or not bool(torch.isfinite(probs).all())
            or not math.isfinite(size)
            or not math.isfinite(baseline)
            or baseline <= 0
            or not math.isclose(float(probs.sum()), 1.0, abs_tol=1e-5)
        ):
            raise ValueError("Invalid checkpoint gate output")
        self.probabilities += probs
        self.hard_counts[int(probs.argmax())] += 1
        self.entropy += float(-(probs * probs.clamp_min(1e-12).log()).sum())
        self.confidence += float(probs.max())
        self.projection_norm += size
        self.projection_to_base_ratio += size / baseline
        self.count += 1


def _summarize(
    engines: dict[str, _EngineGateStats],
    assignments: dict[str, list[int]] | None,
    kind: GateKind,
) -> dict[str, Any]:
    """Summarize engine-balanced occupancy, uncertainty and teacher agreement.

    Training-only pseudo-labels quantify agreement, not external fault-mode
    accuracy. Validation receives no oracle labels.

    Returns:
        JSON-compatible aggregate statistics for one gate and split.

    Raises:
        ValueError: If engine observations or train-only labels are incomplete.
    """
    if not engines or any(item.count == 0 for item in engines.values()):
        raise ValueError("Every audited engine must have observations")
    classes = next(iter(engines.values())).probabilities.numel()
    means = torch.stack([item.probabilities / item.count for item in engines.values()])
    predicted = means.argmax(dim=1).tolist()
    dominant = torch.bincount(means.argmax(dim=1), minlength=classes).tolist()
    window_counts = torch.stack([item.hard_counts for item in engines.values()]).sum(0)
    total = sum(item.count for item in engines.values())
    soft = means.mean(dim=0)
    result: dict[str, Any] = {
        "engines": len(engines),
        "sampled_windows": total,
        "soft_occupancy_engine_mean": soft.tolist(),
        "hard_occupancy_engines": dominant,
        "hard_occupancy_windows": window_counts.tolist(),
        "mean_window_entropy_nats": statistics.mean(
            item.entropy / item.count for item in engines.values()
        ),
        "mean_normalized_entropy": statistics.mean(
            item.entropy / (item.count * math.log(classes)) for item in engines.values()
        ),
        "mean_top_probability": statistics.mean(
            item.confidence / item.count for item in engines.values()
        ),
        "mean_projection_l2": statistics.mean(
            item.projection_norm / item.count for item in engines.values()
        ),
        "mean_projection_to_base_l2": statistics.mean(
            item.projection_to_base_ratio / item.count for item in engines.values()
        ),
        "dominant_engine_fraction": max(dominant) / len(engines),
    }
    if assignments is not None:
        offset = 0 if kind == "working" else 1
        if set(engines) != set(assignments):
            raise ValueError("Training labels must match all and only training engines")
        truth = [assignments[engine][offset] for engine in engines]
        if any(not 0 <= label < classes for label in truth):
            raise ValueError("Invalid training-only cluster pseudo-label")
        majority = max(truth.count(label) for label in range(classes)) / len(truth)
        result["teacher_engine_occupancy"] = [
            truth.count(label) for label in range(classes)
        ]
        result["engine_teacher_accuracy"] = (
            sum(int(a == b) for a, b in zip(predicted, truth, strict=True)) / len(truth)
        )
        result["majority_teacher_accuracy"] = majority
        confidences = means.max(dim=1).values.tolist()
        ece = 0.0
        for bucket in range(10):
            chosen = [
                i for i, confidence in enumerate(confidences)
                if min(int(confidence * 10), 9) == bucket
            ]
            if chosen:
                empirical = statistics.mean(
                    float(predicted[i] == truth[i]) for i in chosen
                )
                expected = statistics.mean(confidences[i] for i in chosen)
                ece += len(chosen) / len(truth) * abs(empirical - expected)
        result["engine_ece_10bins"] = ece
        result["mean_teacher_nll"] = statistics.mean(
            -math.log(max(float(row[label]), 1e-12))
            for row, label in zip(means, truth, strict=True)
        )
    return result


def _extract_module(state: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
    """Extract an exact module's saved parameters.

    Returns:
        Its child-relative state dictionary.
    """
    return {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }


@torch.inference_mode()
def _inspect_gates(
    state: dict[str, Tensor],
    arm: Arms,
    features: Tensor,
    sampled: dict[str, dict[str, list[int]]],
    assignments: dict[str, list[int]],
    cluster: dict[str, Any],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Replay exact trained gate computation on current observed windows only.

    This reproduces the sensor-graph, PatchFormer, pooling, distance logits
    and additive projection in _ClusterConditionedCfCRuntime._predict.

    Returns:
        Per-kind training and validation results.

    Raises:
        ValueError: If trained weights or their normalization are inconsistent.
    """
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    encoder = _PatchFormerModel(30, 15, 64, 5, 2, 0.1)
    encoder.load_state_dict(_extract_module(state, "encoder."), strict=True)
    encoder = encoder.to(device).eval()
    sensor_graph = _SensorGraphAttention(15)
    sensor_graph.load_state_dict(_extract_module(state, "sensor_graph."), strict=True)
    sensor_graph = sensor_graph.to(device).eval()
    reference_mean = state["reference_mean"].to(device)
    reference_std = state["reference_std"].to(device)
    if reference_mean.shape != (30, 128) or reference_std.shape != (30, 128):
        raise ValueError("Incompatible checkpoint feature normalization")
    if not bool((reference_std > 0).all()):
        raise ValueError("Nonpositive feature normalization scale")
    input_weight = state["input_projection.weight"].to(device)
    input_bias = state["input_projection.bias"].to(device)
    positions = state["positions"].to(device)
    if input_weight.shape != (64, 256) or positions.shape != (1, 30, 64):
        raise ValueError("Unexpected CfC base representation shape")

    enabled: tuple[GateKind, ...] = (
        ("working", "failure") if arm == "both" else (cast(GateKind, arm),)
    )
    gates: dict[str, nn.Module] = {}
    projections: dict[str, nn.Module] = {}
    for kind in enabled:
        count = 3 if kind == "working" else 2
        for suffix in ("centroids", "mean", "std"):
            tensor = cast(Tensor, cluster[f"{kind}_{suffix}"])
            if not torch.equal(state[f"{kind}_{suffix}"].cpu(), tensor):
                raise ValueError(f"{kind} checkpoint cluster buffers differ from provenance")
        gate = nn.Sequential(
            nn.LayerNorm(128), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, count)
        )
        gate.load_state_dict(_extract_module(state, f"{kind}_gate."), strict=True)
        projection = nn.Linear(count, 64, bias=False)
        projection.load_state_dict(
            _extract_module(state, f"{kind}_projection."), strict=True
        )
        gates[kind] = gate.to(device).eval()
        projections[kind] = projection.to(device).eval()

    output: dict[str, dict[str, dict[str, Any]]] = {}
    for kind in enabled:
        output[kind] = {}
    for split in ("train", "val"):
        all_rows = [
            (engine, index)
            for engine, indices in sorted(sampled[split].items())
            for index in indices
        ]
        accumulators: dict[str, dict[str, _EngineGateStats]] = {
            kind: {engine: _EngineGateStats.empty(3 if kind == "working" else 2)
                   for engine in sampled[split]}
            for kind in enabled
        }
        for start in range(0, len(all_rows), batch_size):
            part = all_rows[start : start + batch_size]
            indices = [index for _, index in part]
            history = features[indices].to(device)
            normalized = (
                encoder.forward_features(sensor_graph(history)) - reference_mean
            ) / reference_std
            pooled = normalized.mean(dim=1)
            baseline = (
                torch.nn.functional.linear(
                    torch.cat((normalized, torch.zeros_like(normalized)), dim=-1),
                    input_weight, input_bias,
                )
                + positions
            ).norm(dim=-1).mean(dim=1)
            for kind in enabled:
                mean = state[f"{kind}_mean"].to(device)
                std = state[f"{kind}_std"].to(device)
                centers = state[f"{kind}_centroids"].to(device)
                distance = (
                    ((pooled - mean) / std).unsqueeze(1) - centers.unsqueeze(0)
                ).square().mean(dim=-1)
                # The registered NEX-233 experiment fixes temperature=1.0.
                logits = gates[kind](pooled) - distance
                probabilities = logits.softmax(dim=-1)
                projected = projections[kind](probabilities)
                for i, (engine, _) in enumerate(part):
                    accumulators[kind][engine].append(
                        probabilities[i], projected[i], baseline[i]
                    )
        for kind in enabled:
            output[kind][split] = _summarize(
                accumulators[kind],
                assignments if split == "train" else None,
                kind,
            )
    return output


def audit_campaign(
    experiment_root: Path,
    *,
    device: torch.device,
    per_engine: int = 64,
    batch_size: int = 64,
) -> Path:
    """Audit all twelve validation-selected runs, without test observations.

    The reference contributes paired context metrics but has no gate to audit.
    Reports are read-only with respect to all training and checkpoint files.

    Returns:
        Path to the CSV summary generated on persistent experiment storage.

    Raises:
        ValueError: If saved training data or model provenance is inconsistent.
    """
    root = experiment_root.expanduser().resolve()
    prepared = root / "rul/prepared" / PREPARED_NAME
    cluster_file = root / "rul/prepared" / CLUSTER_NAME
    prepared_hash = _sha256(prepared)
    cluster = read_cluster_artifact(
        cluster_file, prepared_sha256=prepared_hash, seed=42,
        working_clusters=3, failure_clusters=2,
    )
    payload = read_rul_artifact(prepared)
    features = cast(Tensor, payload["features"])
    if features.ndim != 3 or features.shape[1:] != (30, 15):
        raise ValueError("Expected the original FD003 30-cycle 15-sensor input")
    sampled = _balanced_windows(cast(list[dict[str, Any]], payload["rows"]), per_engine)
    assignments = cast(dict[str, list[int]], cluster["train_engine_assignments"])
    if set(assignments) != set(sampled["train"]):
        raise ValueError("Cluster pseudo-labels disagree with training engine identities")
    destination = root / OUTPUT_DIRECTORY
    destination.mkdir(parents=True, exist_ok=True)
    cluster_hash = _sha256(cluster_file)
    summary_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            checkpoint, epoch, comparison = _selected_checkpoint(root, arm, seed)
            report: dict[str, Any] = {
                "schema_version": 1,
                "study": "NEX-233-FD003-CfC-cluster-conditioning",
                "arm": arm,
                "seed": seed,
                "selection": "minimum val/loss",
                "checkpoint_epoch": epoch,
                "checkpoint_file": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "prepared_sha256": prepared_hash,
                "cluster_sha256": cluster_hash,
                "sampling": {
                    "method": "deterministic evenly spaced windows per engine",
                    "maximum_windows_per_engine": per_engine,
                    "splits": ["train", "val"],
                    "test_observations_used": False,
                },
                "selected_checkpoint_metrics": comparison,
                "gates": {},
            }
            if arm != "reference":
                state = _model_state(checkpoint, arm, epoch)
                report["gates"] = _inspect_gates(
                    state, arm, features, sampled, assignments, cluster,
                    device=device, batch_size=batch_size,
                )
            output = destination / f"{arm}-s{seed}.json"
            output.write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            if arm == "reference":
                summary_rows.append({
                    "arm": arm,
                    "seed": seed,
                    "gate": "none",
                    "checkpoint_epoch": epoch,
                    "val_eval_mse": comparison["val_eval_mse"],
                    "train_teacher_accuracy": "",
                    "train_majority_accuracy": "",
                    "train_engine_ece_10bins": "",
                    "val_mean_normalized_entropy": "",
                    "val_mean_top_probability": "",
                    "val_dominant_engine_fraction": "",
                    "val_mean_projection_to_base_l2": "",
                })
            for kind, details in cast(dict[str, Any], report["gates"]).items():
                train = details["train"]
                val = details["val"]
                summary_rows.append({
                    "arm": arm,
                    "seed": seed,
                    "gate": kind,
                    "checkpoint_epoch": epoch,
                    "val_eval_mse": comparison["val_eval_mse"],
                    "train_teacher_accuracy": train["engine_teacher_accuracy"],
                    "train_majority_accuracy": train["majority_teacher_accuracy"],
                    "train_engine_ece_10bins": train["engine_ece_10bins"],
                    "val_mean_normalized_entropy": val["mean_normalized_entropy"],
                    "val_mean_top_probability": val["mean_top_probability"],
                    "val_dominant_engine_fraction": val["dominant_engine_fraction"],
                    "val_mean_projection_to_base_l2": val["mean_projection_to_base_l2"],
                })
            print(f"[audited] {arm}-s{seed}: {output}", flush=True)
    summary = destination / "summary.csv"
    if len(summary_rows) != 15:
        raise ValueError("Expected 3 reference rows and 12 conditioned gate audits")
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary


def main() -> None:
    """Audit the completed train/validation experiment without refitting models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root", type=Path,
        default=Path("/mnt/juicefs_prisma/models/nexuml/.experiments"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--windows-per-engine", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but no GPU available")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    result = audit_campaign(
        args.experiment_root, device=device,
        per_engine=args.windows_per_engine, batch_size=args.batch_size,
    )
    print(f"Gate diagnostics summary: {result}")


if __name__ == "__main__":
    main()
