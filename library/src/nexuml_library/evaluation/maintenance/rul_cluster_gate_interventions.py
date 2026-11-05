"""Paired inference-time gate interventions on completed NEX-233 FD003 CfC runs.

Replays trusted, validation-selected checkpoints without constructing a native
training session, changing their parameters, fitting labels, or evaluating the
official test split. The original replay must match the saved full-validation
evaluation-mode MSE before any counterfactual is accepted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

from nexuml_library.data.maintenance.prepared_endpoints import read_prepared_artifact
from nexuml_library.data.maintenance.rul import read_rul_artifact
from nexuml_library.data.maintenance.rul_cluster_conditions import read_cluster_artifact
from nexuml_library.evaluation.maintenance.rul_cluster_gate_audit import (
    ARMS,
    CLUSTER_NAME,
    PREPARED_NAME,
    SEEDS,
    Arms,
    GateKind,
    _balanced_windows,
    _model_state,
    _selected_checkpoint,
    _sha256,
)
from nexuml_library.layers.generative.flow.sequence_vector_field import _CfCSequence
from nexuml_library.layers.model.rul_advanced import _PatchFormerModel
from nexuml_library.layers.model.rul_enhanced_cfc import (
    _AdaptiveSoftSegments,
    _MultiscaleTemporalFeatures,
    _SensorGraphAttention,
)

OUTPUT_DIRECTORY = "rul/enhanced-cfc/cluster-conditioned/fd003/gate-interventions-v1"
AGE_BUCKETS = ("early", "middle", "late")
Intervention = Literal[
    "original", "zero_working", "zero_failure", "zero_both", "shuffle_memberships"
]


def _enabled_gates(arm: Arms) -> tuple[GateKind, ...]:
    """Identify the genuinely trained gates for one arm.

    Returns:
        The enabled working and/or failure gate names.
    """
    if arm == "both":
        return ("working", "failure")
    if arm == "reference":
        return ()
    return (arm,)


def _interventions(arm: Arms) -> tuple[Intervention, ...]:
    """Restrict counterfactuals to the gates actually present in a checkpoint.

    Returns:
        Valid inference-only interventions for the arm.
    """
    if arm == "reference":
        return ("original",)
    if arm == "working":
        return ("original", "zero_working", "shuffle_memberships")
    if arm == "failure":
        return ("original", "zero_failure", "shuffle_memberships")
    return (
        "original",
        "zero_working",
        "zero_failure",
        "zero_both",
        "shuffle_memberships",
    )


class _ReplayCfC(nn.Module):
    """Standalone, strictly loaded replica of the registered all_no_trend head.

    Only the three already trained gate arms and their unchanged reference
    are supported. Loading every checkpoint parameter and buffer strictly is
    essential: missing architectural switches must not silently be ignored.
    """

    def __init__(self, arm: Arms, state: dict[str, Tensor]) -> None:
        super().__init__()
        self.arm = arm
        self.enabled = _enabled_gates(arm)
        self.encoder = _PatchFormerModel(30, 15, 64, 5, 2, 0.1)
        self.sensor_graph = _SensorGraphAttention(15)
        self.register_buffer("reference_mean", torch.zeros(30, 128))
        self.register_buffer("reference_std", torch.ones(30, 128))
        self.input_projection = nn.Linear(256, 64)
        self.positions = nn.Parameter(torch.zeros(1, 30, 64))
        self.temporal = _CfCSequence(64, depth=1, dropout=0.1)
        self.norm = nn.LayerNorm(64)
        self.token_scores = nn.Linear(64, 1)
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )
        self.multiscale = _MultiscaleTemporalFeatures(64, dropout=0.1)
        self.segmentation = _AdaptiveSoftSegments(64, dropout=0.1)
        if arm in ("working", "both"):
            self.working_gate: nn.Module | None = nn.Sequential(
                nn.LayerNorm(128),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 3),
            )
            self.working_projection: nn.Linear | None = nn.Linear(3, 64, bias=False)
            self.register_buffer("working_centroids", torch.zeros(3, 128))
            self.register_buffer("working_mean", torch.zeros(128))
            self.register_buffer("working_std", torch.ones(128))
        else:
            self.working_gate = None
            self.working_projection = None
        if arm in ("failure", "both"):
            self.failure_gate: nn.Module | None = nn.Sequential(
                nn.LayerNorm(128),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 2),
            )
            self.failure_projection: nn.Linear | None = nn.Linear(2, 64, bias=False)
            self.register_buffer("failure_centroids", torch.zeros(2, 128))
            self.register_buffer("failure_mean", torch.zeros(128))
            self.register_buffer("failure_std", torch.ones(128))
        else:
            self.failure_gate = None
            self.failure_projection = None
        # Strict loading enforces the all_no_trend architecture and all parameters.
        self.load_state_dict(state, strict=True)
        self.eval()

    def encode(self, history: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute exactly the original gate inputs and their soft memberships.

        Returns:
            The preconditioning CfC input and enabled membership probabilities.

        Raises:
            ValueError: If the selected sensor input is incompatible.
        """
        if history.ndim != 3 or history.shape[1:] != (30, 15):
            raise ValueError("CfC replay requires [batch,30,15] sensor windows")
        raw = self.sensor_graph(history)
        features = self.encoder.forward_features(raw)
        normalized = (features - self.reference_mean) / self.reference_std
        # Matches _ClusterConditionedCfCRuntime._predict and the unchanged head.
        encoded = (
            self.input_projection(
                torch.cat((normalized, torch.zeros_like(normalized)), dim=-1)
            )
            + self.positions
        )
        pooled = normalized.mean(dim=1)
        probabilities: dict[str, Tensor] = {}
        for kind in self.enabled:
            gate = self.working_gate if kind == "working" else self.failure_gate
            if gate is None:
                raise ValueError(f"Checkpoint is missing its {kind} gate")
            center = cast(Tensor, getattr(self, f"{kind}_centroids"))
            mean = cast(Tensor, getattr(self, f"{kind}_mean"))
            std = cast(Tensor, getattr(self, f"{kind}_std"))
            standardized = (pooled - mean) / std
            distance = (
                standardized.unsqueeze(1) - center.unsqueeze(0)
            ).square().mean(dim=-1)
            # All registered NEX-233 conditioned heads use temperature=1.0.
            probabilities[kind] = (gate(pooled) - distance).softmax(dim=-1)
        return encoded, probabilities

    def predict_from_encoded(
        self,
        encoded: Tensor,
        memberships: dict[str, Tensor],
        *,
        intervention: Intervention,
    ) -> Tensor:
        """Predict RUL from a fixed feature path while intervening only on gates.

        Zero interventions remove the additive projection, not model weights.
        Shuffle interventions replace only the soft membership vectors.

        Returns:
            Capped RUL prediction, shape [batch,1].

        Raises:
            ValueError: If the intervention is unsupported or probabilities are missing.
        """
        if intervention not in _interventions(self.arm):
            raise ValueError(f"Unsupported intervention {intervention} for {self.arm}")
        conditioned = encoded
        for kind in self.enabled:
            if intervention == "zero_both" or intervention == f"zero_{kind}":
                continue
            projector = (
                self.working_projection if kind == "working" else self.failure_projection
            )
            if projector is None or kind not in memberships:
                raise ValueError(f"Missing {kind} membership for CfC replay")
            probability = memberships[kind]
            if probability.ndim != 2 or probability.shape[0] != encoded.shape[0]:
                raise ValueError(f"Invalid {kind} membership dimensions")
            conditioned = conditioned + projector(probability).unsqueeze(1)
        conditioned = self.multiscale(conditioned)
        conditioned = self.segmentation(conditioned)
        tokens = self.norm(self.temporal(conditioned))
        attention = self.token_scores(tokens).softmax(dim=1)
        result = 130.0 * torch.sigmoid(
            self.regressor((tokens * attention).sum(dim=1))
        )
        if result.shape != (encoded.shape[0], 1) or not bool(torch.isfinite(result).all()):
            raise ValueError("Nonfinite or malformed RUL counterfactual prediction")
        return result


def _validation_index(
    rows: list[dict[str, Any]],
) -> tuple[list[int], dict[str, list[int]]]:
    """Create an exhaustive, chronologically ordered validation-engine index.

    Only exact 'val' rows are selected; test rows never enter inference.

    Returns:
        Prepared-array indices in validation order and positions by engine.

    Raises:
        ValueError: If the FD003 validation-engine split is unexpected.
    """
    by_engine: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["split"] == "val":
            by_engine[str(row["asset_id"])].append((int(row["step"]), index))
    if len(by_engine) != 20:
        raise ValueError("Expected exactly 20 held-out FD003 validation engines")
    indices: list[int] = []
    positions: dict[str, list[int]] = {}
    for engine in sorted(by_engine):
        ordered = [index for _, index in sorted(by_engine[engine])]
        if len(ordered) < 2:
            raise ValueError(f"Validation engine {engine} has too few windows")
        positions[engine] = list(range(len(indices), len(indices) + len(ordered)))
        indices.extend(ordered)
    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate validation sample in intervention index")
    return indices, positions


def _donor_engines(engines: list[str], seed: int) -> dict[str, str]:
    """Derange donor engines deterministically, without within-engine donors.

    A cyclic rotation is a bijection, so every donor contributes to exactly
    one recipient. Both gates use the same donor mapping in the combined arm.

    Returns:
        One nonself donor engine per validation engine.

    Raises:
        ValueError: If there are fewer than two engines.
    """
    if len(engines) < 2:
        raise ValueError("Cross-engine shuffle requires multiple validation engines")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shift = int(torch.randint(1, len(engines), (1,), generator=generator).item())
    return {
        engine: engines[(index + shift) % len(engines)]
        for index, engine in enumerate(engines)
    }


def _swap_engine_memberships(
    probabilities: Tensor,
    positions: dict[str, list[int]],
    donors: dict[str, str],
) -> Tensor:
    """Exchange membership trajectories between engines at matched relative age.

    Uses each donor's predicted probabilities only; no labels are shared.
    Recipient sequences retain their own sensor histories and all other
    network inputs. Works with unequal engine observation lengths.

    Returns:
        New probability tensor aligned to the original validation order.

    Raises:
        ValueError: If the donor mapping is not a complete derangement.
    """
    if set(donors) != set(positions) or set(donors.values()) != set(positions):
        raise ValueError("Membership shuffle must be a validation-engine permutation")
    if any(engine == donor for engine, donor in donors.items()):
        raise ValueError("Membership shuffle must exclude the original engine")
    shuffled = torch.empty_like(probabilities)
    for engine, rows in positions.items():
        source = positions[donors[engine]]
        if len(rows) < 2 or len(source) < 2:
            raise ValueError("Cannot age-align an engine with fewer than two samples")
        for index, position in enumerate(rows):
            donor_index = round(index * (len(source) - 1) / (len(rows) - 1))
            shuffled[position] = probabilities[source[donor_index]]
    return shuffled


def _age_bucket(rank: int, length: int) -> str:
    """Classify observation position into thirds of its training-engine life.

    Returns:
        Early, middle or late relative-life stage.

    Raises:
        ValueError: If a sample is outside its engine history.
    """
    if length < 2 or rank < 0 or rank >= length:
        raise ValueError("Invalid engine lifespan rank")
    fraction = rank / (length - 1)
    return AGE_BUCKETS[min(int(fraction * 3), 2)]


def _score_prediction(
    prediction: Tensor,
    targets: Tensor,
    positions: dict[str, list[int]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute exact per-window MSE and macro-engine validation statistics.

    Returns:
        Aggregate summary and all 20 per-engine squared-error means.

    Raises:
        ValueError: If predictions, labels or validation engine indices differ.
    """
    predicted = prediction.detach().float().cpu().flatten()
    expected = targets.detach().float().cpu().flatten()
    if predicted.shape != expected.shape or not bool(torch.isfinite(predicted).all()):
        raise ValueError("Predictions and true RUL targets are not aligned")
    if len(predicted) != sum(len(items) for items in positions.values()):
        raise ValueError("Validation index must cover every predicted window")
    squared = (predicted - expected).square()
    engine_mse = {
        engine: float(squared[indices].mean())
        for engine, indices in positions.items()
    }
    summary = {
        "window_mse": float(squared.mean()),
        "engine_macro_mse": statistics.mean(engine_mse.values()),
        "worst_engine_mse": max(engine_mse.values()),
        "val_engines": float(len(engine_mse)),
        "val_windows": float(len(predicted)),
    }
    if not all(math.isfinite(item) for item in summary.values()):
        raise ValueError("Nonfinite RUL intervention diagnostic")
    return summary, engine_mse


@torch.inference_mode()
def _prepare_validation(
    model: _ReplayCfC,
    features: Tensor,
    indices: list[int],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Cache unchanged encoded sensor histories and original gate memberships.

    All returned tensors reside on CPU to bound GPU memory use.

    Returns:
        Preconditioning features and predicted memberships for every validation window.
    """
    representations: list[Tensor] = []
    probabilities: dict[str, list[Tensor]] = {
        kind: [] for kind in model.enabled
    }
    for start in range(0, len(indices), batch_size):
        selection = indices[start : start + batch_size]
        encoded, predicted = model.encode(features[selection].to(device))
        representations.append(encoded.cpu())
        for kind in model.enabled:
            probabilities[kind].append(predicted[kind].cpu())
    return (
        torch.cat(representations, dim=0),
        {kind: torch.cat(chunks, dim=0) for kind, chunks in probabilities.items()},
    )


@torch.inference_mode()
def _counterfactual_predictions(
    model: _ReplayCfC,
    encoded: Tensor,
    original: dict[str, Tensor],
    shuffled: dict[str, Tensor],
    intervention: Intervention,
    *,
    device: torch.device,
    batch_size: int,
) -> Tensor:
    """Apply one intervention to exactly the same validation sensor embeddings.

    Returns:
        All per-window RUL predictions in validation order.
    """
    output: list[Tensor] = []
    memberships = shuffled if intervention == "shuffle_memberships" else original
    for start in range(0, len(encoded), batch_size):
        batch = encoded[start : start + batch_size].to(device)
        probabilities = {
            kind: values[start : start + batch_size].to(device)
            for kind, values in memberships.items()
        }
        result = model.predict_from_encoded(
            batch, probabilities, intervention=intervention
        )
        output.append(result.cpu())
    return torch.cat(output, dim=0)


@torch.inference_mode()
def _age_stratified_gates(
    model: _ReplayCfC,
    features: Tensor,
    rows: list[dict[str, Any]],
    assignments: dict[str, list[int]],
    *,
    device: torch.device,
    batch_size: int,
    windows_per_engine: int,
) -> list[dict[str, Any]]:
    """Evaluate pseudo-label agreement in early/middle/late TRAIN life stages.

    Terminal pseudo-labels are training-only provenance, not observed failure
    modes or validation labels. Relative age is computed offline from the
    complete training history solely for diagnostics, never as a model input.

    Returns:
        One engine-macro summary per trained gate and age tercile.

    Raises:
        ValueError: If training engine assignments or sampled ages are incomplete.
    """
    if not model.enabled:
        return []
    sampled = _balanced_windows(rows, windows_per_engine)["train"]
    full: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["split"] == "train":
            full[str(row["asset_id"])].append((int(row["step"]), index))
    if set(sampled) != set(assignments) or set(sampled) != set(full):
        raise ValueError("Age study requires original train-only engine assignments")
    stage_of: dict[int, str] = {}
    for engine, observations in full.items():
        indices = [index for _, index in sorted(observations)]
        for rank, index in enumerate(indices):
            stage_of[index] = _age_bucket(rank, len(indices))

    # Sufficient statistics by gate, stage and engine. Per-engine averaging
    # ensures a long-lived engine does not dominate accuracy or confidence.
    totals: dict[tuple[str, str, str], list[float]] = {}
    for engine, indices in sorted(sampled.items()):
        for start in range(0, len(indices), batch_size):
            subset = indices[start : start + batch_size]
            _, probabilities = model.encode(features[subset].to(device))
            for kind in model.enabled:
                correct_label = assignments[engine][0 if kind == "working" else 1]
                for index, probability in zip(subset, probabilities[kind].cpu(), strict=True):
                    stage = stage_of[index]
                    key = (kind, stage, engine)
                    stats = totals.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0.0])
                    p = probability.float()
                    stats[0] += 1
                    stats[1] += float(int(p.argmax()) == correct_label)
                    stats[2] += -math.log(max(float(p[correct_label]), 1e-12))
                    stats[3] += float(p.max())
                    stats[4] += float(
                        -(p * p.clamp_min(1e-12).log()).sum()
                    ) / math.log(len(p))
    summaries: list[dict[str, Any]] = []
    for kind in model.enabled:
        truth_index = 0 if kind == "working" else 1
        count = 3 if kind == "working" else 2
        labels = [value[truth_index] for value in assignments.values()]
        majority = max(labels.count(i) for i in range(count)) / len(labels)
        for stage in AGE_BUCKETS:
            subset = [
                values for (name, age, _), values in totals.items()
                if name == kind and age == stage
            ]
            if len(subset) != len(assignments) or any(values[0] == 0 for values in subset):
                raise ValueError(f"Missing {kind} training-engine age-stratum predictions")
            summaries.append({
                "gate": kind,
                "stage": stage,
                "engines": len(subset),
                "sampled_windows": int(sum(row[0] for row in subset)),
                "teacher_agreement_engine_macro": statistics.mean(
                    row[1] / row[0] for row in subset
                ),
                "teacher_majority_baseline": majority,
                "teacher_nll_engine_macro": statistics.mean(
                    row[2] / row[0] for row in subset
                ),
                "mean_confidence_engine_macro": statistics.mean(
                    row[3] / row[0] for row in subset
                ),
                "mean_normalized_entropy_engine_macro": statistics.mean(
                    row[4] / row[0] for row in subset
                ),
            })
    return summaries


def _verify_original_replay(replayed: float, recorded: float, run: str) -> None:
    """Fail closed when the offline model differs from the native validation run.

    Prevents claims about counterfactual effects if source reconstruction,
    normalization, checkpoint selection or original dataset do not match.

    Raises:
        ValueError: If a selected model's full-validation MSE is inconsistent.
    """
    if not math.isclose(replayed, recorded, rel_tol=0.002, abs_tol=0.25):
        raise ValueError(
            f"{run}: replay full-val MSE {replayed:.6f} differs from native "
            f"selected-checkpoint MSE {recorded:.6f}; no interventions are valid"
        )


def run_interventions(
    experiment_root: Path,
    *,
    device: torch.device,
    batch_size: int = 64,
    age_windows_per_engine: int = 64,
    arms: tuple[Arms, ...] = ARMS,
    seeds: tuple[int, ...] = SEEDS,
) -> Path:
    """Audit all completed checkpoints with paired, test-sealed interventions.

    Saves one JSON report per model, a window/engine-level validation summary,
    per-validation-engine deltas, and training-only life-stage diagnostics.

    Returns:
        The path to the intervention summary CSV on persistent Prisma storage.

    Raises:
        ValueError: If the requested campaign, data or checkpoints are inconsistent.
    """
    if batch_size < 1 or age_windows_per_engine < 3:
        raise ValueError("Require positive batch size and at least 3 age windows per engine")
    if not arms or not seeds or any(arm not in ARMS for arm in arms):
        raise ValueError("Unknown or empty NEX-233 arm selection")
    if any(seed not in SEEDS for seed in seeds):
        raise ValueError("NEX-233 intervention seeds must be 42, 43 or 44")

    root = experiment_root.expanduser().resolve()
    prepared = root / "rul/prepared" / PREPARED_NAME
    endpoint = root / (
        "rul/prepared/rul-FD003-patchformer-sequence-cfc-flow-seed42-endpoints.pt"
    )
    cluster_file = root / "rul/prepared" / CLUSTER_NAME
    payload = read_rul_artifact(prepared)
    rows = cast(list[dict[str, Any]], payload["rows"])
    features = cast(Tensor, payload["features"])
    labels = cast(Tensor, payload["targets"])
    if features.ndim != 3 or features.shape[1:] != (30, 15):
        raise ValueError("NEX-233 intervention requires [N,30,15] prepared FD003 windows")
    indices, engine_positions = _validation_index(rows)
    selected_targets = labels[indices]
    settings = read_prepared_artifact(endpoint)["settings"]
    cluster = read_cluster_artifact(
        cluster_file,
        seed=42,
        working_clusters=3,
        failure_clusters=2,
        prepared_sha256=_sha256(prepared),
        encoder_sha256=str(settings["frozen_patchformer_checkpoint_sha256"]),
    )
    assignments = cast(dict[str, list[int]], cluster["train_engine_assignments"])
    # Checking all engine identities requires no test data to be materialized.
    sampled = _balanced_windows(rows, age_windows_per_engine)
    if set(sampled["train"]) != set(assignments):
        raise ValueError("Cluster pseudo-labels are not train-engine-disjoint")

    output_dir = root / OUTPUT_DIRECTORY
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    engine_rows: list[dict[str, Any]] = []
    age_rows: list[dict[str, Any]] = []
    donors = _donor_engines(sorted(engine_positions), seed=2026)
    for arm in arms:
        for seed in seeds:
            run = f"rul-FD003-enhanced-cfc-cluster-{arm}-s{seed}"
            checkpoint, epoch, reference_metrics = _selected_checkpoint(root, arm, seed)
            state = _model_state(checkpoint, arm, epoch)
            for key, expected in (
                ("reference_mean", settings["feature_mean"]),
                ("reference_std", settings["feature_std"]),
            ):
                if not torch.equal(
                    state[key].cpu(), torch.as_tensor(expected, dtype=torch.float32)
                ):
                    raise ValueError(f"{run}: normalization differs from training-only endpoint")
            for kind in _enabled_gates(arm):
                for suffix in ("centroids", "mean", "std"):
                    if not torch.equal(
                        state[f"{kind}_{suffix}"].cpu(),
                        cast(Tensor, cluster[f"{kind}_{suffix}"]),
                    ):
                        raise ValueError(f"{run}: {kind} centroids differ from cluster artifact")
            model = _ReplayCfC(arm, state).to(device)
            encoded, probabilities = _prepare_validation(
                model, features, indices, device=device, batch_size=batch_size
            )
            shuffled = {
                kind: _swap_engine_memberships(values, engine_positions, donors)
                for kind, values in probabilities.items()
            }
            measurements: dict[str, dict[str, Any]] = {}
            original_by_engine: dict[str, float] = {}
            for variant in _interventions(arm):
                predictions = _counterfactual_predictions(
                    model, encoded, probabilities, shuffled, variant,
                    device=device, batch_size=batch_size,
                )
                scores, per_engine = _score_prediction(
                    predictions, selected_targets, engine_positions
                )
                if variant == "original":
                    _verify_original_replay(
                        scores["window_mse"],
                        reference_metrics["val_eval_mse"],
                        run,
                    )
                    original_by_engine = per_engine
                    original_window_mse = scores["window_mse"]
                    original_engine_mse = scores["engine_macro_mse"]
                delta_by_engine = {
                    engine: mse - original_by_engine[engine]
                    for engine, mse in per_engine.items()
                }
                scores["delta_window_mse_vs_original"] = (
                    scores["window_mse"] - original_window_mse
                )
                scores["delta_engine_macro_mse_vs_original"] = (
                    scores["engine_macro_mse"] - original_engine_mse
                )
                scores["engines_worsened_vs_original"] = sum(
                    value > 0 for value in delta_by_engine.values()
                )
                measurements[variant] = scores
                summary_rows.append({
                    "arm": arm,
                    "seed": seed,
                    "epoch": epoch,
                    "intervention": variant,
                    **scores,
                })
                for engine, mse in per_engine.items():
                    engine_rows.append({
                        "arm": arm,
                        "seed": seed,
                        "intervention": variant,
                        "engine": engine,
                        "mse": mse,
                        "delta_mse_vs_original": delta_by_engine[engine],
                        "windows": len(engine_positions[engine]),
                    })
            age = _age_stratified_gates(
                model, features, rows, assignments,
                device=device, batch_size=batch_size,
                windows_per_engine=age_windows_per_engine,
            )
            age_rows.extend({"arm": arm, "seed": seed, **value} for value in age)
            report = {
                "schema_version": 1,
                "study": "NEX-233 FD003 validation-selected CfC gate interventions",
                "arm": arm,
                "seed": seed,
                "selected_epoch": epoch,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "prepared_sha256": _sha256(prepared),
                "cluster_sha256": _sha256(cluster_file),
                "native_validation_mse": reference_metrics["val_eval_mse"],
                "replay_validation_mse_verified": True,
                "evaluation": {
                    "split": "val",
                    "all_validation_windows": True,
                    "engines": len(engine_positions),
                    "test_data_used": False,
                    "change_to_trained_parameters": False,
                },
                "shuffle": {
                    "kind": "cross-engine derangement",
                    "relative_age_aligned": True,
                    "seed": 2026,
                    "donor_by_engine": donors,
                } if arm != "reference" else None,
                "interventions": measurements,
                "train_age_strata": age,
                "interpretation": (
                    "Interventions quantify dependence on trained soft gates; "
                    "they do not show superiority over the separately trained reference "
                    "or identify physical fault modes."
                ),
            }
            report_file = output_dir / f"{arm}-s{seed}.json"
            report_file.write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            print(f"[interventions] {arm}-s{seed}: {report_file}", flush=True)
    for name, data in (
        ("validation_interventions.csv", summary_rows),
        ("validation_engine_deltas.csv", engine_rows),
        ("train_age_strata.csv", age_rows),
    ):
        if not data:
            continue
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    return output_dir / "validation_interventions.csv"


def main() -> None:
    """Run the read-only NEX-233 checkpoint intervention experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("/mnt/juicefs_prisma/models/nexuml/.experiments"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--age-windows-per-engine", type=int, default=64)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA explicitly requested, but no GPU is available")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    result = run_interventions(
        args.experiment_root,
        device=device,
        batch_size=args.batch_size,
        age_windows_per_engine=args.age_windows_per_engine,
        arms=tuple(args.arms),
        seeds=tuple(args.seeds),
    )
    print(f"Validation interventions: {result}")


if __name__ == "__main__":
    main()
