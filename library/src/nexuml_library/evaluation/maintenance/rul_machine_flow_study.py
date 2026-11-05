"""Strictly causal, per-machine FD003 latent-dynamics comparison.

The three RUL estimators use the same frozen PatchFormer representations:
  1. train-engine-only direct RUL ridge regressor,
  2. a within-machine direct future-state predictor plus calibrated RUL rate,
  3. a within-machine conditional Flow Matching predictor with the SAME rate map.

Each target validation engine adapts its dynamics on its own observed PREFIX,
without any of its RUL labels or future windows. Only its suffix is used for
future-state assessment. The target engine's RUL at cutoff is evaluation-only.
A separate retrospective mode analyzes completed TRAIN engines in-sample;
its outputs must never be interpreted as prospective RUL predictions.

Run: python -m nexuml_library.evaluation.maintenance.rul_machine_flow_study
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
from torch import Tensor

from nexuml_library.data.maintenance.patchformer_flow import (
    _trusted_patchformer_weights,
    resolve_patchformer_checkpoint,
)
from nexuml_library.data.maintenance.prepared_endpoints import read_prepared_artifact
from nexuml_library.data.maintenance.rul import read_rul_artifact
from nexuml_library.data.maintenance.rul_cluster_conditions import (
    _shared_patchformer_checkpoint,
)
from nexuml_library.evaluation.maintenance.rul_cluster_gate_audit import (
    PREPARED_NAME,
    _sha256,
)
from nexuml_library.layers.model.rul_advanced import _PatchFormerModel
from nexuml_library.layers.model.rul_machine_flow_dynamics import (
    DynamicMethod,
    MachineFutureDynamics,
)

Mode = Literal["causal", "retrospective", "both"]
HORIZONS = (1, 5, 10, 20)
RUL_CAP = 130.0
OUTPUT_DIRECTORY = "rul/within-machine-flow/fd003"


@dataclass(frozen=True)
class MachineTrajectory:
    """Immutable chronological observations for one physical FD003 engine."""

    engine: str
    split: Literal["train", "val"]
    steps: Tensor
    latent: Tensor
    rul: Tensor

    def __post_init__(self) -> None:
        if (
            self.steps.ndim != 1
            or self.latent.shape != (len(self.steps), 128)
            or self.rul.shape != (len(self.steps), 1)
            or len(self.steps) < 2
            or not bool((self.steps[1:] > self.steps[:-1]).all())
            or not bool(torch.isfinite(self.latent).all())
            or not bool(torch.isfinite(self.rul).all())
        ):
            raise ValueError("Machine trajectory must contain aligned increasing cycles")


def _evenly_spaced(size: int, maximum: int) -> list[int]:
    """Sample at most maximum observations, retaining chronological endpoints.

    Returns:
        Ordered unique positions.
    """
    if maximum < 2 or size < 2:
        raise ValueError("Need at least two windows to balance an engine")
    n = min(size, maximum)
    return [round(i * (size - 1) / (n - 1)) for i in range(n)]


def _index_engine_rows(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, list[int]]]:
    """Build disjoint train/validation chronological indices without test rows.

    Returns:
        Per-split engine indices into the immutable prepared RUL artifact.

    Raises:
        ValueError: If the original FD003 engine split or chronology is invalid.
    """
    positions: dict[str, dict[str, list[tuple[int, int]]]] = {
        "train": defaultdict(list), "val": defaultdict(list)
    }
    for index, row in enumerate(rows):
        split = row["split"]
        if split in positions:
            positions[split][str(row["asset_id"])].append((int(row["step"]), index))
    if len(positions["train"]) != 80 or len(positions["val"]) != 20:
        raise ValueError("Within-machine study requires 80 train and 20 val FD003 engines")
    if set(positions["train"]) & set(positions["val"]):
        raise ValueError("Train and validation machine identities overlap")
    output: dict[str, dict[str, list[int]]] = {"train": {}, "val": {}}
    for split in ("train", "val"):
        for engine, data in sorted(positions[split].items()):
            ordered = sorted(data)
            if len(ordered) < 2 or any(
                ordered[i][0] >= ordered[i + 1][0]
                for i in range(len(ordered) - 1)
            ):
                raise ValueError(f"Malformed FD003 cycle ordering for {engine}")
            output[split][engine] = [index for _, index in ordered]
    return output


def _validate_prepared_settings(settings: dict[str, Any]) -> None:
    """Reject incompatible data preprocessing and target units.

    Raises:
        ValueError: If preprocessing differs from frozen FD003 CfC inputs.
    """
    if (
        settings.get("subset") != "FD003"
        or settings.get("seed") != 42
        or settings.get("window") != 30
        or settings.get("stride") != 1
        or settings.get("rul_cap") != RUL_CAP
        or float(settings.get("target_scale", -1)) != 1.0
        or settings.get("normalize_targets") is not False
        or settings.get("train_full_windows_only") is not True
    ):
        raise ValueError("Expected original unnormalized FD003 30-cycle RUL protocol")


@torch.inference_mode()
def _encode_windows(
    encoder: _PatchFormerModel,
    raw: Tensor,
    mean: Tensor,
    std: Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> Tensor:
    """Embed frozen sensor windows using trusted train-endpoint normalization.

    Returns:
        CPU [N,128] pooled DPAN features in source window order.
    """
    result: list[Tensor] = []
    for begin in range(0, len(raw), batch_size):
        sample = raw[begin : begin + batch_size].to(device)
        embeddings = encoder.forward_features(sample)
        normalized = (embeddings - mean) / std
        result.append(normalized.mean(dim=1).cpu())
    return torch.cat(result, dim=0)


def load_trajectories(
    experiment_root: Path,
    *,
    device: torch.device,
    batch_size: int = 64,
    calibration_windows: int = 64,
) -> tuple[dict[str, MachineTrajectory], dict[str, MachineTrajectory], dict[str, str]]:
    """Read the exact FD003 study inputs and frozen seed-42 PatchFormer.

    Validation RUL labels remain in returned metadata solely for evaluation.
    Calibration feature normalization is derived only from TRAIN engines.
    Official test windows are never encoded, fitted, or evaluated.

    Returns:
        TRAIN and validation engine dictionaries and immutable source SHA-256s.

    Raises:
        ValueError: If upstream provenance, shapes or engine splits disagree.
    """
    if batch_size < 1 or calibration_windows < 2:
        raise ValueError("Encoding batch size and calibration sample count are invalid")
    root = experiment_root.expanduser().resolve()
    prepared_path = root / "rul/prepared" / PREPARED_NAME
    endpoint_path = (
        root / "rul/prepared/rul-FD003-patchformer-sequence-cfc-flow-seed42-endpoints.pt"
    )
    payload = read_rul_artifact(prepared_path)
    settings = cast(dict[str, Any], payload["settings"])
    _validate_prepared_settings(settings)
    raw = cast(Tensor, payload["features"])
    labels = cast(Tensor, payload["targets"])
    if raw.ndim != 3 or raw.shape[1:] != (30, 15):
        raise ValueError("Expected FD003 sensor histories [N,30,15]")
    engine_indices = _index_engine_rows(cast(list[dict[str, Any]], payload["rows"]))
    endpoint = cast(dict[str, Any], read_prepared_artifact(endpoint_path)["settings"])
    checkpoint = resolve_patchformer_checkpoint(
        "FD003", 42, checkpoint_path=str(_shared_patchformer_checkpoint(root, 42))
    )
    encoder = _PatchFormerModel(30, 15, 64, 5, 2, 0.1)
    digest = _trusted_patchformer_weights(encoder, checkpoint)
    if (
        endpoint.get("frozen_patchformer_checkpoint_sha256") != digest
        or endpoint.get("feature_normalization") != "train_endpoints_only"
        or endpoint.get("subset") != "FD003"
        or endpoint.get("seed") != 42
        or endpoint.get("held_out_test_engine_labels_used") is not False
    ):
        raise ValueError("Frozen PatchFormer or endpoint normalization provenance differs")
    mean = torch.as_tensor(endpoint["feature_mean"], dtype=torch.float32).to(device)
    std = torch.as_tensor(endpoint["feature_std"], dtype=torch.float32).to(device)
    if (
        mean.shape != (30, 128)
        or std.shape != (30, 128)
        or not bool((std > 0).all())
    ):
        raise ValueError("Invalid train-only PatchFormer normalization")
    encoder = encoder.eval().to(device)
    grouped: dict[str, dict[str, tuple[Tensor, Tensor, Tensor]]] = {
        "train": {}, "val": {}
    }
    for split, engines in engine_indices.items():
        for engine, ids in sorted(engines.items()):
            if split == "train":
                keep = _evenly_spaced(len(ids), calibration_windows)
                selected = [ids[i] for i in keep]
            else:
                selected = ids
            # The frozen encoder has no access to evaluation targets.
            encoded = _encode_windows(
                encoder, raw[selected], mean, std,
                device=device, batch_size=batch_size,
            )
            rows = cast(list[dict[str, Any]], payload["rows"])
            steps = torch.tensor([int(rows[i]["step"]) for i in selected])
            grouped[split][engine] = (steps, encoded, labels[selected].cpu())
    # Engine-balanced train-only standardization improves numerical stability
    # while ensuring validation dynamics receive no fitted statistics.
    train_latent = torch.cat([data[1] for data in grouped["train"].values()])
    location = train_latent.mean(dim=0)
    scale = train_latent.std(dim=0).clamp_min(1e-4)
    if not bool(torch.isfinite(scale).all()):
        raise ValueError("Nonfinite train-only machine latent standardization")
    output: dict[str, dict[str, MachineTrajectory]] = {"train": {}, "val": {}}
    for split, engines in grouped.items():
        for engine, (steps, latent, rul) in engines.items():
            output[split][engine] = MachineTrajectory(
                engine, cast(Literal["train", "val"], split),
                steps, (latent - location) / scale, rul,
            )
    return output["train"], output["val"], {
        "prepared_sha256": _sha256(prepared_path),
        "endpoint_sha256": _sha256(endpoint_path),
        "patchformer_checkpoint_sha256": digest,
        "calibration_train_windows_per_engine": str(calibration_windows),
        "test_observations_used": "false",
    }


@dataclass(frozen=True)
class RULCalibration:
    """Train-engine-only linear RUL decoder used identically by all approaches."""

    weights: Tensor
    bias: Tensor

    def raw(self, latent: Tensor) -> Tensor:
        """Decode an unclipped remaining-life estimate.

        Returns:
            Estimated RUL in physical cycles.
        """
        return latent @ self.weights + self.bias

    def capped(self, latent: Tensor) -> Tensor:
        """Clamp the shared baseline to the original 0..130 cycle protocol.

        Returns:
            Nonnegative capped RUL prediction.
        """
        return self.raw(latent).clamp(0.0, RUL_CAP)


def fit_rul_calibration(
    train: dict[str, MachineTrajectory], *, regularization: float = 10.0
) -> RULCalibration:
    """Fit the only supervised RUL mapping on completed TRAIN engines.

    Each engine contributes an equal number of balanced chronological windows.
    No target-validation engine RUL label may influence this model.

    Returns:
        Fixed affine decoder usable by all experimental methods.

    Raises:
        ValueError: If labels or calibration regularization are incompatible.
    """
    if regularization <= 0 or len(train) != 80:
        raise ValueError("Calibrator requires 80 training engines and positive ridge penalty")
    x = torch.cat([trace.latent for trace in train.values()]).double()
    y = torch.cat([trace.rul for trace in train.values()]).double()
    if x.ndim != 2 or x.shape[1] != 128 or y.shape != (len(x), 1):
        raise ValueError("Training-engine calibration features and labels are misaligned")
    if not bool((y >= 0).all()) or not bool((y <= RUL_CAP).all()):
        raise ValueError("Expected capped physical-cycle training RUL labels")
    design = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], dim=-1)
    penalty = torch.eye(design.shape[-1], dtype=design.dtype) * regularization
    penalty[-1, -1] = 0.0
    coefficients = torch.linalg.solve(
        design.T @ design + penalty, design.T @ y
    ).float()
    return RULCalibration(coefficients[:-1, 0], coefficients[-1, 0])


def _prefix_count(length: int, fraction: float, minimum: int = 40) -> int:
    """Choose the last observed window before a strictly held-out suffix.

    Returns:
        Number of chronological prefix observations.

    Raises:
        ValueError: If too few windows are available before or after cutoff.
    """
    if not 0.4 <= fraction <= 0.9 or length < minimum + 10:
        raise ValueError("Invalid causal observation fraction or too-short machine history")
    count = int(math.floor(length * fraction))
    if count < minimum or length - count < 10:
        raise ValueError("Causal machine prefix must have 40 observations and 10 held out")
    return count


def _transition_pairs(
    steps: Tensor,
    *,
    horizons: tuple[int, ...] = HORIZONS,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build same-engine start/end pairs with exact physical cycle offsets.

    The supplied steps must already have been truncated at the observation
    cutoff. Consequently, no pair can cross the future holdout boundary.

    Returns:
        Start positions, endpoint positions, and physical cycle horizons.

    Raises:
        ValueError: If there are insufficient actual within-engine transitions.
    """
    if steps.ndim != 1 or len(steps) < 2 or not bool((steps[1:] > steps[:-1]).all()):
        raise ValueError("Within-engine cycle steps must be strictly increasing")
    if not horizons or any(h < 1 for h in horizons):
        raise ValueError("Transition horizon set must contain positive cycle offsets")
    positions = {int(step): index for index, step in enumerate(steps)}
    pairs = [
        (index, positions[int(step) + horizon], horizon)
        for index, step in enumerate(steps)
        for horizon in horizons
        if int(step) + horizon in positions
    ]
    if not pairs:
        raise ValueError("No chronological sensor transitions available before cutoff")
    start, finish, physical = zip(*pairs, strict=True)
    return (
        torch.tensor(start, dtype=torch.long),
        torch.tensor(finish, dtype=torch.long),
        torch.tensor(physical, dtype=torch.float32).unsqueeze(-1),
    )


def fit_dynamics(
    *,
    method: DynamicMethod,
    trace: MachineTrajectory,
    prefix_count: int,
    seed: int,
    device: torch.device,
    train_steps: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[MachineFutureDynamics, float]:
    """Fit one machine's paired latent transitions using ONLY its prefix.

    Flow and non-flow receive the same pairs, capacity and optimizer budget.
    Only unlabeled feature trajectories are used; target RUL is not accessed.

    Returns:
        Trained frozen-evaluation forecaster and final training objective.

    Raises:
        ValueError: If chronological limits or training settings are invalid.
    """
    if (
        prefix_count > len(trace.steps)
        or prefix_count < 40
        or train_steps < 1
        or batch_size < 1
        or learning_rate <= 0
    ):
        raise ValueError("Invalid machine-specific unsupervised training protocol")
    starts, ends, horizons = _transition_pairs(trace.steps[:prefix_count])
    states = trace.latent[:prefix_count].to(device)
    # Same initialization seed for matched flow/non-flow architecture.
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = MachineFutureDynamics(method=method).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed + 902)
    model.train()
    final_loss = float("nan")
    for _ in range(train_steps):
        selected = torch.randint(
            len(starts), (batch_size,), generator=generator
        )
        left = starts[selected].to(device)
        right = ends[selected].to(device)
        elapsed = horizons[selected].to(device)
        loss = model.training_loss(states[left], states[right], elapsed)
        if not bool(torch.isfinite(loss)):
            raise ValueError(f"Nonfinite within-machine {method} objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach())
    model.eval()
    return model, final_loss


@torch.inference_mode()
def forecast_at_horizons(
    model: MachineFutureDynamics,
    current: Tensor,
    *,
    horizons: tuple[int, ...] = HORIZONS,
    device: torch.device,
    integration_steps: int = 8,
) -> dict[int, Tensor]:
    """Forecast several actual cycle offsets from one last-observed state.

    Returns:
        CPU latent future predictions indexed by physical cycle distance.
    """
    origin = current.reshape(1, -1).to(device)
    return {
        horizon: model.forecast(
            origin,
            torch.tensor([[float(horizon)]], device=device),
            integration_steps=integration_steps,
        ).cpu()[0]
        for horizon in horizons
    }


def estimate_rul_from_dynamics(
    calibration: RULCalibration,
    current: Tensor,
    forecast: dict[int, Tensor],
    *,
    minimum_degradation_rate: float = 0.05,
) -> tuple[float, float, bool]:
    """Convert predicted latent progression to physical-cycle time-to-failure.

    The common train-engine decoder acts as a degradation coordinate. A
    least-squares slope across forecast horizons estimates its local physical
    rate of change. A nondegrading / indistinguishable rate is right-censored
    at the original 130-cycle RUL cap, never labeled as observed failure.

    Returns:
        Capped RUL, local slope (cycles of RUL / cycle), censor flag.

    Raises:
        ValueError: If the dynamic forecast is absent or nonfinite.
    """
    if minimum_degradation_rate <= 0 or not forecast:
        raise ValueError("A positive degradation rate threshold and forecasts are required")
    current_rul = float(calibration.raw(current.reshape(1, -1)).squeeze())
    num = 0.0
    den = 0.0
    for horizon, future in sorted(forecast.items()):
        if horizon <= 0 or not bool(torch.isfinite(future).all()):
            raise ValueError("Physical forecast horizons and states must be finite")
        future_rul = float(calibration.raw(future.reshape(1, -1)).squeeze())
        num += horizon * (future_rul - current_rul)
        den += horizon * horizon
    slope = num / den
    if not math.isfinite(slope) or not math.isfinite(current_rul):
        raise ValueError("Nonfinite RUL degradation coordinate")
    censored = slope >= -minimum_degradation_rate
    estimate = RUL_CAP if censored else current_rul / -slope
    return max(0.0, min(RUL_CAP, estimate)), slope, censored


def _score(values: list[float]) -> tuple[float, float]:
    """Summarize absolute and squared RUL errors across physical machines.

    Returns:
        Mean absolute error and root mean squared error in cycles.
    """
    if not values:
        raise ValueError("Cannot summarize empty machine errors")
    return (
        statistics.mean(abs(value) for value in values),
        math.sqrt(statistics.mean(value * value for value in values)),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a deterministically ordered experiment CSV.

    Raises:
        ValueError: If the requested result table is empty.
    """
    if not rows:
        raise ValueError(f"Cannot write an empty experiment table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _select_engine_names(
    engines: dict[str, MachineTrajectory], maximum: int
) -> list[str]:
    """Select the full study or a deterministic debugging subset.

    Returns:
        Ordered machine identifiers.

    Raises:
        ValueError: If the selected maximum is negative.
    """
    if maximum < 0:
        raise ValueError("Maximum selected engine count cannot be negative")
    names = sorted(engines)
    return names[:maximum] if maximum else names


def _evaluate_causal(
    train: dict[str, MachineTrajectory],
    val: dict[str, MachineTrajectory],
    *,
    output: Path,
    device: torch.device,
    fraction: float,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    max_engines: int,
    integration_steps: int,
    ridge: float,
) -> dict[str, Any]:
    """Compare three causally valid RUL estimators on unseen machine suffixes.

    The *global*, train-only ridge decoder is the direct-RUL baseline. Each
    dynamic model is independently adapted on a validation machine's unlabeled
    prefix. Evaluation accesses its cutoff label only after fitting, and its
    future latent states only after forecasting.

    Returns:
        Study metadata and window/engine-level descriptive metrics.
    """
    calibration = fit_rul_calibration(train, regularization=ridge)
    rows: list[dict[str, Any]] = []
    forecasts: list[dict[str, Any]] = []
    names = _select_engine_names(val, max_engines)
    for index, engine in enumerate(names):
        trace = val[engine]
        cutoff = _prefix_count(len(trace.steps), fraction)
        observed = trace.latent[cutoff - 1].clone()
        # Structural leak barrier: dynamics training receives neither the
        # suffix representations nor ANY target-machine RUL labels.
        observed_only = MachineTrajectory(
            engine=trace.engine,
            split=trace.split,
            steps=trace.steps[:cutoff].clone(),
            latent=trace.latent[:cutoff].clone(),
            rul=torch.zeros_like(trace.rul[:cutoff]),
        )
        # The RUL label is accessed ONLY after all estimators are fitted.
        selected_step = int(trace.steps[cutoff - 1])
        model_predictions: dict[str, float] = {}
        rate_details: dict[str, dict[str, Any]] = {}
        fit_losses: dict[str, float] = {}
        for method in ("direct_future", "flow"):
            trained, loss = fit_dynamics(
                method=cast(DynamicMethod, method),
                trace=observed_only,
                prefix_count=len(observed_only.steps),
                seed=seed + index,
                device=device,
                train_steps=steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )
            fit_losses[method] = loss
            future = forecast_at_horizons(
                trained, observed, device=device,
                integration_steps=integration_steps,
            )
            estimate, slope, censored = estimate_rul_from_dynamics(
                calibration, observed, future,
            )
            model_predictions[method] = estimate
            rate_details[method] = {
                "latent_rul_rate": slope, "rate_censored": censored,
            }
            future_lookup = {
                int(step): position for position, step in enumerate(trace.steps)
            }
            for horizon, prediction in future.items():
                target_pos = future_lookup.get(selected_step + horizon)
                if target_pos is None:
                    continue
                target = trace.latent[target_pos]
                forecasts.append({
                    "machine": engine,
                    "method": method,
                    "cutoff_cycle": selected_step,
                    "horizon_cycles": horizon,
                    "future_latent_mse": float((prediction - target).square().mean()),
                    "persistence_latent_mse": float((observed - target).square().mean()),
                })
        # Engine-specific models never consumed any of these labels.
        true_rul = float(trace.rul[cutoff - 1, 0])
        if not 0 <= true_rul <= RUL_CAP:
            raise ValueError("Expected capped physical RUL label at validation cutoff")
        direct = float(calibration.capped(observed.reshape(1, -1))[0])
        row: dict[str, Any] = {
            "machine": engine,
            "cutoff_cycle": selected_step,
            "observed_windows": cutoff,
            "heldout_windows": len(trace.steps) - cutoff,
            "true_capped_rul": true_rul,
            "direct_rul": direct,
            "direct_error": direct - true_rul,
            "nonflow_rul": model_predictions["direct_future"],
            "nonflow_error": model_predictions["direct_future"] - true_rul,
            "flow_rul": model_predictions["flow"],
            "flow_error": model_predictions["flow"] - true_rul,
            "nonflow_latent_rate": rate_details["direct_future"]["latent_rul_rate"],
            "flow_latent_rate": rate_details["flow"]["latent_rul_rate"],
            "nonflow_rate_censored": rate_details["direct_future"]["rate_censored"],
            "flow_rate_censored": rate_details["flow"]["rate_censored"],
            "nonflow_training_mse": fit_losses["direct_future"],
            "flow_training_mse": fit_losses["flow"],
        }
        rows.append(row)
        print(
            f"[causal {len(rows)}/{len(names)}] {engine}: "
            f"RUL={true_rul:.1f}, direct={direct:.1f}, "
            f"nonflow={row['nonflow_rul']:.1f}, flow={row['flow_rul']:.1f}",
            flush=True,
        )
    _write_csv(output / "causal_per_machine.csv", rows)
    _write_csv(output / "causal_forecast_horizons.csv", forecasts)
    summary: dict[str, Any] = {
        "protocol": "validation_prefix_only_adaptation",
        "status": "exploratory; not a held-out official test result",
        "machines": len(rows),
        "prefix_fraction": fraction,
        "target_val_rul_labels_in_fit": False,
        "validation_future_windows_in_fit": False,
        "upstream_training_engines_only_for_rul_calibration": True,
        "official_test_used": False,
        "model_parameters_per_dynamic_method": MachineFutureDynamics(
            method="flow"
        ).parameter_count(),
        "rul_cap_cycles": RUL_CAP,
        "results": {},
        "future_latent_mse": {},
    }
    for name, error_column in (
        ("direct", "direct_error"),
        ("nonflow", "nonflow_error"),
        ("flow", "flow_error"),
    ):
        mae, rmse = _score([float(row[error_column]) for row in rows])
        summary["results"][name] = {"mae_cycles": mae, "rmse_cycles": rmse}
    for method in ("direct_future", "flow", "persistence"):
        key = "persistence_latent_mse" if method == "persistence" else "future_latent_mse"
        summary["future_latent_mse"][method] = {
            str(horizon): statistics.mean(
                float(row[key])
                for row in forecasts
                if row["horizon_cycles"] == horizon
                and (method == "persistence" or row["method"] == method)
            )
            for horizon in HORIZONS
            if any(
                row["horizon_cycles"] == horizon
                and (method == "persistence" or row["method"] == method)
                for row in forecasts
            )
        }
    (output / "causal_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def _evaluate_retrospective(
    train: dict[str, MachineTrajectory],
    *,
    output: Path,
    device: torch.device,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    max_engines: int,
    integration_steps: int,
    ridge: float,
) -> dict[str, Any]:
    """Fit/score only on completed TRAIN trajectories, explicitly in-sample.

    The per-engine oracle direct readout sees its own full RUL labels. Such
    retrospective reconstruction is never compared as prospective accuracy.

    Returns:
        Per-machine reconstruction diagnostics with in-sample provenance.
    """
    results: list[dict[str, Any]] = []
    for index, name in enumerate(_select_engine_names(train, max_engines)):
        trace = train[name]
        # Oracle full-life readout serves solely as a within-engine upper bound.
        x = torch.cat(
            (trace.latent.double(), torch.ones(len(trace.steps), 1, dtype=torch.float64)),
            dim=-1,
        )
        y = trace.rul.double()
        penalty = torch.eye(x.shape[1], dtype=x.dtype) * ridge
        penalty[-1, -1] = 0.0
        coefficients = torch.linalg.solve(x.T @ x + penalty, x.T @ y).float()
        oracle = RULCalibration(coefficients[:-1, 0], coefficients[-1, 0])
        oracle_errors = (oracle.capped(trace.latent).flatten() - trace.rul.flatten())
        row: dict[str, Any] = {
            "machine": name,
            "full_life_observations": len(trace.steps),
            "oracle_direct_train_mae_cycles": float(oracle_errors.abs().mean()),
            "oracle_direct_train_rmse_cycles": float(oracle_errors.square().mean().sqrt()),
            "fit_and_evaluation_same_engine_full_life": True,
            "prospective_rul_result": False,
        }
        starts, ends, horizons = _transition_pairs(trace.steps)
        for method in ("direct_future", "flow"):
            model, loss = fit_dynamics(
                method=cast(DynamicMethod, method), trace=trace,
                prefix_count=len(trace.steps),
                seed=seed + index, device=device,
                train_steps=steps, batch_size=batch_size,
                learning_rate=learning_rate,
            )
            sample = _evenly_spaced(len(starts), min(len(starts), 128))
            start_idx = starts[sample]
            end_idx = ends[sample]
            offsets = horizons[sample]
            mse: list[float] = []
            with torch.inference_mode():
                for begin in range(0, len(sample), batch_size):
                    begin_idx = start_idx[begin : begin + batch_size]
                    end_idx = end_idx[begin : begin + batch_size]
                    elapsed = offsets[begin : begin + batch_size]
                    predicted = model.forecast(
                        trace.latent[begin_idx].to(device),
                        elapsed.to(device),
                        integration_steps=integration_steps,
                    ).cpu()
                    errors = (
                        predicted - trace.latent[end_idx]
                    ).square().mean(dim=-1)
                    mse.extend(errors.tolist())
            row[f"{method}_train_latent_mse"] = statistics.mean(mse)
            row[f"{method}_training_mse"] = loss
        results.append(row)
        print(f"[retrospective {index + 1}] {name}", flush=True)
    _write_csv(output / "retrospective_per_machine.csv", results)
    summary = {
        "protocol": "full_train_engine_in_sample_reconstruction",
        "prospective_rul_evidence": False,
        "official_test_used": False,
        "machines": len(results),
        "warning": (
            "The oracle direct RUL readout uses full engine lifetime labels. "
            "No reconstruction result is a valid prospective RUL claim."
        ),
    }
    (output / "retrospective_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def run_study(
    experiment_root: Path,
    *,
    device: torch.device,
    mode: Mode = "causal",
    prefix_fraction: float = 0.7,
    seed: int = 42,
    train_steps: int = 300,
    batch_size: int = 64,
    calibration_windows: int = 64,
    max_engines: int = 0,
    integration_steps: int = 8,
    ridge: float = 10.0,
    learning_rate: float = 0.001,
) -> Path:
    """Run the independently adapted, parameter-matched NEX-233 study.

    Outputs remain in their own experimental directory; no previous model
    checkpoints, datasets, cluster audits or scenario files are overwritten.

    Returns:
        Experiment manifest on the shared Prisma volume.

    Raises:
        ValueError: If parameters violate the fixed causal study protocol.
    """
    if (
        mode not in ("causal", "retrospective", "both")
        or seed < 0
        or train_steps < 1
        or batch_size < 1
        or max_engines < 0
        or integration_steps < 1
        or calibration_windows < 2
        or ridge <= 0
        or learning_rate <= 0
    ):
        raise ValueError("Invalid within-machine experiment configuration")
    if not 0.4 <= prefix_fraction <= 0.9:
        raise ValueError("Causal observation fraction must be 0.4..0.9")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA is unavailable")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("medium")
    output = (
        experiment_root.expanduser().resolve()
        / OUTPUT_DIRECTORY
        / (
            f"{mode}-seed{seed}-prefix{round(prefix_fraction * 100):02d}"
            f"-steps{train_steps}-max{max_engines or 'all'}"
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    train, val, provenance = load_trajectories(
        experiment_root,
        device=device,
        batch_size=batch_size,
        calibration_windows=calibration_windows,
    )
    results: dict[str, Any] = {}
    if mode in ("causal", "both"):
        results["causal"] = _evaluate_causal(
            train, val, output=output, device=device, fraction=prefix_fraction,
            seed=seed, steps=train_steps, batch_size=batch_size,
            learning_rate=learning_rate, max_engines=max_engines,
            integration_steps=integration_steps, ridge=ridge,
        )
    if mode in ("retrospective", "both"):
        results["retrospective"] = _evaluate_retrospective(
            train, output=output, device=device,
            seed=seed, steps=train_steps, batch_size=batch_size,
            learning_rate=learning_rate, max_engines=max_engines,
            integration_steps=integration_steps, ridge=ridge,
        )
    manifest = output / "manifest.json"
    manifest.write_text(
        json.dumps({
            "schema_version": 1,
            "study": "NEX-233 within-machine conditional flow vs matched non-flow",
            "mode": mode,
            "seed": seed,
            "train_steps": train_steps,
            "batch_size": batch_size,
            "calibration_windows_per_train_engine": calibration_windows,
            "prefix_fraction": prefix_fraction,
            "max_engines": max_engines,
            "integration_steps": integration_steps,
            "ridge": ridge,
            "learning_rate": learning_rate,
            "data": provenance,
            "train_engines": len(train),
            "validation_engines": len(val),
            "test_used": False,
            "results": results,
        }, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    """Execute the NEX-233 per-machine flow and non-flow study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root", type=Path,
        default=Path("/mnt/juicefs_prisma/models/nexuml/.experiments"),
    )
    parser.add_argument("--mode", choices=("causal", "retrospective", "both"), default="causal")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--prefix-fraction", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--calibration-windows", type=int, default=64)
    parser.add_argument("--max-engines", type=int, default=0)
    parser.add_argument("--integration-steps", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    manifest = run_study(
        args.experiment_root,
        mode=args.mode,
        device=device,
        prefix_fraction=args.prefix_fraction,
        seed=args.seed,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        calibration_windows=args.calibration_windows,
        max_engines=args.max_engines,
        integration_steps=args.integration_steps,
        ridge=args.ridge,
        learning_rate=args.learning_rate,
    )
    print(f"Within-machine RUL study: {manifest}")


if __name__ == "__main__":
    main()
