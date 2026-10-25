"""Post-hoc frozen-CfC RUL failure geometry; never select on test labels.

Rebuild the exact preselected FD001 Gen2 readouts and encoder/Flow features,
verify predictions against native audit CSVs, and export diagnostic-only
per-engine displacement geometry and within-readout channel sensitivity.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext
from nexuml_library.data.maintenance.rul import read_rul_artifact
from nexuml_library.data.maintenance.rul_sequence_flow import (
    AutoPatchFormerSequenceFlowRULFeatures,
)
from nexuml_library.evaluation.maintenance.rul_repro_summary import (
    _read_engine_audit,
)
from nexuml_library.layers.model.rul_sequence_flow import (
    PatchFormerSequenceFlowRULHead,
)
from nexuml_library.scenarios.maintenance.rul_gen2 import (
    cfc_rul_optimization_study,
)


def _run_name(arm: str, seed: int) -> str:
    """Identify the predeclared validation-selected Gen2 configuration.

    Returns:
        Native NexuML run identifier.
    """
    return f"rul-FD001-cfc-gen2-{arm}-d010-adam_1e3-window-head-s{seed}"


def _head_from_checkpoint(path: Path, seed: int) -> PipelineLayer:
    """Reconstruct a trained width-64 readout without running training.

    Args:
        path: Local trusted Lightning checkpoint from a completed Gen2 run.
        seed: Readout initialization seed.

    Returns:
        Frozen evaluation-mode readout.

    Raises:
        ValueError: If the checkpoint has ambiguous, missing or invalid weights.
    """
    runtime = PatchFormerSequenceFlowRULHead(
        hidden_dim=64,
        attention_heads=4,
        dropout=0.1,
        initialization_seed=seed,
    ).build(
        LayerBuildContext(
            input_sizes={"sequence_rul_features": (30, 256)},
            keys_in=["sequence_rul_features"],
            keys_out=["predicted_rul"],
        )
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError(f"Invalid native Gen2 checkpoint: {path}")
    state = cast(dict[str, Any], payload["state_dict"])
    weights: dict[str, Tensor] = {}
    for key, expected in runtime.state_dict().items():
        matched = [
            value
            for name, value in state.items()
            if isinstance(name, str)
            and (name == key or name.endswith(f".{key}"))
            and isinstance(value, Tensor)
        ]
        if len(matched) != 1 or matched[0].shape != expected.shape:
            raise ValueError(f"Missing/ambiguous Gen2 head parameter: {key}")
        weights[key] = matched[0]
    runtime.load_state_dict(weights, strict=True)
    runtime.eval().requires_grad_(False)
    return runtime


def _diagnostic_geometry(
    features: Tensor, training: Tensor, training_rul: Tensor
) -> dict[str, Tensor]:
    """Compute test-independent high-/low-RUL reference geometry.

    The severity axis joins pooled *training* high-RUL and low-RUL centers.
    It is descriptive, not a physical degradation clock. No held-out labels
    enter reference fitting.

    Args:
        features: Frozen features [N,30,256] for an arbitrary evaluation split.
        training: Frozen features [M,30,256] from training engines only.
        training_rul: Aligned capped training RUL targets [M,1].

    Returns:
        Per-observation geometry including displacement-RMS train percentile.

    Raises:
        ValueError: If stage references or feature shapes are invalid.
    """
    if (
        features.ndim != 3
        or training.ndim != 3
        or features.shape[1:] != (30, 256)
        or training.shape[1:] != (30, 256)
        or training_rul.shape != (len(training), 1)
    ):
        raise ValueError("Expected aligned [N,30,256] frozen Gen2 features")
    high = training_rul[:, 0] >= 100.0
    low = training_rul[:, 0] <= 30.0
    if not bool(high.any()) or not bool(low.any()):
        raise ValueError("Need both high- and low-RUL training stages")
    pooled_training = training[:, :, :128].mean(dim=1)
    high_center = pooled_training[high].mean(dim=0)
    low_center = pooled_training[low].mean(dim=0)
    axis = low_center - high_center
    axis_norm = axis.norm()
    if not bool(torch.isfinite(axis_norm)) or float(axis_norm) <= 1e-8:
        raise ValueError("Training stage centers are not separable")
    axis = axis / axis_norm
    z = features[:, :, :128]
    delta = features[:, :, 128:]
    pooled_z = z.mean(dim=1)
    pooled_delta = delta.mean(dim=1)
    delta_rms = delta.square().mean(dim=(1, 2)).sqrt()
    training_delta_rms = training[:, :, 128:].square().mean(dim=(1, 2)).sqrt()
    sorted_train = training_delta_rms.sort().values
    percentile = torch.searchsorted(
        sorted_train, delta_rms.contiguous(), right=True
    ).float() / len(sorted_train)
    nearest_train = (
        torch.cdist(pooled_z.float(), pooled_training.float()).min(dim=1).values
    )
    return {
        "delta_rms": delta_rms,
        "train_delta_rms_percentile": percentile,
        "delta_toward_low_rul": pooled_delta @ axis,
        "baseline_degradation_projection": (pooled_z - high_center) @ axis,
        "transported_degradation_projection": (
            pooled_z + pooled_delta - high_center
        ) @ axis,
        "nearest_train_pooled_embedding_distance": nearest_train,
    }


def _asym_score(prediction: float, target: float) -> float:
    """Compute published-sign-convention NASA score for one engine.

    Returns:
        Nonnegative NASA asymmetric error penalty.
    """
    error = prediction - target
    return math.expm1(error / 10.0 if error >= 0 else -error / 13.0)


def analyze_gen2_failures(
    *,
    audit_root: Path = Path(".experiments/rul/audits"),
    gen2_root: Path = Path(".experiments/rul/gen2"),
    seeds: tuple[int, ...] = (42, 43, 44),
) -> dict[str, Any]:
    """Audit fixed Gen2 heads, test residuals, and frozen displacement geometry.

    All models and hyperparameters were selected previously on validation.
    The official test set is used only for post-hoc diagnosis. The channel
    suppression probe is off-manifold and must not be treated as a new model.

    Returns:
        Per-test-engine diagnostic records and matched seed-level aggregates.

    Raises:
        ValueError: If any native audit, split, or checkpoint is inconsistent.
    """
    summary = json.loads((gen2_root / "summary.json").read_text(encoding="utf-8"))
    if (
        summary.get("best_validation_configuration") != "d010_adam_1e3_window"
        or summary.get("head_seeds") != list(seeds)
        or summary.get("test_predictions_read") is not False
    ):
        raise ValueError("Gen2 validation selection/seeds do not match diagnostic")
    study = cfc_rul_optimization_study("flow", "d010", "adam_1e3", "window", 42)
    source = study.data.datasets[0].source
    if not isinstance(source, AutoPatchFormerSequenceFlowRULFeatures):
        raise ValueError("Gen2 flow source contract has changed")
    # Keep native experiment provenance untouched by the diagnostic rebuild.
    source = source.model_copy(
        update={
            "diagnostics_file": str(audit_root / "gen2-failure-feature-provenance.json")
        }
    )
    dataset = source.build()
    rebuilt = json.loads(Path(source.diagnostics_file).read_text(encoding="utf-8"))
    original = json.loads(
        (audit_root / f"{_run_name('flow', 42)}.geometry.json").read_text(
            encoding="utf-8"
        )
    )
    for key in (
        "patchformer_checkpoint_sha256",
        "flow_checkpoint_sha256",
        "endpoint_artifact_sha256",
    ):
        if rebuilt[key] != original[key]:
            raise ValueError(f"Frozen feature provenance changed: {key}")
    features = cast(Tensor, dataset.data).detach().cpu()
    metadata = dataset.meta.reset_index(drop=True)
    train_indices = [i for i, row in metadata.iterrows() if row["split"] == "train"]
    test_indices = [i for i, row in metadata.iterrows() if row["split"] == "test"]
    if len(test_indices) != 100 or not train_indices:
        raise ValueError("Expected training windows and 100 official FD001 engines")
    ids = [str(metadata.iloc[index]["asset_id"]) for index in test_indices]
    if len(set(ids)) != 100:
        raise ValueError("Test engine identities are not unique")
    test = features[test_indices]
    train = features[train_indices]
    prepared = read_rul_artifact(source.prepared_rul_path)
    train_rul = cast(Tensor, prepared["targets"])[train_indices].detach().cpu()
    geometry = _diagnostic_geometry(test, train, train_rul)
    encoder_features = test.clone()
    encoder_features[:, :, 128:] = 0.0
    per_engine: dict[str, dict[str, Any]] = {
        asset: {key: float(value[index]) for key, value in geometry.items()}
        for index, asset in enumerate(ids)
    }
    per_seed: dict[str, Any] = {}
    for seed in seeds:
        audits = {
            arm: _read_engine_audit(audit_root / f"{_run_name(arm, seed)}.csv")
            for arm in ("encoder", "flow")
        }
        if set(audits["encoder"]) != set(ids) or set(audits["flow"]) != set(ids):
            raise ValueError(f"Audit engines differ from frozen source for seed {seed}")
        predictions: dict[str, Tensor] = {}
        suppressed: Tensor | None = None
        for arm in ("encoder", "flow"):
            run = _run_name(arm, seed)
            manifest = json.loads(
                (audit_root / f"{run}.checkpoint.json").read_text(encoding="utf-8")
            )
            best_key = f"{arm}_d010_adam_1e3_window_s{seed}"
            if manifest.get("checkpoint_epoch") != summary["per_run"][best_key][
                "best_epoch"
            ]:
                raise ValueError(f"Best-validation checkpoint mismatch: {run}")
            path = Path(str(manifest["best_model_path"])).expanduser().resolve()
            if not path.is_file() or path.parent.name != run:
                raise ValueError(f"Missing or nonmatching native checkpoint: {run}")
            model = _head_from_checkpoint(path, seed)
            with torch.no_grad():
                prediction = model.forward_tensor(
                    test if arm == "flow" else encoder_features
                ).flatten()
                if arm == "flow":
                    suppressed = model.forward_tensor(encoder_features).flatten()
            predictions[arm] = prediction
            for index, asset in enumerate(ids):
                expected = audits[arm][asset]["predicted_rul"]
                if not math.isclose(float(prediction[index]), expected, abs_tol=0.05):
                    raise ValueError(
                        f"Rebuilt {arm} prediction differs from native audit: "
                        f"{asset}, seed {seed}"
                    )
        if suppressed is None:
            raise ValueError("Missing displacement-suppressed readout probe")
        changed: list[float] = []
        for index, asset in enumerate(ids):
            reference = audits["flow"][asset]
            encoder = audits["encoder"][asset]
            if any(
                reference[key] != encoder[key]
                for key in ("true_rul_raw", "true_rul_capped")
            ):
                raise ValueError(f"Mismatched native RUL targets: {asset}")
            actual = float(predictions["flow"][index])
            control = float(predictions["encoder"][index])
            raw = reference["true_rul_raw"]
            diff = actual - control
            changed.append(diff)
            per_engine[asset][f"seed_{seed}"] = {
                "true_rul_raw": raw,
                "encoder_prediction": control,
                "flow_prediction": actual,
                "flow_minus_encoder_prediction": diff,
                "flow_minus_encoder_squared_error": (
                    (actual - raw) ** 2 - (control - raw) ** 2
                ),
                "flow_minus_encoder_nasa": (
                    _asym_score(actual, raw) - _asym_score(control, raw)
                ),
                "flow_head_without_displacement": float(suppressed[index]),
                "flow_head_displacement_effect": actual - float(suppressed[index]),
            }
        per_seed[str(seed)] = {
            "flow_higher_rul_count": sum(delta > 0 for delta in changed),
            "mean_flow_minus_encoder_prediction": sum(changed) / len(changed),
            "audit_verified": True,
        }
    ranked = sorted(
        ids,
        key=lambda asset: sum(
            per_engine[asset][f"seed_{seed}"]["flow_minus_encoder_nasa"]
            for seed in seeds
        ),
        reverse=True,
    )
    return {
        "study": "NEX-233 Gen2 frozen CfC post-hoc failure geometry",
        "upstream": "FD001 seed-42 frozen PatchFormer and CfC Flow",
        "configuration": "d010_adam_1e3_window",
        "selection": "locked validation selection; no test tuning",
        "caveat": (
            "High-/low-RUL geometry uses training windows only (>=100 versus <=30) "
            "and pooled embeddings. Displacement suppression is an off-manifold "
            "within-head probe, not a separately trained baseline or a causal effect."
        ),
        "per_seed": per_seed,
        "worst_cross_seed_nasa_engines": ranked[:15],
        "per_engine": per_engine,
    }


def main() -> None:
    """Export post-hoc geometry without changing checkpoints or training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".experiments/rul/audits/gen2-flow-failure-diagnostics.json"),
    )
    args = parser.parse_args()
    report = analyze_gen2_failures()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Gen2 failure diagnostics written to {args.output}")
    for asset in report["worst_cross_seed_nasa_engines"][:10]:
        row = report["per_engine"][asset]
        print(
            f"{asset}: displacement_rms={row['delta_rms']:.3f}, "
            f"train_percentile={row['train_delta_rms_percentile']:.2%}, "
            f"projection_toward_low_rul={row['delta_toward_low_rul']:+.3f}"
        )
        for seed in (42, 43, 44):
            result = row[f"seed_{seed}"]
            print(
                f"  s{seed}: flow_minus_encoder="
                f"{result['flow_minus_encoder_prediction']:+.2f} cycles, "
                f"displacement_probe="
                f"{result['flow_head_displacement_effect']:+.2f} cycles, "
                f"delta_nasa={result['flow_minus_encoder_nasa']:+.2f}"
            )


if __name__ == "__main__":
    main()
