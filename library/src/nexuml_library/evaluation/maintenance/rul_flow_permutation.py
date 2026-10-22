"""Test-time conditional permutation of trained hybrid flow coordinates.

Keep supervised PatchFormer predictions and trained hybrid MLP fixed. Shuffle
only the two flow-derived coordinates among official test engines, jointly,
after validating exact agreement with the native per-engine audit. This is
a post-hoc diagnostic, not test-set-based parameter selection.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext
from nexuml_library.data.maintenance.rul import read_rul_artifact
from nexuml_library.data.maintenance.rul_flow_calibration import (
    AutoPatchFormerFlowRULFeatures,
)
from nexuml_library.evaluation.maintenance.rul_repro_summary import _read_engine_audit
from nexuml_library.layers.model.rul_flow_calibration import (
    PatchFormerFlowRULHead,
)
from nexuml_library.scenarios.maintenance.rul_flow_calibration import (
    patchformer_flow_rul_calibration,
)


def _reconstruct_hybrid(
    checkpoint: Path, seed: int,
) -> PipelineLayer:
    """Restore the frozen native hybrid RUL head from a trusted checkpoint.

    Returns:
        Evaluation-mode trained MLP with no refitting or checkpoint selection.

    Raises:
        ValueError: If the checkpoint has incomplete or ambiguous parameters.
    """
    definition = PatchFormerFlowRULHead(mode="hybrid", initialization_seed=seed)
    runtime = definition.build(
        LayerBuildContext(
            input_sizes={"flow_rul_features": (3,)},
            keys_in=["flow_rul_features"],
            keys_out=["predicted_rul"],
        )
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Expected native hybrid checkpoint dictionary")
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise ValueError("Expected hybrid checkpoint state_dict")
    weights: dict[str, Tensor] = {}
    for key, expected in runtime.state_dict().items():
        matches = [
            value for full_name, value in state.items()
            if isinstance(full_name, str)
            and (full_name == key or full_name.endswith(f".{key}"))
            and isinstance(value, Tensor)
            and value.shape == expected.shape
        ]
        if len(matches) != 1:
            raise ValueError(f"Missing or ambiguous hybrid checkpoint parameter: {key}")
        weights[key] = matches[0]
    runtime.load_state_dict(weights, strict=True)
    runtime.eval()
    runtime.requires_grad_(False)
    return runtime


def _metrics(predicted: Tensor, target: Tensor) -> dict[str, float]:
    """Calculate exact RUL error metrics using the existing NASA convention.

    Returns:
        MAE, RMSE, positive error bias and summed asymmetric NASA score.
    """
    errors = (predicted.flatten() - target.flatten()).double()
    return {
        "rmse": math.sqrt(float(errors.square().mean())),
        "mae": float(errors.abs().mean()),
        "late_bias": float(errors.clamp_min(0).mean()),
        "nasa": float(
            torch.expm1(
                torch.where(errors >= 0, errors / 10, -errors / 13)
            ).sum()
        ),
    }


def permutation_impact(
    model: PipelineLayer,
    features: Tensor,
    targets: dict[str, Tensor],
    *,
    repeats: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """Quantify fixed-model degradation when jointly shuffling flow features.

    Args:
        model: Previously trained hybrid runtime in evaluation mode.
        features: [100,3] time, distance and unchanged PatchFormer prediction.
        targets: Raw and capped official RUL labels, untouched by permutations.
        repeats: Number of deterministic joint-coordinate permutations.
        seed: Reproducible diagnostic random seed.

    Returns:
        Baseline metrics, shuffled means, and paired metric increases.

    Raises:
        ValueError: If test shapes or permutation count are invalid.
    """
    if features.ndim != 2 or features.shape != (100, 3) or repeats < 1:
        raise ValueError("Permutation requires 100 official [t,d,baseline] observations")
    if any(values.shape != (100, 1) for values in targets.values()):
        raise ValueError("Permutation labels must align with 100 test engines")
    with torch.no_grad():
        baseline = model.forward_tensor(features).detach().float()
        original = {
            kind: _metrics(baseline, target) for kind, target in targets.items()
        }
        samples: dict[str, list[dict[str, float]]] = {
            name: [] for name in targets
        }
        gen = torch.Generator().manual_seed(seed)
        for _ in range(repeats):
            shuffled = features.clone()
            shuffle = torch.randperm(len(features), generator=gen)
            shuffled[:, :2] = features[shuffle, :2]
            predictions = model.forward_tensor(shuffled)
            for name, target in targets.items():
                samples[name].append(_metrics(predictions, target))
    return {
        "permutation": "joint_time_distance_across_test_engines",
        "seed": seed,
        "repeats": repeats,
        "results": {
            name: {
                "original": original[name],
                "shuffled_mean": {
                    metric: statistics.mean(row[metric] for row in samples[name])
                    for metric in ("rmse", "mae", "late_bias", "nasa")
                },
                "mean_delta_shuffled_minus_original": {
                    metric: statistics.mean(
                        row[metric] - original[name][metric] for row in samples[name]
                    )
                    for metric in ("rmse", "mae", "late_bias", "nasa")
                },
            }
            for name in targets
        },
    }


def analyze_trained_hybrid(
    subset: Literal["FD001", "FD003"], seed: int, *, repeats: int = 100
) -> dict[str, Any]:
    """Load native frozen source and verify trained baseline against audit.

    Returns:
        A post-hoc permutation report for the specified completed experiment.

    Raises:
        ValueError: If original predictions diverge from saved engine audit.
    """
    spec = patchformer_flow_rul_calibration(subset, seed, "hybrid")
    source = spec.data.datasets[0].source
    if not isinstance(source, AutoPatchFormerFlowRULFeatures):
        raise ValueError("Requires the native hybrid flow-feature data source")
    dataset = source.build()
    prepared = read_rul_artifact(source.prepared_rul_path)
    rows = cast(list[dict[str, Any]], prepared["rows"])
    test_ids = [i for i, row in enumerate(rows) if row["split"] == "test"]
    if len(test_ids) != 100:
        raise ValueError("Official C-MAPSS test population must have 100 engines")
    features = cast(Tensor, dataset.data)[test_ids]
    labels = {
        "capped": cast(Tensor, prepared["targets"])[test_ids],
        "raw": cast(Tensor, prepared["raw_targets"])[test_ids],
    }
    run = spec.name
    manifest = Path(".experiments/rul/audits") / f"{run}.checkpoint.json"
    record = json.loads(manifest.read_text(encoding="utf-8"))
    checkpoint = Path(str(record["best_model_path"])).expanduser().resolve()
    if checkpoint.parent.name != run or not checkpoint.is_file():
        raise ValueError("Hybrid RUL checkpoint manifest does not match native run")
    model = _reconstruct_hybrid(checkpoint, seed)
    audit = _read_engine_audit(Path(".experiments/rul/audits") / f"{run}.csv")
    with torch.no_grad():
        original_predictions = model.forward_tensor(features).flatten()
    for i, index in enumerate(test_ids):
        engine = str(rows[index]["asset_id"])
        if engine not in audit:
            raise ValueError(f"Missing official test engine in hybrid audit: {engine}")
        if not math.isclose(
            float(original_predictions[i]),
            audit[engine]["predicted_rul"],
            abs_tol=0.005,
        ):
            raise ValueError(
                "Restored trained hybrid does not reproduce the official audit "
                f"prediction for {engine}"
            )
    report = permutation_impact(
        model, features, labels, repeats=repeats, seed=seed
    )
    report.update({"subset": subset, "model": run, "audit_verified": True})
    return report


def main() -> None:
    """Write a deterministic post-hoc shuffled-flow evaluation report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=("FD001", "FD003"), default="FD001")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--output", type=Path,
        default=Path(".experiments/rul/audits/hybrid-flow-permutation.json"),
    )
    args = parser.parse_args()
    report = analyze_trained_hybrid(args.subset, args.seed, repeats=args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Flow-coordinate permutation report written to {args.output}")


if __name__ == "__main__":
    main()
