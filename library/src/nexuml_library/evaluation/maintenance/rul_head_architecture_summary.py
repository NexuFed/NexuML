"""Validation-only matched comparison of Transformer, Mamba-style and CfC RUL heads."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Literal

from nexuml.core.components import LayerBuildContext
from nexuml_library.evaluation.maintenance.rul_generalization_summary import (
    _read_history,
)
from nexuml_library.layers.model.rul_sequence_flow import (
    PatchFormerSequenceFlowRULHead,
)

HeadArchitecture = Literal["transformer", "mamba", "cfc"]
_ARCHITECTURES: tuple[HeadArchitecture, ...] = ("transformer", "mamba", "cfc")
_ARMS = ("encoder", "flow")


def _parameter_count(architecture: HeadArchitecture) -> int:
    """Count trainable readout parameters, including shared pooling and MLP.

    Returns:
        Number of trainable RUL head parameters.
    """
    head = PatchFormerSequenceFlowRULHead(
        architecture=architecture,
        hidden_dim=64,
        attention_heads=4,
        dropout=0.1,
        initialization_seed=42,
    ).build(
        LayerBuildContext(
            input_sizes={"sequence_rul_features": (30, 256)},
            keys_in=["sequence_rul_features"],
            keys_out=["predicted_rul"],
        )
    )
    return sum(parameter.numel() for parameter in head.parameters())


def summarize_rul_head_architectures(
    root: Path,
    audit_root: Path,
    seeds: tuple[int, ...] = (42, 43, 44),
    subset: Literal["FD001", "FD003"] = "FD001",
) -> dict[str, Any]:
    """Evaluate all six matched head/feature cells using validation only.

    Args:
        root: Per-epoch full-split training/validation diagnostics.
        audit_root: Frozen-feature provenance and best-checkpoint manifests.
        seeds: Predeclared head-only initialization seeds.
        subset: Official C-MAPSS subset with separately frozen upstream checkpoints.

    Returns:
        Paired head architecture comparison, with frozen-source verification.

    Raises:
        ValueError: If paired runs, validation engines, or upstream hashes differ.
    """
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide unique predeclared RUL head seeds")
    if any(seed not in (42, 43, 44) for seed in seeds):
        raise ValueError("Unknown RUL head architecture seed")
    runs: dict[str, Any] = {}
    validation_engines: set[tuple[str, int]] | None = None
    encoder_digests: set[str] = set()
    endpoint_digests: set[str] = set()
    flow_digests: set[str] = set()
    for seed in seeds:
        for architecture in _ARCHITECTURES:
            for arm in _ARMS:
                name = f"rul-{subset}-cfc-head-{architecture}-{arm}-s{seed}"
                history, legacy = _read_history(
                    root / f"{name}.curves.csv"
                )
                best = min(history, key=lambda row: row["val_eval_mse"])
                last = history[-1]
                report = json.loads(
                    (root / f"{name}.curves.validation_engines.json").read_text(
                        encoding="utf-8"
                    )
                )
                engines = report["validation_engines"]
                identity = {
                    (str(row["asset_id"]), int(row["windows"])) for row in engines
                }
                if not identity or len(identity) != len(engines):
                    raise ValueError(f"Invalid validation engine identities: {name}")
                if validation_engines is None:
                    validation_engines = identity
                if identity != validation_engines:
                    raise ValueError("Architecture runs have different validation engines")
                if (
                    int(report["best_epoch"]) != int(best["epoch"])
                    or report.get("test_labels_used") is not False
                ):
                    raise ValueError(f"Invalid validation epoch/protocol: {name}")
                groups = report["validation_engine_groups"]
                if len(groups) != 4 or sum(
                    int(group["engines"]) for group in groups
                ) != len(engines):
                    raise ValueError(f"Invalid validation stress groups: {name}")
                provenance = json.loads(
                    (audit_root / f"{name}.geometry.json").read_text(
                        encoding="utf-8"
                    )
                )
                if (
                    provenance.get("subset") != subset
                    or provenance.get("seed") != 42
                    or provenance.get("architecture") != "cfc"
                    or provenance.get("use_flow") is not (arm == "flow")
                    or provenance.get("engine_balanced") is not False
                    or provenance.get("test_labels_used_in_feature_fitting") is not False
                ):
                    raise ValueError(f"Invalid frozen upstream RUL protocol: {name}")
                encoder_digests.add(
                    str(provenance["patchformer_checkpoint_sha256"])
                )
                endpoint_digests.add(str(provenance["endpoint_artifact_sha256"]))
                if arm == "flow":
                    flow_digests.add(str(provenance["flow_checkpoint_sha256"]))
                elif provenance["flow_checkpoint_sha256"] is not None:
                    raise ValueError(f"Encoder control unexpectedly loaded Flow: {name}")
                checkpoint = json.loads(
                    (audit_root / f"{name}.checkpoint.json").read_text(
                        encoding="utf-8"
                    )
                )
                if (
                    checkpoint.get("checkpoint_epoch") != int(best["epoch"])
                    or checkpoint.get("loaded_strict_model_state") is not True
                ):
                    raise ValueError(f"Checkpoint does not match validation selection: {name}")
                runs[f"{architecture}_{arm}_s{seed}"] = {
                    "best_epoch": int(best["epoch"]),
                    "last_fitting_epoch": int(last["epoch"]),
                    "best_train_eval_mse": best["train_eval_mse"],
                    "best_val_eval_mse": best["val_eval_mse"],
                    "best_mse_gap": best["mse_gap"],
                    "val_increase_after_best": (
                        last["val_eval_mse"] - best["val_eval_mse"]
                    ),
                    "val_worst_group_mse": max(
                        float(group["window_mse"]) for group in groups
                    ),
                    "val_engine_mean_mse": statistics.mean(
                        float(row["mse"]) for row in engines
                    ),
                    "legacy_postfit_validation_excluded": legacy,
                }
    if (
        len(encoder_digests) != 1
        or len(endpoint_digests) != 1
        or len(flow_digests) != 1
        or "None" in encoder_digests
        or "None" in endpoint_digests
        or "None" in flow_digests
    ):
        raise ValueError("Architecture runs use inconsistent frozen upstream weights")
    ranking: list[dict[str, Any]] = []
    for architecture in _ARCHITECTURES:
        flow = [runs[f"{architecture}_flow_s{seed}"] for seed in seeds]
        encoder = [runs[f"{architecture}_encoder_s{seed}"] for seed in seeds]
        flow_scores = [float(item["best_val_eval_mse"]) for item in flow]
        encoder_scores = [float(item["best_val_eval_mse"]) for item in encoder]
        ranking.append({
            "head_architecture": architecture,
            "trainable_parameters": _parameter_count(architecture),
            "flow_val_mse_mean": statistics.mean(flow_scores),
            "flow_val_mse_std": (
                statistics.stdev(flow_scores) if len(seeds) > 1 else None
            ),
            "encoder_val_mse_mean": statistics.mean(encoder_scores),
            "encoder_val_mse_std": (
                statistics.stdev(encoder_scores) if len(seeds) > 1 else None
            ),
            "paired_flow_minus_encoder_val_mse_mean": statistics.mean(
                flow_scores[i] - encoder_scores[i] for i in range(len(seeds))
            ),
            "flow_val_worst_group_mse_mean": statistics.mean(
                float(item["val_worst_group_mse"]) for item in flow
            ),
            "flow_val_engine_mean_mse_mean": statistics.mean(
                float(item["val_engine_mean_mse"]) for item in flow
            ),
            "flow_post_best_deterioration_mse_mean": statistics.mean(
                float(item["val_increase_after_best"]) for item in flow
            ),
        })
    ranking.sort(key=lambda entry: entry["flow_val_mse_mean"])
    return {
        "study": "NEX-233 frozen CfC Flow RUL estimator architecture comparison",
        "subset": subset,
        "frozen_upstream_seed": 42,
        "head_seeds": list(seeds),
        "architectures": list(_ARCHITECTURES),
        "arms": list(_ARMS),
        "mamba_implementation": (
            "Repository-native unfused PyTorch Mamba-style selective SSM, "
            "not the fused reference mamba_ssm implementation"
        ),
        "protocol": "Adam 1e-3, head width 64, dropout 0.1, window MSE",
        "selection": "mean best fitting-epoch validation-window MSE",
        "test_predictions_read": False,
        "best_validation_head_for_flow": ranking[0]["head_architecture"],
        "ranking": ranking,
        "per_run": runs,
        "caveat": (
            "The architectures have different trainable parameter counts. "
            "All head seeds share a frozen upstream seed-42 model and the same "
            "validation engines; stress groups are not independent splits."
        ),
    }


def main() -> None:
    """Write the complete matched head comparison without reading test CSVs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--subset", choices=("FD001", "FD003"), default="FD001")
    parser.add_argument(
        "--root", type=Path, default=None
    )
    parser.add_argument(
        "--audit-root", type=Path, default=Path(".experiments/rul/audits")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    subset: Literal["FD001", "FD003"] = args.subset
    root = args.root or (
        Path(".experiments/rul/head-architectures")
        / ("fd003" if subset == "FD003" else "")
    )
    output = args.output or root / "summary.json"
    report = summarize_rul_head_architectures(
        root, args.audit_root, tuple(args.head_seeds), subset
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"RUL head architecture comparison written to {output}")


if __name__ == "__main__":
    main()
