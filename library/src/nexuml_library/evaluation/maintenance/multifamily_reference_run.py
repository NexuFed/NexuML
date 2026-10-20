"""Run or summarize multi-seed MIMII sixteen-group zero-shot Flow Matching.

Each seed has independent healthy support/query partitions. Full experiments
compare a DeepSets machine context with the constant-context control using
machine-wise, rather than recording-pooled, SWD. Seeds are resamplings of the
same identities and must not be treated as independent machines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from nexuml_library.evaluation.maintenance.reference_conditioned_run import (
    run_reference_conditioned_folds,
)


def summarize_multifamily_experiments(
    results_root: str | Path,
    *,
    protocol: str,
    seeds: tuple[int, ...],
    modes: tuple[str, ...] = ("reference", "constant"),
) -> dict[str, Any]:
    """Aggregate machine-macro SWD from completed fold/seed reports.

    Returns:
        Per-seed, per-fold, per-family and mode summary without mixing machine
        identity counts with random-seed replication.

    Raises:
        FileNotFoundError: If any requested run has no complete summary.
        ValueError: If the chosen modes, fold keys or protocols disagree.
    """
    if protocol not in {"machine", "family"}:
        raise ValueError("Protocol must be machine or family")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide distinct split seeds")
    if not modes or any(mode not in {"reference", "constant"} for mode in modes):
        raise ValueError("Unknown context modes")
    root = Path(results_root).expanduser().resolve()
    record: dict[str, dict[str, dict[str, Any]]] = {}
    for seed in seeds:
        seed_runs: dict[str, dict[str, Any]] = {}
        for mode in modes:
            source = root / f"seed{seed}" / protocol / mode / f"summary-{mode}.json"
            summary = json.loads(source.read_text())
            expected_protocol = (
                "leave_one_machine_out_healthy_reference"
                if protocol == "machine"
                else "leave_one_family_out_healthy_reference"
            )
            if summary["protocol"] != expected_protocol or summary["context_mode"] != mode:
                raise ValueError("Mode or holdout protocol mismatch in completed summary")
            seed_runs[mode] = summary["folds"]
        if len(modes) == 2 and set(seed_runs[modes[0]]) != set(seed_runs[modes[1]]):
            raise ValueError("Reference and constant controls must evaluate the same folds")
        record[str(seed)] = seed_runs

    by_mode: dict[str, dict[str, Any]] = {}
    for mode in modes:
        records: list[dict[str, Any]] = []
        by_family: dict[str, list[float]] = {}
        for seed in seeds:
            for fold, result in sorted(record[str(seed)][mode].items()):
                per_machine = cast(
                    dict[str, dict[str, float]], result["unseen_by_machine"]
                )
                if not per_machine:
                    raise ValueError("Missing per-machine assessment in completed summary")
                for group, values in sorted(per_machine.items()):
                    family = group.split(":")[0]
                    reduction = float(values["swd_relative_reduction"])
                    by_family.setdefault(family, []).append(reduction)
                    records.append(
                        {
                            "seed": seed,
                            "fold": fold,
                            "machine": group,
                            "family": family,
                            "relative_reduction": reduction,
                        }
                    )
        per_seed = {
            str(seed): sum(
                item["relative_reduction"] for item in records if item["seed"] == seed
            ) / sum(item["seed"] == seed for item in records)
            for seed in seeds
        }
        by_mode[mode] = {
            "per_seed_machine_macro_reduction": per_seed,
            "mean_over_seed_machine_macros": sum(per_seed.values()) / len(per_seed),
            "positive_machine_assessments": sum(
                item["relative_reduction"] > 0 for item in records
            ),
            "machine_assessment_count": len(records),
            "by_family_mean_reduction": {
                name: sum(values) / len(values) for name, values in sorted(by_family.items())
            },
            "records": records,
        }
    paired_comparison: dict[str, Any] | None = None
    if "reference" in by_mode and "constant" in by_mode:
        baseline = {
            (item["seed"], item["fold"], item["machine"]): item["relative_reduction"]
            for item in by_mode["constant"]["records"]
        }
        pairs = [
            {
                "seed": item["seed"],
                "fold": item["fold"],
                "machine": item["machine"],
                "conditioning_advantage_pp": 100 * (
                    item["relative_reduction"]
                    - baseline[(item["seed"], item["fold"], item["machine"])]
                ),
            }
            for item in by_mode["reference"]["records"]
        ]
        paired_comparison = {
            "reference_wins": sum(item["conditioning_advantage_pp"] > 0 for item in pairs),
            "pairs": pairs,
        }
    return {
        "protocol": protocol,
        "seeds": list(seeds),
        "models": by_mode,
        "paired_comparison": paired_comparison,
        "caveat": (
            "Each physical machine/ID is reused across split seeds; "
            "seed-level variation is not independent machine-level evidence."
        ),
    }


def run_multifamily_experiments(
    prepared_index: str | Path,
    *,
    output_root: str | Path,
    protocol: str,
    modes: tuple[str, ...] = ("reference", "constant"),
    seeds: tuple[int, ...] | None = None,
    heldout_machine: str | None = None,
    width: int = 128,
    context_dim: int = 32,
    encoder_width: int = 128,
    num_steps: int = 50,
    lr: float = 3e-4,
    batch_size: int = 16,
    max_epochs: int = 300,
    patience: int = 25,
    accelerator: str = "auto",
) -> dict[str, Any]:
    """Train selected seed/fold variants and summarize their assessments.

    Outputs for each seed/protocol/mode are isolated. A completed run will
    refuse overwriting its summary and checkpoint paths.

    Returns:
        Aggregated SWD assessment over completed modes and seeds.

    Raises:
        FileExistsError: If a completed aggregate report already exists.
    """
    index = json.loads(Path(prepared_index).read_text())
    selected_seeds = seeds if seeds is not None else tuple(int(s) for s in index["seeds"])
    root = Path(output_root).expanduser().resolve()
    for seed in selected_seeds:
        source = Path(index["manifests"][str(seed)][protocol]["manifest"])
        for mode in modes:
            destination = root / f"seed{seed}" / protocol / mode
            run_reference_conditioned_folds(
                source,
                checkpoint_root=destination / "checkpoints",
                reports_dir=destination,
                heldout_machine=heldout_machine,
                context_mode=mode,
                width=width,
                context_dim=context_dim,
                encoder_width=encoder_width,
                num_steps=num_steps,
                lr=lr,
                batch_size=batch_size,
                max_epochs=max_epochs,
                patience=patience,
                accelerator=accelerator,
            )
    result = summarize_multifamily_experiments(
        root, protocol=protocol, seeds=selected_seeds, modes=modes
    )
    summary_path = root / f"aggregate-{protocol}.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite {summary_path}")
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Run independent held-out-machine or family experiments for selected seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", choices=("machine", "family"), required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("reference", "constant"),
        default=["reference", "constant"],
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--heldout-machine", type=str)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--context-dim", type=int, default=32)
    parser.add_argument("--encoder-width", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    selected = tuple(args.seeds) if args.seeds is not None else tuple(
        json.loads(args.prepared_index.read_text())["seeds"]
    )
    if args.summarize_only:
        result = summarize_multifamily_experiments(
            args.output_root, protocol=args.protocol, seeds=selected, modes=tuple(args.modes)
        )
    else:
        result = run_multifamily_experiments(
            args.prepared_index,
            output_root=args.output_root,
            protocol=args.protocol,
            modes=tuple(args.modes),
            seeds=selected,
            heldout_machine=args.heldout_machine,
            width=args.width,
            context_dim=args.context_dim,
            encoder_width=args.encoder_width,
            num_steps=args.num_steps,
            lr=args.lr,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            accelerator=args.accelerator,
        )
    print(json.dumps({
        mode: model["mean_over_seed_machine_macros"]
        for mode, model in result["models"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
