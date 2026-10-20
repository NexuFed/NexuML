"""Paired comparison of known-family and unseen-family zero-shot transport.

All four machine IDs of a selected family appear in both protocols at the
same split seed. For LOMO each ID is unseen and the other 15 machine IDs
train the model, whereas LOFO withholds the entire machine family.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast


def _read_scores(
    path: str | Path,
    protocol: str,
) -> tuple[tuple[int, ...], dict[str, dict[tuple[int, str], float]]]:
    """Read a completed aggregate and validate unique machine/seed identities.

    Returns:
        Sorted split seeds and per-mode mapping of seed/identity to SWD reduction.

    Raises:
        ValueError: If records are duplicated, corrupted or protocol-mismatched.
    """
    aggregate = json.loads(Path(path).expanduser().read_text())
    if aggregate.get("protocol") != protocol:
        raise ValueError(f"Expected {protocol} protocol aggregate: {path}")
    seeds = tuple(sorted(cast(list[int], aggregate["seeds"])))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Aggregate requires distinct nonempty seeds")
    result: dict[str, dict[tuple[int, str], float]] = {}
    for mode in ("reference", "constant"):
        model = cast(dict[str, Any], aggregate["models"][mode])
        records = cast(list[dict[str, Any]], model["records"])
        values: dict[tuple[int, str], float] = {}
        for item in records:
            seed = int(item["seed"])
            group = str(item["machine"])
            family = group.split(":")[0]
            fold = str(item["fold"])
            expected_fold = group if protocol == "machine" else f"family:{family}"
            if (
                seed not in seeds
                or group not in {
                    f"{family}:id_00",
                    f"{family}:id_02",
                    f"{family}:id_04",
                    f"{family}:id_06",
                }
                or family not in {"fan", "pump", "slider", "valve"}
                or fold != expected_fold
                or (seed, group) in values
            ):
                raise ValueError("Invalid, duplicate, or cross-protocol machine record")
            reduction = float(item["relative_reduction"])
            if not (-float("inf") < reduction < float("inf")):
                raise ValueError("Nonfinite relative SWD reduction")
            values[(seed, group)] = reduction
        if not values:
            raise ValueError(f"Missing {mode} measurements in {path}")
        result[mode] = values
    if set(result["reference"]) != set(result["constant"]):
        raise ValueError("Reference and constant contexts lack paired machine results")
    return seeds, result


def compare_machine_family_generalization(
    machine_aggregate: str | Path,
    family_aggregate: str | Path,
    *,
    family: str | None = None,
) -> dict[str, Any]:
    """Compare matched LOMO and LOFO assessments at each machine/seed.

    Positive relative SWD reduction is better than no transport. A positive
    known-family advantage means the fifteen-training-machine protocol
    outperforms training on twelve machines from the other three families.

    Args:
        machine_aggregate: Output of complete/selected machine-fold runner.
        family_aggregate: Output of complete/selected family-fold runner.
        family: Optional family filter, e.g. fan for the first pilot.

    Returns:
        Paired group/seed scores and separately macro-averaged family summaries.

    Raises:
        ValueError: If the folds, seeds or context modes are not matched.
    """
    if family is not None and family not in {"fan", "pump", "slider", "valve"}:
        raise ValueError("Unknown MIMII family")
    machine_seeds, machine_scores = _read_scores(machine_aggregate, "machine")
    family_seeds, family_scores = _read_scores(family_aggregate, "family")
    if machine_seeds != family_seeds:
        raise ValueError("LOMO and LOFO must use identical split seeds")
    selected: set[tuple[int, str]] = set()
    for mode in ("reference", "constant"):
        machine_keys = {
            key
            for key in machine_scores[mode]
            if family is None or key[1].startswith(f"{family}:")
        }
        family_keys = {
            key
            for key in family_scores[mode]
            if family is None or key[1].startswith(f"{family}:")
        }
        if machine_keys != family_keys:
            raise ValueError("LOMO and LOFO need exactly the same machine/seed assessments")
        if selected and selected != machine_keys:
            raise ValueError("Reference and constant scores cover different machines")
        selected = machine_keys
    if not selected:
        raise ValueError("No matched assessments found")
    families = sorted({group.split(":")[0] for _, group in selected})
    for name in families:
        for seed in machine_seeds:
            expected = {
                (seed, f"{name}:{identity}")
                for identity in ("id_00", "id_02", "id_04", "id_06")
            }
            if not expected.issubset(selected):
                raise ValueError(f"Missing complete four-machine family {name} for seed {seed}")
    pairs: list[dict[str, Any]] = []
    for seed, group in sorted(selected):
        machine_reference = machine_scores["reference"][(seed, group)]
        machine_constant = machine_scores["constant"][(seed, group)]
        family_reference = family_scores["reference"][(seed, group)]
        family_constant = family_scores["constant"][(seed, group)]
        pairs.append(
            {
                "seed": seed,
                "machine": group,
                "family": group.split(":")[0],
                "known_family_reference": machine_reference,
                "known_family_constant": machine_constant,
                "unseen_family_reference": family_reference,
                "unseen_family_constant": family_constant,
                "reference_transfer_advantage_pp": 100 * (
                    machine_reference - family_reference
                ),
                "constant_transfer_advantage_pp": 100 * (
                    machine_constant - family_constant
                ),
                "known_family_conditioning_advantage_pp": 100 * (
                    machine_reference - machine_constant
                ),
                "unseen_family_conditioning_advantage_pp": 100 * (
                    family_reference - family_constant
                ),
            }
        )
    score_keys = (
        "known_family_reference",
        "known_family_constant",
        "unseen_family_reference",
        "unseen_family_constant",
    )

    def average(items: list[dict[str, Any]], key: str) -> float:
        """Macro-average an equally weighted collection of machine scores."""
        return sum(float(item[key]) for item in items) / len(items)

    def metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate four model/protocol variants without pooling observations."""
        return {
            "machine_assessments": len(items),
            "mean_relative_reduction": {
                key: average(items, key) for key in score_keys
            },
            "positive_transport_count": {
                key: sum(item[key] > 0 for item in items) for key in score_keys
            },
            "mean_reference_transfer_advantage_pp": average(
                items, "reference_transfer_advantage_pp"
            ),
            "mean_constant_transfer_advantage_pp": average(
                items, "constant_transfer_advantage_pp"
            ),
            "known_family_conditioning_advantage_pp": average(
                items, "known_family_conditioning_advantage_pp"
            ),
            "unseen_family_conditioning_advantage_pp": average(
                items, "unseen_family_conditioning_advantage_pp"
            ),
        }

    return {
        "protocol": "paired_known_family_vs_unseen_family",
        "machine_aggregate": str(Path(machine_aggregate).expanduser().resolve()),
        "family_aggregate": str(Path(family_aggregate).expanduser().resolve()),
        "seeds": list(machine_seeds),
        "families": families,
        "overall": metrics(pairs),
        "by_family": {
            name: metrics([item for item in pairs if item["family"] == name])
            for name in families
        },
        "per_machine_seed": pairs,
        "caveat": (
            "LOMO and LOFO share source machine identities and splitting seeds "
            "but use different training-only standardizers. Relative SWD reductions "
            "are compared; raw SWD units are not directly comparable. "
            "Distinct split seeds do not imply independent physical machines."
        ),
    }


def main() -> None:
    """Write a paired LOMO/LOFO report without overwriting prior work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine-aggregate", type=Path, required=True)
    parser.add_argument("--family-aggregate", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--family", choices=("fan", "pump", "slider", "valve"))
    args = parser.parse_args()
    if args.output_file.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_file}")
    report = compare_machine_family_generalization(
        args.machine_aggregate, args.family_aggregate, family=args.family
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["by_family"], indent=2))


if __name__ == "__main__":
    main()
