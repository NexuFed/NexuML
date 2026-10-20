"""Extract a single frozen FISHER embedding space across all MIMII families.

The original MIMII machine IDs are *family-scoped*. Normal/abnormal
recordings are independently sampled within each family/ID, not longitudinal
health-to-failure trajectories. No normalization or flow training occurs here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from nexuml_library.data.maintenance.mimii import MIMIIAcousticEndpointPairs
from nexuml_library.data.maintenance.prepared_endpoints import read_prepared_artifact
from nexuml_library.layers.model.fisher import FISHEREncoder

MIMII_FAMILIES = ("fan", "pump", "slider", "valve")
MIMII_IDS = ("id_00", "id_02", "id_04", "id_06")


@torch.inference_mode()
def extract_multifamily_fisher(
    *,
    data_root: str | Path = "data/mimii",
    noise_db: int = 6,
    download: bool = True,
    max_pairs_per_machine: int = 128,
    clip_num_samples: int = 16384,
    batch_size: int = 16,
    device: torch.device,
    model_name: str = "jiangab/FISHER-small-0723",
    revision: str | None = None,
    pairing_seed: int = 42,
) -> dict[str, Any]:
    """Extract [N,2,D] FISHER endpoints for all sixteen machine groups.

    An identical frozen FISHER instance encodes *all* four families. Original
    machine-disjoint labels are retained as provenance, but new fold builders
    define their own machine-disjoint and reference/query-disjoint partitions.

    Returns:
        Existing nex233-features-v1 compatible independent/raw tensor payload.

    Raises:
        ValueError: If extraction settings, machine groups or files are invalid.
    """
    if (
        noise_db not in (-6, 0, 6)
        or max_pairs_per_machine <= 0
        or clip_num_samples <= 0
        or batch_size <= 0
    ):
        raise ValueError("Invalid MIMII extraction settings")
    encoder = FISHEREncoder(
        model_name=model_name,
        revision=revision,
        frozen=True,
        sample_rate=16000,
    ).to(device).eval()
    features: list[Tensor] = []
    rows: list[dict[str, Any]] = []
    groups: dict[str, int] = {}
    identities: set[str] = set()
    for family in MIMII_FAMILIES:
        source = MIMIIAcousticEndpointPairs(
            root=Path(data_root).expanduser() / f"{noise_db}_dB_{family}",
            download=download,
            noise_db=cast(Any, noise_db),
            machine_type=cast(Any, family),
            train_machine_ids=["id_00", "id_02"],
            val_machine_ids=["id_04"],
            test_machine_ids=["id_06"],
            include_test=True,
            max_pairs_per_machine=max_pairs_per_machine,
            clip_num_samples=clip_num_samples,
            seed=pairing_seed,
        )
        dataset = source.build()
        if dataset.meta is None:
            raise ValueError("MIMII source lacks machine provenance")
        meta = cast(list[dict[str, Any]], dataset.meta.to_dict(orient="records"))
        for row in meta:
            if row["group_id"] != f"{family}:{row['machine_id']}":
                raise ValueError("Family-qualified MIMII machine identity mismatch")
            for key in ("healthy_file", "faulty_file"):
                path = str(row[key])
                if path in identities:
                    raise ValueError("MIMII recording appears more than once")
                identities.add(path)
            groups[row["group_id"]] = groups.get(row["group_id"], 0) + 1
        print(f"Extracting frozen FISHER: {family} ({len(dataset)} pairs)", flush=True)
        for start in range(0, len(dataset), batch_size):
            items = [
                dataset[index][0]
                for index in range(start, min(start + batch_size, len(dataset)))
            ]
            healthy = torch.stack(
                [cast(Tensor, item["healthy_waveform"]) for item in items]
            )
            faulty = torch.stack(
                [cast(Tensor, item["faulty_waveform"]) for item in items]
            )
            vectors = encoder(torch.cat((healthy, faulty), dim=0).to(device))
            flat = vectors.detach().flatten(1).float().cpu()
            count = len(items)
            features.append(torch.stack((flat[:count], flat[count:]), dim=1))
        rows.extend(meta)
    expected = {f"{family}:{identity}" for family in MIMII_FAMILIES for identity in MIMII_IDS}
    if set(groups) != expected:
        raise ValueError(f"Expected sixteen distinct MIMII groups, got {sorted(groups)}")
    result = torch.cat(features).contiguous()
    if len(rows) != len(result) or not bool(torch.isfinite(result).all()):
        raise ValueError("MIMII multi-family embeddings are not aligned and finite")
    return {
        "format": "nex233-features-v1",
        "features": result,
        "rows": rows,
        "settings": {
            "frontend": "fisher",
            "coupling": "independent",
            "scaling": "raw",
            "seed": pairing_seed,
            "source_pairing_seed": pairing_seed,
            "normalization_fit_split": "none",
            "ot_fit_split": "none",
            "split_protocol": "multifamily_raw_for_reference_holdouts",
            "feature_dim": int(result.shape[2]),
            "model_name": model_name,
            "revision": revision,
            "noise_db": noise_db,
            "clip_num_samples": clip_num_samples,
            "max_pairs_per_machine": max_pairs_per_machine,
            "families": list(MIMII_FAMILIES),
            "machine_groups": sorted(groups),
            "group_pair_counts": dict(sorted(groups.items())),
            "original_id06_test_role": "repurposed_exploratory",
            "source_url": "https://zenodo.org/records/3384388",
        },
    }


def save_multifamily_fisher(
    output_file: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Write an immutable prepared artifact plus lightweight manifest.

    Returns:
        Descriptive summary including absolute feature file and group counts.

    Raises:
        FileExistsError: If the target artifact or manifest already exists.
    """
    output = Path(output_file).expanduser().resolve()
    manifest_file = output.with_suffix(".json")
    if output.exists() or manifest_file.exists():
        raise FileExistsError("Refusing to overwrite existing multi-family FISHER output")
    payload = extract_multifamily_fisher(**kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    read_prepared_artifact(output)
    summary = {
        "protocol": "multifamily_raw_for_reference_holdouts",
        "feature_file": str(output),
        "settings": payload["settings"],
        "pairs": len(payload["features"]),
        "machine_groups": payload["settings"]["machine_groups"],
    }
    manifest_file.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    """Extract all sixteen family-qualified MIMII machine IDs with FISHER."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/mimii"))
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--noise-db", type=int, choices=(-6, 0, 6), default=6)
    parser.add_argument("--max-pairs-per-machine", type=int, default=128)
    parser.add_argument("--clip-num-samples", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model-name", default="jiangab/FISHER-small-0723")
    parser.add_argument("--revision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    summary = save_multifamily_fisher(
        args.output_file,
        data_root=args.data_root,
        noise_db=args.noise_db,
        download=not args.no_download,
        max_pairs_per_machine=args.max_pairs_per_machine,
        clip_num_samples=args.clip_num_samples,
        batch_size=args.batch_size,
        device=torch.device(args.device),
        model_name=args.model_name,
        revision=args.revision,
        pairing_seed=args.seed,
    )
    print(f"Prepared {summary['pairs']} pairs from {len(summary['machine_groups'])} groups")


if __name__ == "__main__":
    main()
