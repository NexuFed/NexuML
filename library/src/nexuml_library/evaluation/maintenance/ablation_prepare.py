"""Offline feature extraction and training-only OT coupling for NEX-233."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext
from nexuml_library.data.maintenance.mimii import MIMIIAcousticEndpointPairs
from nexuml_library.evaluation.maintenance.velocity_diagnostics import (
    neighbor_velocity_alignment,
    velocity_statistics,
)
from nexuml_library.layers.feature.lmbe import LMBE
from nexuml_library.layers.model.fisher import FISHEREncoder

Frontend = Literal["logmel", "fisher"]
Coupling = Literal["independent", "ot"]


def training_normalizer(features: Tensor, rows: list[dict[str, Any]]) -> tuple[Tensor, Tensor]:
    """Fit per-feature pooled endpoint statistics exclusively on training examples.

    Returns:
        Mean and clamped standard deviation, each shaped [1,1,D].

    Raises:
        ValueError: If data are not aligned or training examples are missing.
    """
    if features.ndim != 3 or features.shape[1] != 2 or len(rows) != len(features):
        raise ValueError("Expected aligned [N,2,D] features and provenance")
    indices = [i for i, row in enumerate(rows) if row["split"] == "train"]
    if not indices:
        raise ValueError("At least one training pair is required")
    pooled = features[indices].reshape(-1, features.shape[-1])
    return (
        pooled.mean(dim=0).reshape(1, 1, -1),
        pooled.std(dim=0, unbiased=False).clamp_min(1e-6).reshape(1, 1, -1),
    )


def couple_training_endpoints(
    features: Tensor,
    rows: list[dict[str, Any]],
    *,
    coupling: Coupling,
) -> tuple[Tensor, list[dict[str, Any]], dict[str, float]]:
    """Apply independent or within-training-machine OT assignment.

    Validation/test source and target correspondences remain unchanged.

    Returns:
        Paired tensor, aligned metadata and training pair-cost statistics.

    Raises:
        ValueError: If inputs, metadata or coupling configuration are invalid.
        ImportError: If the optional SciPy Hungarian solver is missing.
    """
    if coupling not in {"independent", "ot"}:
        raise ValueError(f"Invalid coupling mode: {coupling}")
    if features.ndim != 3 or features.shape[1] != 2 or len(features) != len(rows):
        raise ValueError("Coupling requires aligned [N,2,D] endpoints and rows")
    train = [i for i, row in enumerate(rows) if row["split"] == "train"]
    if not train:
        raise ValueError("Coupling requires a training split")
    initial = float((features[train, 1] - features[train, 0]).square().mean())
    updated = features.clone()
    metadata = [dict(row) for row in rows]
    if coupling == "ot":
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:
            raise ImportError("OT coupling requires scipy") from exc
        groups = sorted({rows[i]["group_id"] for i in train})
        for group in groups:
            local = [i for i in train if rows[i]["group_id"] == group]
            squared = torch.cdist(features[local, 0], features[local, 1]).square()
            left, right = linear_sum_assignment(squared.cpu().numpy())
            for j, k in zip(left.tolist(), right.tolist(), strict=True):
                source, target = local[j], local[k]
                updated[source, 1] = features[target, 1]
                metadata[source]["faulty_file"] = rows[target]["faulty_file"]
                metadata[source]["faulty_group"] = rows[target]["faulty_group"]
                metadata[source]["pair_id"] = f"{rows[source]['pair_id']}:ot:{k}"
    final = float((updated[train, 1] - updated[train, 0]).square().mean())
    return updated, metadata, {
        "independent_train_pair_mse": initial,
        "selected_train_pair_mse": final,
        "relative_pair_mse_reduction": 1.0 - final / initial if initial else 0.0,
    }


def feature_variants(
    features: Tensor,
    rows: list[dict[str, Any]],
    *,
    frontend: Frontend,
    seed: int,
    model_name: str = "",
) -> dict[str, dict[str, Any]]:
    """Produce all four coupling/scaling variations for one fixed encoder.

    Returns:
        Variant name to independently serialized feature dataset payload.

    Raises:
        ValueError: If feature inputs are invalid or nonfinite.
    """
    if (
        features.ndim != 3
        or features.shape[1] != 2
        or len(rows) != len(features)
        or not features.is_floating_point()
        or not torch.isfinite(features).all()
    ):
        raise ValueError("Features must be finite floating [N,2,D] with aligned rows")
    mean, std = training_normalizer(features, rows)
    variants: dict[str, dict[str, Any]] = {}
    for scale in ("raw", "standardized"):
        selected = features.clone() if scale == "raw" else (features - mean) / std
        for coupling in ("independent", "ot"):
            paired, metadata, costs = couple_training_endpoints(
                selected, rows, coupling=cast(Coupling, coupling)
            )
            key = f"{frontend}-{coupling}-{scale}"
            train_indices = [
                index for index, row in enumerate(metadata) if row["split"] == "train"
            ]
            train_healthy = paired[train_indices, 0]
            train_faulty = paired[train_indices, 1]
            coherence = velocity_statistics(train_healthy, train_faulty)
            neighbor_alignment = (
                neighbor_velocity_alignment(train_healthy, train_faulty, k=8)
                if len(train_indices) > 1
                else []
            )
            variants[key] = {
                "format": "nex233-features-v1",
                "features": paired.cpu().contiguous(),
                "rows": metadata,
                "training_mean": mean.cpu(),
                "training_std": std.cpu(),
                "settings": {
                    "frontend": frontend,
                    "coupling": coupling,
                    "scaling": scale,
                    "seed": seed,
                    "model_name": model_name,
                    "feature_dim": int(features.shape[-1]),
                    "normalization_fit_split": "train",
                    "ot_fit_split": "train" if coupling == "ot" else "none",
                    "ot_grouping": "machine_id",
                    "training_velocity_statistics": coherence,
                    "training_neighbor_alignment": neighbor_alignment,
                    **costs,
                },
            }
    return variants


def extract_mimii_endpoints(
    source: MIMIIAcousticEndpointPairs,
    *,
    frontend: Frontend,
    device: torch.device,
    batch_size: int,
    model_name: str,
    revision: str | None,
) -> tuple[Tensor, list[dict[str, Any]]]:
    """Extract all machine-disjoint recordings through one frozen frontend.

    Returns:
        CPU features [N,2,D] and source metadata.

    Raises:
        ValueError: If the source or extraction options are invalid.
    """
    if batch_size < 1 or frontend not in {"logmel", "fisher"}:
        raise ValueError("Extraction needs valid frontend and positive batch size")
    dataset = source.build()
    if dataset.meta is None:
        raise ValueError("Expected split and machine metadata in MIMII source")
    encoder: torch.nn.Module
    if frontend == "fisher":
        encoder = FISHEREncoder(
            model_name=model_name,
            revision=revision,
            frozen=True,
            sample_rate=source.sample_rate,
        ).to(device).eval()
    else:
        encoder = LMBE(
            sample_rate=source.sample_rate,
            n_fft=512,
            hop_length=256,
            n_mels=32,
            fmax=source.sample_rate // 2,
        ).build(
            LayerBuildContext(
                input_sizes={"waveform": (source.clip_num_samples,)},
                keys_in=["waveform"],
                keys_out=["embedding"],
            )
        ).to(device).eval()
    batches: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            items = [dataset[i][0] for i in range(start, min(start + batch_size, len(dataset)))]
            healthy = torch.stack([cast(Tensor, item["healthy_waveform"]) for item in items])
            faulty = torch.stack([cast(Tensor, item["faulty_waveform"]) for item in items])
            combined = torch.cat((healthy, faulty), dim=0).to(device)
            if isinstance(encoder, FISHEREncoder):
                vectors = encoder(combined)
            else:
                vectors = cast(Tensor, cast(PipelineLayer, encoder).forward_tensor(combined))
            flat = vectors.detach().flatten(1).float().cpu()
            count = len(items)
            batches.append(torch.stack((flat[:count], flat[count:]), dim=1))
    if not batches:
        raise ValueError("MIMII feature dataset is empty")
    return torch.cat(batches), dataset.meta.to_dict(orient="records")


def prepare_ablation(
    source: MIMIIAcousticEndpointPairs,
    *,
    output_dir: Path,
    frontends: tuple[Frontend, ...],
    device: torch.device,
    batch_size: int,
    model_name: str,
    revision: str | None,
) -> dict[str, Any]:
    """Save reproducible feature variants without overwriting prior artifacts.

    Returns:
        Manifest mapping each experiment variant to its prepared artifact.

    Raises:
        FileExistsError: If an existing variant artifact would be overwritten.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"Experiment manifest already exists: {manifest_path}. "
            "Use a new output directory for independent preparation runs."
        )
    manifest: dict[str, Any] = {}
    for frontend in frontends:
        features, rows = extract_mimii_endpoints(
            source,
            frontend=frontend,
            device=device,
            batch_size=batch_size,
            model_name=model_name,
            revision=revision,
        )
        variants = feature_variants(
            features,
            rows,
            frontend=frontend,
            seed=source.seed,
            model_name=model_name if frontend == "fisher" else "",
        )
        for name, payload in variants.items():
            settings = payload["settings"]
            settings.update(
                {
                    "revision": revision or "",
                    "machine_type": source.machine_type,
                    "noise_db": source.noise_db,
                    "train_machine_ids": source.train_machine_ids,
                    "val_machine_ids": source.val_machine_ids,
                    "test_machine_ids": source.test_machine_ids,
                    "clip_num_samples": source.clip_num_samples,
                }
            )
            file = output_dir / f"{name}-seed{source.seed}.pt"
            if file.exists():
                raise FileExistsError(f"Prepared artifact exists: {file}")
            torch.save(payload, file)
            manifest[name] = {"path": str(file.resolve()), **settings}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    """Run eight-configuration feature preparation from the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--machine-type", choices=("fan", "pump", "slider", "valve"), default="fan"
    )
    parser.add_argument("--noise-db", type=int, choices=(-6, 0, 6), default=6)
    parser.add_argument("--frontend", choices=("logmel", "fisher", "both"), default="both")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-pairs-per-machine", type=int, default=128)
    parser.add_argument("--clip-num-samples", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", default="jiangab/FISHER-small-0723")
    parser.add_argument("--revision")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    options: tuple[Frontend, ...] = (
        ("logmel", "fisher") if args.frontend == "both" else (cast(Frontend, args.frontend),)
    )
    source = MIMIIAcousticEndpointPairs(
        root=args.root,
        download=not args.no_download,
        machine_type=args.machine_type,
        noise_db=args.noise_db,
        max_pairs_per_machine=args.max_pairs_per_machine,
        clip_num_samples=args.clip_num_samples,
        seed=args.seed,
    )
    manifest = prepare_ablation(
        source,
        output_dir=args.output_dir,
        frontends=options,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        model_name=args.model_name,
        revision=args.revision,
    )
    print(f"Prepared {len(manifest)} variants in {args.output_dir}")


if __name__ == "__main__":
    main()
