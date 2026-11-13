"""Recover the minimal FD003 prerequisites for the within-machine flow study.

This is deliberately separate from the study: recovering raw observations and
training a frozen upstream encoder must never happen implicitly during scoring.
Existing artifacts are validated and never replaced. Missing upstream weights
can only be recreated by explicitly requesting a new native NexuML training run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any, cast

import torch

from nexuml_library.data.maintenance.patchformer_flow import (
    AutoPatchFormerFlowEndpoints,
    _trusted_patchformer_weights,
)
from nexuml_library.data.maintenance.prepared_endpoints import read_prepared_artifact
from nexuml_library.data.maintenance.rul import read_rul_artifact
from nexuml_library.data.maintenance.rul_cluster_conditions import (
    _shared_patchformer_checkpoint,
)
from nexuml_library.data.maintenance.rul_download import (
    cmapss_source_sha256,
    ensure_cmapss,
)
from nexuml_library.evaluation.maintenance.rul_cluster_gate_audit import PREPARED_NAME
from nexuml_library.evaluation.maintenance.rul_machine_flow_study import (
    _index_engine_rows,
    _validate_prepared_settings,
)
from nexuml_library.evaluation.maintenance.rul_prepare import (
    prepare_cmapss,
    save_rul_artifact,
)
from nexuml_library.layers.model.rul_advanced import _PatchFormerModel
from nexuml_library.scenarios.maintenance.rul import _GOEL2026_FD003_SENSORS

SCENARIO = "maintenance-rul-controlled-fd003-patchformer-s42"
ENDPOINT_NAME = "rul-FD003-patchformer-sequence-cfc-flow-seed42-endpoints.pt"
UPSTREAM_RUN = "rul-FD003-controlled-patchformer-seed42"
PRISMA_MOUNT = Path("/mnt/juicefs_prisma")


def _locations(experiment_root: Path) -> dict[str, Path]:
    """Locate all immutable inputs within an explicit experiment root.

    Returns:
        Paths for the official source, prepared windows, audit and embeddings.
    """
    root = experiment_root.expanduser().resolve()
    return {
        "experiment_root": root,
        "raw_dir": root.parent / "data" / "cmapss",
        "prepared": root / "rul" / "prepared" / PREPARED_NAME,
        "audit": root / "rul" / "audits" / f"{UPSTREAM_RUN}.checkpoint.json",
        "checkpoint_dir": root / "rul" / "checkpoints" / UPSTREAM_RUN,
        "endpoint": root / "rul" / "prepared" / ENDPOINT_NAME,
    }


def _sha256(path: Path) -> str:
    """Calculate the complete SHA-256 digest of a local artifact.

    Returns:
        Hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recovery_status(experiment_root: Path) -> dict[str, Any]:
    """Inspect prerequisites without reading or modifying their contents.

    Returns:
        Path and existence indicators suitable for a read-only inventory.
    """
    locations = _locations(experiment_root)
    raw = locations["raw_dir"]
    status: dict[str, Any] = {
        "experiment_root": str(locations["experiment_root"]),
        "prisma_mounted": PRISMA_MOUNT.is_mount(),
        "official_fd003_tables": {
            name: (raw / name).is_file()
            for name in ("train_FD003.txt", "test_FD003.txt", "RUL_FD003.txt")
        },
        "prepared": {
            "path": str(locations["prepared"]),
            "exists": locations["prepared"].is_file(),
        },
        "audit": {"path": str(locations["audit"]), "exists": locations["audit"].is_file()},
        "endpoint": {
            "path": str(locations["endpoint"]),
            "exists": locations["endpoint"].is_file(),
        },
    }
    try:
        checkpoint = _shared_patchformer_checkpoint(locations["experiment_root"], 42)
    except (FileNotFoundError, ValueError) as exc:
        status["checkpoint"] = {"available": False, "reason": str(exc)}
    else:
        status["checkpoint"] = {"available": True, "path": str(checkpoint)}
    return status


def _check_mount(root: Path) -> None:
    """Prevent accidental writes into an unmounted, similarly named directory.

    Raises:
        RuntimeError: If a Prisma destination is not backed by its expected mount.
    """
    if root.is_relative_to(PRISMA_MOUNT) and not PRISMA_MOUNT.is_mount():
        raise RuntimeError(
            "Refusing recovery: /mnt/juicefs_prisma is not mounted. "
            "Check findmnt before creating any files."
        )


def _train_patchformer(locations: dict[str, Path]) -> None:
    """Launch the native supervised upstream experiment with a unique config.

    The training run uses the official C-MAPSS data source and validation-best
    checkpoint selection. It never reuses earlier, now-lost model weights.

    Raises:
        RuntimeError: If an orphaned audit could be overwritten by a rerun.
        subprocess.CalledProcessError: If NexuML resolve or training fails.
    """
    if locations["audit"].exists():
        raise RuntimeError(
            "Upstream checkpoint audit exists but its selected weights are missing. "
            "Restore the checkpoint from backup; no existing audit will be overwritten."
        )
    if locations["checkpoint_dir"].is_dir() and any(
        locations["checkpoint_dir"].glob("*.ckpt")
    ):
        raise RuntimeError(
            "Existing partial PatchFormer checkpoints found without an audit. "
            "Inspect or restore them before starting a new training run."
        )
    recovery_dir = locations["experiment_root"] / "rul" / "recovery"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    config = recovery_dir / f"{SCENARIO}-{uuid.uuid4().hex[:12]}.yaml"
    env = os.environ.copy()
    env["NEXUML_RUL_CHECKPOINT_ROOT"] = str(
        locations["experiment_root"] / "rul" / "checkpoints"
    )
    workdir = locations["experiment_root"].parent
    subprocess.run(
        ["nexuml", "resolve", SCENARIO, "--output", str(config)],
        cwd=workdir,
        env=env,
        check=True,
    )
    subprocess.run(
        ["nexuml", "train", "--config", str(config)],
        cwd=workdir,
        env=env,
        check=True,
    )


def rebuild_prerequisites(
    experiment_root: Path, *, retrain_missing: bool = False
) -> dict[str, Any]:
    """Recreate missing source, windows, selected encoder and flow endpoints.

    Existing files are never overwritten. A missing PatchFormer checkpoint
    requires ``retrain_missing=True`` and creates a *new* training provenance.

    Args:
        experiment_root: Explicit writable .experiments directory.
        retrain_missing: Permit expensive native PatchFormer training if necessary.

    Returns:
        Verified paths and SHA-256 hashes for the recovered study inputs.

    Raises:
        FileNotFoundError: If the checkpoint requires training without permission.
        RuntimeError: If the Prisma mount is absent or an old audit is orphaned.
        ValueError: If any preserved artifact fails scientific provenance checks.
    """
    locations = _locations(experiment_root)
    root = locations["experiment_root"]
    _check_mount(root)

    # The official NASA archive is a public reproducible source; the local
    # prepared artifact is a deterministic transformation of these tables.
    raw = ensure_cmapss(locations["raw_dir"], "FD003", download=True)
    prepared = locations["prepared"]
    if not prepared.exists():
        payload = prepare_cmapss(
            raw,
            subset="FD003",
            seed=42,
            val_fraction=0.2,
            window=30,
            stride=1,
            rul_cap=130.0,
            sensor_ids=_GOEL2026_FD003_SENSORS,
            normalization="zscore",
            padding_mode="zero",
            train_full_windows_only=True,
            normalize_targets=False,
        )
        save_rul_artifact(payload, prepared)

    stored = read_rul_artifact(prepared)
    settings = cast(dict[str, Any], stored["settings"])
    _validate_prepared_settings(settings)
    if (
        settings.get("sensor_ids") != _GOEL2026_FD003_SENSORS
        or settings.get("normalization") != "zscore"
        or settings.get("padding_mode") != "zero"
        or settings.get("scaler_fit_scope") != "training_assets_only"
    ):
        raise ValueError("Preserved FD003 preparation differs from the controlled protocol")
    if cast(torch.Tensor, stored["features"]).shape[1:] != (30, 15):
        raise ValueError("Recovered FD003 windows do not have [30,15] sensor layout")
    _index_engine_rows(cast(list[dict[str, Any]], stored["rows"]))

    try:
        checkpoint = _shared_patchformer_checkpoint(root, 42)
    except FileNotFoundError as exc:
        if not retrain_missing:
            raise FileNotFoundError(
                "FD003 windows are ready, but the seed-42 PatchFormer is missing. "
                "Restore its original checkpoint and audit or rerun with --retrain-missing."
            ) from exc
        _train_patchformer(locations)
        checkpoint = _shared_patchformer_checkpoint(root, 42)

    # Independently verify the selected model before trusting any endpoints.
    encoder = _PatchFormerModel(30, 15, 64, 5, 2, 0.1)
    checkpoint_sha = _trusted_patchformer_weights(encoder, checkpoint)
    endpoint = locations["endpoint"]
    if not endpoint.exists():
        AutoPatchFormerFlowEndpoints(
            subset="FD003",
            seed=42,
            root=str(raw),
            checkpoint_path=str(checkpoint),
            prepared_rul_path=str(prepared),
            output_file=str(endpoint),
            healthy_min_rul=100.0,
            faulty_max_rul=30.0,
            train_pairs=512,
            val_pairs=128,
            embedding_batch_size=64,
            sequence_embeddings=True,
        ).build()

    endpoint_settings = cast(dict[str, Any], read_prepared_artifact(endpoint)["settings"])
    if (
        endpoint_settings.get("subset") != "FD003"
        or endpoint_settings.get("seed") != 42
        or endpoint_settings.get("feature_layout") != "sequence"
        or endpoint_settings.get("feature_normalization") != "train_endpoints_only"
        or endpoint_settings.get("frozen_patchformer_checkpoint_sha256") != checkpoint_sha
        or endpoint_settings.get("held_out_test_engine_labels_used") is not False
    ):
        raise ValueError("Preserved flow endpoints do not match selected seed-42 encoder")

    return {
        "status": "ready",
        "official_archive_sha256": cmapss_source_sha256(locations["raw_dir"]),
        "prepared": str(prepared),
        "prepared_sha256": _sha256(prepared),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "endpoint": str(endpoint),
        "endpoint_sha256": _sha256(endpoint),
        "note": (
            "A retrained checkpoint has new provenance; "
            "prior results are not numerically identical."
        ),
    }


def main() -> None:
    """Inspect or rebuild the minimal frozen FD003 machine-flow inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Fetch missing NASA data and reconstruct the minimal prerequisites",
    )
    parser.add_argument(
        "--retrain-missing", action="store_true",
        help="Permit full native PatchFormer training when its checkpoint is lost",
    )
    args = parser.parse_args()
    if args.retrain_missing and not args.rebuild:
        parser.error("--retrain-missing requires --rebuild")
    result = (
        rebuild_prerequisites(args.experiment_root, retrain_missing=args.retrain_missing)
        if args.rebuild
        else recovery_status(args.experiment_root)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
