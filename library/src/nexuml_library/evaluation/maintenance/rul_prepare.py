"""Leakage-safe run-to-failure preprocessing for supervised NexuML RUL.

C-MAPSS labels are cycles until failure (official test RUL offsets included).
XJTU-SY labels are remaining *recordings*, not seconds or a hidden time scale.
All transforms are fitted on training assets only. Windows never span assets.
The resulting immutable artifact is consumed by a typed NexuML data source.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


def _windowize(
    sequences: list[tuple[str, str, np.ndarray, np.ndarray]],
    *,
    window: int,
    stride: int,
    final_only_test: bool,
) -> tuple[Tensor, Tensor, list[dict[str, Any]]]:
    """Build causal left-edge-padded windows without joining asset sequences.

    Returns:
        Features [N,L,D], labels [N,1], and sample provenance.

    Raises:
        ValueError: For inconsistent shapes, labels or nonfinite values.
    """
    features: list[Tensor] = []
    labels: list[float] = []
    metadata: list[dict[str, Any]] = []
    for asset, split, observations, rul in sequences:
        if (
            observations.ndim != 2
            or len(observations) == 0
            or rul.shape != (len(observations),)
            or not np.isfinite(observations).all()
            or not np.isfinite(rul).all()
            or np.any(rul < 0)
        ):
            raise ValueError(f"Malformed run-to-failure trajectory: {asset}")
        positions = (
            [len(observations) - 1]
            if split == "test" and final_only_test
            else list(range(0, len(observations), stride))
        )
        if positions[-1] != len(observations) - 1:
            positions.append(len(observations) - 1)
        for stop in positions:
            start = max(0, stop - window + 1)
            segment = observations[start : stop + 1]
            if len(segment) < window:
                segment = np.concatenate(
                    [np.repeat(segment[:1], window - len(segment), axis=0), segment],
                    axis=0,
                )
            features.append(torch.from_numpy(segment.copy()).float())
            labels.append(float(rul[stop]))
            metadata.append(
                {"asset_id": asset, "split": split, "step": int(stop)}
            )
    return (
        torch.stack(features),
        torch.tensor(labels, dtype=torch.float32).unsqueeze(-1),
        metadata,
    )


def _scale_sequences(
    sequences: list[tuple[str, str, np.ndarray, np.ndarray]],
) -> tuple[list[tuple[str, str, np.ndarray, np.ndarray]], dict[str, list[float]]]:
    """Standardize features using training assets only.

    Returns:
        Scaled sequences and the exact training-only normalization parameters.

    Raises:
        ValueError: If there are no training assets or channels disagree.
    """
    training = [samples for _, split, samples, _ in sequences if split == "train"]
    if not training:
        raise ValueError("At least one training asset is required")
    concat = np.concatenate(training, axis=0)
    if concat.ndim != 2 or not np.isfinite(concat).all():
        raise ValueError("Nonfinite training observations")
    mean = concat.mean(axis=0)
    std = concat.std(axis=0)
    std = np.maximum(std, 1e-6)
    scaled = [
        (asset, split, (samples - mean) / std, rul)
        for asset, split, samples, rul in sequences
    ]
    return scaled, {"mean": mean.tolist(), "std": std.tolist()}


def _split_assets(
    names: list[str],
    *,
    seed: int,
    val_fraction: float,
) -> tuple[set[str], set[str]]:
    """Assign disjoint training and validation engines.

    Returns:
        Sets of train and validation asset IDs.

    Raises:
        ValueError: If a valid asset-disjoint partition is impossible.
    """
    if len(names) < 3 or not 0 < val_fraction < 0.5:
        raise ValueError("Need at least three assets and val_fraction in (0,0.5)")
    unique = sorted(set(names))
    if len(unique) != len(names):
        raise ValueError("Duplicate asset IDs")
    permuted = np.random.default_rng(seed).permutation(unique).tolist()
    n_val = max(1, min(len(unique) - 2, round(len(unique) * val_fraction)))
    return set(permuted[n_val:]), set(permuted[:n_val])


def _cmapss_table(path: Path) -> np.ndarray:
    """Read one official C-MAPSS whitespace-delimited sensor table.

    Returns:
        Two-dimensional finite numeric matrix.

    Raises:
        FileNotFoundError: If the requested sensor table does not exist.
        ValueError: If the table has invalid columns or nonfinite entries.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    array = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if array.shape[1] != 26 or not np.isfinite(array).all():
        raise ValueError(f"Expected 26 finite columns in {path}")
    return array


def prepare_cmapss(
    root: str | Path,
    *,
    subset: str = "FD001",
    seed: int = 42,
    val_fraction: float = 0.2,
    window: int = 30,
    stride: int = 1,
    rul_cap: float | None = 125.0,
    final_only_test: bool = True,
) -> dict[str, Any]:
    """Construct supervised C-MAPSS RUL windows with official test targets.

    Returns:
        Portable nexuml-rul-windows-v1 payload. The RUL unit is cycles.
        Targets are divided by a saved scale for stable optimization.

    Raises:
        FileNotFoundError: If a required official C-MAPSS file is missing.
        ValueError: If engine indices, official labels or arguments are invalid.
    """
    if subset not in {"FD001", "FD002", "FD003", "FD004"}:
        raise ValueError("Unknown C-MAPSS subset")
    if window < 1 or stride < 1 or (rul_cap is not None and rul_cap <= 0):
        raise ValueError("Invalid window, stride or RUL cap")
    root = Path(root).expanduser()
    train = _cmapss_table(root / f"train_{subset}.txt")
    test = _cmapss_table(root / f"test_{subset}.txt")
    rul_file = root / f"RUL_{subset}.txt"
    if not rul_file.is_file():
        raise FileNotFoundError(rul_file)
    terminal_rul = np.loadtxt(rul_file, dtype=np.float64, ndmin=1)
    if terminal_rul.ndim != 1 or not np.isfinite(terminal_rul).all():
        raise ValueError("Malformed official C-MAPSS test RUL labels")
    engine_ids = np.unique(train[:, 0].astype(np.int64))
    train_ids, val_ids = _split_assets(
        [str(i) for i in engine_ids], seed=seed, val_fraction=val_fraction
    )
    sequences: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for origin, table in (("train", train), ("test", test)):
        ids = np.unique(table[:, 0].astype(np.int64))
        if origin == "test" and (
            len(ids) != len(terminal_rul)
            or list(ids) != list(range(1, len(terminal_rul) + 1))
        ):
            raise ValueError("Official test RUL count/engine IDs disagree")
        for engine in ids:
            part = table[table[:, 0] == engine]
            cycles = part[:, 1]
            if (
                len(cycles) != len(np.unique(cycles))
                or not np.array_equal(cycles, np.arange(1, len(part) + 1))
            ):
                raise ValueError(f"Engine {engine} must have consecutive unique cycles")
            name = f"{origin}:{subset}:{engine}"
            if origin == "test":
                split = "test"
                remaining = terminal_rul[engine - 1] + cycles[-1] - cycles
            else:
                split = "train" if str(engine) in train_ids else "val"
                remaining = cycles[-1] - cycles
            if rul_cap is not None:
                remaining = np.minimum(remaining, rul_cap)
            # Operational settings (3) and sensor channels (21): 24 features.
            sequences.append((name, split, part[:, 2:26], remaining))
    sequences, scaler = _scale_sequences(sequences)
    scale = float(rul_cap) if rul_cap is not None else float(
        max(np.max(rul) for _, split, _, rul in sequences if split == "train")
    )
    scale = max(scale, 1.0)
    normalized = [
        (asset, split, values, rul / scale)
        for asset, split, values, rul in sequences
    ]
    features, targets, rows = _windowize(
        normalized, window=window, stride=stride, final_only_test=final_only_test
    )
    return {
        "format": "nexuml-rul-windows-v1",
        "features": features,
        "targets": targets,
        "rows": rows,
        "settings": {
            "dataset": "cmapss",
            "subset": subset,
            "window": window,
            "stride": stride,
            "seed": seed,
            "target_unit": "cycles",
            "target_scale": scale,
            "rul_cap": rul_cap,
            "feature_names": [
                *(f"setting_{i}" for i in range(1, 4)),
                *(f"sensor_{i}" for i in range(1, 22)),
            ],
            "scaler": scaler,
            "split_assets": {
                split: sorted({row["asset_id"] for row in rows if row["split"] == split})
                for split in ("train", "val", "test")
            },
            "official_test_split": True,
            "test_terminal_only": final_only_test,
        },
    }


def _bearing_features(path: Path) -> np.ndarray:
    """Compute causal per-recording vibration features from two accelerometers.

    Returns:
        Twelve features: six per vibration channel.

    Raises:
        ValueError: If the vibration file is missing valid sensor channels.
    """
    import pandas as pd

    frame = pd.read_csv(path)
    if frame.shape[1] < 2:
        raise ValueError(f"Expected two accelerometer columns in {path}")
    # Standard XJTU-SY CSVs have Horizontal_vibration/Vertical_vibration.
    preferred = ["Horizontal_vibration", "Vertical_vibration"]
    if all(name in frame.columns for name in preferred):
        arrays = [frame[name].to_numpy(dtype=np.float64) for name in preferred]
    else:
        arrays = [frame.iloc[:, i].to_numpy(dtype=np.float64) for i in (0, 1)]
    result: list[float] = []
    for channel in arrays:
        if len(channel) < 8 or not np.isfinite(channel).all():
            raise ValueError(f"Invalid vibration recording {path}")
        centered = channel - channel.mean()
        rms = float(np.sqrt(np.mean(channel**2)))
        stdev = max(float(np.std(centered)), 1e-9)
        spectrum = np.abs(np.fft.rfft(centered))
        frequencies = np.fft.rfftfreq(len(centered), d=1.0)
        centroid = float(
            np.sum(frequencies * spectrum) / max(np.sum(spectrum), 1e-9)
        )
        result.extend(
            [
                rms,
                stdev,
                float(np.max(np.abs(channel))),
                float(np.mean((centered / stdev) ** 3)),
                float(np.mean((centered / stdev) ** 4)),
                centroid,
            ]
        )
    return np.asarray(result, dtype=np.float64)


def prepare_xjtu_sy(
    root: str | Path,
    *,
    seed: int = 42,
    window: int = 20,
    stride: int = 1,
) -> dict[str, Any]:
    """Build XJTU-SY recording-wise RUL windows from all fifteen bearings.

    Bearings 1-5 of each of three conditions are assigned deterministically
    as 3 train, 1 val and 1 test per condition. This is an explicitly defined
    *research split*, not an official XJTU-SY prognostics leaderboard split.

    Returns:
        Window payload with target in number of remaining recordings.

    Raises:
        ValueError: If the complete 3x5 run-to-failure corpus is unavailable.
    """
    if window < 1 or stride < 1:
        raise ValueError("Window and stride must be positive")
    root = Path(root).expanduser()
    found: dict[str, Path] = {}
    for folder in root.rglob("*"):
        match = re.fullmatch(r"Bearing([123])_([1-5])", folder.name)
        if match and folder.is_dir():
            if folder.name in found:
                raise ValueError(f"Duplicate bearing folder {folder.name}")
            found[folder.name] = folder
    expected = {f"Bearing{condition}_{i}" for condition in (1, 2, 3) for i in range(1, 6)}
    if set(found) != expected:
        raise ValueError("XJTU-SY must contain exactly Bearing1_1 through Bearing3_5")
    sequences: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for condition in (1, 2, 3):
        machines = sorted(name for name in expected if name.startswith(f"Bearing{condition}_"))
        permutation = np.random.default_rng(seed + condition).permutation(machines)
        roles = {
            name: ("train" if i < 3 else "val" if i == 3 else "test")
            for i, name in enumerate(permutation)
        }
        for name in machines:
            paths = list(found[name].glob("*.csv"))
            if not paths:
                raise ValueError(f"No vibration CSV recordings for {name}")

            def ordinal(path: Path) -> int:
                """Return a recording's chronological position from its filename.

                Returns:
                    Integer recording index.

                Raises:
                    ValueError: If the filename does not contain a numeric index.
                """
                if not path.stem.isdecimal():
                    raise ValueError(f"Unknown XJTU-SY recording order: {path.name}")
                return int(path.stem)

            paths.sort(key=ordinal)
            indices = [ordinal(path) for path in paths]
            if len(indices) != len(set(indices)):
                raise ValueError(f"Duplicate recording number in {name}")
            features = np.stack([_bearing_features(path) for path in paths])
            rul = np.arange(len(paths) - 1, -1, -1, dtype=np.float64)
            sequences.append((name, roles[name], features, rul))
    sequences, scaler = _scale_sequences(sequences)
    scale = max(
        1.0, max(float(rul.max()) for _, split, _, rul in sequences if split == "train")
    )
    normalized = [
        (asset, split, features, rul / scale)
        for asset, split, features, rul in sequences
    ]
    features, targets, rows = _windowize(
        normalized, window=window, stride=stride, final_only_test=False
    )
    return {
        "format": "nexuml-rul-windows-v1",
        "features": features,
        "targets": targets,
        "rows": rows,
        "settings": {
            "dataset": "xjtu_sy",
            "window": window,
            "stride": stride,
            "seed": seed,
            "target_unit": "recordings",
            "target_scale": scale,
            "feature_names": [
                f"{name}_{channel}"
                for channel in ("horizontal", "vertical")
                for name in ("rms", "std", "peak", "skew", "kurtosis", "spectral_centroid")
            ],
            "scaler": scaler,
            "split_assets": {
                split: sorted({row["asset_id"] for row in rows if row["split"] == split})
                for split in ("train", "val", "test")
            },
            "official_test_split": False,
        },
    }


def save_rul_artifact(payload: dict[str, Any], path: str | Path) -> dict[str, Any]:
    """Persist a new immutable NexuML RUL tensor artifact and JSON metadata.

    Returns:
        Artifact path, dimensions, counts and scientific provenance.

    Raises:
        FileExistsError: If either output exists.
    """
    target = Path(path).expanduser().resolve()
    manifest = target.with_suffix(".json")
    if target.exists() or manifest.exists():
        raise FileExistsError("Refusing to overwrite a prepared RUL artifact")
    # Delay writing until content has passed the exact datasource contract.
    from nexuml_library.data.maintenance.rul import validate_rul_payload

    validate_rul_payload(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    result = {
        "artifact": str(target),
        "num_windows": len(payload["features"]),
        "shape": list(payload["features"].shape),
        "settings": payload["settings"],
    }
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Prepare labeled C-MAPSS or XJTU-SY lifetimes without training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("cmapss", "xjtu_sy"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--subset", default="FD001", choices=("FD001", "FD002", "FD003", "FD004"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--rul-cap", type=float, default=125.0)
    args = parser.parse_args()
    if args.dataset == "cmapss":
        payload = prepare_cmapss(
            args.root, subset=args.subset, seed=args.seed,
            val_fraction=args.val_fraction, window=args.window,
            stride=args.stride, rul_cap=args.rul_cap,
        )
    else:
        payload = prepare_xjtu_sy(
            args.root, seed=args.seed, window=args.window, stride=args.stride
        )
    result = save_rul_artifact(payload, args.output_file)
    print(json.dumps({
        "artifact": result["artifact"],
        "shape": result["shape"],
        "split_assets": result["settings"]["split_assets"],
    }, indent=2))


if __name__ == "__main__":
    main()
