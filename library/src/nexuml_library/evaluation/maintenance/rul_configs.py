"""Resolve supervised RUL scenarios into immutable NexuML YAML configurations.

This utility ONLY compiles and saves configurations. Every optimization run
is performed separately with `nexuml train --config ...` (normal CLI path).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nexuml.core.compiler import compile as compile_pipeline
from nexuml_library.data.maintenance.rul import read_rul_artifact
from nexuml_library.scenarios.maintenance.rul import maintenance_rul_benchmark

MODEL_CHOICES = ("feature_mlp", "lstm", "tcn", "transformer", "tsmixer")


def resolve_rul_configs(
    prepared_files: list[str | Path],
    output_dir: str | Path,
    *,
    architectures: tuple[str, ...] = MODEL_CHOICES,
    objectives: tuple[str, ...] = ("huber",),
    hidden_dim: int = 96,
    depth: int = 3,
    dropout: float = 0.1,
    max_epochs: int = 150,
    batch_size: int = 64,
    lr: float = 3e-4,
    patience: int = 20,
) -> dict[str, Any]:
    """Generate resolved, shape-validated configurations from RUL artifacts.

    Each config locks the prepared training data path, asset split seed, model,
    optimization settings, preprocessing scale and checkpoint directory.

    Args:
        prepared_files: One or more independent already-prepared RUL tensors.
        output_dir: Directory for YAML configs and manifest.
        architectures: Supported sequence model choices.
        objectives: Regression objective choices including optional quantiles.
        hidden_dim: Sequence hidden width.
        depth: Network depth.
        dropout: Encoder dropout rate.
        max_epochs: Maximum training epochs.
        batch_size: NexuML native mini-batch size.
        lr: Optimizer learning rate.
        patience: Validation loss patience.

    Returns:
        Manifest of resolved YAML configurations and exact CLI commands.

    Raises:
        ValueError: If inputs collide or contain unknown configurations.
        FileExistsError: If an output would be overwritten.
    """
    if (
        not prepared_files
        or not architectures
        or not objectives
        or len(set(architectures)) != len(architectures)
        or len(set(objectives)) != len(objectives)
        or any(a not in MODEL_CHOICES for a in architectures)
        or any(o not in {"mse", "mae", "huber", "quantile"} for o in objectives)
    ):
        raise ValueError("Choose valid nonempty independent model/objective combinations")
    target = Path(output_dir).expanduser().resolve()
    manifest_path = target / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {manifest_path}")
    records: list[dict[str, Any]] = []
    planned: list[tuple[Path, Any]] = []
    names: set[str] = set()
    for raw in prepared_files:
        artifact = Path(raw).expanduser().resolve()
        payload = read_rul_artifact(artifact)
        settings = payload["settings"]
        for architecture in architectures:
            for objective in objectives:
                scenario = maintenance_rul_benchmark(
                    artifact,
                    architecture=architecture,
                    objective=objective,
                    hidden_dim=hidden_dim,
                    depth=depth,
                    dropout=dropout,
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                    lr=lr,
                    patience=patience,
                    seed=int(settings["seed"]),
                    checkpoint_root=target.parent / "checkpoints",
                )
                if scenario.name in names:
                    raise ValueError(f"Scenario output would collide: {scenario.name}")
                names.add(scenario.name)
                resolved = compile_pipeline(scenario).resolved_config
                yaml_path = target / f"{scenario.name}.yaml"
                if yaml_path.exists():
                    raise FileExistsError(f"Refusing to overwrite {yaml_path}")
                planned.append((yaml_path, resolved))
                records.append(
                    {
                        "name": scenario.name,
                        "prepared_file": str(artifact),
                        "architecture": architecture,
                        "objective": objective,
                        "config": str(yaml_path),
                        "build_command": f"nexuml build {yaml_path}",
                        "train_command": f"nexuml train --config {yaml_path}",
                    }
                )
    target.mkdir(parents=True, exist_ok=True)
    for path, resolved in planned:
        resolved.save(path)
    result = {
        "format": "nexuml-rul-configs-v1",
        "configs": records,
        "note": (
            "No training was performed by the configuration generator. "
            "Execute NexuML build/train commands in manifest order."
        ),
    }
    manifest_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Write reproducible RUL YAML scenarios without starting training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-files", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--architectures", nargs="+", default=list(MODEL_CHOICES), choices=MODEL_CHOICES)
    parser.add_argument(
        "--objectives", nargs="+", default=["huber"],
        choices=("mse", "mae", "huber", "quantile"),
    )
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()
    result = resolve_rul_configs(
        args.prepared_files,
        args.output_dir,
        architectures=tuple(args.architectures),
        objectives=tuple(args.objectives),
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )
    print(json.dumps(result["configs"], indent=2))


if __name__ == "__main__":
    main()
