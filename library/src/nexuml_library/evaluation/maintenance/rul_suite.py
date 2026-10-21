"""Run independent registered RUL scenarios through the public NexuML CLI.

This module orchestrates CLI processes only. It does not construct pipelines,
train models, or evaluate checkpoints outside NexuML's normal lifecycle.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

RUL_SUITE_SCENARIOS: tuple[str, ...] = (
    "maintenance-rul-feature-mlp",
    "maintenance-rul-lstm",
    "maintenance-rul-tcn",
    "maintenance-rul-transformer",
    "maintenance-rul-tsmixer",
    "maintenance-rul-quantile-tcn",
)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Persist the status of the current independent benchmark runs."""
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run_rul_scenario_suite(
    prepared_files: Sequence[str | Path],
    output_dir: str | Path,
    *,
    scenarios: Sequence[str] = RUL_SUITE_SCENARIOS,
    max_epochs: int | None = None,
) -> dict[str, Any]:
    """Resolve, build, and train each registered RUL scenario independently.

    Invoke NexuML's public CLI for every dataset/scenario pair; preserve
    separate YAML, checkpoint directories, logs, and exit statuses. Stop
    immediately on a CLI failure rather than reporting a partial success.

    Args:
        prepared_files: Immutable prepared C-MAPSS or XJTU-SY artifacts.
        output_dir: New suite directory for resolved YAML, logs and manifest.
        scenarios: Registered RUL scenarios to execute, in order.
        max_epochs: Optional NexuML CLI training-epoch override.

    Returns:
        Persisted manifest with one outcome per independent training run.

    Raises:
        ValueError: If datasets, scenario names, or max_epochs are invalid.
        FileNotFoundError: If a prepared artifact does not exist.
        FileExistsError: If the output directory already exists.
        RuntimeError: If any NexuML CLI command fails or cannot launch.
    """
    artifacts = [Path(path).expanduser().resolve() for path in prepared_files]
    selected = list(scenarios)
    if (
        not artifacts
        or len(set(artifacts)) != len(artifacts)
        or not selected
        or len(set(selected)) != len(selected)
        or any(name not in RUL_SUITE_SCENARIOS for name in selected)
        or (max_epochs is not None and max_epochs < 1)
    ):
        raise ValueError("Provide unique datasets, supported scenarios and positive max_epochs")
    for artifact in artifacts:
        if not artifact.is_file():
            raise FileNotFoundError(f"Prepared RUL artifact does not exist: {artifact}")

    root = Path(output_dir).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to reuse RUL suite output directory: {root}")

    runs: list[dict[str, Any]] = []
    for dataset_index, artifact in enumerate(artifacts):
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", artifact.stem).strip("-") or "dataset"
        for scenario_name in selected:
            run_id = f"{dataset_index:02d}-{slug}-{scenario_name}"
            config = root / "configs" / f"{run_id}.yaml"
            log = root / "logs" / f"{run_id}.log"
            commands = [
                ["nexuml", "resolve", scenario_name, "--output", str(config)],
                ["nexuml", "build", str(config)],
                ["nexuml", "train", "--config", str(config)],
            ]
            if max_epochs is not None:
                commands[-1].extend(["--max-epochs", str(max_epochs)])
            runs.append(
                {
                    "run_id": run_id,
                    "prepared_file": str(artifact),
                    "scenario": scenario_name,
                    "config": str(config),
                    "log": str(log),
                    "checkpoint_root": str(root / "checkpoints" / run_id),
                    "commands": commands,
                    "status": "pending",
                }
            )

    (root / "configs").mkdir(parents=True)
    (root / "logs").mkdir()
    manifest_path = root / "manifest.json"
    manifest: dict[str, Any] = {
        "format": "nexuml-rul-suite-v1",
        "max_epochs_override": max_epochs,
        "runs": runs,
    }
    _write_manifest(manifest_path, manifest)

    for record in runs:
        env = os.environ.copy()
        env["NEXUML_RUL_PREPARED_FILE"] = record["prepared_file"]
        env["NEXUML_RUL_CHECKPOINT_ROOT"] = record["checkpoint_root"]
        log_path = Path(record["log"])
        for phase, command in zip(("resolve", "build", "train"), record["commands"], strict=True):
            record["status"] = phase
            _write_manifest(manifest_path, manifest)
            with log_path.open("a", encoding="utf-8") as output:
                output.write(f"$ {' '.join(command)}\n")
                output.flush()
                try:
                    completed = subprocess.run(
                        command,
                        env=env,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                except OSError as exc:
                    record["status"] = "failed"
                    record["failure"] = str(exc)
                    _write_manifest(manifest_path, manifest)
                    raise RuntimeError(f"Cannot launch {phase} for {record['run_id']}") from exc
            if completed.returncode != 0:
                record["status"] = "failed"
                record["failed_phase"] = phase
                record["returncode"] = completed.returncode
                _write_manifest(manifest_path, manifest)
                raise RuntimeError(
                    f"NexuML {phase} failed for {record['run_id']} "
                    f"(exit {completed.returncode}); see {log_path}"
                )
        record["status"] = "completed"
        _write_manifest(manifest_path, manifest)
    return manifest


def main() -> None:
    """Execute the registered benchmark suite via standard NexuML commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-files", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scenarios", nargs="+", choices=RUL_SUITE_SCENARIOS,
        default=list(RUL_SUITE_SCENARIOS),
    )
    parser.add_argument("--max-epochs", type=int)
    args = parser.parse_args()
    manifest = run_rul_scenario_suite(
        args.prepared_files, args.output_dir,
        scenarios=args.scenarios, max_epochs=args.max_epochs,
    )
    print(json.dumps({
        "manifest": str(Path(args.output_dir).expanduser().resolve() / "manifest.json"),
        "completed": sum(run["status"] == "completed" for run in manifest["runs"]),
    }, indent=2))


if __name__ == "__main__":
    main()
