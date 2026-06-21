"""Optional local provenance snapshots for scenario-file runs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from nexuml.core.config import ResolvedConfig
from nexuml.core.scenario_loader import LoadedScenarioFile, project_root_for


def _git_state(root: Path) -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=root, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return None

    return {
        "root": str(root),
        "commit": run(["rev-parse", "HEAD"]),
        "branch": run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "status": run(["status", "--short"]),
    }


def _best_trial(study: Any) -> dict[str, Any] | None:
    if study is None:
        return None
    try:
        trial = study.best_trial
    except Exception:
        return None
    return {
        "number": getattr(trial, "number", None),
        "value": getattr(trial, "value", None),
        "params": dict(getattr(trial, "params", {}) or {}),
    }


def snapshot_scenario_file_run(
    loaded: LoadedScenarioFile,
    artifact_dir: str | Path | None,
    *,
    project_root: str | Path | None = None,
    command: str,
    command_args: dict[str, Any] | None = None,
    search_space: dict[str, dict[str, Any]] | None = None,
    study: Any = None,
) -> Path | None:
    """Snapshot source, built scenario YAML, metadata, git, and tuning summary.

    Returns:
        Path to the snapshot directory, or ``None`` if *artifact_dir* is ``None``.
    """
    if artifact_dir is None:
        return None

    root = (
        Path(project_root).expanduser().resolve() if project_root else project_root_for(loaded.path)
    )
    out = Path(artifact_dir).expanduser()
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)

    (out / "scenario.py").write_text(loaded.source, encoding="utf-8")
    (out / "resolved_scenario.yaml").write_text(
        ResolvedConfig.from_scenario(loaded.scenario).to_yaml(), encoding="utf-8"
    )
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "source_path": str(loaded.path),
                "hypothesis": loaded.hypothesis,
                "parent": loaded.parent,
                "tags": loaded.tags,
                "scenario_name": loaded.scenario.name,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    (out / "git.json").write_text(json.dumps(_git_state(root), indent=2), encoding="utf-8")
    (out / "command.json").write_text(
        json.dumps(
            {"command": command, "args": command_args or {}, "project_root": str(root)},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    if search_space is not None:
        (out / "search_space.json").write_text(
            json.dumps(search_space, indent=2, default=str), encoding="utf-8"
        )
    if (best_trial := _best_trial(study)) is not None:
        (out / "best_trial.json").write_text(
            json.dumps(best_trial, indent=2, default=str), encoding="utf-8"
        )

    return out
