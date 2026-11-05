"""Immutable-source GPU Job contracts for FD003 cluster conditioning."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / ".k8s/scenarios/submit_rul_cfc_cluster_conditioning.py"
)


def _launcher(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Load the standalone Kubernetes launcher with sibling imports."""
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("nex233_cluster_job_launcher", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cluster_job_matrix_and_source_attestation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Twelve Jobs share immutable source and do not touch completed studies."""
    module = _launcher(monkeypatch)
    trials = module.cluster_trials()
    assert len(trials) == 12
    assert len(set(trials)) == 12
    names: set[str] = set()
    digest = "a" * 64
    source = f"job-staging/nexuml-nex233/{digest}"
    for arm, seed in trials:
        name = module.job_name(arm, seed)
        assert name not in names
        names.add(name)
        manifest = module.render_cluster_job(
            arm, seed, digest, source, ("gh200-1", "student-worker-1")
        )
        assert f"name: {name}" in manifest
        assert "nexuml-study: rul-cfc-cluster" in manifest
        assert f"maintenance-rul-cfc-cluster-fd003-{arm}-s{seed}" in manifest
        assert f"subPath: {source}" in manifest
        assert 'nvidia.com/gpu: "1"' in manifest
        assert "values: [gh200-1, student-worker-1]" in manifest
        assert "ghcr.io/nexufed/nexuml:sha-f8d5490-cuda12.8.1" in manifest
        assert "cmapss-FD003-seed42-cfc-conditional-clusters-v1.pt" in manifest
        assert "__SCENARIO__" not in manifest
        assert "__SOURCE_DIGEST__" not in manifest
        assert "rul-cfc-followup" not in manifest
    assert len(names) == 12


def test_cluster_job_rejects_unknown_studies_and_bad_digests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent node-unsafe or incorrectly labeled GPU experiments."""
    module = _launcher(monkeypatch)
    with pytest.raises(ValueError, match="Unsupported"):
        module.job_name("invalid", 42)
    with pytest.raises(ValueError, match="Unsupported"):
        module.job_name("both", 45)
    with pytest.raises(ValueError, match="Unsafe"):
        module.render_cluster_job(
            "both", 42, "invalid", "job-staging/nexuml-nex233/invalid",
            ("student-worker-1",)
        )
    with pytest.raises(ValueError, match="Unsafe"):
        module.render_cluster_job(
            "both", 42, "a" * 64,
            "job-staging/nexuml-nex233/" + "a" * 64,
            ("gh200-1",)
        )
