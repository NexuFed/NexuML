#!/usr/bin/env python3
"""Submit NEX-233 CfC reliability, factor-ablation and FD001 replication Jobs.

Each Job mounts an immutable content-hashed NexuML source snapshot and has
an independent checkpoint, best-validation audit and completion marker.
Never replace existing Jobs or reuse the preceding confirmation namespace.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess

from rul_confirm_gpu_pool import discover_excluded_hostnames
from submit_rul_enhanced_cfc_fd003 import (
    DEFAULT_STAGE,
    NAME_PATTERN,
    NAMESPACE,
    RENDERED,
    digest_source,
    existing_job,
    stage_snapshot,
)

TEMPLATE = Path(__file__).with_name("job_rul_enhanced_cfc_followup.yaml.template")
ARMS = ("baseline", "all_no_trend")
ABLATIONS = (
    "without_finetune", "without_graph", "without_multiscale",
    "without_segmentation", "without_trajectory",
)
RELIABILITY_SEEDS = tuple(range(42, 52))
ABLATION_SEEDS = (42, 43, 44)
FD001_SEEDS = (42, 43, 44)
CAMPAIGNS = ("reliability", "ablation", "fd001")


def trials(campaign: str) -> tuple[tuple[str, str, int], ...]:
    """Build the immutable experiment matrix.

    Returns:
        Valid subset, configuration and readout-seed triples.

    Raises:
        ValueError: If the campaign is not registered.
    """
    if campaign == "reliability":
        return tuple(
            ("FD003", config, seed)
            for config in ARMS for seed in RELIABILITY_SEEDS
        )
    if campaign == "ablation":
        return tuple(
            ("FD003", config, seed)
            for config in ABLATIONS for seed in ABLATION_SEEDS
        )
    if campaign == "fd001":
        return tuple(
            ("FD001", config, seed)
            for config in ARMS for seed in FD001_SEEDS
        )
    if campaign == "all":
        return tuple(item for phase in CAMPAIGNS for item in trials(phase))
    raise ValueError(f"Unknown follow-up campaign: {campaign}")


def phase_for(subset: str, configuration: str, seed: int) -> str:
    """Validate one trial is among the prespecified combinations.

    Returns:
        Campaign identifier for the validated trial.

    Raises:
        ValueError: For unsupported subset/configuration/seed.
    """
    for phase in CAMPAIGNS:
        if (subset, configuration, seed) in trials(phase):
            return phase
    raise ValueError(f"Unsupported RUL follow-up: {subset}/{configuration}/{seed}")


def followup_job_name(subset: str, configuration: str, seed: int) -> str:
    """Build one DNS-compatible and experiment-unique Job name.

    Returns:
        Job name associated with exactly one NexuML scenario.
    """
    phase_for(subset, configuration, seed)
    job = (
        f"nexuml-rul-cfc-followup-{subset.lower()}-"
        f"{configuration.replace('_', '-')}-s{seed}"
    )
    if len(job) > 63:
        raise ValueError(f"Kubernetes Job name too long: {job}")
    return job


def render_followup_manifest(
    subset: str,
    configuration: str,
    seed: int,
    digest: str,
    source_subpath: str,
    excluded_hosts: tuple[str, ...],
    namespace: str = NAMESPACE,
) -> str:
    """Render a one-GPU Job with strict original-source provenance.

    Args:
        subset: FD003 main study or FD001 replication.
        configuration: One of the registered arms/factor omissions.
        seed: Independently initialized model/readout seed.
        digest: SHA-256 source snapshot attestation.
        source_subpath: Immutable location relative to the JuiceFS source PVC.
        excluded_hosts: Cluster-derived GH200 and student-worker-1 hostnames.
        namespace: Target Kubernetes namespace.

    Returns:
        Kubernetes Job manifest with all variables resolved.

    Raises:
        ValueError: For unsafe or incomplete identifiers.
    """
    phase = phase_for(subset, configuration, seed)
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("Expected source SHA-256 digest")
    if (
        not excluded_hosts
        or "student-worker-1" not in excluded_hosts
        or any(re.fullmatch(r"[A-Za-z0-9.-]+", h) is None for h in excluded_hosts)
    ):
        raise ValueError("Missing or invalid GPU node exclusions")
    if not NAME_PATTERN.fullmatch(namespace):
        raise ValueError("Invalid Kubernetes namespace")
    if source_subpath != f"job-staging/nexuml-nex233/{digest}":
        raise ValueError("Unexpected immutable source snapshot mount")
    job = followup_job_name(subset, configuration, seed)
    scenario = (
        f"maintenance-rul-enhanced-cfc-followup-{subset.lower()}-"
        f"{configuration}-s{seed}"
    )
    mapping = {
        "__JOB_NAME__": job,
        "__NAMESPACE__": namespace,
        "__SEED__": str(seed),
        "__CONFIGURATION__": configuration,
        "__CONFIG_SLUG__": configuration.replace("_", "-"),
        "__SUBSET__": subset,
        "__SUBSET_LOWER__": subset.lower(),
        "__CAMPAIGN__": phase,
        "__SCENARIO__": scenario,
        "__SOURCE_DIGEST__": digest,
        "__SOURCE_SHORT__": digest[:12],
        "__STAGE_SUBPATH__": source_subpath,
        "__EXCLUDED_NODES__": ", ".join(sorted(set(excluded_hosts))),
    }
    rendered = TEMPLATE.read_text(encoding="utf-8")
    for key, value in mapping.items():
        rendered = rendered.replace(key, value)
    if re.search(r"__[A-Z_]+__", rendered):
        raise ValueError("Unresolved RUL follow-up Job template variable")
    return rendered


def main() -> None:
    """Render/submit selected native Jobs without modifying other experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign", choices=(*CAMPAIGNS, "all"), default="reliability",
    )
    parser.add_argument("--namespace", default=NAMESPACE)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGE)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Optional subset of the chosen campaign's prespecified seeds",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not NAME_PATTERN.fullmatch(args.namespace):
        parser.error("Invalid Kubernetes namespace")
    stage = args.staging_root.expanduser().absolute()
    if stage == Path("/mnt/juicefs") or not stage.is_relative_to(Path("/mnt/juicefs")):
        parser.error("Staging root must be inside the JuiceFS source PVC")
    chosen = trials(args.campaign)
    if args.seeds is not None:
        if len(set(args.seeds)) != len(args.seeds):
            parser.error("Duplicate head seeds")
        allowed = {seed for _, _, seed in chosen}
        if any(seed not in allowed for seed in args.seeds):
            parser.error(f"Seed(s) outside {args.campaign} protocol: {sorted(allowed)}")
        chosen = tuple(item for item in chosen if item[2] in args.seeds)
    exclusions = discover_excluded_hostnames()
    print(f"Excluded GPU hosts: {', '.join(exclusions)}")
    digest = digest_source()
    snapshot = stage_snapshot(stage, digest, dry_run=args.dry_run)
    subpath = snapshot.relative_to("/mnt/juicefs").as_posix()
    RENDERED.mkdir(parents=True, exist_ok=True)
    for subset, config, seed in chosen:
        job = followup_job_name(subset, config, seed)
        path = RENDERED / f"{job}.yaml"
        path.write_text(
            render_followup_manifest(
                subset, config, seed, digest, subpath, exclusions, args.namespace
            ),
            encoding="utf-8",
        )
        if args.dry_run:
            print(f"[dry-run] {job}: {path}")
            continue
        if existing_job(job, args.namespace):
            print(f"[skip] Existing Job is unchanged: {job}")
            continue
        subprocess.run(
            ["kubectl", "-n", args.namespace, "create",
             "--dry-run=client", "-f", str(path)],
            check=True,
        )
        subprocess.run(
            ["kubectl", "-n", args.namespace, "create", "-f", str(path)],
            check=True,
        )
        print(f"[submitted] {job} snapshot={digest}", flush=True)
    print(f"source_sha256={digest} staging={snapshot}")


if __name__ == "__main__":
    main()
