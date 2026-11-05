#!/usr/bin/env python3
"""Submit twelve paired NEX-233 FD003 cluster-conditioned CfC GPU Jobs.

Require the independently prepared, train-engine-only cluster artifact
before submission. No completed reliability/ablation Job is modified.
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

TEMPLATE = Path(__file__).with_name("job_rul_cfc_cluster_conditioning.yaml.template")
ARMS = ("reference", "working", "failure", "both")
SEEDS = (42, 43, 44)
CLUSTERS = Path(
    "/mnt/juicefs_prisma/models/nexuml/.experiments/rul/prepared/"
    "cmapss-FD003-seed42-cfc-conditional-clusters-v1.pt"
)


def cluster_trials() -> tuple[tuple[str, int], ...]:
    """Enumerate the predeclared paired 4x3 experiment.

    Returns:
        Every arm and head seed exactly once.
    """
    return tuple((arm, seed) for arm in ARMS for seed in SEEDS)


def job_name(arm: str, seed: int) -> str:
    """Return one immutable, DNS-compatible Job name.

    Returns:
        Cluster-conditioning Job name.

    Raises:
        ValueError: If arm or seed is not in the registered matrix.
    """
    if (arm, seed) not in cluster_trials():
        raise ValueError("Unsupported FD003 cluster-conditioning experiment")
    return f"nexuml-rul-cfc-cluster-fd003-{arm}-s{seed}"


def render_cluster_job(
    arm: str,
    seed: int,
    digest: str,
    source_subpath: str,
    excluded_hosts: tuple[str, ...],
    namespace: str = NAMESPACE,
) -> str:
    """Render one source-attested native CfC Job.

    Args:
        arm: Reference, working-only, failure-only or combined conditioning.
        seed: One of the three independent readout seeds.
        digest: Content-addressed source snapshot SHA-256.
        source_subpath: Path inside the source JuiceFS PVC.
        excluded_hosts: All excluded GH200 nodes and student-worker-1.
        namespace: Target Kubernetes namespace.

    Returns:
        Fully substituted one-GPU Kubernetes Job manifest.

    Raises:
        ValueError: If any prerequisite identifier is invalid.
    """
    name = job_name(arm, seed)
    if (
        re.fullmatch(r"[a-f0-9]{64}", digest) is None
        or source_subpath != f"job-staging/nexuml-nex233/{digest}"
        or not excluded_hosts
        or "student-worker-1" not in excluded_hosts
        or any(re.fullmatch(r"[A-Za-z0-9.-]+", host) is None for host in excluded_hosts)
        or NAME_PATTERN.fullmatch(namespace) is None
    ):
        raise ValueError("Unsafe cluster Job provenance or node constraints")
    mapping = {
        "__JOB_NAME__": name,
        "__NAMESPACE__": namespace,
        "__SEED__": str(seed),
        "__CONFIG_SLUG__": arm,
        "__CONFIGURATION__": arm,
        "__SUBSET__": "FD003",
        "__SUBSET_LOWER__": "fd003",
        "__SOURCE_DIGEST__": digest,
        "__SOURCE_SHORT__": digest[:12],
        "__STAGE_SUBPATH__": source_subpath,
        "__EXCLUDED_NODES__": ", ".join(sorted(set(excluded_hosts))),
        "__SCENARIO__": f"maintenance-rul-cfc-cluster-fd003-{arm}-s{seed}",
    }
    rendered = TEMPLATE.read_text(encoding="utf-8")
    for token, value in mapping.items():
        rendered = rendered.replace(token, value)
    if re.search(r"__[A-Z_]+__", rendered):
        raise ValueError("Unresolved cluster-conditioning Job template variable")
    return rendered


def main() -> None:
    """Validate the shared cluster data and submit only new experiment Jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=NAMESPACE)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()
    if NAME_PATTERN.fullmatch(args.namespace) is None:
        parser.error("Invalid namespace")
    stage = args.staging_root.expanduser().absolute()
    if stage == Path("/mnt/juicefs") or not stage.is_relative_to(Path("/mnt/juicefs")):
        parser.error("Source staging must be inside /mnt/juicefs")
    if not CLUSTERS.is_file():
        parser.error(
            f"Missing train-only cluster artifact: {CLUSTERS}. "
            "Prepare it once with the NexuML cluster source before launching."
        )
    choices = cluster_trials()
    if args.seeds is not None:
        if len(set(args.seeds)) != len(args.seeds) or any(
            seed not in SEEDS for seed in args.seeds
        ):
            parser.error(f"Seeds must be a unique subset of {SEEDS}")
        choices = tuple(item for item in choices if item[1] in args.seeds)
    exclusions = discover_excluded_hostnames()
    digest = digest_source()
    snapshot = stage_snapshot(stage, digest, dry_run=args.dry_run)
    source_subpath = snapshot.relative_to("/mnt/juicefs").as_posix()
    RENDERED.mkdir(parents=True, exist_ok=True)
    for arm, seed in choices:
        name = job_name(arm, seed)
        manifest = RENDERED / f"{name}.yaml"
        manifest.write_text(
            render_cluster_job(
                arm, seed, digest, source_subpath, exclusions, args.namespace
            ),
            encoding="utf-8",
        )
        if args.dry_run:
            print(f"[dry-run] {name}: {manifest}")
            continue
        if existing_job(name, args.namespace):
            print(f"[skip] Existing Job unchanged: {name}")
            continue
        subprocess.run(
            ["kubectl", "-n", args.namespace, "create",
             "--dry-run=client", "-f", str(manifest)],
            check=True,
        )
        subprocess.run(
            ["kubectl", "-n", args.namespace, "create", "-f", str(manifest)],
            check=True,
        )
        print(f"[submitted] {name} source={digest}", flush=True)
    print(f"source_sha256={digest} source_snapshot={snapshot}")


if __name__ == "__main__":
    main()
