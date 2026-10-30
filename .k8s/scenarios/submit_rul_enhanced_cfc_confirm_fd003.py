#!/usr/bin/env python3
"""Submit nine isolated native FD003 CfC confirmation experiments to Kubernetes.

Reuse the verified, immutable NEX-233 source staging implementation. Only
the nine confirmation Jobs are managed; prior composite Jobs are untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess

from submit_rul_enhanced_cfc_fd003 import (
    DEFAULT_STAGE,
    NAME_PATTERN,
    NAMESPACE,
    RENDERED,
    SEEDS,
    digest_source,
    existing_job,
    stage_snapshot,
)

TEMPLATE = Path(__file__).with_name(
    "job_rul_enhanced_cfc_confirm_fd003.yaml.template"
)
CONFIGURATIONS = ("baseline", "finetune", "all_no_trend")


def confirmation_job_name(configuration: str, seed: int) -> str:
    """Name one distinct prespecified native training Job.

    Returns:
        Kubernetes-compatible name using the configuration and seed.

    Raises:
        ValueError: For unsupported confirmation combinations.
    """
    if configuration not in CONFIGURATIONS or seed not in SEEDS:
        raise ValueError("Unexpected independent CfC confirmation combination")
    return f"nexuml-rul-fd003-cfc-confirm-{configuration.replace('_', '-')}-s{seed}"


def render_confirmation_manifest(
    configuration: str,
    seed: int,
    digest: str,
    source_subpath: str,
    namespace: str = NAMESPACE,
) -> str:
    """Render a single-GPU Job for an individually selected scenario.

    Args:
        configuration: The baseline or one of two screening winners.
        seed: Independent readout initialization seed.
        digest: Verified source snapshot digest.
        source_subpath: Path relative to the JuiceFS source PVC.
        namespace: Namespace for the submitted Job.

    Returns:
        Rendered manifest with no unresolved variable placeholders.

    Raises:
        ValueError: For unsupported configuration/seed or invalid template.
    """
    job = confirmation_job_name(configuration, seed)
    scenario = f"maintenance-rul-enhanced-cfc-confirm-fd003-{configuration}-s{seed}"
    replacements = {
        "__JOB_NAME__": job,
        "__CONFIGURATION__": configuration,
        "__CONFIG_SLUG__": configuration.replace("_", "-"),
        "__SEED__": str(seed),
        "__NAMESPACE__": namespace,
        "__SCENARIO__": scenario,
        "__SOURCE_DIGEST__": digest,
        "__SOURCE_SHORT__": digest[:12],
        "__STAGE_SUBPATH__": source_subpath,
    }
    rendered = TEMPLATE.read_text(encoding="utf-8")
    for old, new in replacements.items():
        rendered = rendered.replace(old, new)
    if re.search(r"__[A-Z_]+__", rendered):
        raise ValueError("Unresolved NEX-233 confirmation template placeholder")
    return rendered


def main() -> None:
    """Render, verify and optionally submit independently selected RUL Jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=NAMESPACE)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument(
        "--configurations", nargs="+", choices=CONFIGURATIONS,
        default=list(CONFIGURATIONS),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not NAME_PATTERN.fullmatch(args.namespace):
        parser.error("Invalid Kubernetes namespace")
    if len(args.seeds) != len(set(args.seeds)) or any(
        seed not in SEEDS for seed in args.seeds
    ):
        parser.error("Seeds must be unique and selected from 42, 43, 44")
    if len(args.configurations) != len(set(args.configurations)):
        parser.error("Configurations must be unique")
    staging_root = args.staging_root.expanduser().absolute()
    if (
        staging_root == Path("/mnt/juicefs")
        or not staging_root.is_relative_to(Path("/mnt/juicefs"))
    ):
        parser.error("Staging root must be under the JuiceFS source PVC")
    digest = digest_source()
    target = stage_snapshot(staging_root, digest, dry_run=args.dry_run)
    source_subpath = target.relative_to("/mnt/juicefs").as_posix()
    RENDERED.mkdir(parents=True, exist_ok=True)

    for configuration in args.configurations:
        for seed in args.seeds:
            name = confirmation_job_name(configuration, seed)
            manifest = RENDERED / f"{name}.yaml"
            manifest.write_text(
                render_confirmation_manifest(
                    configuration, seed, digest, source_subpath, args.namespace
                ),
                encoding="utf-8",
            )
            if args.dry_run:
                print(f"[dry-run] {name}: {manifest} (no upload or submission)")
                continue
            if existing_job(name, args.namespace):
                print(f"[skip existing job] {name}: delete explicitly to replace")
                continue
            subprocess.run(
                [
                    "kubectl", "-n", args.namespace, "apply",
                    "--dry-run=client", "-f", str(manifest),
                ],
                check=True,
            )
            subprocess.run(
                ["kubectl", "-n", args.namespace, "apply", "-f", str(manifest)],
                check=True,
            )
            print(f"[submitted] {name} snapshot={digest}", flush=True)
    print(f"source_sha256={digest} staging={target}")


if __name__ == "__main__":
    main()
