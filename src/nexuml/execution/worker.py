"""Canonical remote worker entrypoint; never submit the outer job recursively."""

import argparse
import os
from pathlib import Path
from typing import Any

from nexuml.core.config import ResolvedConfig
from nexuml.execution import run
from nexuml.execution.definitions import LocalExecution, RayClusterExecution
from nexuml.execution.semantics import ensure_distributed_semantics


def launch_options(scenario: Any, environment: dict[str, str]) -> dict[str, Any]:
    """Validate torchrun rank/world-size rather than run independent replicas.

    Returns:
        Native Lightning placement preserving the reviewed training request.

    Raises:
        ValueError: For incomplete or inconsistent native distributed launch settings.
    """
    if "LOCAL_WORLD_SIZE" not in environment:
        if int(environment.get("WORLD_SIZE", "1")) > 1:
            raise ValueError(
                "Distributed replicas must launch through torchrun, not independent Python."
            )
        return {}
    local = int(environment["LOCAL_WORLD_SIZE"])
    world = int(environment.get("WORLD_SIZE", "0"))
    rank = int(environment.get("RANK", "-1"))
    local_rank = int(environment.get("LOCAL_RANK", "-1"))
    if (
        local < 1
        or world < local
        or world % local
        or not 0 <= rank < world
        or not 0 <= local_rank < local
    ):
        raise ValueError("Inconsistent torchrun rank/world-size environment.")
    if not isinstance(scenario.execution, LocalExecution):
        raise ValueError("External torchrun must use the derived Local worker configuration.")
    if scenario.training.devices not in ("auto", local):
        raise ValueError("torchrun local world-size differs from reviewed training.devices.")
    if world > 1:
        ensure_distributed_semantics(scenario)
    return {"devices": local, "num_nodes": world // local, "enable_loggers": rank == 0}


def main() -> None:
    """Read one explicit worker config and dispatch the existing lifecycle.

    Raises:
        ValueError: If the worker receives an outer job instead of a derived config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    if args.config.startswith("s3://"):
        from nexuml.storage.s3 import S3Client

        text = S3Client().read_bytes(args.config).decode("utf-8")
    else:
        text = Path(args.config).read_text(encoding="utf-8")
    scenario = ResolvedConfig.from_yaml(text).to_scenario()
    if not isinstance(scenario.execution, (LocalExecution, RayClusterExecution)):
        raise ValueError(
            "Remote worker requires a derived Local or RayCluster config, not an outer job."
        )
    run(scenario, **launch_options(scenario, dict(os.environ)))


if __name__ == "__main__":
    main()
