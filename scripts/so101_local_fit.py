"""Simulator-free SO-101 bounded-fit, numerical-update and complete-export reference."""

import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
from typing import cast

from safetensors.torch import save_file
import torch

from nexuml.core.export import export_package
from nexuml.training import NexuSession, local_fit, local_fit_manifest
from nexuml_library.data.lerobot import LeRobotDataSource
from nexuml_library.scenarios.imitation.so101 import (
    so101_behavior_cloning,
    validate_so101_demonstrations,
)


def main():
    """Fit two externally initialized states twice; retain only local run artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new directory; existing evidence is never overwritten")
    recipe = so101_behavior_cloning(str(args.data.resolve()))
    validate_so101_demonstrations(cast(LeRobotDataSource, recipe.data.datasets[0].source))
    recipe.training = recipe.training.model_copy(update={"seed": args.seed})
    torch.manual_seed(args.seed)
    session = NexuSession(recipe, enable_progress_bar=False, enable_loggers=False)
    initial = {
        name: value.detach().cpu().clone() for name, value in session.pipeline.state_dict().items()
    }
    assert session.runtime.policy is not None
    args.output.mkdir(parents=True)
    export_package(session.runtime.policy, args.output / "untrained")
    first_manifest = local_fit_manifest(recipe, initial)

    def fit(state, manifest):
        return local_fit(
            trusted_recipe=recipe,
            initial_tensors=state,
            model_manifest=manifest,
            local_data=recipe.data,
            budget={"steps": args.steps},
            seed=args.seed,
            optimizer_mode="reset",
        )

    first = fit(initial, first_manifest)
    repeated = fit(initial, first_manifest)
    assert all(torch.equal(first.updated_tensors[k], repeated.updated_tensors[k]) for k in initial)
    assert any(not torch.equal(initial[k], first.updated_tensors[k]) for k in initial)
    second_initial = {name: value + 0.01 for name, value in initial.items()}
    second_manifest = local_fit_manifest(recipe, second_initial)
    second = fit(second_initial, second_manifest)
    second_repeated = fit(second_initial, second_manifest)
    assert all(
        torch.equal(second.updated_tensors[k], second_repeated.updated_tensors[k]) for k in initial
    )
    assert any(
        not torch.equal(first.updated_tensors[k], second.updated_tensors[k]) for k in initial
    )
    session.pipeline.load_state_dict(first.updated_tensors, strict=True)
    export_package(session.runtime.policy, args.output / "learned")
    for name, state in (
        ("initial", initial),
        ("updated", first.updated_tensors),
        ("second_initial", second_initial),
        ("second_updated", second.updated_tensors),
    ):
        save_file(state, args.output / f"{name}.safetensors")
    receipt = {}
    for name, result in (("first", first), ("second", second)):
        public = asdict(result)
        del public["updated_tensors"]
        receipt[name] = public
    receipt["bitwise_repeatable"] = True
    receipt["versions"] = {
        package: importlib.metadata.version(package)
        for package in ("nexuml", "nexuml-library", "torch", "lightning", "lerobot")
    }
    (args.output / "fit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
