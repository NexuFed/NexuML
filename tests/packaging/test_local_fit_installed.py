"""Installed-wheel SDK proof with server and simulator imports unavailable."""

import importlib.abc
import importlib.util
import os
from pathlib import Path
import runpy
import sys

import pytest

pytestmark = pytest.mark.skipif(
    "NEXUML_INSTALLED_DIST_TEST" not in os.environ,
    reason="installed distribution environment was not requested",
)


def test_installed_library_fit_without_server_or_simulator(tmp_path, monkeypatch):
    forbidden = {"fastapi", "uvicorn", "grpc", "isaacsim", "isaaclab", "omni", "pxr"}

    class BlockOptional(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in forbidden:
                raise ModuleNotFoundError(f"optional dependency blocked: {fullname}")

    assert not forbidden.intersection(sys.modules)
    original_find_spec = importlib.util.find_spec

    def absent_spec(name, package=None):
        if name.split(".")[0] in forbidden:
            return None
        return original_find_spec(name, package)

    # Availability probes see genuine absence; actual optional imports still fail.
    monkeypatch.setattr(importlib.util, "find_spec", absent_spec)
    blocker = BlockOptional()
    sys.meta_path.insert(0, blocker)
    try:
        import nexuml
        import nexuml_library
        from nexuml.training import local_fit

        roots = {Path("src").resolve(), Path("library/src").resolve()}
        assert roots.isdisjoint(Path(p).resolve() for p in sys.path if p)
        assert not any(
            str(Path(module.__file__).resolve()).startswith(str(root))
            for module in (nexuml, nexuml_library)
            for root in roots
        )
        unit = runpy.run_path(str(Path(__file__).resolve().parents[1] / "core/test_local_fit.py"))
        args = unit["request"](tmp_path)
        result = local_fit(**args)
        assert result.train_windows == 8 and result.episodes == 2
        assert result.metrics["optimizer_steps"] == 3
        assert result.provenance["optimizer_mode"] == "reset"
        if dataset_root := os.environ.get("NEXUML_SO101_DATA"):
            from nexuml.core.compiler import compile
            from nexuml.training import local_fit_manifest
            from nexuml_library.scenarios.imitation.so101 import so101_behavior_cloning

            recipe = so101_behavior_cloning(dataset_root)
            state = compile(recipe).state_dict()
            real = local_fit(
                trusted_recipe=recipe,
                initial_tensors=state,
                model_manifest=local_fit_manifest(recipe, state),
                local_data=recipe.data,
                budget={"steps": 3},
                seed=42,
            )
            assert real.episodes == 48 and real.train_windows > 48
        assert not forbidden.intersection(sys.modules)
    finally:
        sys.meta_path.remove(blocker)
