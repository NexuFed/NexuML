"""Exercise package policy without importing application or training runtimes."""

from __future__ import annotations

import ast
import fnmatch
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture
def policy() -> dict[str, Any]:
    source = Path(__file__).resolve().parents[2] / "src/nexuml/core/export.py"
    names = {
        "_RUNTIME_EXTERN_PATTERNS",
        "_RUNTIME_TOP_LEVELS",
        "_STDLIB_MODULES",
        "_is_runtime_module",
        "_discover_pipeline_module_packages",
        "_apply_package_policy",
        "_validate_package_policy",
        "_is_stdlib_module",
        "_normalize_module_name",
        "_collect_external_dependencies",
        "_policy_requirement_modules",
    }
    tree = ast.parse(source.read_text())
    nodes: list[ast.stmt] = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
        )
    ]
    namespace: dict[str, Any] = {
        "sys": sys,
        "fnmatch": fnmatch,
        "ast": ast,
        "_resolve_distribution": lambda module: (module.split(".")[0], "1.0"),
    }
    # Compile only policy definitions; even module-level NexuML/Torch imports stay unexecuted.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def _apply(policy, module_names):
    actions = {"extern": [], "intern": [], "deny": []}
    exporter = SimpleNamespace(**{name: values.append for name, values in actions.items()})
    layers = [type("Layer", (), {"__module__": name})() for name in module_names]
    custom = policy["_apply_package_policy"](exporter, SimpleNamespace(modules=lambda: layers))
    return actions, custom


def _matches(module, patterns):
    return any(
        module == pattern.removesuffix(".**") or fnmatch.fnmatchcase(module, pattern)
        for pattern in patterns
    )


@pytest.mark.parametrize(
    "module",
    [
        "lightning_utilities.core.rank_zero",
        "tensorboard.compat.proto.summary_pb2",
        "uvicorn",
        "websockets",
        "fastapi",
        "fastapi.exceptions",
        "fastapi.middleware.cors",
        "fastapi.responses",
        "starlette.responses",
        "starlette.datastructures",
        "starlette.concurrency",
        "imageio",
        "imageio.v2",
        "imageio_ffmpeg",
    ],
)
def test_deferred_cli_tracking_dependencies_are_explicit_runtime_externals(policy, module):
    actions, custom = _apply(policy, [module])
    assert _matches(module, actions["extern"]), module
    assert policy["_is_runtime_module"](module)
    assert not custom


def test_owned_and_custom_sources_remain_hermetic_without_a_catch_all(policy):
    actions, custom = _apply(policy, ["nexuml.core.policy", "custom_actor.layers"])
    assert custom == {"custom_actor"}
    for module in ("nexuml.core.policy", "nexuml_library.layers.actor", "custom_actor.layers"):
        assert _matches(module, actions["intern"])
        assert not _matches(module, actions["extern"])
    assert not _matches("unlisted_dependency.runtime", actions["extern"])
    assert not policy["_is_runtime_module"]("unlisted_dependency.runtime")
    assert _matches("tests.actor", actions["deny"])


def test_owned_source_externalization_is_still_rejected(policy):
    exporter = SimpleNamespace(
        externed_modules=lambda: ["nexuml.core.policy"], interned_modules=lambda: []
    )
    with pytest.raises(RuntimeError, match="NexuML-owned modules were externalized"):
        policy["_validate_package_policy"](exporter, set())


@pytest.fixture
def dependency_exporter():
    root = Path(__file__).resolve().parents[2] / "src"
    edges = {
        "<policy>": ["nexuml.core.policy", "custom_actor.runtime"],
        "nexuml.core.policy": [
            "torch",
            "nexuml.execution.definitions",
            "nexuml.execution",
            "nexuml.core.factory",
            "nexuml.data.loaders.definitions",
        ],
        "nexuml.core.factory": ["lightning.pytorch.callbacks"],
        "nexuml.data.loaders.definitions": ["nexuml.data.loaders.dali_backend"],
        "nexuml.data.loaders.dali_backend": ["nvidia"],
        "nexuml.execution.definitions": [
            "pydantic",
            "nexuml.cli.main",
            "nexuml.training.lightning",
        ],
        "nexuml.execution": ["nexuml.execution.service"],
        "nexuml.execution.service": ["ray"],
        "nexuml.cli.main": ["fastapi", "uvicorn", "websockets", "starlette.responses"],
        "nexuml.training.lightning": ["torchrl", "lightning", "tensorboard"],
        "custom_actor.runtime": ["PIL.Image", "custom_actor.helper"],
        "custom_actor.helper": ["torch"],
    }
    externed = {
        "torch",
        "pydantic",
        "ray",
        "fastapi",
        "uvicorn",
        "websockets",
        "starlette.responses",
        "torchrl",
        "lightning",
        "lightning.pytorch.callbacks",
        "nvidia",
        "tensorboard",
        "PIL.Image",
    }
    for module in externed:
        edges[module] = []
    nodes: dict[str, dict[str, Any]] = {module: {} for module in edges}
    nodes["<policy>"]["is_pickle"] = True
    for module, filename in (
        ("nexuml.execution", "nexuml/execution/__init__.py"),
        ("nexuml.execution.definitions", "nexuml/execution/definitions.py"),
        ("nexuml.core.factory", "nexuml/core/factory.py"),
        ("nexuml.data.loaders.definitions", "nexuml/data/loaders/definitions.py"),
    ):
        nodes[module].update(
            source=(root / filename).read_text(), is_package=filename.endswith("__init__.py")
        )

    def source_dependencies(source, module, is_package):
        imports = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
                imports.update(f"{node.module}.{alias.name}" for alias in node.names)
        return [name for name in edges[module] if name in imports]

    graph = SimpleNamespace(
        nodes=nodes,
        successors=lambda module: iter(edges[module]),
        predecessors=lambda module: (parent for parent, deps in edges.items() if module in deps),
    )
    return SimpleNamespace(
        dependency_graph=graph,
        externed_modules=lambda: sorted(externed),
        _get_dependencies=source_dependencies,
        edges=edges,
    )


def test_policy_requirements_exclude_dispatch_only_deps_but_keep_source_contract(
    policy, dependency_exporter
):
    collect = policy["_collect_external_dependencies"]
    original_sources = {
        module: dict(attrs) for module, attrs in dependency_exporter.dependency_graph.nodes.items()
    }
    declared, supervised_requirements = collect(dependency_exporter)
    inferred, requirements = collect(dependency_exporter, inference_only=True)
    assert {entry["module"] for entry in inferred} == {"torch", "pydantic", "PIL"}
    assert set(requirements) == {"torch==1.0", "pydantic==1.0", "PIL==1.0"}
    assert "torchrl==1.0" in supervised_requirements
    assert "fastapi==1.0" in supervised_requirements
    assert collect(dependency_exporter) == (declared, supervised_requirements)
    assert dependency_exporter.dependency_graph.nodes == original_sources


@pytest.mark.parametrize("owner", ["custom_actor.runtime", "custom_actor.helper"])
@pytest.mark.parametrize("dependency", ["torchrl", "nexuml.training.lightning"])
def test_genuine_custom_actor_dependencies_are_never_dropped(
    policy, dependency_exporter, owner, dependency
):
    dependency_exporter.edges[owner].append(dependency)
    _, requirements = policy["_collect_external_dependencies"](
        dependency_exporter, inference_only=True
    )
    assert "torchrl==1.0" in requirements


def test_custom_import_of_lazy_execution_export_retains_its_dependencies(
    policy, dependency_exporter
):
    dependency_exporter.edges["custom_actor.runtime"].append("nexuml.execution")
    dependency_exporter.edges["nexuml.execution"].append("nexuml.execution.ray")
    dependency_exporter.edges["nexuml.execution.ray"] = ["lightning"]
    dependency_exporter.dependency_graph.nodes["nexuml.execution.ray"] = {}
    _, requirements = policy["_collect_external_dependencies"](
        dependency_exporter, inference_only=True
    )
    assert "lightning==1.0" in requirements


@pytest.mark.parametrize("statement", ["import torchrl", "class Eager:\n    import torchrl"])
def test_eager_settings_imports_are_not_masked(policy, dependency_exporter, statement):
    settings = dependency_exporter.dependency_graph.nodes["nexuml.execution.definitions"]
    settings["source"] += "\n" + statement + "\n"
    dependency_exporter.edges["nexuml.execution.definitions"].append("torchrl")
    _, requirements = policy["_collect_external_dependencies"](
        dependency_exporter, inference_only=True
    )
    assert "torchrl==1.0" in requirements
