"""CPU-only native configuration authoring, persistence and declaration checks."""

from __future__ import annotations

from collections.abc import Generator
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING, Any, cast

import pytest
from pydantic import ValidationError

from nexuml.core.types import FactorySpec
from nexuml_library.environments.isaaclab_config import (
    IsaacBindings,
    IsaacTaskConfigSpec,
    ViserSpec,
    isaac_task_config,
)

if TYPE_CHECKING:
    # Deliberately absent: forward annotations must not require the native package.
    import uninstalled_native  # ty: ignore[unresolved-import]


def _mutate(cfg: dict, *, values: list, enabled: bool = True) -> None:
    cfg.update(values=values, enabled=enabled)


def _replace(cfg: dict, scale: float = 2.0) -> dict:
    return {**cfg, "scale": scale}


@pytest.mark.parametrize(
    "args,kwargs", [((), {}), (({}, {}), {}), (({},), {"cfg": {}}), ((), {"other": {}})]
)
def test_runtime_customization_requires_exactly_one_config(args, kwargs):
    spec = isaac_task_config(_replace)
    with pytest.raises(TypeError, match="exactly one config"):
        spec.build(*args, **kwargs)
    assert spec.build(cfg={}) == {"scale": 2.0}


def _forward_annotations(
    cfg: "uninstalled_native.Config", *, option: "uninstalled_native.Option"
) -> "uninstalled_native.Config":
    return cfg


def _no_config() -> None:
    pass


def _keyword_config(*, cfg: dict) -> None:
    pass


def _unannotated_config(cfg) -> None:
    pass


def _unannotated_option(cfg: dict, option) -> None:
    pass


def _unannotated_return(cfg: dict):
    pass


def _positional_option(cfg: dict, option: int = 1, /) -> None:
    pass


def _variadic_args(cfg: dict, *args: int) -> None:
    pass


def _variadic_kwargs(cfg: dict, **kwargs: int) -> None:
    pass


async def _async_config(cfg: dict) -> None:
    pass


def _generator_config(cfg: dict) -> Generator[dict, None, None]:
    yield cfg


class _CallableConfig:
    def __call__(self, cfg: dict) -> None:
        pass

    def configure(self, cfg: dict) -> None:
        pass


def test_capture_and_build_mutation_and_replacement():
    spec = isaac_task_config(_mutate, values=(1, {"nested": (2, 3)}))
    assert isinstance(spec, FactorySpec)
    assert spec.factory == f"{__name__}:_mutate"
    assert spec.args == []
    assert spec.kwargs == {"values": [1, {"nested": [2, 3]}]}
    cfg = {"original": True}
    assert spec.build(cfg) is cfg
    assert cfg == {"original": True, "values": [1, {"nested": [2, 3]}], "enabled": True}

    replacement = isaac_task_config(_replace, scale=3.0).build(cfg)
    assert replacement is not cfg
    assert replacement == {**cfg, "scale": 3.0}
    assert "scale" not in cfg
    assert isaac_task_config(_replace).build({}) == {"scale": 2.0}


def test_forward_annotations_are_not_evaluated_or_coerced():
    cfg = {}
    spec = isaac_task_config(_forward_annotations, option={"native": "inert"})
    assert spec.build(cfg) is cfg


@pytest.mark.parametrize(
    "fn",
    [
        _no_config,
        _keyword_config,
        _unannotated_config,
        _unannotated_option,
        _unannotated_return,
        _positional_option,
        _variadic_args,
        _variadic_kwargs,
        _async_config,
        _generator_config,
        _CallableConfig,
    ],
)
def test_incompatible_functions_are_rejected(fn):
    with pytest.raises(TypeError):
        isaac_task_config(fn)


def test_nonimportable_targets_are_rejected(monkeypatch):
    def local(cfg: dict) -> None:
        pass

    for fn in (local, lambda cfg: None, _CallableConfig().configure):
        with pytest.raises(ValueError, match="top-level|bound methods"):
            isaac_task_config(fn)
    with pytest.raises(TypeError, match="callable"):
        isaac_task_config(cast(Any, None))
    monkeypatch.setattr(_replace, "__module__", "__main__")
    with pytest.raises(ValueError, match="__main__"):
        isaac_task_config(_replace)


@pytest.mark.parametrize("kwargs", [{}, {"unknown": 1}, {"values": [], "cfg": {}}])
def test_missing_unknown_and_runtime_config_kwargs_are_rejected(kwargs):
    with pytest.raises(TypeError):
        isaac_task_config(_mutate, **kwargs)


@pytest.mark.parametrize(
    "value", [object(), {1: "not a string key"}, {"bad": float("inf")}, float("nan"), {1, 2}]
)
def test_nonportable_arguments_are_rejected(value):
    with pytest.raises(ValidationError, match="unsupported|string keys|finite"):
        isaac_task_config(_mutate, values=value)


def test_positional_configuration_arguments_are_rejected():
    with pytest.raises(ValidationError, match="args must be empty"):
        IsaacTaskConfigSpec.from_factory(_replace, 3.0)
    with pytest.raises(ValidationError, match="args must be empty"):
        IsaacTaskConfigSpec(factory="not_installed:configure", args=[3.0])


@pytest.mark.parametrize(
    ("target", "kwargs", "error"),
    [
        (f"{__name__}:_unannotated_return", {}, TypeError),
        (f"{__name__}:_mutate", {}, TypeError),
        (f"{__name__}:_replace", {"unknown": 1}, TypeError),
        (f"{__name__}:_CallableConfig", {}, TypeError),
        ("__main__:configure", {}, ValueError),
        ("not_a_target", {}, ValueError),
        ("not_installed_native_task:configure", {}, ValueError),
    ],
)
def test_restored_callable_is_checked_at_execution(target, kwargs, error):
    spec = IsaacTaskConfigSpec.model_validate({"factory": target, "kwargs": kwargs})
    with pytest.raises(error):
        spec.build({})


def test_static_json_and_yaml_round_trip_blocks_targets_native_and_cuda():
    payload = {
        "task_config": isaac_task_config(_mutate, values=(1, {"nested": (2, 3)})).model_dump(
            mode="json"
        ),
        "bindings": IsaacBindings(
            observations={"observation": ("policy",), "image": ("sensors", "rgb")}
        ).model_dump(mode="json"),
        "viser": ViserSpec(purposes=("train", "eval"), visible_env_indices=(0, 2)).model_dump(
            mode="json"
        ),
    }
    code = textwrap.dedent(
        """
        import importlib.abc
        import io
        import json
        import sys

        payload = json.loads(sys.argv[1])
        target_module = payload['task_config']['factory'].partition(':')[0]
        attempts = []
        class BlockImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                root = fullname.split('.')[0]
                if (
                    fullname == target_module
                    or root.startswith(('isaaclab', 'isaacsim'))
                    or root in {'omni', 'pxr', 'warp', 'newton', 'viser'}
                    or fullname.startswith('nexuml_library.environments._isaaclab')
                ):
                    attempts.append(fullname)
                    raise AssertionError('forbidden import: ' + fullname)
        sys.meta_path.insert(0, BlockImports())

        import torch
        def forbidden_cuda(*args, **kwargs):
            raise AssertionError('static restoration must not allocate CUDA')
        torch.cuda._lazy_init = forbidden_cuda
        torch.cuda.init = forbidden_cuda

        from pydantic import BaseModel
        from ruamel.yaml import YAML
        from nexuml.core.serialization import lower_model, restore_model_data
        from nexuml_library.environments.isaaclab_config import (
            IsaacBindings, IsaacTaskConfigSpec, ViserSpec,
        )
        class Authoring(BaseModel):
            task_config: IsaacTaskConfigSpec
            bindings: IsaacBindings
            viser: ViserSpec

        authored = Authoring.model_validate(payload)
        assert authored.model_dump(mode='json') == payload
        assert Authoring.model_validate_json(authored.model_dump_json()) == authored
        yaml = YAML()
        stream = io.StringIO()
        yaml.dump(lower_model(authored), stream)
        restored = Authoring.model_validate(
            restore_model_data(yaml.load(stream.getvalue()), Authoring)
        )
        assert restored == authored
        assert restored.task_config.factory == payload['task_config']['factory']
        assert restored.bindings.observations['image'] == ('sensors', 'rgb')
        assert target_module not in sys.modules
        assert not attempts, attempts
        assert not torch.cuda.is_initialized()
        print('static JSON/YAML restore: no target, native, viewer or CUDA execution')
        """
    )
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", code, json.dumps(payload)],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root / "library/src")]),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_bindings_require_explicit_nonempty_literal_mapping_paths():
    bindings = IsaacBindings.model_validate(
        {"observations": {"observation": ["policy"], "image": ["sensors", "camera.rgb"]}}
    )
    assert bindings.observations == {
        "observation": ("policy",),
        "image": ("sensors", "camera.rgb"),
    }
    assert bindings.action == "action"
    with pytest.raises(ValidationError, match="observations"):
        IsaacBindings.model_validate({})


@pytest.mark.parametrize(
    "fields",
    [
        {"observations": {}},
        {"observations": {"": ("policy",)}},
        {"observations": {"observation": ()}},
        {"observations": {"observation": ("",)}},
        {"observations": {"observation": (1,)}},
        {"observations": {"observation": "policy"}},
        {"observations": {"observation": ("policy",)}, "action": ""},
    ],
)
def test_invalid_bindings_are_rejected(fields):
    with pytest.raises(ValidationError):
        IsaacBindings.model_validate(fields)


def test_viser_defaults_and_explicit_selection_are_portable():
    assert ViserSpec().model_dump(mode="json") == {
        "purposes": ["train"],
        "bind_address": "127.0.0.1",
        "port": 8080,
        "eye": [4.0, -4.0, 3.0],
        "lookat": [0.0, 0.0, 0.0],
        "visible_env_indices": None,
        "max_visible_envs": None,
        "enable_markers": True,
        "enable_live_plots": True,
        "live_plots_update_interval": 5,
    }
    for port in (1, 64535):
        spec = ViserSpec(
            purposes=("eval",),
            bind_address="localhost",
            port=port,
            visible_env_indices=(0, 2),
            max_visible_envs=1,
            enable_markers=False,
            enable_live_plots=False,
            live_plots_update_interval=1,
        )
        assert ViserSpec.model_validate_json(spec.model_dump_json()) == spec


@pytest.mark.parametrize(
    "fields",
    [
        {"purposes": ()},
        {"purposes": ("train", "train")},
        {"purposes": ("replay",)},
        {"bind_address": "0.0.0.0"},
        {"bind_address": "::"},
        {"bind_address": "192.168.1.2"},
        {"port": 0},
        {"port": -1},
        {"port": 64536},
        {"port": 65535},
        {"port": True},
        {"port": 8080.0},
        {"port": "8080"},
        {"eye": (float("inf"), 0, 0)},
        {"lookat": (0, float("nan"), 0)},
        {"eye": (0, 0)},
        {"visible_env_indices": (0, 0)},
        {"visible_env_indices": (-1,)},
        {"visible_env_indices": (True,)},
        {"visible_env_indices": (1.0,)},
        {"max_visible_envs": 0},
        {"max_visible_envs": True},
        {"live_plots_update_interval": 0},
        {"live_plots_update_interval": True},
        {"share": True},
        {"public": True},
        {"unknown": True},
    ],
)
def test_unsafe_or_invalid_viser_declarations_are_rejected(fields):
    with pytest.raises(ValidationError):
        ViserSpec.model_validate(fields)


@pytest.mark.parametrize(
    ("model", "field", "value"),
    [
        (isaac_task_config(_replace), "factory", "not_installed:configure"),
        (IsaacBindings(observations={"observation": ("policy",)}), "action", "control"),
        (ViserSpec(), "port", 9000),
    ],
)
def test_models_are_frozen_and_forbid_extra_fields(model, field, value):
    with pytest.raises(ValidationError, match="frozen"):
        setattr(model, field, value)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        type(model).model_validate({**model.model_dump(), "unexpected": True})
