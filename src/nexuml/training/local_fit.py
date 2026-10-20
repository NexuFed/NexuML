"""Bounded, in-process fits of an operator-owned recipe; no transport or code loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Any

import lightning as L
import torch

from nexuml.core.compiler import compile
from nexuml.core.policy import CompiledPolicy
from nexuml.core.serialization import lower_model
from nexuml.core.types import DataSpec, ScenarioSpec, TrainingSpec
from nexuml.training.lightning import NexuSession


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _state_hash(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _validate_tensors(tensors, expected) -> None:
    if not isinstance(tensors, Mapping) or set(tensors) != set(expected):
        raise ValueError("Model state requires exactly the approved named tensors")
    for name, reference in expected.items():
        tensor = tensors[name]
        if (
            type(tensor) is not torch.Tensor
            or tensor.layout != torch.strided
            or tensor.device.type == "meta"
            or tensor.shape != reference.shape
            or tensor.dtype != reference.dtype
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError("Model tensor shape, dtype, layout or finiteness differs")


def _check_recipe(recipe) -> None:
    if not isinstance(recipe, ScenarioSpec) or not isinstance(recipe.training, TrainingSpec):
        raise ValueError(
            "trusted_recipe must be an operator-installed dataset-training ScenarioSpec"
        )
    if recipe.checkpoint is not None or recipe.callbacks or recipe.execution.kind != "local":
        raise ValueError(
            "Local fit requires a fresh local session without checkpoints or callbacks"
        )
    if recipe.data.preprocessing.enabled or not isinstance(recipe.training.batch_size, int):
        raise ValueError("Local fit requires explicit prepared inputs and a fixed batch size")
    loader = recipe.data.loader
    if loader.num_workers or loader.weighted_sampling or loader.backend.component_name != "torch":
        raise ValueError("Local fit requires the deterministic single-process TorchLoader")


def _apply_trainable_scope(
    pipeline: torch.nn.Module, trainable_names: tuple[str, ...] | None
) -> set[str]:
    eligible = {name for name, param in pipeline.named_parameters() if param.requires_grad}
    trainable = eligible
    if trainable_names is not None:
        if (
            not isinstance(trainable_names, tuple)
            or not trainable_names
            or any(type(name) is not str for name in trainable_names)
            or len(set(trainable_names)) != len(trainable_names)
        ):
            raise ValueError("Trainable scope requires a nonempty tuple of unique parameter names")
        trainable = set(trainable_names)
        if not trainable <= eligible:
            raise ValueError("Trainable scope requires originally trainable parameter names")
    if not trainable:
        raise ValueError("Local fit requires a nonempty trainable scope")
    for name, param in pipeline.named_parameters():
        param.requires_grad_(name in trainable)
    return trainable


def local_fit_manifest(
    trusted_recipe: ScenarioSpec,
    initial_tensors: Mapping[str, torch.Tensor],
    *,
    trainable_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Describe exact numeric state and locally installed model/physical semantics.

    The caller, not a remote request, owns ``trusted_recipe``. Build the expected
    manifest locally; never resolve an import, recipe or dataset from this manifest.
    ``trainable_names`` may select only originally trainable parameters by exact name.
    Unsupported recipes, initial states or scopes raise ``ValueError``.

    Returns:
        A numeric-state schema with content-pinned architecture and base identity.
    """
    _check_recipe(trusted_recipe)
    # Compilation needs explicit input shapes, not access to private episodes.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        pipeline = compile(trusted_recipe)
    state = pipeline.state_dict()
    _validate_tensors(initial_tensors, state)
    trainable = _apply_trainable_scope(pipeline, trainable_names)
    code_types: set[type] = {type(module) for module in pipeline.modules()}
    code_types.update(
        type(layer.component)
        for layers in trusted_recipe.pipeline.stages.values()
        for layer in layers
    )
    if trusted_recipe.policy is not None:
        code_types.add(type(trusted_recipe.policy.action_adapter))
    code = sorted(hashlib.sha256(inspect.getsource(cls).encode()).hexdigest() for cls in code_types)
    contract = (
        lower_model(trusted_recipe.interaction.environment.describe())
        if trusted_recipe.interaction is not None
        else None
    )
    pipeline_schema = lower_model(trusted_recipe.pipeline)
    # Stages execute in insertion order; sorted JSON object keys must not erase that order.
    pipeline_schema["stages"] = list(pipeline_schema["stages"].items())
    semantics = {
        "pipeline": pipeline_schema,
        "inputs": lower_model(trusted_recipe.data)["input_contracts"],
        "input_shapes": trusted_recipe.data.input_shapes,
        "policy": lower_model(trusted_recipe.policy) if trusted_recipe.policy else None,
        "interaction_contract": contract,
        "code_sha256": code,
        "training": lower_model(trusted_recipe.training),
        "loader": lower_model(trusted_recipe.data.loader),
    }
    if trainable_names is not None:
        semantics["trainable_names"] = sorted(trainable)
    return {
        "schema_version": 1,
        "update_kind": "full",
        "recipe_hash": _hash(semantics),
        "physical_schema_hash": _hash(contract),
        "base_version": _state_hash(initial_tensors),
        "tensors": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
                "role": "trainable" if name in trainable else "frozen",
            }
            for name, value in state.items()
        },
    }


@dataclass(frozen=True)
class LocalFitResult:
    """Shareable full numeric update; no session, paths, episodes or optimizer state."""

    updated_tensors: dict[str, torch.Tensor]
    metrics: dict[str, float]
    train_windows: int
    episodes: int
    provenance: dict[str, Any]


class _Consumption(L.Callback):
    def __init__(self):
        self.samples = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if outputs is None or not torch.isfinite(outputs["loss"]).all():
            raise ValueError("Local fit produced a nonfinite loss")
        self.samples += batch[0].batch_size[0]


def _episode_receipt(dataset, recipe) -> tuple[int, int, str]:
    meta = dataset.meta
    if meta is None or not {"episode_index", "frame_index", "timestamp", "split"} <= set(meta):
        raise ValueError("Local fit requires explicit episodic split/clock metadata")
    if (
        meta.empty
        or meta[["episode_index", "frame_index", "timestamp", "split"]].isna().any().any()
    ):
        raise ValueError("Local episode metadata is incomplete")
    if not set(meta["split"]) <= {"train", "val", "test"}:
        raise ValueError("Local fit requires fixed train/val/test partitions")
    for _, episode in meta.groupby("episode_index", sort=False):
        if episode["split"].nunique() != 1:
            raise ValueError("An episode crosses local data partitions")
        frames = episode["frame_index"].tolist()
        times = episode["timestamp"].tolist()
        if frames != list(range(len(frames))) or not all(math.isfinite(t) for t in times):
            raise ValueError("Local episodes require complete ordered frames and finite timestamps")
        if times[0] != 0 or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("Local episodes require a reset clock and increasing timestamps")
    train = meta[meta["split"] == "train"]
    if train.empty:
        raise ValueError("Local fit has no eligible training windows")
    contract = recipe.interaction.environment.describe() if recipe.interaction is not None else None
    if contract is not None:
        recorded = getattr(dataset, "info", {}).get("provenance", {}).get("interaction_contract")
        expected = lower_model(contract)
        if recorded != expected:
            raise ValueError("Local episode physical/processor contract differs from the recipe")
    digest = hashlib.sha256()
    digest.update(meta[["episode_index", "frame_index", "timestamp", "split"]].to_json().encode())
    for index in range(len(dataset)):
        x, y = dataset[index]
        if y is None:
            raise ValueError("Local fit requires numeric target windows")
        if contract is not None:
            if set(x.keys()) != set(contract.observations):
                raise ValueError("Local data includes undeclared actor inputs")
            CompiledPolicy._validate_fields(x, contract.observations, "Local observation")
            CompiledPolicy._validate_fields(y, contract.actions, "Local target")
        for side, values in (("x", x), ("y", y)):
            numeric = dict(values.items())
            for value in numeric.values():
                if type(value) is not torch.Tensor or not torch.isfinite(value).all():
                    raise ValueError("Local data must contain finite numeric windows")
            digest.update(side.encode())
            digest.update(_state_hash(numeric).encode())
    return len(train), int(train["episode_index"].nunique()), digest.hexdigest()


def local_fit(
    *,
    trusted_recipe: ScenarioSpec,
    initial_tensors: Mapping[str, torch.Tensor],
    model_manifest: Mapping[str, Any],
    local_data: DataSpec,
    budget: Mapping[str, int],
    seed: int,
    optimizer_mode: str = "reset",
    trainable_names: tuple[str, ...] | None = None,
    device: str = "cpu",
) -> LocalFitResult:
    """Fit approved private episodes with NexuSession and a fresh optimizer each call.

    ``local_data`` must exactly match the operator-owned recipe binding. Budgets
    contain exactly one positive integer, ``steps`` or ``epochs``. One frame is
    one eligible window in this first, non-chunked contract; counts are not steps.
    Calls are sequential single-process fits on CPU or one explicitly selected
    available ``cuda:N`` device, independent of any simulator. ``trainable_names``
    must match the operator-selected scope used to build the manifest.

    Returns:
        Validated CPU numeric tensors, allowlisted scalars and opaque fingerprints.

    Raises:
        ValueError: If authorization, compatibility, budget or numerical checks fail.
    """
    _check_recipe(trusted_recipe)
    if optimizer_mode != "reset":
        raise ValueError("Only optimizer_mode='reset' is supported")
    devices: int | str = 1
    if device == "cpu":
        accelerator = "cpu"
    elif isinstance(device, str) and re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device):
        index = int(device.split(":")[1])
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError("Selected CUDA device is not available")
        accelerator = "gpu"
        devices = f"{index},"  # Lightning interprets an unseparated integer as a device count.
    else:
        raise ValueError("device must be 'cpu' or one explicit 'cuda:N' device")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if not isinstance(budget, Mapping) or set(budget) not in ({"steps"}, {"epochs"}):
        raise ValueError("budget requires exactly one of steps or epochs")
    amount = next(iter(budget.values()))
    if type(amount) is not int or amount <= 0:
        raise ValueError("Local fit budget must be a positive integer")
    if not isinstance(local_data, DataSpec) or lower_model(local_data) != lower_model(
        trusted_recipe.data
    ):
        raise ValueError("Local data binding is not authorized by the installed recipe")
    # shortcut: one complete episode source, extend only with tested multi-source accounting.
    if local_data.source is not None or len(local_data.datasets) != 1:
        raise ValueError("Local fit requires one explicit episode source")
    source = local_data.datasets[0]
    root = getattr(source.source, "root", None)
    if (
        source.split_type != "keep"
        or source.max_samples is not None
        or not isinstance(root, str)
        or "://" in root
        or not Path(root).is_dir()
    ):
        raise ValueError(
            "Local fit requires complete approved on-disk episodes with preserved splits"
        )
    expected = local_fit_manifest(trusted_recipe, initial_tensors, trainable_names=trainable_names)
    if dict(model_manifest) != expected:
        raise ValueError("Model manifest/base/physical semantics differ from the installed recipe")
    state = {name: value.detach().cpu().clone() for name, value in initial_tensors.items()}
    training = trusted_recipe.training.model_copy(
        update={
            "max_epochs": amount if "epochs" in budget else -1,
            "max_steps": amount if "steps" in budget else None,
            "seed": seed,
            "deterministic": True,
            "accelerator": accelerator,
            "devices": devices,
            "strategy": "auto",
            "precision": "32-true",
        }
    )
    scenario = trusted_recipe.model_copy(deep=True, update={"training": training})
    with TemporaryDirectory(prefix="nexuml-local-fit-") as private_run:
        session = NexuSession(
            scenario,
            accelerator=accelerator,
            devices=devices,
            log_dir=private_run,
            enable_loggers=False,
            enable_progress_bar=False,
        )
        session.pipeline.load_state_dict(state, strict=True)
        _apply_trainable_scope(session.pipeline, trainable_names)
        windows, episodes, data_hash = _episode_receipt(session.data_module.dataset, scenario)
        consumption = _Consumption()
        getattr(session.trainer, "callbacks").append(consumption)
        session.fit()
        if not session.trainer.global_step:
            raise ValueError("Local fit executed no optimizer updates")
        metrics = {
            "optimizer_steps": float(session.trainer.global_step),
            "consumed_windows": float(consumption.samples),
        }
        for key, value in session.trainer.callback_metrics.items():
            if key == "train/loss":
                metrics[key] = float(value)
        meta = session.data_module.dataset.meta
        assert meta is not None  # Checked before fitting by _episode_receipt.
        if (meta["split"] == "val").any():
            for key, value in session.validate()[0].items():
                if key == "val/loss":
                    metrics[key] = float(value)
        updated = {
            name: value.detach().cpu().contiguous().clone()
            for name, value in session.pipeline.state_dict().items()
        }
    _validate_tensors(updated, state)
    if any(
        not torch.equal(updated[name], state[name])
        for name, schema in expected["tensors"].items()
        if schema["role"] == "frozen"
    ):
        raise ValueError("Local fit changed a tensor outside the shared trainable scope")
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Local fit produced a nonfinite approved metric")
    return LocalFitResult(
        updated,
        metrics,
        windows,
        episodes,
        {
            "schema_version": 1,
            "update_kind": "full",
            "optimizer_mode": "reset",
            "seed": seed,
            "budget": dict(budget),
            "model_manifest": expected,
            "updated_version": _state_hash(updated),
            "data_hash": data_hash,
            "fit_config_hash": _hash(lower_model(training)),
        },
    )
