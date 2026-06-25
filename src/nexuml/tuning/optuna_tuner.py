"""Optuna-based hyperparameter tuning for NexuML."""

from __future__ import annotations

import ast
import copy
import logging
from contextlib import nullcontext
from typing import Any, Callable, cast

from nexuml.core.types import ScenarioSpec, TuningSpec

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "training.lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "training.batch_size": {"type": "categorical", "choices": [32, 64, 128]},
}


def _set_nested(obj: Any, path: str, value: Any) -> None:
    """Set a nested attribute using dot-separated path (e.g. 'training.lr')."""
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _get_nested(obj: Any, path: str) -> Any:
    """Get a nested attribute using dot-separated path.

    Returns:
        Value at the resolved attribute path.
    """
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def build_objective(
    base_scenario: ScenarioSpec,
    search_space: dict[str, dict[str, Any]],
    metric_key: str = "val/loss",
    enable_progress_bar: bool = False,
    build_factory: Callable[..., ScenarioSpec] | None = None,
) -> Callable:
    """Build an Optuna objective function from a base scenario and search space.

    Args:
        base_scenario: The base ScenarioSpec to tune.
        search_space: Dict mapping dotted-path param names to Optuna suggest kwargs.
            Each entry must have a "type" key (float, int, categorical) and the
            remaining keys are forwarded to the Optuna suggest method.
            Example::
                {
                    "training.lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
                    "training.max_epochs": {"type": "int", "low": 5, "high": 50},
                    "training.batch_size": {"type": "categorical", "choices": [32, 64, 128]},
                }
        metric_key: Lightning logged metric to optimise (must be logged with self.log()).
        enable_progress_bar: Show Lightning progress bar during each trial.
        build_factory: Optional factory to rebuild the scenario from sampled params.

    Returns:
        Callable: objective(trial) -> float | list[float]

    Raises:
        ImportError: If the ``optuna`` package is not installed.
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("Install optuna: pip install optuna")

    from nexuml.core.registry import get_registry
    from nexuml.training.lightning import train

    def objective(trial: optuna.Trial) -> float | list[float]:
        scalar_params, arch_params = _resolve_search_space(trial, search_space)
        scenario = (
            build_factory(**arch_params)
            if build_factory and arch_params
            else copy.deepcopy(base_scenario)
        )

        for param_path, value in scalar_params.items():
            _set_nested(scenario, param_path, value)

        logger.info(f"Trial {trial.number}: {trial.params}")

        registry = get_registry()
        result = train(
            scenario,
            registry=registry,
            enable_progress_bar=enable_progress_bar,
            run_name=f"trial_{trial.number}",
        )

        # Prefer eval_algorithm_results (contains AUC etc. not in logged_metrics
        # when no external logger is configured), then fall back to logged_metrics.
        eval_results = getattr(result, "eval_algorithm_results", None) or {}
        logged = result.trainer.logged_metrics
        combined = {**logged, **eval_results}
        # Also check with test/ prefix used by NexuSession
        test_eval = {f"test/{k}": v for k, v in eval_results.items()}
        combined.update(test_eval)

        if metric_key not in combined:
            available = sorted(combined.keys())
            raise ValueError(f"Metric '{metric_key}' not found. Available: {available}")

        return float(combined[metric_key])

    return objective


def _resolve_search_space(
    trial: Any,
    search_space: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    scalar_params: dict[str, Any] = {}
    arch_params: dict[str, Any] = {}
    sampled: dict[str, Any] = {}

    for name, spec in search_space.items():
        _resolve_entry(trial, name, spec, sampled, scalar_params, arch_params)

    return scalar_params, arch_params


def _is_scalar_override(name: str, spec: dict[str, Any]) -> bool:
    return "." in name and not ({"choices", "when", "derived"} & set(spec))


def _resolve_entry(
    trial: Any,
    name: str,
    spec: dict[str, Any],
    sampled: dict[str, Any],
    scalar_params: dict[str, Any],
    arch_params: dict[str, Any],
    force_arch: bool = False,
) -> None:
    if "derived" in spec:
        value = _evaluate_derived(spec["derived"], sampled)
    else:
        value = _suggest(trial, name, spec)

    sampled[name] = value
    if not force_arch and _is_scalar_override(name, spec):
        scalar_params[name] = value
    else:
        _set_param_dict(arch_params, name, value)

    when = spec.get("when") or {}
    branch = when.get(value)
    if branch is None:
        return
    for child_name, child_spec in branch.items():
        _resolve_entry(
            trial,
            child_name,
            child_spec,
            sampled,
            scalar_params,
            arch_params,
            force_arch=True,
        )


def _set_param_dict(params: dict[str, Any], path: str, value: Any) -> None:
    target = params
    parts = path.split(".")
    for part in parts[:-1]:
        existing = target.setdefault(part, {})
        if not isinstance(existing, dict):
            raise ValueError(
                f"Cannot assign nested search parameter {path!r}; {part!r} is not a dict"
            )
        target = existing
    target[parts[-1]] = value


def _suggest(trial: Any, name: str, spec: dict[str, Any]) -> Any:
    kwargs = dict(spec)
    kwargs.pop("when", None)
    suggest_type = kwargs.pop("type", None)
    if suggest_type is None and "choices" in kwargs:
        suggest_type = "categorical"

    if suggest_type == "float":
        return trial.suggest_float(name, **kwargs)
    if suggest_type == "int":
        return trial.suggest_int(name, **kwargs)
    if suggest_type == "categorical":
        return trial.suggest_categorical(name, **kwargs)
    raise ValueError(
        f"Unknown suggest type {suggest_type!r} for param {name!r}. "
        "Use 'float', 'int', or 'categorical'."
    )


def _evaluate_derived(rule: str | Callable[..., Any], sampled: dict[str, Any]) -> Any:
    if not isinstance(rule, str):
        fn = cast(Callable[..., Any], rule)
        return fn(**sampled)
    tree = ast.parse(rule, mode="eval")
    allowed_nodes = (
        ast.Expression,
        ast.IfExp,
        ast.Compare,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Eq,
        ast.NotEq,
        ast.In,
        ast.NotIn,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.UnaryOp,
        ast.Not,
        ast.List,
        ast.Tuple,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Unsupported derived expression node: {type(node).__name__}")
    return eval(
        compile(tree, "<derived-search-space>", "eval"), {"__builtins__": {}}, dict(sampled)
    )


def tune(
    scenario: ScenarioSpec,
    search_space: dict[str, dict[str, Any]],
    tuning_spec: TuningSpec | None = None,
    metric_key: str = "val/loss",
    enable_progress_bar: bool = False,
    build_factory: Callable[..., ScenarioSpec] | None = None,
) -> Any:
    """Run Optuna hyperparameter search.

    Args:
        scenario: Base ScenarioSpec to tune.
        search_space: Dict mapping dotted-path param names to Optuna suggest kwargs.
        tuning_spec: TuningSpec controlling n_trials, storage, pruning, etc.
        metric_key: Logged metric to optimise.
        enable_progress_bar: Show progress bar during each trial.
        build_factory: Optional factory to rebuild the scenario from sampled params.

    Returns:
        Completed ``optuna.Study``.

    Raises:
        ImportError: If the ``optuna`` package is not installed.
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("Install optuna: pip install optuna")

    if tuning_spec is None:
        tuning_spec = TuningSpec()

    # Set up storage
    from nexuml.core.log_paths import resolve_logs_root

    storage_path = resolve_logs_root(tuning_spec.storage)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_path.with_suffix('.db')}"

    # Print service info with Optuna dashboard command
    from nexuml.tracking.logger import print_service_info

    print_service_info(
        trainer_loggers=[],
        scenario_name=scenario.name,
        tuning_spec=tuning_spec,
    )

    # Pruner
    pruner = optuna.pruners.MedianPruner() if tuning_spec.prune else optuna.pruners.NopPruner()
    sampler = optuna.samplers.TPESampler(group=True, multivariate=True)

    # Direction(s)
    directions = tuning_spec.directions
    if len(directions) == 1:
        study = optuna.create_study(
            direction=directions[0],
            storage=storage_url,
            study_name=scenario.name,
            pruner=pruner,
            sampler=sampler,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            directions=directions,
            storage=storage_url,
            study_name=scenario.name,
            pruner=pruner,
            sampler=sampler,
            load_if_exists=True,
        )

    objective = build_objective(
        base_scenario=scenario,
        search_space=search_space,
        metric_key=metric_key,
        enable_progress_bar=enable_progress_bar,
        build_factory=build_factory,
    )

    mlflow_context = _maybe_start_mlflow_study_run(scenario)
    with mlflow_context:
        study.optimize(objective, n_trials=tuning_spec.n_trials)

    if len(directions) == 1:
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best params: {study.best_trial.params}")
        logger.info(f"Best value: {study.best_trial.value}")
    else:
        logger.info(f"Completed {len(study.trials)} trials (multi-objective)")

    return study


def _maybe_start_mlflow_study_run(scenario: ScenarioSpec):
    """Start an MLflow parent run for Optuna studies when enabled.

    Returns:
        Context manager — ``mlflow.start_run(...)`` when MLflow is active,
        otherwise ``nullcontext()``.
    """
    logging_spec = getattr(scenario, "logging", None)
    mlflow_spec = getattr(logging_spec, "mlflow", None) if logging_spec is not None else None
    if mlflow_spec is None or not mlflow_spec.auto_nested_runs or mlflow_spec.run_id is not None:
        return nullcontext()

    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        from nexuml.tracking.logger import (
            _augment_mlflow_tags,
            _configure_mlflow_tracking_uri,
            _get_or_create_mlflow_experiment_id,
            _resolve_mlflow_artifact_location,
        )
    except ImportError:
        return nullcontext()

    if mlflow.active_run() is not None:
        return nullcontext()

    assert logging_spec is not None
    tracking_uri = _configure_mlflow_tracking_uri(mlflow_spec.tracking_uri)
    experiment_name = mlflow_spec.experiment_name or logging_spec.experiment_name
    artifact_location = _resolve_mlflow_artifact_location(
        tracking_uri,
        mlflow_spec.artifact_location,
    )
    experiment_id = _get_or_create_mlflow_experiment_id(
        MlflowClient(tracking_uri=tracking_uri),
        experiment_name,
        artifact_location,
    )

    return mlflow.start_run(
        experiment_id=experiment_id,
        run_name="study",
        tags={key: str(value) for key, value in _augment_mlflow_tags(mlflow_spec.tags).items()},
        log_system_metrics=logging_spec.log_system_metrics,
    )
