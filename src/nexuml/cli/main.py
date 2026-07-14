"""NexuML CLI — entry point for all pipeline operations."""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path
from typing import Any, Optional

# Suppress Lightning internal deprecation warnings we can't fix
warnings.filterwarnings(
    "ignore",
    message=r"`isinstance\(treespec, LeafSpec\)` is deprecated",
    category=FutureWarning,
)

# Silence Lightning marketing tips (litlogger / litmodels) before any Lightning import
import lightning_utilities.core.rank_zero as _lu_rank_zero  # noqa: E402

_original_rank_zero_info = _lu_rank_zero.rank_zero_info


def _filtered_rank_zero_info(*args: Any, **kwargs: Any) -> None:
    if args and isinstance(args[0], str):
        msg = args[0]
        if "litlogger" in msg or "litmodels" in msg:
            return
    _original_rank_zero_info(*args, **kwargs)


_lu_rank_zero.rank_zero_info = _filtered_rank_zero_info

import typer  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

app = typer.Typer(name="nexuml", help="NexuML pipeline framework CLI")
console = Console()


def _get_scenario_fn(name: str):
    """Look up a scenario function by name.

    Returns:
        Callable scenario function matching *name*.

    Raises:
        typer.Exit: If no scenario with the given name is registered.
    """
    from nexuml.core.scenario_registry import get_scenario_registry

    registry = get_scenario_registry()
    try:
        return registry.get(name)
    except KeyError as exc:
        available = ", ".join(sorted(registry.list().keys()))
        console.print(f"[red]Unknown scenario '{name}'. Available: {available}[/red]")
        raise typer.Exit(1) from exc


def _load_scenario(
    scenario_name: Optional[str],
    config_path: Optional[Path],
    scenario_file: Optional[Path] = None,
):
    sources = [bool(scenario_name), bool(config_path), bool(scenario_file)]
    if sum(sources) > 1:
        console.print("[red]Provide only one of scenario name, --config, or --scenario-file[/red]")
        raise typer.Exit(1)
    if scenario_name:
        scenario_fn = _get_scenario_fn(scenario_name)
        return scenario_fn()
    if config_path:
        from nexuml.core.config import ResolvedConfig
        from nexuml.core.types import ScenarioSpec

        config = ResolvedConfig.load(config_path)
        return ScenarioSpec.model_validate(config.model_dump())
    if scenario_file:
        from nexuml.core.scenario_loader import load_scenario_file

        return load_scenario_file(scenario_file).scenario

    console.print("[red]Provide a scenario name, --config path, or --scenario-file path[/red]")
    raise typer.Exit(1)


@app.command(name="resolve", help="Resolve a scenario to YAML config")
def scenario_resolve(
    name: str = typer.Argument(help="Scenario name"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output YAML path"),
):
    """Resolve a named scenario to a YAML config file."""
    from nexuml.core.compiler import compile
    from nexuml.core.registry import get_registry

    scenario_fn = _get_scenario_fn(name)
    scenario = scenario_fn()

    registry = get_registry()
    pipeline = compile(scenario, registry)

    if output is None:
        output = Path(f"configs/{name}.yaml")

    pipeline.resolved_config.save(output)
    console.print(f"[green]Resolved config saved to {output}[/green]")


@app.command(name="build", help="Compile and validate a pipeline from config")
def build(
    config_path: Path = typer.Argument(help="Path to resolved YAML config"),
):
    """Compile and validate a pipeline from a resolved YAML config."""
    from nexuml.core.compiler import compile
    from nexuml.core.config import ResolvedConfig
    from nexuml.core.registry import get_registry
    from nexuml.core.types import ScenarioSpec

    config = ResolvedConfig.load(config_path)
    scenario = ScenarioSpec.model_validate(config.model_dump())

    registry = get_registry()
    pipeline = compile(scenario, registry)

    console.print("[green]Pipeline compiled successfully![/green]")
    console.print(f"  Stages: {list(pipeline.stages.keys())}")
    console.print(f"  Loss keys: {pipeline.loss_keys}")
    console.print(f"  Input sizes: {pipeline.input_sizes}")

    if (
        scenario.logging is not None
        and scenario.logging.diagram is not None
        and scenario.logging.diagram.enabled
    ):
        try:
            from nexuml.core.diagram import export_mermaid_diagram
            from nexuml.core.log_paths import resolve_logs_root

            dspec = scenario.logging.diagram
            output_path = resolve_logs_root(dspec.output_dir) / f"{scenario.name}.md"
            export_mermaid_diagram(
                pipeline,
                output_path,
                depth=dspec.depth,
                direction=dspec.direction,
                show_params=dspec.show_params,
                show_shapes=dspec.show_shapes,
                show_metrics=dspec.show_metrics,
            )
            console.print(f"[green]Diagram exported to {output_path}[/green]")
        except Exception as exc:
            console.print(f"[yellow]Warning: diagram export failed: {exc}[/yellow]")


@app.command(name="train", help="Train a pipeline from a scenario or config")
def train_cmd(
    scenario_name: Optional[str] = typer.Argument(None, help="Scenario name"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Config YAML path"),
    scenario_file: Optional[Path] = typer.Option(
        None,
        "--scenario-file",
        help="Trusted Python file exposing scenario() -> ScenarioSpec",
    ),
    artifact_dir: Optional[Path] = typer.Option(
        None,
        "--artifact-dir",
        help="Optional directory for scenario-file provenance snapshots",
    ),
    max_epochs: Optional[int] = typer.Option(None, "--max-epochs", help="Override max epochs"),
    trainer_checkpoint: Optional[Path] = typer.Option(
        None,
        "--trainer-checkpoint",
        help="Optional Lightning Trainer checkpoint to resume from.",
    ),
    override: Optional[list[str]] = typer.Option(
        None,
        "--override",
        "-O",
        help="Override a scenario field: key.path=value. Repeatable.",
    ),
):
    """Train a pipeline from a scenario name or config file.

    Raises:
        typer.Exit: If input validation fails or training encounters an error.
    """
    from nexuml.training.lightning import NexuSession

    scenario = None
    loaded_file = None
    if scenario_name or config_path or scenario_file:
        if sum([bool(scenario_name), bool(config_path), bool(scenario_file)]) > 1:
            console.print(
                "[red]Provide only one of scenario name, --config, or --scenario-file[/red]"
            )
            raise typer.Exit(1)
        if scenario_file:
            try:
                from nexuml.core.scenario_loader import load_scenario_file

                loaded_file = load_scenario_file(scenario_file)
                scenario = loaded_file.scenario
            except Exception as exc:
                console.print(f"[red]Invalid scenario file: {exc}[/red]")
                raise typer.Exit(1) from exc
        else:
            scenario = _load_scenario(scenario_name, config_path)
    elif trainer_checkpoint is None:
        console.print("[red]Provide a scenario/config/scenario-file or --trainer-checkpoint[/red]")
        raise typer.Exit(1)

    if scenario is not None and max_epochs is not None:
        scenario.training.max_epochs = max_epochs
        if (
            scenario.training.scheduler is not None
            and "max_epochs" in scenario.training.scheduler.params
        ):
            scenario.training.scheduler.params["max_epochs"] = max_epochs

    if scenario is not None and override:
        from nexuml.cli.overrides import apply_overrides

        try:
            apply_overrides(scenario, override)
        except (KeyError, ValueError) as exc:
            console.print(f"[red]Override error: {exc}[/red]")
            raise typer.Exit(1) from exc

    scenario_label = scenario.name if scenario is not None else str(trainer_checkpoint)
    console.print(f"[blue]Training scenario: {scenario_label}[/blue]")

    if (
        scenario is not None
        and scenario.logging is not None
        and scenario.logging.diagram is not None
        and scenario.logging.diagram.enabled
    ):
        try:
            from nexuml.core.compiler import compile as compile_pipeline
            from nexuml.core.diagram import export_mermaid_diagram
            from nexuml.core.log_paths import resolve_logs_root
            from nexuml.core.registry import get_registry

            pipeline = compile_pipeline(scenario, get_registry())
            dspec = scenario.logging.diagram
            output_path = resolve_logs_root(dspec.output_dir) / f"{scenario.name}.md"
            export_mermaid_diagram(
                pipeline,
                output_path,
                depth=dspec.depth,
                direction=dspec.direction,
                show_params=dspec.show_params,
                show_shapes=dspec.show_shapes,
                show_metrics=dspec.show_metrics,
            )
            console.print(f"[green]Diagram exported to {output_path}[/green]")
        except Exception as exc:
            console.print(f"[yellow]Warning: diagram export failed: {exc}[/yellow]")

    session = NexuSession(
        scenario=scenario,
        trainer_checkpoint=trainer_checkpoint,
    )
    result = session.run()
    if loaded_file is not None and artifact_dir is not None:
        from nexuml.core.provenance import snapshot_scenario_file_run

        snapshot_scenario_file_run(
            loaded_file,
            artifact_dir,
            command="train",
            command_args={
                "scenario_file": str(scenario_file),
                "max_epochs": max_epochs,
                "trainer_checkpoint": str(trainer_checkpoint) if trainer_checkpoint else None,
            },
        )

    if scenario is not None and scenario.exports:
        from nexuml.core.export import export_package

        for spec in scenario.exports:
            if spec.kind == "train_package":
                export_path = Path(spec.output) if spec.output else Path("exported_model")
                export_package(
                    result.pipeline,
                    export_path,
                    lightning_module=result.lightning_module,
                    trainer=result.trainer,
                )
                console.print(f"[green]Exported train package to {export_path}[/green]")
            else:
                console.print(f"[yellow]Skipping unsupported export kind: {spec.kind}[/yellow]")

    console.print("[green]Training complete![/green]")
    if result.test_results:
        console.print(f"  Test results: {result.test_results}")


@app.command(name="export-dataset", help="Export a dataset view from a scenario or config")
def export_dataset_cmd(
    scenario_name: Optional[str] = typer.Argument(None, help="Scenario name"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Config YAML path"),
    output: Path = typer.Option(
        Path("exported_dataset"), "--output", "-o", help="Export directory"
    ),
    backend: str = typer.Option("numpy", "--backend", help="Dataset export backend"),
    split: list[str] | None = typer.Option(
        None,
        "--split",
        help="Split to export. Repeat the option to export multiple splits.",
    ),
    preprocess: bool = typer.Option(
        False,
        "--preprocess/--no-preprocess",
        help="Run the compiled pipeline until the requested preprocessing keys exist.",
    ),
    preprocess_until_key: list[str] | None = typer.Option(
        None,
        "--preprocess-until-key",
        help="TensorDict x key marking the preprocessing boundary. Repeatable.",
    ),
    x_key: list[str] | None = typer.Option(
        None,
        "--x-key",
        help="x TensorDict keys to persist. Repeatable. Default: all.",
    ),
    y_key: list[str] | None = typer.Option(
        None,
        "--y-key",
        help="label TensorDict keys to persist. Repeatable. Default: all.",
    ),
    include_labels: bool = typer.Option(
        True,
        "--labels/--no-labels",
        help="Include labels from the batch y TensorDict.",
    ),
    dtype: str | None = typer.Option(
        None,
        "--dtype",
        help="Optional storage dtype passed to the export backend, e.g. float16.",
    ),
):
    """Export a dataset view to disk from a scenario or config.

    Raises:
        typer.Exit: If preprocessing is enabled without --preprocess-until-key.
    """
    from nexuml.data.export import export_data_module
    from nexuml.training.lightning import (
        LightningFeatureExtractor,
        create_runtime_artifacts,
    )

    scenario = _load_scenario(scenario_name, config_path)
    preprocess_until_key = preprocess_until_key or []
    use_preprocessing = preprocess or bool(preprocess_until_key)
    if use_preprocessing and not preprocess_until_key:
        console.print(
            "[red]Provide at least one --preprocess-until-key when --preprocess is enabled.[/red]"
        )
        raise typer.Exit(1)

    console.print(f"[blue]Preparing dataset export for scenario: {scenario.name}[/blue]")
    runtime = create_runtime_artifacts(scenario)

    transform = None
    if use_preprocessing:
        transform = LightningFeatureExtractor(
            runtime.lightning_module,
            x_keys=preprocess_until_key,
        )
        console.print(
            f"[blue]Preprocessing enabled until keys: {', '.join(preprocess_until_key)}[/blue]"
        )
    else:
        console.print("[blue]Exporting raw data-module batches (no preprocessing).[/blue]")

    export_data_module(
        runtime.data_module,
        output,
        backend=backend,
        splits=split,
        transform=transform,
        x_keys=x_key,
        y_keys=y_key,
        include_labels=include_labels,
        dtype=dtype,
    )
    console.print(f"[green]Dataset exported to {output}[/green]")


@app.command(name="export", help="Export a trained pipeline")
def export_cmd(
    scenario_name: str = typer.Argument(help="Scenario name to train and export"),
    output: Path = typer.Option(Path("exported_model"), "--output", "-o", help="Export directory"),
    checkpoint: Optional[Path] = typer.Option(
        None, "--checkpoint", help="Optional checkpoint to export"
    ),
):
    """Export a trained pipeline to a portable model package."""
    from nexuml.core.export import export_package
    from nexuml.training.lightning import NexuSession

    scenario_fn = _get_scenario_fn(scenario_name)
    scenario = scenario_fn()
    if scenario.checkpoint is None:
        from nexuml.core.types import CheckpointLoadSpec

        scenario.checkpoint = CheckpointLoadSpec()
    scenario.checkpoint.source = str(checkpoint) if checkpoint is not None else None

    console.print(f"[blue]Creating pipeline with scenario: {scenario.name}[/blue]")
    session = NexuSession.from_scenario(scenario)
    session.setup()
    console.print("[green]Pipeline created successfully![/green]")

    source_metadata: dict[str, Any] = {}
    if checkpoint is not None:
        source_metadata["source"] = {"checkpoint": str(checkpoint)}

    export_package(
        session.pipeline,
        output,
        lightning_module=session.lightning_module,
        trainer=session.trainer,
        checkpoint_path=checkpoint,
        source_metadata=source_metadata,
    )
    console.print(f"[green]Exported to {output}[/green]")


@app.command(
    name="smoke", help="Run full smoke test: resolve → build → train → export → reload → infer"
)
def smoke(
    scenario_name: str = typer.Argument(
        default="synthetic-linear-ae-reconstruction",
        help="Scenario name",
    ),
    max_epochs: int = typer.Option(3, "--max-epochs", help="Training epochs"),
    download: bool = typer.Option(False, "--download", help="Download datasets before training"),
):
    """Run a full smoke-test: resolve → build → train → export → reload → infer."""
    from tensordict import TensorDict

    from nexuml.core.compiler import compile
    from nexuml.core.export import export_package, infer, load_package
    from nexuml.core.log_paths import resolve_logs_root
    from nexuml.core.registry import get_registry
    from nexuml.training.lightning import create_data_module_from_spec, train

    scenario_fn = _get_scenario_fn(scenario_name)

    # Only pass download=True when the scenario accepts it (avoid TypeError
    # on scenarios that do not declare a download parameter).
    sig = inspect.signature(scenario_fn)
    kwargs: dict[str, Any] = {}
    if "download" in sig.parameters:
        kwargs["download"] = download
    scenario = scenario_fn(**kwargs)
    scenario.training.max_epochs = max_epochs

    # 1. Resolve
    console.print("[blue]1. Resolving scenario...[/blue]")
    registry = get_registry()
    pipeline = compile(scenario, registry)
    config_path = Path(f"configs/{scenario_name}.yaml")
    pipeline.resolved_config.save(config_path)
    console.print(f"   Config saved to {config_path}")

    # 2. Build (already done in compile)
    console.print("[blue]2. Pipeline built successfully[/blue]")
    console.print(f"   Stages: {list(pipeline.stages.keys())}")

    # 3. Train
    console.print("[blue]3. Training...[/blue]")
    result = train(scenario, registry=registry, enable_progress_bar=True)
    console.print("   Training complete!")

    # 4. Export
    console.print("[blue]4. Exporting...[/blue]")
    export_dir = resolve_logs_root(f".experiments/{scenario_name}_export")
    export_package(result.pipeline, export_dir)
    console.print(f"   Exported to {export_dir}")

    # 5. Reload
    console.print("[blue]5. Reloading...[/blue]")
    loaded_pipeline, loaded_config, metadata = load_package(export_dir, registry)
    console.print(f"   Config hash: {metadata.get('config_hash', 'N/A')}")

    # 6. Inference
    console.print("[blue]6. Running inference...[/blue]")
    data_module = create_data_module_from_spec(scenario)
    data_module.setup()
    x_sample, y_sample = data_module.dataset[0]
    x_batch = TensorDict(
        {k: v.unsqueeze(0) for k, v in x_sample.items()},
        batch_size=[1],
    )
    x_out = infer(loaded_pipeline, x_batch)
    console.print(f"   Inference output keys: {list(x_out.keys())}")

    console.print("[bold green]Smoke test PASSED![/bold green]")


@app.command(name="tune", help="Run Optuna hyperparameter search on a scenario")
def tune_cmd(
    scenario_name: Optional[str] = typer.Argument(None, help="Scenario name"),
    scenario_file: Optional[Path] = typer.Option(
        None,
        "--scenario-file",
        help="Trusted Python file exposing scenario() -> ScenarioSpec",
    ),
    artifact_dir: Optional[Path] = typer.Option(
        None,
        "--artifact-dir",
        help="Optional directory for scenario-file provenance snapshots",
    ),
    n_trials: Optional[int] = typer.Option(None, "--n-trials", help="Number of Optuna trials"),
    metric_key: Optional[str] = typer.Option(
        None, "--metric", help="Metric to optimise (default: tuning_spec.metric_key or val/loss)"
    ),
    direction: Optional[str] = typer.Option(
        None,
        "--direction",
        help="Optuna direction: minimize or maximize (default: from tuning_spec)",
    ),
    storage: Optional[str] = typer.Option(None, "--storage", help="Optuna storage path"),
    prune: Optional[bool] = typer.Option(None, "--prune/--no-prune", help="Enable Optuna pruning"),
    override: Optional[list[str]] = typer.Option(
        None,
        "--override",
        "-O",
        help="Override a scenario field: key.path=value. Repeatable.",
    ),
):
    """Tune a scenario's hyperparameters with Optuna using a default search space.

    Raises:
        typer.Exit: If scenario input is ambiguous or loading fails.
    """
    from nexuml.core.types import TuningSpec
    from nexuml.tuning.optuna_tuner import tune

    if bool(scenario_name) == bool(scenario_file):
        console.print("[red]Provide exactly one of scenario name or --scenario-file[/red]")
        raise typer.Exit(1)

    loaded_file = None
    if scenario_file:
        try:
            from nexuml.core.scenario_loader import load_scenario_file

            loaded_file = load_scenario_file(scenario_file)
            scenario = loaded_file.scenario
        except Exception as exc:
            console.print(f"[red]Invalid scenario file: {exc}[/red]")
            raise typer.Exit(1) from exc
    else:
        scenario_fn = _get_scenario_fn(scenario_name or "")
        scenario = scenario_fn()

    if loaded_file and loaded_file.search_space:
        search_space = loaded_file.search_space
    else:
        from nexuml.tuning.optuna_tuner import DEFAULT_SEARCH_SPACE

        search_space = DEFAULT_SEARCH_SPACE

    tuning_spec = (
        loaded_file.tuning_spec
        if loaded_file and loaded_file.tuning_spec
        else TuningSpec(
            n_trials=n_trials if n_trials is not None else 50,
            directions=[direction or "minimize"],
            metric_key=metric_key or "val/loss",
            storage=storage or ".experiments/optuna/optuna.log",
            prune=bool(prune) if prune is not None else False,
        )
    )
    if loaded_file and loaded_file.tuning_spec:
        if n_trials is not None:
            tuning_spec.n_trials = n_trials
        if storage is not None:
            tuning_spec.storage = storage
        if prune is not None:
            tuning_spec.prune = prune
        if direction is not None:
            tuning_spec.directions = [direction]
        if metric_key is not None:
            tuning_spec.metric_key = metric_key

    if override:
        from nexuml.cli.overrides import apply_overrides

        try:
            apply_overrides(scenario, override)
        except (KeyError, ValueError) as exc:
            console.print(f"[red]Override error: {exc}[/red]")
            raise typer.Exit(1) from exc

    effective_metric = tuning_spec.metric_key
    console.print(
        f"[blue]Tuning scenario: {scenario.name} "
        f"({tuning_spec.n_trials} trials, metric={effective_metric}, "
        f"direction={tuning_spec.directions[0]})[/blue]"
    )
    study = tune(
        scenario=scenario,
        search_space=search_space,
        tuning_spec=tuning_spec,
        metric_key=effective_metric,
        build_factory=loaded_file.build_factory if loaded_file else None,
    )
    if loaded_file is not None and artifact_dir is not None:
        from nexuml.core.provenance import snapshot_scenario_file_run

        snapshot_scenario_file_run(
            loaded_file,
            artifact_dir,
            command="tune",
            command_args={
                "scenario_file": str(scenario_file),
                "n_trials": tuning_spec.n_trials,
                "metric": effective_metric,
                "storage": tuning_spec.storage,
                "prune": tuning_spec.prune,
            },
            search_space=search_space,
            study=study,
        )

    console.print("[green]Tuning complete![/green]")
    console.print(f"  Best trial: {study.best_trial.number}")
    console.print(f"  Best params: {study.best_trial.params}")
    console.print(f"  Best value: {study.best_trial.value:.6f}")


registry_app = typer.Typer(help="Inspect discovered registry contents")
app.add_typer(registry_app, name="registry")

backend_app = typer.Typer(help="Inspect available backend implementations")
app.add_typer(backend_app, name="backend")


def _print_discovery_errors(registry: Any, verbose: bool = False) -> None:
    """Surface modules that failed to import/register during discovery.

    Discovery is resilient — a broken module is skipped, not fatal — so the
    only place the failure becomes visible is here. Without this the list would
    silently omit items with no hint as to why.
    """
    errors = getattr(registry, "errors", None)
    if not errors:
        return
    table = Table(title=f"[bold yellow]Discovery errors ({len(errors)})[/bold yellow]")
    table.add_column("Module", style="yellow")
    table.add_column("Phase", style="magenta")
    table.add_column("Error", style="red")
    for err in sorted(errors, key=lambda e: (e.module, e.key or "")):
        where = err.module if err.key is None else f"{err.module}\n[dim]→ {err.key}[/dim]"
        table.add_row(where, err.phase, f"{err.error_type}: {err.message}")
    console.print(table)
    if verbose:
        for err in errors:
            console.print(f"[dim]── {err.module} ──[/dim]")
            console.print(err.traceback)
    else:
        console.print(
            "[dim]Run with --verbose for full tracebacks. "
            "Fix or remove the listed modules to restore them.[/dim]"
        )


@registry_app.command(name="list", help="List registered items by kind: layers|data|scenarios|eval")
def registry_list(
    kind: str = typer.Argument("layers", help="Kind to list: layers, data, scenarios, eval"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show full tracebacks for discovery errors"
    ),
):
    """List registered items by kind (layers, data, scenarios, eval).

    Raises:
        typer.Exit: If *kind* is not one of the recognised values.
    """
    kind = kind.lower()
    if kind == "layers":
        from nexuml.core.registry import get_registry

        registry = get_registry()
        items = registry.list()
        table = Table(title="Registered Layers")
        table.add_column("Type Key", style="cyan")
        table.add_column("Module", style="green")
        table.add_column("Constructor Params", style="white")
        for type_key, cls in sorted(items.items()):
            module_path = f"{cls.__module__}.{cls.__name__}"
            sig = inspect.signature(cls.__init__)

            def _ann(p: inspect.Parameter) -> str:
                return (
                    p.annotation.__name__
                    if hasattr(p.annotation, "__name__")
                    else str(p.annotation)
                )

            params = [
                f"{name}: {_ann(p)}"
                for name, p in sig.parameters.items()
                if name not in ("self", "input_sizes", "keys_in", "keys_out", "kwargs")
                and p.default is not inspect.Parameter.empty
            ]
            table.add_row(type_key, module_path, ", ".join(params[:5]))
        console.print(table)
        _print_discovery_errors(registry, verbose)
    elif kind == "data":
        from nexuml.data.registry import get_dataset_registry

        registry = get_dataset_registry()
        items = registry.list()
        table = Table(title="Registered Datasets")
        table.add_column("Type Key", style="cyan")
        table.add_column("Module", style="green")
        for type_key, cls in sorted(items.items()):
            module_path = f"{cls.__module__}.{cls.__name__}"
            table.add_row(type_key, module_path)
        console.print(table)
        _print_discovery_errors(registry, verbose)
    elif kind == "scenarios":
        from nexuml.core.scenario_registry import get_scenario_registry

        registry = get_scenario_registry()
        items = registry.list()
        table = Table(title="Registered Scenarios")
        table.add_column("Name", style="cyan")
        table.add_column("Module", style="green")
        for name, fn in sorted(items.items()):
            table.add_row(
                name, f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__name__', repr(fn))}"
            )
        console.print(table)
        _print_discovery_errors(registry, verbose)
    elif kind == "eval":
        from nexuml.evaluation.registry import get_eval_registry

        registry = get_eval_registry()
        items = registry.list()
        table = Table(title="Registered Evaluation Algorithms")
        table.add_column("Type Key", style="cyan")
        table.add_column("Module", style="green")
        for type_key, cls in sorted(items.items()):
            module_path = f"{cls.__module__}.{cls.__name__}"
            table.add_row(type_key, module_path)
        console.print(table)
        _print_discovery_errors(registry, verbose)
    else:
        console.print(f"[red]Unknown kind '{kind}'. Use: layers, data, scenarios, eval[/red]")
        raise typer.Exit(1)


@backend_app.command(name="list", help="List available backend implementations")
def backend_list(
    category: Optional[str] = typer.Argument(
        None,
        help="Optional category filter, e.g. data-export, data-loader, training",
    ),
):
    """List available backend implementations.

    Raises:
        typer.Exit: If *category* does not match any registered backends.
    """
    rows: list[tuple[str, str, str]] = []

    def add(row_category: str, name: str, implementation: str) -> None:
        if category is None or row_category == category:
            rows.append((row_category, name, implementation))

    from nexuml.data.export import get_export_backend, list_export_backends
    from nexuml.data.loaders import get_loader_backend, list_loader_backends

    for name in sorted(list_export_backends()):
        backend_cls = get_export_backend(name)
        add("data-export", name, f"{backend_cls.__module__}.{backend_cls.__name__}")

    for name in sorted(list_loader_backends()):
        backend = get_loader_backend(name)
        add("data-loader", name, f"{backend.__class__.__module__}.{backend.__class__.__name__}")

    add("training", "lightning", "nexuml.training.lightning.NexuSession")
    add("tracking", "tensorboard", "nexuml.tracking.logger")
    add("tracking", "dvclive", "nexuml.tracking.logger")
    add("tracking", "mlflow", "nexuml.tracking.logger")
    add("eval-storage", "memory", "nexuml.evaluation.storage")
    add("eval-storage", "memmap", "nexuml.evaluation.storage")
    add("pipeline-export", "package", "nexuml.core.export.export_package")
    add("pipeline-export", "safetensors", "nexuml.core.export.export_safetensors")
    add("pipeline-export", "onnx", "nexuml.core.export.export_onnx")

    if category is not None and not rows:
        console.print(f"[red]Unknown backend category or no backends found: {category}[/red]")
        raise typer.Exit(1)

    table = Table(title="Available Backends")
    table.add_column("Category", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Implementation", style="white")
    for row in sorted(rows):
        table.add_row(*row)
    console.print(table)


# ---------------------------------------------------------------------------
# Library subcommands
# ---------------------------------------------------------------------------

library_app = typer.Typer(help="Manage local library roots")
app.add_typer(library_app, name="library")


@library_app.command(name="add", help="Add a local library root path")
def library_add(
    path: Path = typer.Argument(
        ..., help="Path to local library root", exists=True, file_okay=False, dir_okay=True
    ),
):
    """Add a local library root path to the NexuML config."""
    from nexuml.core.discovery import LibraryConfig

    config = LibraryConfig.load()
    config.add_root(str(path))
    config.save()
    console.print(f"[green]Added library root: {path.resolve()}[/green]")


@library_app.command(name="delete", help="Remove a local library root path")
def library_delete(
    path: Path = typer.Argument(..., help="Path to local library root"),
):
    """Remove a local library root path from the NexuML config.

    Raises:
        typer.Exit: If the path is not currently configured as a library root.
    """
    from nexuml.core.discovery import LibraryConfig

    config = LibraryConfig.load()
    normalized = str(Path(path).expanduser().resolve())
    if normalized not in config.roots:
        console.print(f"[red]Library root not configured: {normalized}[/red]")
        raise typer.Exit(1)
    config.remove_root(normalized)
    config.save()
    console.print(f"[green]Removed library root: {normalized}[/green]")


@library_app.command(name="list", help="List available library sources")
def library_list():
    """List all available library sources (installed and local)."""
    from nexuml.core.discovery import LibraryConfig, discover_entry_point_packages

    config = LibraryConfig.load()
    entry_point_packages = discover_entry_point_packages()

    if not config.roots and not entry_point_packages:
        console.print("[yellow]No libraries are available or configured.[/yellow]")
        return

    table = Table(title="Library Sources")
    table.add_column("Kind", style="magenta", no_wrap=True)
    table.add_column("Source", style="cyan", no_wrap=True, overflow="ignore")

    for package in entry_point_packages:
        kind = "base" if package == "nexuml_library" else "entry-point"
        table.add_row(kind, package)

    for root in config.roots:
        table.add_row("path", root)

    console.print(table)


if __name__ == "__main__":
    app()

# Expose as a Click command for mkdocs-click documentation generation
click_app = typer.main.get_command(app)
