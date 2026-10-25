# nexuml.cli.main

*module*

NexuML CLI — entry point for all pipeline operations.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/cli/main.py#L1-L1)

## nexuml.cli.main.app

*attribute* · *module attribute*

```python
app = typer.Typer(name='nexuml', help='NexuML pipeline framework CLI')
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/cli/main.py#L14-L14)

## nexuml.cli.main.backend_app

*attribute* · *module attribute*

```python
backend_app = typer.Typer(help='Inspect available backend implementations')
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/cli/main.py#L649-L649)

## nexuml.cli.main.click_app

*attribute* · *module attribute*

```python
click_app = typer.main.get_command(app)
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/cli/main.py#L936-L936)

## nexuml.cli.main.console

*attribute* · *module attribute*

```python
console = Console()
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/cli/main.py#L15-L15)

## nexuml.cli.main.library_app

*attribute* · *module attribute*

```python
library_app = typer.Typer(help='Manage local library roots')
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/cli/main.py#L814-L814)

## nexuml.cli.main.registry_app

*attribute* · *module attribute*

```python
registry_app = typer.Typer(help='Inspect discovered registry contents')
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/cli/main.py#L646-L646)

## nexuml.cli.main.backend_list

*function*

```python
def backend_list(category: Optional[str] = typer.Argument(None, help='Optional category filter, e.g. data-export, data-loader, training'))
```

List available backend implementations.

**Raises**

- `typer.Exit` — If *category* does not match any registered backends.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/cli/main.py#L735-L762)

## nexuml.cli.main.build

*function*

```python
def build(config_path: Path = typer.Argument(help='Path to resolved YAML config'))
```

Compile and validate a pipeline from a resolved YAML config.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/cli/main.py#L110-L151)

## nexuml.cli.main.execution_catalog

*function*

```python
def execution_catalog()
```

List registered execution schemas and installation support without probing targets.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/cli/main.py#L765-L771)

## nexuml.cli.main.execution_discover

*function*

```python
def execution_discover(config: Path = typer.Option(..., '--config', '-c'), timeout: float = typer.Option(10, min=0.1, max=60))
```

Read the configured target's sourced capacity, without starting training.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/cli/main.py#L774-L788)

## nexuml.cli.main.execution_inspect

*function*

```python
def execution_inspect(reference: Path = typer.Argument(...))
```

Inspect a recorded native reference, refusing a same-named UID replacement.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/cli/main.py#L801-L811)

## nexuml.cli.main.execution_preflight

*function*

```python
def execution_preflight(config: Path = typer.Option(..., '--config', '-c'))
```

Validate the exact configured execution selection without submitting a job.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/cli/main.py#L791-L798)

## nexuml.cli.main.export_cmd

*function*

```python
def export_cmd(scenario_name: str = typer.Argument(help='Scenario name to train and export'), output: Path = typer.Option(Path('exported_model'), '--output', '-o', help='Export directory'), checkpoint: Optional[Path] = typer.Option(None, '--checkpoint', help='Optional checkpoint to export'))
```

Export a trained pipeline to a portable model package.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/cli/main.py#L406-L444)

## nexuml.cli.main.export_dataset_cmd

*function*

```python
def export_dataset_cmd(scenario_name: Optional[str] = typer.Argument(None, help='Scenario name'), config_path: Optional[Path] = typer.Option(None, '--config', '-c', help='Config YAML path'), output: str = typer.Option('exported_dataset', '--output', '-o', help='Path or s3:// URI'), backend: str = typer.Option('numpy', '--backend', help='Dataset export backend'), split: list[str] | None = typer.Option(None, '--split', help='Split to export. Repeat the option to export multiple splits.'), preprocess: bool = typer.Option(False, '--preprocess/--no-preprocess', help='Run the compiled pipeline until the requested preprocessing keys exist.'), preprocess_until_key: list[str] | None = typer.Option(None, '--preprocess-until-key', help='TensorDict x key marking the preprocessing boundary. Repeatable.'), x_key: list[str] | None = typer.Option(None, '--x-key', help='x TensorDict keys to persist. Repeatable. Default: all.'), y_key: list[str] | None = typer.Option(None, '--y-key', help='label TensorDict keys to persist. Repeatable. Default: all.'), include_labels: bool = typer.Option(True, '--labels/--no-labels', help='Include labels from the batch y TensorDict.'), dtype: str | None = typer.Option(None, '--dtype', help='Optional storage dtype passed to the export backend, e.g. float16.'), samples_per_shard: int = typer.Option(256, '--samples-per-shard', help='Samples per WebDataset tar shard'), s3_endpoint_url: str | None = typer.Option(None, '--s3-endpoint-url', help='Optional S3-compatible endpoint'), s3_region: str | None = typer.Option(None, '--s3-region', help='Optional S3 region'), s3_profile: str | None = typer.Option(None, '--s3-profile', help='Optional AWS credential profile name'))
```

Export a dataset view to disk or S3 from a scenario or config.

**Raises**

- `typer.Exit` — If preprocessing is enabled without --preprocess-until-key.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/cli/main.py#L292-L403)

## nexuml.cli.main.library_add

*function*

```python
def library_add(path: Path = typer.Argument(..., help='Path to local library root', exists=True, file_okay=False, dir_okay=True))
```

Add a local library root path to the NexuML config.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/cli/main.py#L818-L831)

## nexuml.cli.main.library_delete

*function*

```python
def library_delete(path: Path = typer.Argument(..., help='Path to local library root'))
```

Remove a local library root path from the NexuML config.

**Raises**

- `typer.Exit` — If the path is not currently configured as a library root.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/cli/main.py#L834-L853)

## nexuml.cli.main.library_list

*function*

```python
def library_list()
```

List all available library sources (installed and local).

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/cli/main.py#L856-L880)

## nexuml.cli.main.registry_list

*function*

```python
def registry_list(kind: str = typer.Argument('layers', help='Kind to list: layers, data, scenarios, eval'), verbose: bool = typer.Option(False, '--verbose', '-v', help='Show full tracebacks for discovery errors'))
```

List registered items by kind (layers, data, scenarios, eval).

**Raises**

- `typer.Exit` — If *kind* is not one of the recognised values.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/cli/main.py#L682-L732)

## nexuml.cli.main.scenario_resolve

*function*

```python
def scenario_resolve(name: str = typer.Argument(help='Scenario name'), output: Optional[Path] = typer.Option(None, '--output', '-o', help='Output YAML path'))
```

Resolve a named scenario to a YAML config file.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/cli/main.py#L89-L107)

## nexuml.cli.main.serve_cmd

*function*

```python
def serve_cmd(directory: Path = typer.Option(Path('.'), '--directory', exists=True, file_okay=False, resolve_path=True), port: int = typer.Option(8000, '--port', min=1, max=65535), origin: str = typer.Option('http://127.0.0.1:3000', '--origin'), token_file: Optional[Path] = typer.Option(None, '--token-file', exists=True, dir_okay=False))
```

Start a loopback API using this installation; never install dependencies.

**Raises**

- `typer.Exit` — If API dependencies or protected startup settings are missing.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/cli/main.py#L883-L930)

## nexuml.cli.main.smoke

*function*

```python
def smoke(scenario_name: str = typer.Argument(default='synthetic-linear-ae-reconstruction', help='Scenario name'), max_epochs: int = typer.Option(3, '--max-epochs', help='Training epochs'), download: bool = typer.Option(False, '--download', help='Download datasets before training'))
```

Run a full smoke-test: resolve → build → train → export → reload → infer.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/cli/main.py#L447-L509)

## nexuml.cli.main.train_cmd

*function*

```python
def train_cmd(scenario_name: Optional[str] = typer.Argument(None, help='Scenario name'), config_path: Optional[Path] = typer.Option(None, '--config', '-c', help='Config YAML path'), scenario_file: Optional[Path] = typer.Option(None, '--scenario-file', help='Trusted Python file exposing scenario() -> ScenarioSpec'), artifact_dir: Optional[Path] = typer.Option(None, '--artifact-dir', help='Optional directory for scenario-file provenance snapshots'), max_epochs: Optional[int] = typer.Option(None, '--max-epochs', help='Override max epochs'), trainer_checkpoint: Optional[Path] = typer.Option(None, '--trainer-checkpoint', help='Optional Lightning Trainer checkpoint to resume from.'), override: Optional[list[str]] = typer.Option(None, '--override', '-O', help='Override a scenario field: key.path=value. Repeatable.'))
```

Train a pipeline from a scenario name or config file.

**Raises**

- `typer.Exit` — If input validation fails or training encounters an error.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/cli/main.py#L154-L289)

## nexuml.cli.main.tune_cmd

*function*

```python
def tune_cmd(scenario_name: Optional[str] = typer.Argument(None, help='Scenario name'), scenario_file: Optional[Path] = typer.Option(None, '--scenario-file', help='Trusted Python file exposing scenario() -> ScenarioSpec'), artifact_dir: Optional[Path] = typer.Option(None, '--artifact-dir', help='Optional directory for scenario-file provenance snapshots'), n_trials: Optional[int] = typer.Option(None, '--n-trials', help='Number of Optuna trials'), metric_key: Optional[str] = typer.Option(None, '--metric', help='Metric to optimise (default: tuning_spec.metric_key or val/loss)'), direction: Optional[str] = typer.Option(None, '--direction', help='Optuna direction: minimize or maximize (default: from tuning_spec)'), storage: Optional[str] = typer.Option(None, '--storage', help='Optuna storage path'), prune: Optional[bool] = typer.Option(None, '--prune/--no-prune', help='Enable Optuna pruning'), override: Optional[list[str]] = typer.Option(None, '--override', '-O', help='Override a scenario field: key.path=value. Repeatable.'))
```

Tune a scenario's hyperparameters with Optuna using a default search space.

**Raises**

- `typer.Exit` — If scenario input is ambiguous or loading fails.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/cli/main.py#L512-L643)
