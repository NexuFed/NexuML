# nexuml.core.scenario_loader

*module*

Trusted Python scenario/tuning file loading.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/scenario_loader.py#L1-L1)

## nexuml.core.scenario_loader.LoadedScenarioFile

*class* · *dataclass*

```python
class LoadedScenarioFile
```

Loaded trusted scenario file plus optional agent metadata.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/scenario_loader.py#L15-L27)

### nexuml.core.scenario_loader.LoadedScenarioFile.build_factory

*attribute* · *class attribute* · *instance attribute*

```python
build_factory: Callable[..., ScenarioSpec] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/scenario_loader.py#L27-L27)

### nexuml.core.scenario_loader.LoadedScenarioFile.hypothesis

*attribute* · *class attribute* · *instance attribute*

```python
hypothesis: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/scenario_loader.py#L22-L22)

### nexuml.core.scenario_loader.LoadedScenarioFile.parent

*attribute* · *class attribute* · *instance attribute*

```python
parent: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/scenario_loader.py#L23-L23)

### nexuml.core.scenario_loader.LoadedScenarioFile.path

*attribute* · *instance attribute*

```python
path: Path
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/scenario_loader.py#L19-L19)

### nexuml.core.scenario_loader.LoadedScenarioFile.scenario

*attribute* · *instance attribute*

```python
scenario: ScenarioSpec
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/scenario_loader.py#L20-L20)

### nexuml.core.scenario_loader.LoadedScenarioFile.search_space

*attribute* · *class attribute* · *instance attribute*

```python
search_space: dict[str, dict[str, Any]] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/scenario_loader.py#L25-L25)

### nexuml.core.scenario_loader.LoadedScenarioFile.source

*attribute* · *instance attribute*

```python
source: str
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/scenario_loader.py#L21-L21)

### nexuml.core.scenario_loader.LoadedScenarioFile.tags

*attribute* · *class attribute* · *instance attribute*

```python
tags: list[str] = field(default_factory=list)
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/scenario_loader.py#L24-L24)

### nexuml.core.scenario_loader.LoadedScenarioFile.tuning_spec

*attribute* · *class attribute* · *instance attribute*

```python
tuning_spec: TuningSpec | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/scenario_loader.py#L26-L26)

## nexuml.core.scenario_loader.load_scenario_file

*function*

```python
def load_scenario_file(path: str | Path) -> LoadedScenarioFile
```

Load a trusted Python file exposing ``scenario() -> ScenarioSpec``.

**Returns**

- (`LoadedScenarioFile`) — Loaded scenario file with metadata, search space, and tuning spec.

**Raises**

- `FileNotFoundError` — If *path* does not exist.
- `ValueError` — If the file is not a ``.py`` file or lacks a callable ``scenario()``.
- `TypeError` — If ``scenario()`` does not return a ``ScenarioSpec``.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/scenario_loader.py#L166-L205)

## nexuml.core.scenario_loader.project_root_for

*function*

```python
def project_root_for(path: Path) -> Path
```

Return the project root directory for a given path.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/scenario_loader.py#L30-L36)
