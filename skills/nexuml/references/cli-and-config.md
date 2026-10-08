# Public CLI and configuration boundaries

These probes describe NexuML's typed 0.2 API. Run from the user's project with
the selected installation; check its help before relying on a command/option.
No framework checkout or source `PYTHONPATH` is needed.

## Select the interpreter

For a project installation, `PYTHON` below is the explicit project interpreter
(for example `.venv/bin/python`; on Windows use `.venv\Scripts\python.exe`).

```sh
"$PYTHON" -c 'import sys, nexuml; from importlib.metadata import version; print(sys.executable); print(version("nexuml")); print(nexuml.__file__)'
"$PYTHON" -m nexuml.cli.main --help
"$PYTHON" -m nexuml.cli.main library list
"$PYTHON" -m nexuml.cli.main registry list layers --verbose
"$PYTHON" -m nexuml.cli.main registry list data --verbose
"$PYTHON" -m nexuml.cli.main registry list eval --verbose
"$PYTHON" -m nexuml.cli.main registry list scenarios --verbose
"$PYTHON" -m nexuml.cli.main backend catalog
```

For a uv tool, invoke the explicit `nexuml` executable instead. An unrelated
`python` on PATH cannot inspect that isolated environment. If authoring/tests
need an importable SDK, prefer an explicitly selected project environment rather
than reaching inside uv's implementation directories.

Registry tables report definitions and field names, not complete schemas. For
structured inspection use the installed public definition:

```python
import json
from my_library.layers.encoder import MyEncoder

print(json.dumps(MyEncoder.model_json_schema(), indent=2))
```

Or inspect installed registry entries without inventing a `registry --json`
flag:

```python
import json
from nexuml.core.registry import get_component_registry

registry = get_component_registry()
print(json.dumps([
    {"kind": e.kind, "name": e.name, "version": e.version,
     "import_target": e.import_target, "schema": e.definition_type.model_json_schema()}
    for e in registry.entries(kind="layer")
], indent=2))
for error in registry.errors:
    print(error.short())
```

Discovery imports trusted libraries. Missing entries plus diagnostics can mean
an import failure, not an absent algorithm.

## Portable configuration without compilation

```python
from nexuml.core.config import ResolvedConfig
from my_library.scenarios.baseline import baseline

scenario = baseline()  # trusted Python factory, not necessarily side-effect-free
config = ResolvedConfig.from_scenario(scenario)
text = config.to_yaml()
restored = ResolvedConfig.from_yaml(text).to_scenario()
assert list(restored.pipeline.stages) == list(scenario.pipeline.stages)
```

Use `ResolvedConfig.load(path)` for existing YAML. Persist through this boundary,
not ordinary Pydantic `model_dump()` or a hand-built selector/parameter bag.
Registered definitions serialize as kind-specific `type`, `version`, `params`;
their library must remain discoverable when restored.

## Executable checks and runs

```sh
"$PYTHON" -m nexuml.cli.main resolve my-scenario --output configs/baseline.yaml
"$PYTHON" -m nexuml.cli.main build configs/baseline.yaml
"$PYTHON" -m nexuml.cli.main train --scenario-file experiments/candidate.py --artifact-dir experiments/run-001 --max-epochs 1
"$PYTHON" -m nexuml.cli.main train --config configs/baseline.yaml
```

`resolve` calls the compiler; both it and `build` execute constructors/dummy
forwards. They can allocate resources or access external dependencies. Neither
proves the actual dataset or scientific protocol is correct. Output destinations
must be fresh or explicitly approved; CLI resolve is not a revision-safe editor.

`--artifact-dir` records provenance for **scenario-file** runs. It is not a
universal output-directory switch for logger/checkpoint placement, and a named/
YAML invocation does not acquire those snapshots merely by passing the flag.
Keep the command/overrides and emitted logs too. For failure diagnostics a
snapshot may be absent; preserve the input and failure log yourself.

Training/checkpoint/export semantics belong to NexuML/Lightning and its installed
execution backend. Verify backend capabilities before remote execution, resume,
stop or export; a driver exit does not prove remote workers stopped.

## Version-matched documentation

Start with the installed schemas/help and public documentation:

- https://nexufed.github.io/NexuML/start/install/
- https://nexufed.github.io/NexuML/how-to/custom-library/
- https://nexufed.github.io/NexuML/reference/tuning-file/
- https://nexufed.github.io/NexuML/explanation/library-discovery/

The site may describe a newer release than the installed package. Confirm
signatures and availability locally rather than silently upgrading or copying
legacy examples. A tutorial checkout is reference material, not a runtime
dependency unless the user actually installs its library.
