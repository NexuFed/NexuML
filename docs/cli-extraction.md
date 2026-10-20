# Lightweight CLI extraction

The CLI reference is generated from the actual `nexuml.cli.main.app` using Typer's
Click command tree. The pinned Typer version vendors Click; extraction uses its
own context, group, and parameter interfaces, not a separately installed Click.
It converts the app afresh so new registered commands, nested groups, aliases,
and parameters appear without a documentation list or handwritten flags table.

## Frozen setup and checks

Run from the repository root:

```sh
uv sync --project docs/python --frozen
uv pip install --python docs/python/.venv/bin/python --no-deps -e .
docs/python/.venv/bin/python docs/scripts/extract_cli.py --output docs/.generated/cli.json
docs/python/.venv/bin/python -m pytest --noconftest tests/test_cli_lightweight.py docs/tests/test_cli.py
```

The standalone `docs/python` project owns the locked help dependencies. The
editable core installation supplies real package-version metadata and the
`nexuml` console entry point, **without installing its runtime dependencies**.
Invoke this interpreter directly afterward: another `uv sync`/`uv run` can remove
the metadata-only installation, in which case repeat the `--no-deps` install.
Never install root extras or ML stacks to make documentation extraction pass.
This environment supports help, not training or other runtime commands.

`--noconftest` deliberately bypasses the repository runtime suite's shared
fixtures, which import PyTorch. The guarded subprocess tests reject attempted
imports of torch, Lightning/Lightning utilities, TensorDict, NumPy, pandas, and
optional backends during root and recursively discovered subcommand help.
They also check real installed metadata, factory identity, and unknown attributes.
The docs tests compare the complete command tree against real `--help`, register
temporary nested/aliased commands, and prove static Griffe export visibility.
Runtime-only export-identity and warning-suppression tests skip when their actual
dependencies are absent; run those alongside the existing CLI/import tests in the
normal runtime environment. No fake ML modules are used.

## Generated data contract

The extractor requires an explicit `--output` path, creates its parent directory,
and writes deterministic UTF-8 JSON. Keep this generated file outside authored
content and version control. It does not execute command callbacks, evaluate
callable defaults, execute source fragments, or construct a replacement CLI.
Hidden commands/groups and parameters are excluded. Public parameters include
Click's automatic help option and the real shell-completion options.

The top level is an array, in recursive command order. Each entry contains:

```json
{
  "path": "nexuml train",
  "help": "Train a pipeline from a scenario or config",
  "parameters": [
    {
      "name": "override",
      "kind": "option",
      "opts": ["--override", "-O"],
      "secondary_opts": [],
      "type": "text",
      "required": false,
      "multiple": true,
      "nargs": 1,
      "default": null,
      "help": "Override a scenario field: key.path=value. Repeatable."
    }
  ]
}
```

`kind` is `argument` or `option`. `type` comes from the real Click parameter
type's metavar, falling back to its type name; choices retain their accepted
values. Option aliases remain in `opts`, negative paired forms in
`secondary_opts`. `multiple` and `nargs` describe repetition and tuple arity.
Defaults retain JSON scalars, recursively convert tuples/lists/maps, use enum
values and path strings, and represent callable defaults as
`<dynamic: module.qualified_name>` without invoking them. Unsupported defaults
or non-finite JSON numbers fail extraction instead of emitting unstable data.

The site renderer must treat every string as text, never as MDX/HTML source.
Help/defaults may legitimately contain `<`, `&`, braces, or markup-looking text.

## Import boundary

Package/core initializers retain their `__all__` exports and static type-checking
imports. Only runtime-heavy adapters/layers are loaded on attribute access;
returned objects are the original definitions, not wrappers. Package version
lookup still uses installed distribution metadata. Each actual command applies
the existing warning/Lightning marketing suppression before its runtime imports.
Help parsing never reaches that setup or the command implementation.
