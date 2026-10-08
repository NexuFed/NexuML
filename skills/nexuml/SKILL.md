---
name: nexuml
description: >-
  Use installed NexuML correctly from a user's external library or project:
  select the Python runtime, inspect components and schemas, compose typed
  scenarios, validate, build, train and inspect evidence. Use whenever a task
  mentions NexuML, ScenarioSpec, TensorDict pipeline routing, NexuML YAML,
  component discovery or NexuML CLI errors, even without a framework checkout.
  For authoring components, paper reproduction or iterative research, combine
  this guidance with the corresponding NexuML skill when available.
compatibility: Python >=3.12 with a compatible NexuML installation; filesystem and shell access.
---

# Operate installed NexuML

Keep the user's library and ordinary NexuML scenarios authoritative. Use the
existing CLI/Python lifecycle, not a new training loop, scheduler or API server.

## Establish the boundary

1. Locate the user's project, library package, package metadata and existing
   scenarios. The working directory need not be NexuML's repository.
2. Select one runtime explicitly. Prefer the project environment for development.
   Use its Python for imports, checks and CLI calls; for an isolated uv tool use
   that tool's executable. Report the actual version/executable before execution.
3. Inspect that installation's `--help`, library sources, component definitions
   and discovery errors. Do not infer available components from this skill's
   examples or a globally installed base library.
4. Establish data, compute, downloads, output paths and budget from the request.
   Ask only for missing consequential boundaries. Once established, execute
   routine work within that scope without per-run approval. An inspect/build-only
   request or explicit pause still excludes training.

Read [CLI and config boundaries](references/cli-and-config.md) for concrete
probes and supported forms. Examples describe the typed 0.2 API, not an assurance
that every installed version supports it.

## Environment and discovery

- A project `.venv` and `uv tool install nexuml` are separate installations.
  Packages installed into one do not become dependencies of the other.
- Declare NexuML/custom-library dependencies in project metadata. Use an existing
  uv project's lock/sync workflow; use `uv pip install --python ...` for an
  explicitly selected existing environment. Do not provision/upgrade it merely
  because inspection finds a mismatch. Install declared project dependencies
  when that is part of the requested authoring/research workflow; ask before
  changes outside its scope, such as unrelated upgrades or global tool changes.
- Prefer an editable installed library exposing `nexuml.libraries`. A configured
  local root is an alternative, not an extra step required after installation.
  `library add` does not install dependencies and modifies user-level discovery
  configuration. Avoid duplicate imports under different package names.
- `nexuml[library]` includes the base library; core-only installs are legitimate.
  Add only required extras, with platform-specific CUDA/DALI instructions verified
  against the selected release. Do not require `api` for ordinary agent workflows.

## Compose before extending

Use existing recipes/components and the user's public imports first. In Python,
construct concrete definitions directly, for example
`LayerSpec(component=MyEncoder(width=64), keys_in=["features"], keys_out=["z"])`.
Use `nn_module` for suitable importable ordinary PyTorch modules. Registry names
are persistence/discovery identities, not a substitute for navigable Python.

Preserve stage and layer order: TensorDict connections route keys but do not
topologically schedule the pipeline. Check preceding producers, overwritten keys,
feature/label/metadata provenance, declared shapes, losses and metrics. Only select
metrics actually produced by the scenario; names do not implement an evaluator.
Use distinct output keys by default; intentional overwrites must preserve the
meaning expected by every later consumer.

## Check and execute deliberately

Distinguish these proofs:

| Check | What it establishes |
| --- | --- |
| Typed construction / `ResolvedConfig` round-trip | Valid fields and portable identities; may import trusted definitions. |
| Compiler / CLI `build` | Constructors and dummy forwards work for declared inputs. |
| Dataset/batch checks | Real data, labels, shapes and partition membership satisfy the contract. |
| Small execution | The configured lifecycle completes and produces actual metrics/artifacts. |
| Protocol evaluation | Results meet the declared scientific and resource criteria. |

CLI `resolve` also compiles: do not present it as validation without execution.
Imports, scenario factories, builds and checkpoint loading execute trusted Python;
they are not sandboxed just because run in another process.

For training, use exactly one source: registered scenario, `--config`, or
`--scenario-file`. Preserve the launched source, resolved configuration, dependency
identity and actual overrides. Use a fresh artifact directory for each run; the
scenario-file snapshot is helpful but does not snapshot every imported library or
dataset. Record those identities separately. Keep resource-consuming work within
the approved budget and use the host's supported completion/observation mechanism.

## Diagnose at the responsible boundary

Inspect discovery errors, data contracts, keys/shapes, dependencies and runtime
selection before editing. Correct user-owned code/configuration when responsible.
If evidence points to a framework bug, retain a minimal reproducer and report the
installed version; do not patch site-packages or hide it with a monkeypatch.

End with the selected runtime, changed/used files, exact commands, passed/failed/
unavailable checks and artifact paths. Successful execution is not proof of
convergence, paper parity or final-test generalization.
