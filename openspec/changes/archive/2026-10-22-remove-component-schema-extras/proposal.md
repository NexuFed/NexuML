# Proposal

## Why

Component authors currently describe outputs, routing, categories and choices through hand-written JSON Schema extras. This leaks Studio/transport details into typed Python definitions and duplicates facts already used by runtime construction.

## What Changes

- Remove authored `json_schema_extra` and the `x-nexuml-outputs`, `x-nexuml-routing` and `x-nexuml-category` schema extensions. Keep ordinary generated parameter schemas.
- Use native `Literal`/enum fields for actual fixed choices, including validation rather than UI-only restrictions.
- Make known dataset labels and evaluator input/default rules small native domain contracts, shared by runtime construction and catalog generation. Derive intrinsic key-parameter bindings from semantic Python annotations rather than a second field-name map.
- Keep placement/wiring on the existing scenario specs and preserve valid YAML, component identities, routing overrides, collection values and runtime defaults. Unknown dynamic contracts remain unresolved; discovery does not build datasets or execute models.
- Use Studio's existing kind/module grouping instead of component-specific category overrides. Custom subgroups such as Models → Vision may disappear.
- **BREAKING:** replace schema-carried I/O hints with explicit native-contract catalog fields and advance the API interface. Update backend, launcher and browser together; reject mismatched interfaces instead of maintaining the old metadata format.
- Verify schema/contract/runtime agreement, external-library authoring, graph editing and both installed package modes. Keep contributor guidance concise and command-first.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `typed-component-authoring`: Native domain contracts and genuine typed choices without authored JSON Schema extras or duplicated runtime facts.
- `nexuml-local-api`: Expose parameter schemas separately from discoverable native I/O contracts, without constructing runtimes.
- `studio-workbench`: Consume those contracts for graph authoring and use module-derived grouping while preserving editing behavior.
- `studio-local-distribution`: Coordinate the new API interface across Python, launcher, browser and installed-package checks.

## Impact

Core component contracts, evaluation materialization and their public exports; affected library data sources/evaluators/model choices; API catalog/interface identity; Studio catalog/graph/types/compatibility; fixtures, authoring guidance and regression/package tests. No new dependency, parallel registry, graph DSL, configuration migration, package publication, backend redesign or platform-matrix expansion.
