# Tasks

## 1. Native domain contracts

- [x] 1.1 Map every current schema-extra producer/consumer and interface-version assertion to its retained behavior; establish focused Python/Studio baselines. Verify the inventory covers four image-source declarations, five evaluator route declarations, three model category overrides, both fixed-choice fields, external libraries and the existing schema/runtime validation assertions.
- [x] 1.2 Add the lightweight semantic key annotations, optional known-label declaration and shared immutable evaluator context-input rules/resolver beside existing component contracts; expose only necessary authoring imports and document them in docstrings. Add/run focused tests for scalar/optional/list annotation discovery, unknown versus known-empty, required/defaulted context slots, direct build-path use and unchanged plain-value serialization; introduce no JSON Schema hooks or per-component model hierarchy.

## 2. Built-in migration

- [x] 2.1 Migrate the four image sources to one definition-owned label declaration consumed by runtime label names and metadata columns; remove duplicated label facts and schema extras. Add/run dataset fixtures with supplied tensors/targets proving advertised labels equal actual runtime labels without downloads, and add a short native-label example to the existing data-source authoring guide.
- [x] 2.2 Migrate the five evaluators to intrinsic semantic key types and shared context-input rules, retaining surrounding spec wiring and existing evaluation materialization. Add/run tests for actual runtime keys, explicit overrides, omitted/null/empty values, required inputs, optional/collection parameters, unknown external contracts and unchanged axis provenance; replace old schema-metadata assertions without dropping their coverage and update the existing evaluator authoring example.
- [x] 2.3 Replace both authored choice enums with native closed string types and remove the three model category overrides and resulting unused imports. Add/run tests proving supported choices round-trip, unsupported choices fail before construction, generated schemas contain ordinary enums, and inherited frozen/extra-field validation remains intact; update only affected parameter documentation.

## 3. Catalog and Studio cutover

- [x] 3.1 Extend existing catalog entries with typed native I/O descriptions separate from ordinary parameter schemas. Add/run API tests for generated context/parameter bindings, domains, requiredness, defaults, collection shape, known/unknown labels, required-parameter definitions and an external library; use build/download/scan sentinels to prove discovery does not construct runtimes and retain all unrelated catalog fields and diagnostics.
- [x] 3.2 Update Studio catalog/types/graph consumers together and remove custom schema-key readers and category overrides. Add/run model and browser regressions for label ports, scalar/nullable/list parameter routes, inherited context defaults, explicit overrides, connection removal/reconnect, domain/order/overwritten-producer rules, module grouping, invalid-buffer retention and unknown-contract manual editing; verify YAML/save/undo/build signatures remain unchanged and ordinary schema-backed dropdowns still work.
- [x] 3.3 Advance Python, shared launcher/browser compatibility and all affected helpers/fixtures/package assertions to API interface 3 while keeping endpoint paths and the supported package range. Add/run old/missing/mismatched identity and malformed native-catalog tests proving actionable failure before catalog-dependent editing; retain bearer/first-frame/proxy authentication and owned/attached lifecycle checks, with no legacy metadata adapter or automatic environment mutation. Update only affected compatibility guidance.

## 4. Integrated acceptance and handoff

- [x] 4.1 Run strict docs, affected Python Ruff/format/type checks and pytest, Studio typecheck/lint/unit/build, and valid configuration round-trip regressions. Verify authoring examples use native contracts, contributor command cheatsheets remain intact, no dependency/lockfile or unrelated security-header changes occurred, and all unavailable optional checks are reported rather than suppressed.
- [x] 4.2 Build matching Python distributions and one npm tarball after production edits settle; run the existing isolated project and tool package/browser workflows against those exact artifacts. Verify native-contract forms/ports, an external library, CPU example/build/train/reconnect, save/reopen, artifacts, offline assets, installed dependency immutability and hosted HTTP/WebSocket/ownership behavior; keep existing assertions and pending platform gates, and start no real cluster or GPU jobs.
- [x] 4.3 Audit the completed source and generated built-in schemas for remaining authored extras or custom schema-key consumers, distinguishing deliberate negative-test assertions from actual uses. Record concise current evidence, intentional category/early-validation/interface changes and limits; verify no parallel registry, field-name heuristic, runtime introspection or new inspection endpoint was introduced, `git diff --check` passes, and `openspec validate remove-component-schema-extras --strict` succeeds.

## Workflow follow-up

- Apply only after the user approves this proposal; do not infer commit, push, publication or archival authorization.
- Archive after implementation/review acceptance, preserving unrelated release and cross-platform gates.
