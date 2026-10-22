# Design

## Context

See `proposal.md` for motivation and the four spec deltas for acceptance. The current consumers are concentrated in `studio/model/{graph,catalog,types}.ts`; the API currently forwards definition schemas from `api/worker.py`.

The extensions mix four different concerns:

| Existing use | Native replacement |
| --- | --- |
| Fixed-choice schema enums | `Literal` or enum parameter types |
| Category overrides on three model definitions | Existing kind/module grouping |
| Label outputs on four image sources | A definition-owned label declaration consumed by the dataset runtime |
| Context/parameter routes on five evaluators | Shared typed context-input rules and semantic key-parameter annotations |

Most graph wiring already comes from `LayerSpec`, `DataSpec` and `EvalAlgorithmSpec`; keep it there. The existing evaluation key-contract spec also prohibits resurrecting removed tensor-key bags. No active change or configuration migration is needed as a prerequisite.

## Goals / Non-Goals

**Goals:** Remove the authored extras and their consumers; keep one owner for domain facts; retain the current valid configuration representation and UI operations.

**Non-Goals:** A new registry, graph-description language, generated per-component configuration/context hierarchy, runtime-constructor reflection, automatic dataset/model construction, new inspection round-trips, dependency additions, UI redesign or platform-matrix expansion. Do not remove unrelated `X-NexuML-*` security headers or drag/drop MIME types. Preserve contributor command cheatsheets and existing trust/process/file boundaries.

## Decisions

### 1. Use native types where the domain already has one

- Use `Literal` for the current string-valued visualization methods and model variants. This keeps ordinary Python string authoring and serialized values; introduce an enum only if an existing domain API already needs one. Standard schema generation already produces enums understood by `fields.tsx`.
- Introduce lightweight `FeatureKey` and `LabelKey` annotations for intrinsic evaluator parameters. They remain ordinary string values; use semantic `Annotated` metadata, not wrapper values, schema hooks or custom JSON Schema keywords. Inspect public definition field metadata, including optional and list forms, not runtime signatures or `_key` suffixes.
- Requiredness and parameter defaults come from the existing typed fields. Do not maintain a second parameter-name/required/default map. Unannotated strings are not guessed to be routes.

This makes choices actual validation: unsupported values fail earlier than today. Valid serialized values remain unchanged. Remove choice tables/imports only when this change makes them redundant; retain genuine runtime computation data.

### 2. Add only the missing native label/context contract

Keep these small declarations beside the existing component bases in `core/components.py`, with only the exports needed for ordinary library authoring.

**Dataset labels:** expose an optional typed fixed-label declaration on `DataSourceDefinition`. `None` means unknown; an empty tuple means known no labels. Move the four image sources' label facts to their public definitions and reuse them in runtime label names and the corresponding metadata columns. Do not keep a runtime `LABEL_NAMES` list and another tooling list with the same facts. Existing scenario-declared targets and feature keys continue to drive the graph as before.

Do not grow a generic expression language for arbitrary dynamic outputs. Labels already explicit in configuration remain usable; information requiring runtime data stays unresolved. Other source implementations need not be rewritten merely because their outputs are not statically known.

**Evaluator context:** retain `EvalBuildContext` for runtime values. Add one shared immutable context-input declaration for the existing feature/label slots, with typed required/default rules; do not introduce a context/config class per evaluator. The five affected definitions declare only context slots they actually consume. Undeclared contracts are unknown, not a claim that an external component reads nothing.

A shared core resolver applies those rules. Affected `build()` paths use that resolver, including direct calls, so tooling and runtime use the same defaults instead of repeating `or "latent"` / `or "y_true"`. Preserve explicit override precedence and each existing absent/null/empty-value behavior. The ordinary evaluation factory continues to supply surrounding spec wiring; it must not add another set of defaults. Requiredness does not authorize new blanket static validation of runtime-only keys or changing axis provenance/accessor rules.

Rejected alternatives: moving dictionaries to `ui_metadata` changes only their spelling; analyzing arbitrary `build()` code cannot reliably establish contracts; constructing datasets to discover labels can download or mutate files.

### 3. Extend the existing catalog, not the schemas or registry

Keep `schema` as ordinary parameter JSON Schema and add a separate typed `io` description to each existing catalog entry. It carries known source labels and evaluator input bindings. Bindings distinguish context from parameters and contain a generated field reference, domain, collection shape, requiredness and applicable default. Component authors do not write serialized paths such as `algorithm.params.reconstructed_key`; the transport maps local field bindings into the existing scenario representation.

Distinguish unknown from known-empty explicitly. A valid component without a native contract remains listed with its ordinary schema and unresolved I/O. No legacy-schema parser, schema-field stripping shim, second discovery store or frontend component-name table is introduced.

Descriptors are static declarations/bindings, not a cache of resolved draft values. Studio resolves them against its current draft; required definition parameters do not justify instantiating default objects during catalog generation. Configured source targets remain ordinary graph inputs; dynamic information not expressible by the bounded contract stays unknown rather than triggering a new per-keystroke API call.

### 4. Reuse Studio's projection and generic forms

Update `Entry`/catalog types and `graph.ts` to read native I/O descriptions, then remove the three extension properties from `Schema` and all associated fixtures. Keep schema resolution for ordinary refs, unions, collections, defaults and enums.

Connection edits still write the existing fields. Preserve domains, aliases, overwritten-key producer order, evaluator collection types, required-empty ports, undo and save/build revision behavior. Context fallbacks retain their inherited-edge removal guard; a native parameter default is not automatically an undeletable context route. Unknown I/O retains manual/raw editing and honest diagnostics, not fabricated readiness.

Remove the category override branch in `catalog.ts` and keep its existing module fallback. Search/insertion remain unchanged; grouping labels may change where an explicit Vision subgroup previously overrode the package layout. Do not compensate with another taxonomy map.

### 5. Make the wire cutover explicit

Advance the existing API interface identity from 2 to 3. Update Python's interface constant, shared launcher/browser compatibility policy and every affected positive/mismatch fixture, runtime setup helper and package-smoke assertion together. Keep REST endpoint paths and the existing supported Python-package range; package releases and interface identity remain separate.

Both launcher and browser must reject old, missing or incompatible interface identity before catalog-dependent editing. A matching interface must supply the native catalog contract; malformed/missing contract data must produce an actionable error, not a render crash or guessed compatibility. No automatic package installation, hidden environment switching or dual metadata format.

## Risks / Trade-offs

- Wrong label or default after deduplication → compare published native facts with actual runtime objects, including explicit overrides and absent/null/empty context cases.
- Lost routes through nullable/list annotations → cover scalar, optional, collection and context bindings with model and real browser tests, including an external library not known to Studio.
- Native enums reject previously accepted nonsense → make the earlier validation intentional, retain valid string serialization, and preserve invalid editor buffers with field-addressable errors.
- Category labels change → accept module-derived groups; retain search, kind grouping and insertion rather than building another taxonomy.
- A nicer-looking metadata system grows into another framework → only implement the labels, intrinsic key roles and context rules required by the current cases; reuse existing specs, discovery, materialization and graph projection.
- Catalog/import code is mistaken for safe execution → library imports remain trusted code, but contract inspection must not call builds, scan dataset paths, allocate models or download data.
- New and old packages appear compatible → test both mismatch directions and matching installed project/tool modes through the existing compatibility seam.

## Migration Plan

1. Establish focused contract/schema/runtime and UI baselines; add the native primitives and regression tests.
2. Migrate the affected built-ins and catalog/graph consumers in the same change; update the interface guard and all callers/fixtures. Do not publish an intermediate state.
3. Update concise authoring guidance, remove legacy expectations and verify no authored extras or custom schema-key readers remain. Do not change unrelated security-header handling.
4. Build matching Python distributions and one npm tarball after code settles. Run the existing CPU browser/package workflow against those artifacts in both project and tool environments; preserve all existing safety and lifecycle assertions.
5. No YAML/checkpoint migration or release action is performed. Rollback means restoring a matching previous frontend/backend pair, not adding a compatibility adapter. Existing platform/owner release gates remain unchanged.
