# Verification — remove-component-schema-extras

## Behavior

- Native `FeatureKey` / `LabelKey` annotations retain ordinary string, null and list serialization, including constrained/nullable lists. No schema hooks or parameter-name heuristics.
- `fixed_label_keys` is separate from existing configurable `label_keys` fields. CIFAR labels remain `class_labels`, MNIST `digit`, FashionMNIST `category`; runtime labels and metadata columns share the definitions' declarations.
- Five evaluators use shared frozen context rules in direct and factory builds. Existing missing/null/empty-value semantics, explicit overrides, grouping fallback and axis provenance are covered.
- Catalog `io` is separate from parameter `schema`. Unknown labels/context consumption remain distinct from known-empty contracts. Discovery sentinels reject builds, definition instantiation and dataset scans.
- Studio consumes native bindings, preserves scalar/list routing and inherited-context removal protection, and blocks malformed native catalogs before editing. `__proto__` bindings are rejected.
- Intentional changes: module-derived categories replace custom subgroups; invalid closed choices fail earlier; API interface is 3. Endpoint paths, supported package range and valid YAML identities/wiring remain unchanged.
- No dependencies/lockfiles, contributor command cheatsheet, unrelated security headers, filesystem boundaries or process ownership rules changed. Existing editor/runtime work is excluded.

## Completed checks

- Baseline: 75 focused Python tests; 42 Studio unit tests.
- `NEXUML_API_WHEEL=<final wheel> .venv/bin/python -m pytest tests/api -q`: **37 passed**, including built-wheel startup, CPU build/train/export, transport security and observation.
- Focused source command covering API, evaluation, data and typed component/choice tests: **288 passed, 2 skipped**. Final API and constrained-annotation checks were rerun after review fixes; these counts overlap, not totals.
- Isolated CPU interpreter: `python -m pytest tests/_registry tests/evaluation tests/data -q`: **295 passed, 4 skipped**. Declared image/audio dependencies are installed there; no real data downloads or GPU/cluster jobs.
- `npm test`: **124 passed**; `npm run typecheck`, `npm run lint`, `npm run prepack`: passed.
- `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .`: passed (253 Python files).
- `.venv/bin/ty check --python .venv src/nexuml/core/components.py src/nexuml/api/contracts.py src/nexuml/api/worker.py --output-format concise`: passed.
- `.venv/bin/mkdocs build --strict --site-dir /tmp/opencode/nexuml-native-6rtdal/final/docs`: passed. Both final native authoring examples also execute and agree with runtime contracts.
- Static source audit: no production authored schema extras or removed custom-key readers. Isolated installed catalog audit: **57 components**, no removed extensions, original image labels intact.
- Independent review found constrained-annotation discovery and prototype-binding defects; both corrected with regressions and verified in a scoped re-review.
- `openspec validate remove-component-schema-extras --strict`: passed after package acceptance. Tracked and new-file whitespace checks passed; final artifact hashes and both browser-run success records rechecked.

## Installed artifacts

Retained under `/tmp/opencode/nexuml-native-6rtdal/`:

- Final backend wheel/sdist: `final/nexuml/`.
- Unchanged-after-build library wheel/sdist: `nexuml-library/`.
- Final frontend tarball: `final/nexufed-nexuml-studio-0.1.0.tgz`.
- Exact final artifact hashes: `final/artifact-sha256.json`.
- Byte comparison: all 187 packaged Python modules and 155 frontend package files match the final source/build.
- Both final isolated project/tool runtime handshakes passed using those wheels.
- Final tool browser/package acceptance: **20/20 browser tests passed**, plus runtime selection, offline assets, CPU workflows, external native routing, save/reopen, reconnection/export, dependency immutability, hosted HTTP/WebSocket authentication and owned/attached shutdown. Retained installed run: `/tmp/opencode/nexuml packaged smoke AUnall`; screenshots/results: `final/tool-browser-results/`.
- Final project browser/package acceptance: **20/20 browser tests passed**, plus the same full runtime/security/immutability/lifecycle checks against the identical final tarball and wheels. Retained installed run: `/tmp/opencode/nexuml packaged smoke 4D4nrl`; screenshots/results: `final/project-browser-results/`.

The initial package attempt hit a short shell deadline and exposed an ambiguous new browser-test locator; it was scoped to the inspector. Final artifacts were rebuilt after production review fixes and are tested with a longer deadline. The first complete tool attempt passed 19/20 tests; the added external-library test needed to select Components after using Structure. That test-only navigation step was corrected without changing the final artifacts. Neither attempt is counted as successful package acceptance.

## Limits

The original checkout remains incompletely provisioned. Its broad run had 12 missing-`torchvision`/`torchaudio` scenario failures; isolated conformance passed once those declared dependencies were available. Nothing was installed into the checkout's `.venv` for this change.

Library typing still reports exactly **16 diagnostics reproduced on starting HEAD (`6f64597`) files**: 15 missing optional image/plotting imports and one existing TensorDict typing error. Comparison evidence is in `type-baseline/`; no new diagnostics or suppressions. Full-repository type health is not claimed.

Validation did not include exact-head remote CI, macOS/Windows acceptance, real cluster/GPU execution, publication or merge. Archival is not performed. Existing release/platform gates and prior environment/security follow-ups remain unchanged.
