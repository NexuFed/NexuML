# Tasks

## 1. Establish migration coverage and preservation baselines

- [ ] 1.1 Add a reviewed content/legacy-route inventory covering every current authored page, important fragments, generated API route shapes, and the three downloadable templates; verify each page has a retained, merged, or explicitly obsolete destination and every existing feature category remains covered.
- [ ] 1.2 Capture the current `gh-pages` structure, compatible version manifest, latest target, and release-tree digests in migration evidence without remote writes; verify the baseline includes all existing release directories rather than only the versions observed during planning.
- [ ] 1.3 Record the live corporate CSS/font/logo source and vendor the approved reusable assets with provenance; verify they match the current preview rather than the older local design and can be served locally.

## 2. Bootstrap the locked site and prove both build targets

- [ ] 2.1 Create the isolated `docs/` npm project and standalone `docs/python` uv environment, pin compatible tools/runtimes, and add ignored generated-output paths; verify clean frozen installs do not resolve/install root ML extras and document setup in `docs/README.md`.
- [ ] 2.2 Implement shared origin/mount/version/source configuration, target-selective collections/routes, and the standard dev/check/build/preview commands; verify a small versioned/global fixture builds at `/NexuML/` and `/`, with no duplicate routes or leaked search entries.
- [ ] 2.3 Configure `starlight-pydocs` for only the two public package roots using the frozen static Griffe runner and revision-correct source links; verify both trees, Google docstrings, signatures, annotations, configuration fields/defaults, and private filtering without runtime imports.
- [ ] 2.4 Add API tests for representative root re-exports, library classes/methods, source links, Autodoc, and temporary new public modules/symbols in both roots; verify new pages/navigation/symbols appear after rebuild without stubs or nav edits and test fixtures do not leak into production output.

## 3. Generate the real CLI reference without loading ML stacks

- [ ] 3.1 Trace and minimally defer heavy imports in the package/core initializers and CLI runtime-warning setup while preserving exported names and behavior; verify no-heavy-import help subprocess tests and existing public-import/runtime CLI tests pass.
- [ ] 3.2 Add actual Click-tree extraction and safely rendered structured CLI data, including nested groups, arguments, aliases, paired options, defaults, and repeated values; verify recursive comparisons with real root/subcommand `--help` and a newly registered fixture command.
- [ ] 3.3 Document and automate the frozen help environment, including authentic package metadata/entry-point installation without dependencies where required; verify clean extraction uses no source-fragment execution, replacement CLI definitions, mocked runtime modules, or accidental ML dependency installation.

## 4. Consolidate and migrate the user journey

- [ ] 4.1 Convert useful editorial content to validated MDX and implement the four primary navigation sections plus secondary links; verify the migration inventory is complete, the home page no longer includes the full README, and obsolete routing articles are not competing reading paths.
- [ ] 4.2 Consolidate scenario/lifecycle/run instructions and library/discovery/registration instructions into canonical task destinations while keeping specialized tasks focused; verify source support, registry/library commands, typed definition/runtime boundaries, direct module authoring, and security limitations against current implementation and inventory.
- [ ] 4.3 Write the short Why NexuML and canonical architecture/data-flow explanation using one coherent example and diagram; verify coverage of scenario composition, persistence, materialization, TensorDict routing, Lightning training/evaluation, export, feature scope, and the distinction between current functionality and future direction.
- [ ] 4.4 Replace mandatory CIFAR onboarding with the one-epoch synthetic CPU walkthrough, retain optional CIFAR access, and connect the Studio alternative; add installed-distribution CPU smoke assertions for the exact commands, explicit config path, completion, and real outputs, and verify no recipe/runtime defaults changed.
- [ ] 4.5 Give retained/consolidated feature guides clear prerequisites, copyable commands/code, observable outputs and next steps; verify data/loading/export, tracking/tuning, evaluation/checkpoints, export/inference, advanced execution, autoresearch, and contributor coverage, with hosted Studio/API details kept outside the initial run path and all safety notes retained.

## 5. Apply branding and preserve documentation interactions

- [ ] 5.1 Apply local Geist/branding and corporate tokens using shared CSS and narrow supported Starlight overrides; verify readable dark/light prose, API signatures, CLI output, callouts, logo treatment, and responsive alignment without a shell fork or decorative marketing UI.
- [ ] 5.2 Replace MkDocs admonitions/snippets with working MDX/Starlight content, preserve code-copy controls and tabbed examples, and enable Mermaid; add browser fixtures and verify actual architecture/generated-pipeline diagrams render alongside working tabs, copy controls, and edit/source links.
- [ ] 5.3 Add built-preview desktop/mobile visual and keyboard/accessibility checks for representative prose/API/CLI pages; verify sidebar, skip/focus behavior, theme controls, code overflow, and contrast, and retain screenshots and browser diagnostics as review artifacts.

## 6. Add the main-published global MDX changelog

- [ ] 6.1 Implement the changelog collection/schema and stable dated route derivation; add rejection tests for invalid/missing metadata, unsafe links, mismatched dates, duplicate routes, missing assets, and release announcements without matching release identifiers/links.
- [ ] 6.2 Build the newest-first index and rich detail pages with deterministic equal-date ordering and metadata-derived grouping; verify adding one MDX entry updates index/detail/search without nav edits and leaves existing entry URLs unchanged.
- [ ] 6.3 Add screenshot/video/MDX support with accessible alternatives and production-prefix-safe assets; verify keyboard controls, responsive rendering, shared brand styling, documentation/version links, and search over the actual published global output.
- [ ] 6.4 Add excluded sample fixtures/template and concise entry-authoring/publishing instructions, plus verified initial editorial content; verify the documented add/preview/check procedure works, no sample is presented as a real release, and `.github/release.yml`/GitHub Release generation remain unchanged.

## 7. Compose portable versioned publication safely

- [ ] 7.1 Implement pure local archive composition with explicit main/tag output ownership, recorded source provenance, tree-digest protection, and failure on unreadable/missing expected archives; verify fixture tests preserve all historical paths/bytes/modes and reject changes outside the allowed mutable paths.
- [ ] 7.2 Preserve the mike-compatible version manifest, materialize the latest alias without symlinks, and implement semantic stable ordering/idempotent retry handling; verify legacy selectors still work, initial main publication keeps the existing latest target, older tags never roll it backward, and conflicting release revisions cannot overwrite snapshots.
- [ ] 7.3 Generate mapped unversioned/current-page compatibility redirects and implement accessible version switching with corresponding-page/home fallback; verify important README/API/tutorial routes, mapped fragments/query strings, fallback links, reserved global paths, and downloadable/non-HTML latest endpoints.
- [ ] 7.4 Add deterministic rendered-link/fragment/asset checks and assembled-preview smoke tests, including prefix-aware search, sitemap/canonical links, and cross-target navigation; verify deliberate broken MDX/local links fail and the same sources pass project-prefix and alternate-origin/root-mount builds without content rewrites.
- [ ] 7.5 Implement/test fast-forward staging with bounded refetch/recompose retries and stale-main protection; verify concurrent main/tag/tag fixtures retain every successfully staged release and newer global content, and document dry-run, retry, rollback, CNAME/custom-domain, and archive-prefix compatibility procedures.

## 8. Wire trusted CI and retire the old toolchain after parity

- [ ] 8.1 Add reusable GitHub-hosted read-only documentation checks for trusted CI and untrusted pull requests, covering frozen installs, extraction, lint/type, build, links, and browser evidence; verify the existing required CI aggregate includes docs and no broad PR trigger exposes institute runners or deployment credentials.
- [ ] 8.2 Update `docs.yml` to validate exact source revisions, stage archives before the deployment queue, and fetch/deploy the newest complete archive through Pages actions; verify job dependencies, artifact provenance, narrow token permissions, non-dropping staging semantics, and no reliance on bot pushes triggering legacy Pages builds.
- [ ] 8.3 Document the maintainer Pages Source cutover to GitHub Actions while retaining the existing URL/provider and archive; verify the procedure includes a preservation dry run, live smoke checks, failed-deployment recovery, and no production CNAME/domain/provider switch.
- [ ] 8.4 After the replacement parity gates pass, remove current MkDocs config/generator/dependencies and update root locks, local commands, `.openchamber/project.json`, contributor guides, README and package metadata links; verify frozen root/docs installs, all active contributor commands and links, unchanged release-note/publishing behavior, and no blanket edits to archived OpenSpec records.

## 9. Integrated acceptance evidence

- [ ] 9.1 Run the documented clean-checkout setup, `npm --prefix docs run check`, build, preview, and browser checks plus relevant import/CLI/installed-distribution regressions; verify every acceptance scenario has passing evidence and clearly identify any environment-blocked check.
- [ ] 9.2 Dry-run the complete main and stable-tag publication flow against a local copy of the real archive; verify release digests, the initial stable target, main-only global changelog ownership, latest endpoints, version fallback, and both hosting-mount configurations with retained reports/screenshots.
- [ ] 9.3 Review the simplified first-run/task/architecture journeys against the coverage inventory and corporate visual source; verify the result removes duplicated reading without losing unique commands/features/safety guidance, and run `openspec validate migrate-starlight-docs --strict` on the final implementation evidence.

## Workflow follow-up

- After review, synchronize the approved deltas into their existing main capability paths and the new changelog capability; update obsolete Purpose text in the affected main specs at that point without rewriting historical archives.
- Perform the maintainer-controlled Pages Source cutover and first validated main publication after merge; verify the live dev/global changelog, historical URLs, stable latest target, and release-tree preservation before closing NEX-247.
- Publish the first immutable Starlight release at the next normal stable tag; do not create a release or change package versions solely to make this proposal appear complete.
- Archive the OpenSpec change only after implementation verification and the project's review requirements are satisfied.
