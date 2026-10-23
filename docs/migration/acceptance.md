# NEX-247 local acceptance evidence

## Scope and preservation

- All 57 legacy authored pages and three downloads are accounted for in
  `route-map.json`, `baseline.json`, and `../content-review.md`.
- Four primary navigation sections replace overlapping learning/lifecycle paths.
  The canonical starting recipe is `synthetic-linear-ae-reconstruction`, one CPU
  epoch and one device. Optional CIFAR and specialized tasks remain available.
- Historical source baseline: `fbc074595f2d993edf879661b7b70d851aced9cd`.
  Published archive baseline: `f553f5ab04862cd59951b6422473b0a526fe29fb`.
  Existing releases are `v0.1.0` and `v0.2.1`; initial latest stays on `v0.2.1`.
- Implementation validation performed no commits, remote writes, or production
  publication. Pages settings, DNS, package versions, release-note categories, and
  release-publishing behavior remain unchanged. Local build provenance records the
  checkout's HEAD; previews are not claims of a published source revision.

## Reproducible checks

All 34 implementation tasks are checked. Final post-review project, alternate-root,
and stable-version runs passed; both real-archive preservation reports were refreshed.
Independent review confirmed both runtime-registration and filename-collision fixes.

Use the Node selection in `docs/.node-version` and commands from `docs/README.md`.
This host used `npm exec --yes --package=node@22.23.4 -- ...` without changing its
system runtime.

| Check | Evidence |
| --- | --- |
| Frozen docs setup | `npm --prefix docs ci` and `npm --prefix docs run setup:python`; no ML packages installed in the extraction environment |
| Fresh authoring/type checking | Removed generated CLI JSON before `dev` and `typecheck`; both lifecycle hooks regenerated it; authoring URL returned HTTP 200 |
| Main docs gate | `npm --prefix docs run check`: content/schema checks, authentic CLI tests, Python lint/type checks, Astro checks, both static builds, rendered links/fragments/assets |
| Automatic APIs | `npm --prefix docs run test:api`: both packages, new public symbols, private filtering, re-exports, configuration fields, source links, Autodoc, project/root mounts |
| Rejection gates | `npm --prefix docs run test:gates`, schema tests, and rendered-link tests reject malformed MDX/frontmatter, unsafe or missing media, invalid dates/releases/routes, broken links and fragments |
| Browser evidence | `npm --prefix docs run setup:browser` then `npm --prefix docs run test:browser`: six desktop/mobile dark/light checks; API/CLI/prose, search, tabs/copying, actual pipeline Mermaid output, changelog ordering/media, version fallback and keyboard/accessibility |
| Tag namespace | Repeated check/API/rejection/browser/check gates with `DOCS_VERSION=v999.0.0`; fixture version expectations use the actual selected version |
| Hosting portability | Check/browser/composition with `DOCS_SITE=https://docs.nexufed.ai DOCS_MOUNT=''`; no editorial rewrite or live host change |
| Publication invariants | 19 local publication tests, including real bare-Git competing pushes, retry/stale-main checks, semantic latest selection, immutable bytes/modes, and main-only global ownership |
| Real archive | `docs/tests/archive-proof.py` against the baseline archive at project and root mounts; main plus simulated tag retain historical release digests and initial stable target |
| Runtime regression | 65 focused import/CLI/factory/adapter/registry/discovery tests, including fresh-process restoration without pre-importing the adapter |
| Installed artifacts | Rebuilt both wheels from source distributions; installed-wheel library smoke and the exact documented discovery/resolve/build/CPU-train sequence pass |
| Static checks | Repository Ruff lint/format, targeted runtime typing, isolated docs typing, root lock validation and frozen all/dev resolution dry-run |
| Entry links | 15 root/Studio README and package URLs/fragments checked against actual frozen stable HTML or the new dev/global output |
| Sitemaps | 902 assembled sitemap locations resolve under the configured production origin/mount, including global and latest endpoints |
| Specifications | Seven affected main capabilities synchronized, preserving untouched requirements; their strict validation and the change's strict validation pass |

Generated proof reports, build logs, browser screenshots, accessibility reports,
and traces live under ignored `docs/.output/`, `docs/test-results/`, and
`docs/playwright-report/`. Reports include `api-proof.json`, `entry-link-proof.json`,
`archive-proof-project.json`, `archive-proof-root.json`, and `sitemap-proof.json`.

## Review corrections

- Stable-version acceptance tests no longer assume `dev`.
- Actual light-mode signature and copy-feedback contrast failures were corrected,
  not excluded from accessibility checks.
- Browser installation uses an npm project script; repo-root `npm --prefix ... exec`
  did not resolve the installed Playwright executable on this host.
- Core discovery now scans the built-in adapter at runtime. Fresh-process
  restoration tests no longer import that adapter before trying to restore it.
- Changelog IDs preserve raw filenames instead of letting Astro normalize and
  silently merge them. An actual build regression test rejects case-colliding files.

## Limits and maintainer follow-up

### Branch publication follow-up

Main and dev now publish separate unreleased snapshots from matching branches;
only main owns the global changelog/support. Stable releases and latest retain
their previous contracts. This supersedes the original main-to-dev naming.

Follow-up verification: 23 publication/workflow tests (including divergent branch
histories and competing main/dev pushes), complete documentation gates in both
namespaces (34 passed, two runtime-only skips), eight browser tests per namespace,
API proofs at project/root mounts, and the updated real-archive main/tag dry-run
all pass. Historical release hashes and initial stable latest remain unchanged.
The preview was rebuilt and `/NexuML/dev/` returns HTTP 200. Independent review
found no material defects. Hosted deployment was not performed; see the rollout
and protected-environment requirements in `docs/publication.md`.

### Remaining environment and owner checks

- Full institute/GPU/optional-stack CI and hosted Actions artifact transfer/Pages
  deployment have not run here. Root frozen dependency resolution was checked
  without installing the full ML environment. Representative examples were checked;
  every advanced backend and external service was not executed.
- Additional compiler/export tests produced 19 passes, two optional ONNX skips,
  and one WebDataset test blocked by the absent DALI `wds2idx` utility; CIFAR was
  deliberately excluded. No runtime workaround or DALI installation was added.
- npm reports four low-severity transitive advisories in the Mermaid/ELK/KaTeX
  dependency graph. No unsupported forced major upgrade was applied.
- Repository-wide strict OpenSpec validation also exposes 11 unrelated existing
  spec failures (placeholder purposes/long requirements and an empty
  `staged-eval-fit-runner` spec). Those unchanged capabilities were not rewritten.
- Specs are already synchronized. The change remains active; when review permits
  archival, use `openspec archive migrate-starlight-docs --skip-specs` to avoid
  reapplying its ADDED requirements. No archive operation was performed here.
- After merge, follow `docs/publication.md`: maintainer-controlled Pages Source
  cutover, first validated main publication, live smoke and historical-tree checks.
  Close NEX-247 only after that verification. The next normal stable tag creates
  the first immutable Starlight release; no migration-only release is needed.
