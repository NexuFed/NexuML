# Design

## Branch publication follow-up

The approved follow-up separates `main` and `dev` snapshots by their actual source
branches, superseding the original convention where main published `/dev/`.
Both branch sites are unreleased and appear alongside immutable releases in the
version selector. Main alone owns the global changelog/support; dev and tag events
must preserve it. Each branch's stale-build checks use only its own history, not
the other branch's potentially divergent commits. Stable latest and the root
entry remain tied to the newest release. Durable staging and preservation checks
remain unchanged. Roll out the branch-aware workflow to both branches and disable
the competing legacy publisher before the maintainer-controlled Pages cutover;
initialize global content from main before enabling dev deployments.

## Context

See `proposal.md` for motivation and `specs/` for the behavior contracts. Repository exploration established these constraints:

- The current site has 57 authored Markdown pages, repeated scenario/lifecycle/concept explanations, six admonitions, a README include, and three downloadable job templates. Tab and Mermaid extensions are configured, but the current prose does not itself provide rendering fixtures for them.
- `scripts/gen_ref_pages.py` scans only `src/nexuml` and `library/src/nexuml_library`. Preserve that boundary, not an indiscriminate repository scan. Export declarations are uneven; public API coverage cannot mean only the handful of root exports.
- `nexuml` and `nexuml.core` initializers eagerly reach runtime modules. CLI module initialization also patches Lightning utilities. Most actual command handlers already defer their implementation imports.
- Docs currently build inside full-stack CI and publish separately through mike without an exact-revision documentation gate. Institute runners must not execute untrusted fork code.
- Live inspection found `dev`, `v0.1.0`, `v0.2.1`, a `latest` symlink to `v0.2.1`, and a mike-format `versions.json`. GitHub Pages currently serves the `gh-pages` root in legacy branch mode. Unversioned README links such as `/NexuML/start/install/` currently return 404 while their `/latest/` equivalents work.
- The live corporate preview is accessible. Its served CSS uses Geist, `#101010` background, `#181818` surfaces, `#f3f3f3` text, `#aaa` muted text, `#303030` lines, `#0da2e7` blue, and `#20cb98` teal. Its current wordmark is served as a local brand image. The older local IBM Plex design is not the source of truth.
- User decisions: publish an always-current global changelog from main; leave historical releases untouched; reduce documentation complexity without losing commands, features, or architecture/vision.

## Goals / Non-Goals

**Goals:**

- Make one useful default visible before optional complexity, with a canonical location for each task.
- Separate source analysis/help extraction from ML execution, and separate mutable editorial publication from immutable release sites.
- Preserve the existing archive as data; make deployment composition testable without remote writes.
- Make origin, project mount, version namespace, and source revision explicit inputs rather than content conventions.

**Non-Goals:**

- No runtime feature changes, public Python API redesign, invented roadmap, or new training/export behavior.
- No CMS, automatic changelog prose, duplicated tutorial corpus, UI framework added for ordinary docs content, or replacement search service.
- No migration of old releases to Starlight, rewriting archived OpenSpec evidence, production domain switch, or Cloudflare deployment in this change.

## Decisions

### 1. One docs project, two static build targets

Use npm, matching the existing frontend tooling, with an independent `docs/package.json` and committed `package-lock.json`. Pin compatible Astro, Starlight, MDX, pydocs, and required diagram/check tooling; pin the Node runtime in machine-readable project configuration. Do not depend on `studio/node_modules` or introduce a repository-wide JavaScript workspace for this task.

Use this structure:

```text
docs/
  README.md                       # setup, commands, and links to authoring/deployment help
  package.json / package-lock.json
  astro.config.mjs                 # shared theme and explicit build-target selection
  python/                         # standalone, frozen uv extraction/check environment
  src/
    content.config.ts             # target-specific collection loading and schemas
    content/docs/                 # authored documentation MDX
    content/changelog/            # one authored MDX entry per update
    routes/changelog/             # statically generated global index/detail route templates
    components/                   # shared brand, CLI rendering, version selector
    styles/brand.css
    assets/brand/                 # local brand/font/media sources as appropriate
  public/                         # downloadable files and non-transformed public assets
  scripts/                        # extraction, checks, build composition, publication
  tests/                          # fixtures and focused unit/browser checks
  templates/changelog.mdx         # excluded from published collections
```

`versioned` builds load the documentation collection, pydocs, and generated CLI data, with an effective base of `<mount>/<version>/`. `changelog` builds load the dedicated changelog collection and shared Starlight shell at `<mount>/`, with no API extraction or version-frozen timeline. Use supported target-specific content loading and Astro route injection for the global custom pages; do not fork Starlight routing. Exclude irrelevant collection entries/routes from each target and test the resulting search indexes as well as HTML.

Both targets share components, tokens, and dependencies. New versioned docs link to the global changelog using the mount configuration, not a second copy inside every release. A tag builds only its versioned site; main builds `dev` and the global changelog.

Standard contributor entry points are `npm --prefix docs ci`, `npm --prefix docs run dev`, `npm --prefix docs run check`, `npm --prefix docs run build`, `npm --prefix docs run preview`, and `npm --prefix docs run test:browser`. Document the frozen Python bootstrap used by these scripts. The default build assembles a local preview containing both targets and working version/navigation entry points without fetching production archives. Preview fixtures must never be publishable as real release snapshots.

**Alternative rejected:** a changelog inside each version build would keep normal readers on stale release editorial content. A second site/framework would duplicate branding and contributor tooling. A single build of all historical source versions would make immutable archives depend on today's toolchain.

### 2. Consolidate content around tasks, not a page-count target

Primary navigation is fixed at four sections. Keep generated API trees and advanced execution details collapsed initially, with direct links and search still available. Preserve tutorial access under Start here, and place Changelog/Contributing/version selection in secondary navigation.

| Reading need | Canonical destination | Consolidation |
| --- | --- | --- |
| What is it and how do I start? | Short overview, installation, first run, Studio, external tutorials | Replace the README include and multiple competing entry-path introductions |
| How do I create/run an experiment? | One create/run guide | Merge repeated scenario composition, source-choice, common train/override, and lifecycle instructions; keep exact source support and trust warnings |
| What happens during training? | Focused evaluation/checkpoint sections linked from the run guide | Retain distinct resume versus weight reuse and actual metric/artifact semantics |
| How do I use a feature? | Focused data/export, tracking, tuning, model export/inference task pages | Keep different tasks separate; do not create one enormous manual |
| How do I extend it? | One library/extension entry guide with role-specific examples | Consolidate package layout, entry-point registration, local roots, and discovery; retain specialized layer/data/evaluation guidance |
| Why this design? | Why NexuML and one architecture/data-flow explanation | Merge mental model, scenario anatomy, TensorDict, lifecycle, and useful Lightning comparisons |
| What is the exact contract? | CLI, configuration/integrations, Python API | Preserve specialized reference contracts without copying their tables into task pages |
| How do I operate advanced setups? | Advanced execution and contributor procedures | Move hosted Studio/proxy details away from its initial install/run flow, retaining all relevant security rules |

Before moving content, commit a machine-readable old-route-to-destination mapping covering all authored pages, existing public API route shapes, assets, and important fragments. Each authored page is retained, merged into a named destination/section, or explicitly classified as an obsolete redirect-only placeholder. The mapping is also the reviewable feature-coverage inventory. Keep established slugs where they still describe a useful canonical page; navigation labels do not require a wholesale URL rename.

Each task has a goal/prerequisites, copyable commands or code, expected output and artifact-discovery instructions, then a clear next action. Put optional integrations after the working default. Preserve trusted Python/checkpoint warnings, authentication and filesystem boundaries, backend limitations, export versus dataset-export distinctions, and all existing documented feature categories. Keep the externally maintained tutorial projects as links, not copied implementations.

The architecture example connects typed definitions and scenario composition to persisted identities/configuration, materialized TensorDict stages, the Lightning session, evaluation, and portable artifacts. Explain the value of reusable components over duplicated project glue. Mark future direction explicitly; do not add speculative capabilities.

**Alternative rejected:** merely collapsing the existing sidebar hides duplication without removing it. Aggressively deleting advanced material or merging unrelated tasks into giant pages would reduce discoverability rather than complexity.

### 3. A CPU-first example without changing runtime defaults

Use `synthetic-linear-ae-reconstruction` as the common CLI/Studio first-run example. Resolve to an explicitly named configuration path rather than assuming how recipe keys become filenames. Select one epoch, CPU, and a single device using existing supported overrides. Do not change the recipe's application defaults to make documentation pass.

Validate discovery, resolve, build, training/evaluation completion, and claimed artifacts in the normal-user core-plus-library environment. This runtime smoke belongs with existing installed-distribution/CPU checks, not in the lightweight docs extraction environment. Do not promise a checkpoint or diagram when the example does not enable one. If demonstrating trained-model export, explicitly obtain a real training artifact or configure the existing train-package export; never label an initialized pipeline as trained. Keep CIFAR reachable as an optional follow-up with its additional prerequisites.

**Alternative rejected:** CIFAR as the mandatory first task adds downloads and setup before demonstrating the core model. Changing training APIs or defaults is unnecessary.

### 4. Static API analysis with an explicit public-surface policy

Configure one `starlight-pydocs` instance with two package entries, rooted only at `../src` and `../library/src`. Use Google docstrings and bases `reference/api/nexuml` and `reference/api/nexuml_library` to preserve the existing public route family and dotted symbol anchors where supported.

Use a standalone locked uv project in `docs/python`, outside the root ML workspace's dependency resolution. Invoke the frozen Griffe executable explicitly through the integration's runner configuration; do not allow its default unpinned `uvx --from griffe` fallback. Disable runtime inspection and prove extraction while heavy imports are unavailable/blocked. Add a compatible static Pydantic extension only if needed to preserve configuration field/default documentation, and lock it with the extraction tools.

Visibility policy:

- Exclude underscore-prefixed modules/symbols and known nonpublic implementation paths.
- Honor `__all__` for a module's exported members, including resolvable public aliases.
- In modules without `__all__`, include public definitions, not every imported third-party name.
- Preserve public submodule navigation even where the root export list is intentionally small.
- Verify representative root re-exports, `ScenarioSpec` fields/defaults, library definitions such as `LinearEncoder`, methods, and private runtime exclusions. Add a temporary public fixture module/symbol in each package during tests and prove automatic discovery without navigation edits.

Use revision-correct GitHub source links: the checked-out commit for dev and the tagged commit for releases, with repository-relative file paths. Use supported Autodoc/SymbolSearch components where useful; retain Starlight/Pagefind full-text search over generated HTML. Generated API details come from Python, never from MDX prose. Keep dumps/caches and generated CLI data out of authored content and version control unless a test fixture explicitly needs them.

**Alternative rejected:** handwritten API stubs or copied dumps for the current API would drift. Prose-snapshot versioning plugins do not snapshot injected API routes; whole-site release output does.

### 5. Authentic lightweight CLI extraction

Traverse the actual Click app produced by `nexuml.cli.main`, including nested groups and both positive/negative option forms. Emit structured command metadata and render it from a small MDX/ Astro component, so help strings/defaults are escaped text rather than accidentally executable MDX. Preserve arguments, required/repeated values, aliases, defaults, and descriptions. Do not scrape Rich line wrapping as the primary data model.

Make only the import changes needed to make actual CLI help independent of ML execution:

- Preserve public exports and callable identity while deferring runtime-heavy imports in both `nexuml/__init__.py` and `nexuml/core/__init__.py` where the actual import trace requires it.
- Preserve static export visibility for Griffe and normal installed-package metadata behavior.
- Defer the existing Lightning warning/marketing suppression until runtime work begins, preserving its behavior on execution paths without importing Lightning utilities during root or subcommand help.
- Keep actual command registration and handlers authoritative. Do not execute source fragments, create docs-only replacement apps, or inject fake torch/Lightning modules.

Install the real core package editable with `--no-deps` into the already frozen extraction environment when required for authentic package metadata/console entry points. Invoke its interpreter without an automatic resync that would remove that metadata-only installation. This is a docs/help environment, not a supported environment for running training.

Tests recursively compare extracted command/option contracts with real `nexuml --help` and subcommand help, and fail if new commands are missed. Use subprocess import guards in the lightweight environment and existing runtime tests to prove exported names, command behavior, and warning suppression did not regress.

**Alternative rejected:** importing the full runtime violates the docs-build boundary; AST-reconstructing an independent CLI risks incorrect Typer/Click defaults and behavior.

### 6. Branded defaults, not a custom docs application

Vendor the current corporate wordmark and record its source/provenance; serve branding and fonts locally rather than depend on the preview hostname or a hashed website asset URL. Use Geist, the observed corporate surface/text/line tokens, modest heading tracking, flat code surfaces, a tight reading grid, and sparse blue/teal emphasis. Brand the API/CLI/changelog consistently through shared tokens.

Start with Starlight `customCss`, its existing responsive sidebar/search/TOC, and built-in code-copy/tabs/asides components. Use only narrow supported component overrides needed for the branded title/header and version selector, reusing default components rather than copying entire shells. Do not add Tailwind or React solely to recreate standard docs UI.

Keep dark as the branded default and preserve an intentional light theme with separately contrast-checked text, code, logo treatment, callouts, and diagrams. Maintain keyboard navigation, visible focus, skip links, reduced-motion behavior, and mobile layouts. Enable a compatible Mermaid integration and retain representative architecture/generated-pipeline fixtures for browser testing.

**Alternative rejected:** a generic recolor does not meet the corporate reference; a forked layout or marketing hero would sacrifice documentation usability and upstream accessibility fixes.

### 7. Global editorial changelog data and routes

Use a dedicated MDX content collection with this contract:

- `date`: a valid ISO calendar date, interpreted consistently rather than in the builder's local timezone.
- `title` and `summary`: nonempty editorial strings.
- `category`: a bounded documented enum, including a release announcement category.
- `version`: optional exact release identifier; required for a release announcement.
- `links`: typed labels and safe HTTPS or site-local destinations; release announcements require a matching tag/GitHub Release link.
- Optional local/HTTPS media with required accessibility metadata appropriate to its type. Video rendering uses controls and captions or a text alternative, not autoplay-only content.

Use stable `/changelog/<date>/<entry-slug>/` detail routes, deriving the slug from the committed filename. Reject date/filename inconsistencies, duplicate routes, invalid/unsafe links, and missing local assets. Sort descending by date, then ascending stable slug for equal dates. Derive month/year and displayed version grouping from the entries, not a hand-maintained list. Moving an already published date/slug requires a compatibility redirect.

Build the index and detail pages with the shared Starlight shell. Keep global navigation/search scoped to real global output and link back to stable documentation/version selection. Screenshots, short videos, code, and ordinary supported MDX components work as article content. Store the template and demonstrative test entries outside the published collection; seed real editorial content only from verified existing release/product facts or an approved migration announcement.

The changelog publishes only from main. Tags never copy their collection into the global timeline. Keep `.github/release.yml` and existing GitHub Release note generation/publishing behavior intact.

**Alternative rejected:** a Git commit feed is not an editorial changelog; global content copied from a tag can erase newer announcements.

### 8. Version archives are merged, never replaced wholesale

Treat the existing `gh-pages` branch as the durable static archive. Read the real branch before staging; a fetch failure is not permission to deploy an empty archive. Keep the mike-compatible array shape and alias fields in `versions.json` so old Material version selectors continue to work. Add a small separate publication-state file for source provenance and route availability; do not add metadata files to retained release directories.

Pure local composition receives an archive tree, validated candidate outputs, source identity, and hosting configuration. It writes a separate candidate tree and verifies the allowed changes before any git push or Pages deployment:

| Event | Allowed mutable output | Forbidden output changes |
| --- | --- | --- |
| Main | `dev/`, global `changelog/` and its root-level assets/search/sitemap support, shared metadata and compatibility redirects | Every release directory and the stable latest target |
| Stable tag | One previously absent `vX.Y.Z/`, latest materialization when it is the newest stable release, shared metadata/default redirects | Existing release directories, `dev/`, global changelog and its assets/search |
| Retry | Idempotent confirmation using recorded source identity | Replacing an archived version with a different build/revision |

Record exact source commit and build-target provenance for new outputs. Existing archives have unknown provenance unless independently known; never treat them as replaceable because they lack new metadata. Check retained file contents, paths, and modes with tree digests before and after composition. A source mismatch for an existing release is an error, not an overwrite option.

Build each release at its canonical `<mount>/vX.Y.Z/` base. Materialize `latest/` as a copy of the selected immutable site's static tree, not a new source build or a symlink. This preserves non-HTML endpoints and downloads as well as pages, works in GitHub Pages deployment artifacts and future static providers, and keeps canonical/source links tied to the real release. Its modest storage duplication is preferable to provider-dependent symlinks or a custom redirect rule for every asset. New canonical links may navigate from the latest entry alias into the pinned version; document that behavior.

Compute newest stable using semantic version ordering, not event arrival time. Backfilling an older tag never rolls `latest` backward. Keep root documentation defaulting to stable `latest` while it exists; use dev only as an explicitly labeled local/bootstrap fallback when no stable archive exists. During this migration, `v0.1.0` and `v0.2.1` remain MkDocs and stable readers keep the old release until the next tag.

Generate page-preserving static compatibility redirects for important unversioned URLs and merged current-version pages. Protect reserved global paths from documentation alias generation. Keep query/fragment information through client navigation where a mapping exists and provide canonical metadata, meta refresh, and an accessible direct fallback link; do not claim these files are server-side HTTP 301 responses. Unknown obsolete anchors get a documented mapping/limitation rather than fabricated headings.

Version selection reads shared compatible version metadata plus route availability. Retain the corresponding page when present; otherwise announce a fallback to the chosen version's home. Archived HTML is not rewritten to add the new selector.

**Alternative rejected:** replacing `gh-pages` with one `dist/`, deleting unfamiliar directories, rebuilding all releases, or treating Starlight prose snapshots as complete versioning would lose required history or API correctness.

### 9. Trusted validation and reliable Pages publication

Extract a reusable docs-check workflow running on GitHub-hosted CPU runners with read-only permissions. Let trusted push CI and the main/tag publisher call it, and run it directly for pull requests. Do not add a broad `pull_request` trigger to the existing workflow with institute/GPU jobs. Include the docs result in the existing required CI aggregate for trusted commits. The reusable checks bootstrap only pinned docs/CLI tools, not root `dev,all,dali` extras.

Keep exact-revision provenance on validated build artifacts. The trusted publisher stages only artifacts for that source revision and uses narrow permissions: archive staging needs contents write; Pages deployment needs pages write and OIDC. No deployment token or trusted runner is exposed to fork builds. The CPU first-run check remains in installed-distribution/runtime CI; it is not an excuse to install the ML stack in the docs-check job.

Use two publication stages to avoid losing release events through GitHub Actions' single pending concurrency slot:

1. After checks, stage each candidate into `gh-pages` with a normal fast-forward git push. If another candidate won the race, refetch and recompose against that archive with bounded retries; never force-push. Source/ancestry checks prevent an older main candidate from replacing a newer published dev/changelog. Immutable snapshots already staged by other runs survive every retry.
2. Serialize only the Pages deployment job, with cancellation of in-progress deployment disabled. After obtaining the slot, fetch the latest complete `gh-pages` tree and publish that assembled tree. A superseded pending deploy is safe because its release output was durably staged before entering the queue and the surviving deploy includes it. Do not queue whole workflows before staging and assume `cancel-in-progress: false` preserves every pending tag.

Use GitHub's static Pages upload/deploy actions for the assembled output and switch the repository Pages **Source** from legacy branch publishing to **GitHub Actions** during cutover. The provider and project URL stay unchanged; `gh-pages` remains the archive. This avoids relying on a bot-token branch push to trigger legacy Pages rebuilding. Artifacts contain no symlinks/hardlinks. Maintain `.nojekyll` for archive compatibility. Test composition/conflict handling locally; deployment itself requires a trusted maintainer-controlled workflow run.

**Alternative rejected:** an unvalidated independent deploy races required checks; resetting the archive loses history; serializing whole workflows can silently drop pending tags; a token-driven legacy branch push is not a reliable Pages deployment mechanism.

### 10. Origin/mount portability and link validation

Keep `site` origin, deployment mount, documentation version, source revision, and build target in one configuration boundary. The effective docs base is `<mount>/<version>/`; global changelog/search assets use `<mount>/`. Content uses base-independent page links, local asset imports, or shared URL helpers. No authored content embeds the Pages hostname or `/NexuML/` prefix.

Validate built output, not only content source. Check internal hrefs and fragments, generated API/source anchors, CSS/JS/image/video/download paths, and navigation across the assembled version/global boundary. Do not assume successful MDX compilation validates links. Use a known-route inventory for cross-target links before assembly, then crawl the complete assembled preview to catch alias/prefix errors. Reject escapes outside the configured mount. External service availability checks must not replace deterministic local validation.

Keep version-specific sitemaps and canonical links tied to their revision. Generate the global sitemap/index from global entries and the selected stable documentation without indexing duplicate compatibility redirects as content. Verify Pagefind indexes and loaded search/diagram assets at both a non-root project mount and a root custom-domain configuration.

The future migration guide changes origin/mount and the deployment adapter, then covers DNS and, for branch archives on GitHub Pages, root-level CNAME placement versus Actions Pages custom-domain settings. No CNAME or provider change is applied now. Frozen HTML may contain the old absolute base/canonical URLs: DNS alone cannot repair it. Retain and serve the archived project-prefix paths through a compatibility tree/provider rules when moving to a root domain, or explicitly perform a separately approved archive relocation/rebuild. The compatibility approach preserves old bytes and requires no editorial rewrite.

**Alternative rejected:** hardcoding a repository prefix into MDX or assuming a hostname-only change makes already generated archives portable would recreate the current broken-entry-link problem.

## Risks / Trade-offs

- **Early pydocs/plugin compatibility or incomplete static Pydantic/re-export output** → Pin a compatible set and prove representative symbols, both module trees, Autodoc, prefix paths, and search before mass migration or removing MkDocs.
- **CLI import refactoring changes normal Python imports or warning behavior** → Limit edits to proven import boundaries; test public export identity, normal runtime commands, authentic help, and no-heavy-import subprocesses.
- **Content reduction hides behavior or produces giant articles** → Review the old-route/feature inventory, merge only overlap, retain focused specialized tasks, and keep safety notes next to their operations.
- **Two target builds leak duplicate routes/search results or collide on assets** → Filter collections/routes per target, namespace versioned assets, protect global paths during publication, and test the assembled tree.
- **Queue races lose snapshots or stale builds win** → Stage durable immutable output before the deploy queue; use fast-forward/recompose retries, monotonic main provenance, semantic latest selection, and concurrency fixtures.
- **Latest materialization increases archive size** → Accept one duplicated static tree rather than provider-specific alias infrastructure. Do not duplicate every historical site into the source branch.
- **New publication fails after archive staging** → Keep the previously live Pages deployment available and retry deployment from the complete staged archive; do not delete the new snapshot or reset git history.
- **Historical markup has old design/canonical assumptions** → Preserve it as the user requested; document the initial dev-only theme cutover and archive-prefix constraints on a future domain move.

## Migration Plan

1. Inventory content, routes/fragments/downloads, public symbols, current publication state, and corporate assets. Save archive-preservation baseline evidence without altering production.
2. Prove the locked docs environment, dual build targets, static API rendering, and actual lightweight recursive CLI help. Keep the old site/build intact until these gates pass.
3. Consolidate and convert editorial content, establish the CPU-first tutorial, implement branding/components/diagrams, and add the global changelog schema/routes/template. Verify feature coverage and redirects against the inventory.
4. Implement local publication composition and dry-run it against a copy of the existing archive. Verify unchanged release trees, latest behavior, main/tag ordering, aliases, route/search/assets, and root-mount portability.
5. Wire exact-revision read-only checks, trusted archive staging, and explicit Pages deployment. Document the maintainer Pages Source setting change before running the first deploy; do not change the domain.
6. After parity evidence passes, remove current MkDocs config/generation/dependencies and replace local/contributor/OpenChamber commands. Update affected spec purposes during synchronization; leave historical OpenSpec records and release-note automation untouched.
7. Publish validated main output to dev and the global changelog. Check the live site and confirm existing release-tree hashes and stable latest selection. At the next stable tag, publish the first new immutable Starlight release normally.

Rollback is a redeployment of a previously validated assembled Pages artifact or a forward commit restoring only owned mutable publication paths. Retain all immutable release directories, never force-push/reset the archive, and keep the prior source revision available for investigation. Before cutover, the old pipeline remains the fallback; after cutover, use the archive/deployment procedure rather than resurrecting two competing publishers.
