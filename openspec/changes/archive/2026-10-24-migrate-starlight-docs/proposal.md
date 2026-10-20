# Proposal

## Why

NexuML's documentation repeats explanations and entry paths, making it harder to find a working command or understand the framework. [NEX-247](https://linear.app/nexufed/issue/NEX-247/migrate-nexuml-docs-to-astro-starlight-mdx-with-branded-api-reference) replaces the MkDocs stack with branded Astro Starlight documentation while simplifying the reading journey and preserving public reference coverage and published releases.

## What Changes

- Replace the current site with an isolated, locked Astro Starlight + MDX project in `docs/`, retaining uv for lightweight Python extraction.
- Reduce primary navigation to **Start here**, **Use NexuML**, **Understand NexuML**, and **Reference**. Keep tutorials in the starting path; expose Changelog, Contributing, and version selection as secondary navigation.
- Consolidate duplicate scenario, lifecycle, architecture, and library instructions. Keep one runnable procedure per task, expected outputs, necessary safety notes, feature coverage, and a clear account of NexuML's vision and implemented architecture. Use a one-epoch synthetic CPU example for the default first run; retain CIFAR as an optional example.
- Apply the live NexuFed corporate identity using supported Starlight styling and narrow component overrides, preserving search, accessible navigation, code copying, diagrams, tabs, and edit/source links.
- Generate Python API trees for both public packages using `starlight-pydocs`/Griffe. Generate the CLI reference from the actual Typer/Click command tree without importing heavy runtime backends.
- Add schema-validated editorial changelog MDX entries with automatic chronological index/detail routes. Publish the global `/changelog/` from validated main commits, independently of release snapshots. Preserve GitHub Releases note generation.
- Keep GitHub Pages at the existing project URL. Preserve `dev`, immutable tagged snapshots, and `latest`; leave existing release archives untouched. The initial Starlight cutover updates `dev` and the global changelog, not the existing latest release.
- Add validation, prefix-aware links/assets, legacy URL mappings/redirects, publication safety tests, desktop/mobile visual QA, and a documented config/deployment-only future domain/provider migration.
- **BREAKING (contributor tooling only):** replace MkDocs/mike commands, configuration, generators, and dependencies after replacement parity is demonstrated. Preserve public Python behavior and supported documentation entry URLs.

## Capabilities

### New Capabilities

- `documentation-changelog`: Validated MDX release/update entries, rich media, generated chronological routes, release links, and main-published global changelog behavior.

### Modified Capabilities

- `documentation`: Starlight builds, corporate styling, static API extraction, accurate lightweight CLI extraction, diagrams, checks, versioned publication, compatibility routes, and hosting portability.
- `documentation-journey`: Four clear primary sections, consolidated task instructions and architecture/vision, with dated changelog entries exempt from evergreen version-number restrictions.
- `first-run-tutorial`: A consistent synthetic CPU-first walkthrough replaces mandatory CIFAR training, with verified commands and artifact expectations.
- `scenario-learning-docs`: Embed scenario explanation in the canonical architecture and create/run journey rather than require a separate, duplicated learning page.
- `decorator-discovery-docs`: Explain discovery within the canonical extension guide, retaining typed-definition/runtime semantics and verification commands without a prerequisite learning detour.
- `reference-docs`: Compact reference entry points and generated Starlight API/CLI output replace MkDocs-specific reference tooling.

## Impact

- Site content/config/assets under `docs/`; `mkdocs.yml`; `scripts/gen_ref_pages.py`; documentation extraction, checks, publication scripts, and tests.
- `pyproject.toml`, `uv.lock`, and new docs JavaScript/isolated extraction lockfiles; minimal package/CLI import-boundary changes only where necessary for authentic lightweight CLI extraction.
- `.github/workflows/ci.yml`, `.github/workflows/docs.yml`, the existing `gh-pages` archive, and Pages deployment configuration. Release-note categories and Python/Studio publishing remain unchanged.
- README/package metadata links, `.openchamber/project.json`, contributor/publication instructions, and the affected OpenSpec requirements. Archived OpenSpec records remain historical evidence, not targets for blanket replacement.
- No hosting/domain switch, runtime feature redesign, second tutorial repository, CMS, automatic editorial prose generation, or historical release rebuilds.
