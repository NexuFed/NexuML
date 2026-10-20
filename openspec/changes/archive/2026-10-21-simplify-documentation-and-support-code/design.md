# Design

## Context

See `proposal.md` for motivation and the two documentation deltas for acceptance. The existing Studio/API/backend changes remain authoritative for runtime behavior. The audit identified repeated instructions, brittle skill checks, write-only draft state, an unused event-history array, and repeated graph/catalog construction.

## Goals / Non-Goals

**Goals:** Delete repetition and unused work while keeping the normal install-to-first-run path complete and preserving tested behavior.

**Non-Goals:** Backend redesign, a workbench state-management rewrite, repository-wide formatting, new tooling/dependencies, package bumps, publishing, removing hosted support, or deleting the manual agent-benchmark harness. Do not modify other changes' proposals, designs, specs, or task acceptance gates.

## Decisions

### One home for each explanation

| Location | Responsibility |
| --- | --- |
| `README.md` | Product purpose, ordinary installation, links to first use and Studio |
| `studio/README.md` | Setup, launch, one small example, links; roughly one screen of guidance |
| `docs/studio.md` | Canonical first-run tutorial, runtime selection, and advanced hosting procedure |
| `docs/how-to/local-api.md` | Concise manual startup and transport-specific reference |
| `docs/development/install.md` | Contributor installation |
| `docs/how-to/development.md` | Checks and contribution workflow, linking to installation |

Review the remaining authored docs and skill guidance for the same issues; do not rewrite concise, correct pages merely for consistency. Reduce repeated library-registration recipes by linking their existing owner pages. Keep generated CLI/Python reference and use the existing authenticated OpenAPI contract for REST details. Handwritten API notes retain WebSocket authentication/replay, revision conflicts, and lifecycle limits. Keeping every existing paragraph under new headings is not a reduction.

### Write for released packages, without version claims

Use normal PyPI/npm commands, including `pip install "nexuml[api,library]"`; link advanced uv-tool selection instead of showing every installer variant in the quickstart. Remove preview/candidate installation detours, numbered prerequisite/compatibility prose, release-number badges, pinned image tags, and release filenames from authored guidance, including skill bodies. Keep deliberate TestPyPI publishing instructions in Development.

For containers, use an explicit user-selected registry tag, not an invented `latest`. Preserve protocol endpoint paths, required component identity fields, and numerical configuration values so examples still work. Package manifests, lockfiles, runtime checks/UI identity reporting, skill frontmatter, generated references, and historical OpenSpec specs are not targets of this editorial rule. Versioned documentation deployment remains unchanged. Prefer a review/search checklist to a new version-scanning framework.

### Retain useful checks, not prose-format policing

Keep the lightweight skill structure/link check and execution of the actual documented example. Validate frontmatter values with the installed YAML tooling rather than requiring one scalar style. Keep genuine identity, payload, link, and eval-input checks.

Move unique fitted-state/checkpoint regression assertions from `skills/tests/check_runtime.py` into ordinary pytest coverage; remove tuning assertions already covered there, carrying over any unique cases first. Keep the example smoke in the existing script and update its descriptions/CI caller if needed. Preserve independently installable skill safety guidance and the minimal-library starter. Leave the manual workflow grader, eval inputs, and ignored workspaces intact; their removal is a separate product decision.

### Restrict code cleanup to verified duplication

- Remove the unread `semanticRevision` draft-store field; retain document revisions used for save conflicts, layout matching, and build freshness.
- Replace the unused generic event array with the minimum accounting needed for its truncation notice. Preserve independent metric/terminal bounds, replay cursors, gap reporting, reconnect, and terminal invalidation. Do not remove disk-backed history.
- Reuse the current graph for connection checks rather than projecting it per candidate. Update all internal callers/tests together; retain domain, order, repeated-key, and evaluator-default rules. No global cache or new graph representation.
- Construct the execution catalog once per API catalog response and derive both existing representations from it. Preserve all response fields; deleting an apparently unused wire field is unnecessary compatibility risk.
- Expand touched dense expressions when necessary for readability; do not minify code or introduce wrapper layers to lower line counts.

### Keep evidence current and honest

Condense only the existing Studio `verification.md` into current commit/check results, reproduction commands, durable evidence links, and known limits. Preserve pending cross-platform gates and distinguish older evidence from current-head results. Historical detail remains recoverable through Git; do not invent fresh results or delete local artifacts.

## Risks / Trade-offs

- Essential guidance lost while cutting prose → Walk the first-run tutorial and follow every advanced link; retain safety details beside the relevant action.
- Moving tests loses coverage → Map each assertion to its retained or new pytest case before removing it.
- Graph/event cleanup changes behavior → Add focused regression checks before editing, then run existing graph, observation, save/reconnect, and package checks.
- Concurrent release/install deltas → Preserve the PyPI, optional-extra, and portability requirements of the existing changes; this change adds editorial constraints, not a competing release process.

## Migration Plan

No user data migration or release action. Implement docs and checks first, then the bounded support-code edits. Keep established pages or repair every navigation/reference link. Verify with the existing project commands; stop once the documented paths are concise and the focused behavior regressions pass. Revert only this cleanup's edits if a regression cannot be resolved without expanding scope.
