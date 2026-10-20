# Proposal

## Why

The pending Studio pull request repeats setup, UI behavior, release history, and implementation details across several guides. Before merging, reduce that maintenance burden and remove verified support-code duplication without weakening the working product.

## What Changes

- Review all human-facing documentation, prioritizing the PR additions. Keep short explanations and outcome-based tutorials; link to reference material instead of repeating it.
- Remove documented release numbers, dependency/runtime version ranges, numbered compatibility notes, and pinned installation examples. Keep machine-readable package metadata and runtime compatibility enforcement unchanged.
- Describe the released product: ordinary PyPI/npm installation, no TestPyPI beginner setup, unpublished-package warnings, or candidate-wheel detours.
- Reduce the Studio README to setup, launch, a first run, and links. Keep one canonical tutorial and one advanced hosting procedure; shorten the manual API guide around generated reference and essential transport rules.
- Consolidate contributor setup, correct stale commands/CI descriptions, and summarize the Studio verification diary without discarding outstanding acceptance gates.
- Keep useful skill/example checks, relax formatting-only assertions, and consolidate framework regression tests into the normal test suite. Do not delete the optional agent-benchmark harness without a separate decision.
- Remove unused draft/event state and repeated graph/catalog work. Preserve transport responses, training behavior, security, persistence, and observation semantics.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `documentation-journey`: Task-focused, nonduplicated documentation without hard-coded release/runtime version claims.
- `user-install-docs`: Release-state package installation with short Studio onboarding and a separate contributor path.

## Impact

- Documentation: root/package READMEs, `docs/`, skill bodies/references, navigation, and the existing Studio verification summary.
- Code/checks: Studio draft, observation and graph callers/tests; API catalog construction; skill validators, framework tests, and affected CI commands.
- No new dependencies, package/version bumps, release actions, public API removals, or backend redesign. Existing OpenSpec requirements and pending platform gates remain intact.
