# Tasks

## 1. Shared version preparation

- [x] 1.1 Add shared stable-tag parsing and release preparation under `scripts/`, using existing uv/npm version operations without commits, tags, lifecycle scripts, or environment synchronization; verify lightweight/annotated tags, invalid/deleted/branch refs, source `0.2.1` to release `0.2.2`, and repeated preparation with temporary-repository tests.
- [x] 1.2 Extend `scripts/check-release-versions.py` to validate the explicit prepared release identity across both Python projects, Studio, and owned lock entries while retaining branch validation; verify deliberate drift in each owned field is rejected and fixtures prove third-party lock records, compatibility bounds, interface versions, and historical text are unchanged.
- [x] 1.3 Document the fixed local-development identity and tag-selected publication version in `docs/src/content/docs/development/release.mdx`; verify the documented workflow no longer requires manual metadata bumps or a `SET_BY_CI` placeholder.

## 2. Tagged CI and immutable candidate production

- [x] 2.1 Extend `ci.yml` with tag-push handling and a read-only GitHub-hosted eligibility gate before institute-runner tag jobs; add workflow/API-fixture tests proving exact main-branch CI and protected-main ancestry are required, while normal branch checks, required check name `CI`, GPU assertions, and fork restrictions remain intact.
- [x] 2.2 Apply version preparation before all tagged source installs, tests, and publication builds, disable tag-build cache restoration, and retain the existing package acceptance graph; verify workflow tests cover every relevant checkout and that a synthetic stamped wheel/sdist/npm candidate reports the chosen version without rebuilding the publication tarball in consumers.
- [x] 2.3 Emit the immutable candidate manifest only after all mandatory tag checks pass, recording repository/tag/peeled commit, CI run/attempt, artifact IDs, expected filenames, and file integrity; verify fixture tests reject mismatched metadata/bytes, unsafe paths, and failed/skipped/cancelled gate outcomes.
- [x] 2.4 Update the operator guide's tag-build and candidate-artifact sections; verify it identifies the seven-day release window, explains the distinction between branch and tag artifacts, and does not claim macOS/Windows or live registry acceptance.

## 3. Candidate resolution and publisher integration

- [x] 3.1 Implement one read-only resolver shared by both publishing workflows with bounded waiting and strict tag/commit/run/attempt/artifact validation; verify mocked API cases for pending-to-success, timeout, failure, cancellation, missing/expired candidates, moved tags, two tags on one commit, and duplicate successful identities without real sleeps or registry writes.
- [x] 3.2 Replace pre-tag commit-only artifact selection in `release.yml` and `publish-studio.yml`, split Studio's credential-free validation from protected publication, and preserve publisher filenames/environments; verify workflow tests require validated candidate outputs before OIDC jobs, no branch fallback, no publisher build/stamping steps, and explicit bounded validation timeouts.
- [x] 3.3 Make manual publication tag-only and compare packed versions with the candidate rather than unstamped source metadata; verify dispatch and metadata tests accept an eligible existing tag, reject branch dispatch, and preserve checks on non-version publication metadata.
- [x] 3.4 Update manual-publication/retry instructions in the operator guide to use `--ref vX.Y.Z`; verify examples select existing candidates and neither create tags automatically nor rebuild a successful candidate.

## 4. Candidate-to-production identity

- [x] 4.1 Pass candidate versions into installed-distribution/TestPyPI verification, compare downloaded Python files with original candidate hashes, and retain verification evidence bound to the manifest identity; verify tests cover matching evidence, mismatched files/versions, and absent or failed verification without contacting a registry.
- [x] 4.2 Require matching TestPyPI evidence before PyPI upload and preserve exact-version npm integrity-checked promotion to `latest`; extend mocked publisher tests to prove missing/mismatched evidence or npm packages block writes, production never rebuilds, and GitHub assets attach only after both Python uploads succeed.
- [x] 4.3 Document the unchanged tag-to-candidate/GitHub-Release-to-production stages and fix-forward behavior; verify the guide forbids retagging, replacement bytes, automatic unpublish, and misleading success claims after partial publication.

## 5. Integrated verification

- [x] 5.1 Run the targeted packaging/workflow suite and `npm --prefix studio test -- tests/compatibility.test.ts`; record results showing tag-derived installed versions and unchanged rejection of unsupported backend versions/interfaces, with no live publish or registry-setting changes.
- [x] 5.2 Run `npm --prefix docs run check` and `openspec validate add-tag-driven-release-versioning --strict`; verify the updated operator guide and both deltas pass their checks.
- [x] 5.3 Record implementation evidence and any unavailable hosted/institute CI or owner-only tag/registry checks in the change directory; verify no required check is reported as passed merely from fixture tests and no existing user changes are included in the implementation diff.

## Workflow follow-up

- An authorized owner performs the first real tagged candidate/production release after implementation and required CI succeed; this plan does not authorize publishing or changing registry/environment settings.
- Archive this change after implementation review and verification, retaining any explicitly outstanding owner-only release evidence.
