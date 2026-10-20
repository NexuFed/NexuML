# Design

## Context

See `proposal.md` for motivation and the two spec deltas for the behavior contract.

Both Python projects use Hatchling with static `project.version` fields. Studio has a static npm version and two corresponding lockfile root versions; `uv.lock` also records both workspace versions. `scripts/check-release-versions.py` verifies equality today. Python runtime reporting already uses installed distribution metadata.

`ci.yml` currently builds and tests immutable artifacts on branch pushes. `release.yml` and `publish-studio.yml` start on tag pushes, GitHub Release publication, or manual dispatch, but resolve successful branch-CI artifacts by commit. Those artifacts cannot acquire a newly chosen tag version without rebuilding. Studio additionally assumes the packed version equals the original checkout's `package.json` version.

The existing trusted publisher filenames, protected environments, GitHub-hosted publisher jobs, institute CPU/GPU runners, required `CI` check, and intentionally disabled macOS/Windows targets are constraints. The documentation publication workflow already derives its display/archive version from the Git ref and needs no new global replacement mechanism.

## Goals / Non-Goals

**Goals:**
- Make one stable tag sufficient to choose all three release versions, with no source version-bump commit.
- Preserve validation of the actual publication files and immutable candidate-to-production identity.
- Resolve cross-workflow startup races without exposing publishing credentials to builds.
- Keep normal branch development and local installation working without Git-derived versions or release environment variables.

**Non-Goals:**
- Automatic version-number selection, prereleases/nightlies, a release bot, or automatic GitHub Release creation.
- A new build backend, `hatch-vcs`, or a release-management dependency.
- Rewriting dependency compatibility ranges, interface versions, historical documentation, or test example versions.
- Changing runtime compatibility, registry accounts/settings, institute-runner policy, or supported platform coverage.
- A new long-term artifact storage service or automatic recovery of already published versions.

## Decisions

### 1. Stamp valid metadata in disposable tag checkouts

Accept only ASCII stable tags matching `v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)`. Remove the single leading `v` to obtain the shared release version. Resolve annotated and lightweight tags to their peeled commit. Reject branch refs, tag deletions, malformed tags, and a tag moved away from its candidate commit.

Keep the existing checked-in version values as fixed, valid development metadata. They are not the release authority and need not be bumped for a release. Local installs continue to report that development identity; uniquely identifying local revisions is explicitly deferred.

Add one shared preparation entrypoint alongside the existing release validator. Use existing `uv version` functionality with `--frozen` to set both Python workspace project versions before refreshing the lock once. Use `npm version` for Studio with commit/tag creation and lifecycle scripts disabled, allowing the same version for idempotence. Preparation must not create/sync a development environment and must fail before builds if its owned fields do not agree. Install the necessary existing tools in the jobs that prepare a release checkout rather than assuming runner-global versions.

The only version-bearing preparation targets are:

| Target | Owned release identity |
| --- | --- |
| `pyproject.toml` | `project.version` for `nexuml` |
| `library/pyproject.toml` | `project.version` for `nexuml-library` |
| `uv.lock` | The two editable workspace package versions and consequent owned metadata |
| `studio/package.json` | The package's `version` |
| `studio/package-lock.json` | Top-level `version` and `packages[""].version` |

Check the committed lock state before preparation, refresh owned workspace metadata as needed, and verify that third-party package resolutions, sources, hashes, and compatibility declarations have not changed. A resolver change to unrelated packages is an error, not permission to upgrade them. Do not use a repository-wide textual substitution or hand-rewrite third-party lock records.

Run the same preparation before tagged source installations, tests, and package builds so no test executes against a different version identity. Other jobs only consume the version/candidate outputs they need. Ordinary branch jobs retain current valid metadata and lock validation.

**Alternatives:** `SET_BY_CI` breaks ordinary packaging unless every entrypoint substitutes it. Git-derived Python metadata would not solve npm coordination and would add local-build/sdist behavior outside the requested release-only scope. Preparing metadata only in publishing jobs is too late to test the published bytes.

### 2. Reuse the existing CI graph for tagged candidates

Extend `ci.yml` to handle stable tag pushes as well as existing branch pushes. Add a credential-free GitHub-hosted release-identity/eligibility job before any tag execution on institute runners. For a tag it verifies protected-main ancestry and an already successful required main-branch CI run for the exact peeled commit; for ordinary branches it preserves normal development CI behavior. Every institute job that can run for a tag depends on this eligibility result.

The existing source, supported-Python/GPU, static, strict documentation, distribution, rebuilt-sdist, and Studio project/tool checks remain mandatory. Reuse the existing job graph instead of copying a second release-specific suite. Tag package jobs prepare metadata first, build each publication artifact once, and test those exact files. The separate wheel-from-sdist validation remains a check; it does not replace the original candidate files. Disable dependency/build-cache restoration for tag release builds; branch caching remains unchanged.

Preserve the aggregate check name `CI`. Only tag-push CI is eligible to issue a release candidate. A manual CI run against a tag is not an alternate candidate source; use GitHub's rerun of the failed original tag run when no successful candidate exists.

**Alternatives:** reusing a pre-tag artifact cannot change its version while preserving its bytes. Running a second ad hoc build in either publisher would duplicate packaging logic and could bypass installed-package acceptance.

### 3. Add a small immutable candidate manifest, not a release service

After all mandatory checks succeed, the aggregate gate assembles a `release-candidate` artifact that binds:

- Schema version, repository, full tag ref, normalized version, and peeled source commit.
- CI workflow identity, run ID, and run attempt.
- Immutable `python-distributions` and `studio-distribution` artifact IDs from that attempt.
- Expected relative filenames and SHA-256 hashes of the four Python publication files and npm tarball, plus npm's existing SHA-512 `pack.integrity`.

The manifest is part of the successful gate: an upload/validation failure must make `CI` fail. All artifacts retain the existing seven-day lifetime; this design does not promise production promotion after the originals expire.

Both publishers use the same read-only resolver to find the unique successful tag-push candidate and verify its repository/workflow origin, run/attempt success, tag, peeled commit, artifact IDs, package metadata, and file hashes. Allowlisted expected package filenames prevent metadata from selecting arbitrary paths. A manifest is data, never executable build instructions.

Only one distinct successful candidate identity is allowed per tag. Two tags on one commit remain distinct candidates. Multiple successful identities for one tag, unavailable original artifacts, or changed tag identity fail closed; do not select the newest run, overwrite artifacts, or silently fall back to another attempt. Failed attempts without a successful candidate may be retried. Once a candidate succeeds, retry publication against it rather than rebuild it. If an irrecoverable duplicate/expiry occurs, use a fresh version tag.

**Alternatives:** commit-only lookup confuses versions when tags share a commit. Selecting the latest successful run can silently replace candidate bytes between TestPyPI/`next` testing and production. New durable storage and automatic recovery are not needed for the stated workflow.

### 4. Keep publisher identities and use bounded credential-free waiting

Keep `release.yml` and `publish-studio.yml` as the actual publisher entrypoints with their current tag-push and GitHub Release triggers. Do not call publishing jobs from `ci.yml` through a new reusable-workflow chain; that risks changing the identity validated by npm's trusted publisher.

The tag-push workflows may start before tag CI. Their credential-free validation jobs run the shared resolver with a bounded wait and an explicit 120-minute job timeout. The resolver polls pending/queued work at a fixed interval, reports the selected tag/run, and terminates on failed/cancelled CI or timeout. A successful candidate must be fully available before outputs are passed to protected publishing jobs. If a slow build exceeds the wait, a maintainer retries publication on the same tag after CI finishes; no rebuild is necessary.

Split Studio validation from its environment-protected publish job, as Python already does. Validation has only read permissions and no `id-token: write`. Protected publish jobs recheck downloaded identity/integrity immediately before registry writes. They do not run dependency installation, a build, package lifecycle scripts, or version stamping. Compare package versions with the manifest/tag; for other publication metadata compare with the tagged source, replacing only the expected version in memory rather than requiring the committed development version.

Manual publisher dispatch is retained only for existing stable tags. Branch dispatch fails with actionable guidance. No arbitrary version input, automatic tag creation, or publisher-secret fallback is added. Keep npm's supported CLI, OIDC/provenance settings, protected environment checks, and independent publish/dist-tag permissions unchanged.

### 5. Preserve the two-stage release and verify its candidate

Tag publication uses the unique verified candidate for both Python and npm. TestPyPI verification passes the expected candidate version into installed-package tests rather than reading unstamped development metadata from a fresh checkout. Verify downloaded Python files against the candidate hashes as well as testing isolated installs.

Publishing the matching GitHub Release requests production. Python requires successful TestPyPI verification for the same candidate identity, then uploads the original candidate files to PyPI. Bind that verification evidence to the manifest identity in a small immutable result artifact from the successful TestPyPI verification run; production rejects unrelated or missing evidence. npm checks the existing registry version's integrity against the candidate and moves only that version to `latest`. Registry writes remain independently approved and non-atomic; this change does not add an automatic cross-registry transaction.

Attach Python files to the existing GitHub Release only after successful production publication. The PyPI delta aligns the older workflow-created-release wording with the existing `release.published` operator flow; it does not add automatic release creation.

Update existing packaging tests to receive the explicit expected tag version where needed. Preserve metadata-based runtime reporting, mandatory compatibility rejection, independent core/library installs, and source-distribution rebuildability without Git or CI variables. A `v0.3.0` tag remains blocked by today's `0.2` compatibility policy until that policy is deliberately reviewed in a separate change.

## Risks / Trade-offs

- [Tag CI repeats branch validation] → Reuse the same jobs; the second run proves the actual stamped release identity and bytes, not just the source revision.
- [Polling occupies GitHub-hosted validation runners] → Bound the wait to 120 minutes, fail without credentials, and support tag-only publication retries after CI completes.
- [Lock refresh changes dependency resolution] → Assert third-party lock records are unchanged and never broaden compatibility ranges automatically.
- [Development metadata is not a unique revision] → Document the fixed local identity; do not present branch artifacts as release candidates.
- [Seven-day artifacts expire or successful tag CI is rebuilt] → Refuse substitution, document the release window and rerun rules, and use a new version tag when original identity is unrecoverable.
- [Two separately approved registries partially publish] → Keep explicit failure reporting and fix-forward recovery; never overwrite/unpublish existing versions automatically.
- [Existing live registry/environment setup is incomplete] → Keep owner setup as a prerequisite. Planning and local validation must not mutate settings or publish packages.

## Migration Plan

1. Implement the preparation and candidate-resolution changes with local fixture-based tests; update workflow assertions and the operator guide in the same change.
2. Run packaging/workflow tests, unchanged compatibility tests, and strict documentation checks. Exercise successful, pending, failed, cancelled, duplicate, retagged, and expired candidates with mocked API/registry responses rather than real publication.
3. Merge to protected `main` and let normal branch CI pass without a version bump. Do not change existing tags or published files.
4. An authorized owner chooses the next unused compatible patch tag, waits for its stamped CI and candidate approvals, verifies TestPyPI/npm `next`, then publishes the matching GitHub Release and approves production as before.
5. Stop on any version, integrity, compatibility, or identity failure. Retry only publication for an already successful candidate; do not rebuild a published candidate under the same version.

Before any candidate upload, rollback is a normal reviewed revert of the workflow/helper changes. After any registry upload, preserve its tag/artifacts and fix forward with a fresh version; do not force-move tags, rewrite published metadata, or delete registry versions.
