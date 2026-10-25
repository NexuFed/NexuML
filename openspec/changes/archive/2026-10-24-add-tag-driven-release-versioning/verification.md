# Implementation verification

## Local evidence

- Version preparation: `python3 -m pytest -c /dev/null --rootdir=tests/packaging --confcutdir=tests/packaging -q tests/packaging/test_release_versions.py` — 41 passed. Temporary repositories cover annotated/lightweight tags, seven-field drift, idempotence, stale locks, and preservation of non-version metadata.
- Integrated packaging/workflow suite: `TMPDIR=/tmp/opencode python3 -m pytest -c /dev/null --rootdir=tests/packaging --confcutdir=tests/packaging --basetemp=/tmp/opencode/tag-release-final -q tests/packaging` — 157 passed, 4 skipped. Skips require explicit built/installed distribution environments; the separate real-artifact check below exercised built metadata.
- Candidate validation: 101 synthetic cases cover immutable IDs, every historical attempt, pending/failed/cancelled/expired/duplicate candidates, two tags on one commit, moved refs, safe archive paths, wrong metadata/hash/origin, and matching/missing/mismatched TestPyPI evidence. Fake GET API/clock/sleep responses do not publish or wait in real time.
- All four version/candidate scripts and their tests plus workflow tests passed targeted Ruff lint/format and `uvx ty check --python /env/bin/python3`. Including installed-distribution tests in the type check exposed the environment's missing existing `torch.nn` dependency; full-project type/runtime checks still require the CI environment.
- `npm --prefix studio test -- tests/compatibility.test.ts` — passed (1 test); existing backend/interface policy unchanged.
- `npm --prefix docs run check` initially stopped because the active Node 22.16.0 is below docs' required 22.23.4. Retried explicitly with `npm exec --yes --package=node@24 -- npm --prefix docs run check`: passed, including content/tests/static checks, Astro diagnostics, both builds, and rendered links/assets. No dependency files changed.
- `openspec validate add-tag-driven-release-versioning --strict` — passed.
- Official actionlint 1.7.12 release binary, archive checked against its published SHA-256 list: changed workflows passed with a temporary config declaring the existing `ika-runner`/`ika-runner-gpu` labels. Shellcheck/pyflakes integrations were disabled; shell gate behavior is covered by pytest.
- Disposable real-source checkout at `/tmp/opencode/release-stamp.c2PNIU`: unchanged development metadata was tagged locally as `v0.2.2`; the preparation CLI succeeded against the full real `uv.lock` and all preservation checks. Both `uv build --package ...` commands produced `0.2.2` wheels and source distributions. Artifact-content/metadata test passed (1 test). Rebuilding each source distribution without checkout Git metadata retained version `0.2.2`.
- Original and rebuilt wheels were installed with `uv pip install --no-deps` into separate isolated temporary environments. From outside the checkout, `python -I` proved `nexuml.__version__ == version('nexuml') == version('nexuml-library') == '0.2.2'` in both. This verifies installed identity, not dependency/runtime/training acceptance.
- Read-only GitHub GETs checked the real run/attempt/job/artifact response fields against completed branch CI run `40095737423`, including its exact successful `CI` job. Existing Python/Studio artifact zip member layouts match the resolver allowlists. Historical run `38915697742` confirmed attempt-specific start timestamps. These were API-shape/archive-layout checks, not eligible release-candidate or registry acceptance.
- Independent workflow integration and final candidate-helper reviews found no actionable defects. The helper review also checked live read-only API response fields and tag-name filtering; it did not perform publication or independently rerun tests.
- The five checked-in package/lock files remain unchanged; no version-bump commit, source tag, push, publication, or registry-setting mutation was made. The `v0.2.2` tag above exists only in the disposable test repository.

## Owner-only verification

- GitHub-hosted tag/publisher execution, institute CPU/GPU suites, full installed Python/Studio browser acceptance, actual OIDC/provenance, registry file acceptance, and live environment/trusted-publisher configuration are not proven by local fixtures. An authorized owner must perform the first real release after merge and exact-main CI.
- macOS/Windows remain intentionally disabled and unverified. Production requires the original seven-day candidate artifacts and matching TestPyPI evidence; expiry or conflicting successful identities require a new version tag, never replacement bytes.

All 17 implementation tasks are complete. No commit, push, real release tag, archive, or live publication was requested or performed.
