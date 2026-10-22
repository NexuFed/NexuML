# Documentation publication

GitHub Pages remains at **https://nexufed.github.io/NexuML/**. The existing
`gh-pages` branch remains the durable archive, not a disposable build directory.
This procedure changes neither the provider nor the domain. Maintainers perform
the Pages setting change and first production publication **after merge**.

## Inputs and ownership

Run `docs/scripts/publication.py` with Python 3.12+ and Git. It uses only Python's
standard library. Build and validate the exact source revision first, using the
[documentation checks](README.md). The trusted workflow must verify that the
downloaded build artifacts belong to that revision and passed its required checks;
the composer records provenance but cannot authenticate an arbitrary build folder.
Local preview/bootstrap fixtures are not eligible publication artifacts.

| Event | Owned output | Retained output |
| --- | --- | --- |
| Main (`--version dev`) | `dev/`; optionally global changelog/support; shared metadata and root compatibility redirects | Every release; the existing stable latest selection; unrelated archive files |
| Stable tag (`--version vX.Y.Z`) | One new immutable snapshot; latest if newest; shared metadata and root redirects | Existing snapshots, dev, global changelog and all its support |
| Retry | Confirm exact revision **and** prepared snapshot digest | Never replace a conflicting or unknown-provenance release |

Global support is exactly `changelog/`, `_astro/`, `pagefind/`, `404.html`,
`sitemap-index.xml`, `sitemap-0.xml`, `favicon.png`, and the exact-revision
`build.json` identity record. A global candidate may contain only those
root entries and must include `changelog/index.html`. Supplying `--changelog` on a
tag fails. Main should supply both builds; omitting it deliberately retains global
content. Versioned builds must contain `index.html`.

All archive directories are retained, including stable snapshots not listed in
the original `versions.json`. That file keeps mike's array, title and alias fields.
Stable releases sort numerically, not lexically or by arrival. Main preserves the
original stable selection (`v0.2.1` at migration); only a normal stable tag advances
it. Backfilling an older tag never rolls latest backward. Without any stable
snapshot, root entry redirects use dev as a bootstrap fallback.

`latest/` is an ordinary full copy of the chosen release, including assets,
downloads and permissions, not a rebuild or symlink. Canonical links may leave
latest for the pinned release. The original legacy latest symlink is the only
accepted input symlink; its target must be an existing root stable directory.
Other archive/candidate symlinks and special files fail. Copies break hardlinks.

`publication.json` is shared state outside the frozen release directories:

```json
{
  "schemaVersion": 1,
  "versions": {
    "v0.2.1": {"revision": null, "digest": "<sha256>", "routes": ["", "start/install/"]},
    "dev": {"revision": "<full-source-commit-sha>", "digest": "<sha256>", "routes": [""]}
  },
  "global": {"revision": "<full-source-commit-sha-or-null>", "digest": "<sha256>"}
}
```

Digests cover relative paths, file types, all permission bits and file bytes,
including directory entries; timestamps are excluded. Global digests cover only
the enumerated global/support roots. Snapshot digests include compatibility pages
added before the new snapshot freezes. Routes are actual directory-index routes,
relative to each snapshot: home is `""`, not `"/"`. Existing snapshots initially
have `revision: null`; missing provenance never makes them replaceable. Subsequent
compositions verify recorded digests/routes against the archive before merging.

## Preservation dry run (no remote writes)

Fetch the existing archive; a missing/unreadable archive or missing `versions.json`
is an error, never an instruction to start empty. Fetch full source history and a
fresh trusted main/tag ref too. Default source refs are `refs/heads/main` for dev
and `refs/tags/<version>` for a release. For detached Actions checkouts, pass an
explicit fetched ref such as `refs/remotes/origin/main`. Main's revision must be
an ancestor of its trusted ref; a release must equal its peeled tag commit.

Example, from the repository root, after exact-revision checks/builds:

```sh
git fetch --no-tags origin refs/heads/gh-pages
archive_commit=$(git rev-parse FETCH_HEAD)
work=$(mktemp -d /tmp/opencode/docs-publication.XXXXXX)
python3 - "$archive_commit" "$work/archive" <<'PY'
from pathlib import Path
import runpy
import sys
publication = runpy.run_path('docs/scripts/publication.py')
publication['extract_archive'](Path.cwd(), sys.argv[1], Path(sys.argv[2]))
PY

# Use the actual validated build directories and full source SHA, not these labels.
python3 docs/scripts/publication.py \
  --archive "$work/archive" \
  --candidate /absolute/path/to/validated-versioned-build \
  --changelog /absolute/path/to/validated-global-build \
  --output "$work/assembled" \
  --version dev --revision "$VALIDATED_SOURCE_SHA" \
  --mount /NexuML/ --site https://nexufed.github.io \
  --repository "$PWD" --source-ref refs/remotes/origin/main
```

The archive, candidate, optional global build and output must not overlap. Output
must be absent, with an already existing parent. Root paths, traversal and symlink
path components are rejected.
Composition reads inputs without changing them and writes only its new output.
It verifies the allowed change set and retained snapshot digests, deleting its
new output on a composition failure. For a tag, omit `--changelog`, use the real
stable version, and pass its exact tag commit/ref. Do not create a migration-only
release to move latest.

Before cutover, compare all historical tree digests/modes to the fetched archive
and [recorded migration baseline](migration/baseline.json), not just two familiar
releases. Serve/crawl the assembled tree under `/NexuML/`; test home, install,
API, templates, old release URLs, latest assets/downloads, dev, global changelog,
search, sitemap/canonical links and corresponding-page version switching. Retain
the report and source/archive SHAs. Repeat new builds/composition at an alternate
origin with `--mount /` for portability proof; frozen old HTML remains unchanged.

Compatibility pages cover reviewed authored mappings plus actual selected-site
routes, including generated API pages. Root aliases prefer a real page in the
selected stable snapshot, then its mapped equivalent/nearest existing section,
then home. Missing mapped current-version pages are added **before freezing new
output**, never inside historical releases. Global/changelog/support and root API
endpoints are reserved. Redirect files contain canonical metadata, meta refresh,
and an accessible direct link; they are **not HTTP 301 responses**. JavaScript
retains the incoming query and fragment, using the mapping's fragment when there
is no incoming fragment. Removed legacy heading IDs cannot always be preserved;
no nonexistent replacement headings are fabricated.

Run the isolated invariant tests:

```sh
python3 -m unittest discover -s docs/tests -p test_publication.py -v
```

They use temporary local bare repositories only (Node, when available, additionally
executes the redirect JavaScript against query/fragment examples). The races exercise real pushes,
rejections and refetch/recomposition, not a mocked remote state. They do not build
Starlight, crawl a real archive or deploy Pages; those remain integrated gates.

## Durable staging, then queued deployment

Only a trusted main/tag publisher may run this mode with archive-write credentials:

```sh
python3 docs/scripts/publication.py --stage --remote origin --branch gh-pages \
  --candidate /absolute/path/to/validated-versioned-build \
  --changelog /absolute/path/to/validated-global-build \
  --version dev --revision "$VALIDATED_SOURCE_SHA" \
  --mount /NexuML/ --site https://nexufed.github.io \
  --repository "$PWD" --source-ref refs/remotes/origin/main --attempts 4
```

For a tag, omit `--changelog` and use its stable version/tag ref. Do not pass
`--archive` or `--output` in stage mode. Temporary Git databases and trees live
under `TMPDIR`, or `/tmp/opencode` by default, and are cleaned afterward. Staging
does not checkout, reset or clean the source working tree. The isolated Git
repository needs transport authentication supplied through Git's environment or
credential helper; source-checkout-local `http.*.extraheader` settings are not
automatically inherited. Keep tokens out of remote URLs and command logs.

Every attempt fetches the existing archive, composes afresh, validates preservation,
and creates a child commit of that fetched archive. The push is normal, never
forced. A competing update triggers refetch/recompose, up to `--attempts` (1–10,
default 4). A rejected push without a changed remote head fails immediately.
Exhaustion fails safely: rerun the already validated candidate; do not overwrite
the branch. Git-inexpressible permission modes, empty directories or attributes/
filters that change staged bytes/modes are rejected rather than silently lost.

Main provenance uses `git merge-base --is-ancestor`. A candidate older than already
published dev/global returns `status: "stale"` without replacing newer output.
Divergent history or missing required commit history fails for investigation.
An immutable retry needs both identical source SHA and identical prepared bytes/
modes. Rebuilding the same tag differently is not permission to mutate it.

Success prints one JSON object: `status`, `version`, `revision`, `latest`, plus
`archiveCommit`, `attempts` and `staged` in stage mode. `staged: false` means the
complete fetched tree was already correct (including a stale-main no-op), not that
the publisher should deploy an old per-run candidate.

**Stage every validated event before the deployment queue.** Serialize only the
Pages deployment job, with cancellation of an in-progress deployment disabled.
GitHub Actions can replace a pending queued job even with that setting; serializing
whole staging workflows can therefore lose release events. Durable archive commits
make a superseded pending deploy safe.

After acquiring the deployment slot, **fetch the newest complete `gh-pages` again**,
extract and validate that archive, and upload that tree with GitHub's static Pages
upload/deploy actions. Do not upload the earlier candidate or simply deploy the
commit returned by a run that waited in the queue. Preserve `.nojekyll`. Artifacts
must contain no symlinks or hardlinks. Stage permissions are contents-write;
deployment needs pages-write and OIDC. Untrusted PR checks get neither credentials
nor institute/self-hosted runners.

## Maintainer-controlled Pages cutover and recovery

1. Merge the replacement publisher only after preservation dry runs and parity
   checks pass. Retain the previous validated assembled artifact and archive SHA.
2. Ensure the legacy branch publisher will not compete with the new publisher.
   In repository **Settings → Pages → Build and deployment → Source**, switch
   **Deploy from a branch** to **GitHub Actions**. Keep the existing project URL,
   provider and archive branch. Do not add a CNAME or change DNS/custom-domain
   settings during this migration.
3. Run the trusted, validated main publisher (rerun the merged main commit's
   **Deploy docs** Actions run if it started before the Source setting changed).
   It stages dev/global only; old
   `v0.1.0`, `v0.2.1` and stable latest stay on their old technology/bytes. A bot-token
   push to gh-pages is not relied on to trigger a legacy Pages build.
4. After the queued latest-archive deployment succeeds, smoke-test the live URLs
   listed above and compare historical hashes/modes. Record the deployed archive
   SHA and artifact identity. The first Starlight stable release waits for the next
   normal tag.

If deployment fails **after staging**, retain the staged snapshots and the currently
live deployment. Fix/retry deployment from the freshly fetched complete archive.
Rollback means redeploying a previous validated assembled artifact, or a reviewed
**forward commit restoring only owned mutable paths** while keeping every immutable
release. Preserve provenance consistently with any restored dev/global output.
Never force-push/reset the archive, delete a release or revive two competing
publishers. Before cutover, the old pipeline remains the fallback; after cutover,
use this archive/deployment recovery process.

## Future origin, domain or provider migration (not part of cutover)

Change the shared origin/mount build configuration and deployment adapter, then
validate new output at that location; do not edit editorial links. DNS and HTTPS
configuration are separate maintainer actions. For legacy **branch-source** Pages,
the custom-domain `CNAME` belongs at the archive root, not in each version. For
**Actions-source** Pages, configure the repository's Pages custom-domain setting
and DNS/verification; do not assume adding a CNAME inside an uploaded artifact sets
the domain. Preserve any existing root CNAME as archive data; none is added here.

Frozen MkDocs/Starlight snapshots may embed the old `/NexuML/` base and absolute
canonical URLs. Changing DNS alone cannot repair those bytes. Keep serving their
old project-prefix compatibility paths through a copied compatibility tree or
provider routing rules, including assets/search/downloads. Alternatively seek a
separate approved archive relocation/rebuild. Keep original snapshots untouched;
do not silently rewrite the historical archive during a hostname move.
