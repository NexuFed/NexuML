# Publish a release

NexuML publishes `nexuml` and `nexuml-library` through `.github/workflows/release.yml`.
Studio publishes `@nexufed/nexuml-studio` independently through `publish-studio.yml`.
Both consume immutable artifacts from successful CI on the **exact release commit**,
not rebuilt files. Use registry Trusted Publishing; never add publishing tokens to repository secrets.

## One-time configuration

Create or claim both project names on [PyPI](https://pypi.org/) and [TestPyPI](https://test.pypi.org/), then configure these trusted publishers:

| Registry | Distribution | GitHub environment |
| --- | --- | --- |
| TestPyPI | `nexuml` | `testpypi` |
| TestPyPI | `nexuml-library` | `testpypi-library` |
| PyPI | `nexuml` | `pypi` |
| PyPI | `nexuml-library` | `pypi-library` |

Every publisher uses owner `NexuFed`, repository `NexuML`, and workflow `release.yml`. Create all four matching environments in the GitHub repository and protect them with the required maintainer reviewers. The workflow grants `id-token: write` only to publication jobs, so no long-lived publishing credential is stored.

## TestPyPI candidate

Run the **Release** workflow manually from the frozen candidate on `main`. It selects the non-expired `python-distributions` artifact from a successful CI run for that exact commit, publishes core before the library to TestPyPI through their separate protected environments, downloads those wheels again, and verifies both the core-only and `nexuml[library]` installation paths. Third-party dependencies resolve from production PyPI.

CI retains the validated distributions for seven days. If the artifact expires, rerun CI for the exact candidate before starting Release; Release never rebuilds publication files.

TestPyPI files are immutable. Bump both project versions before rerunning a candidate whose artifacts were already uploaded.

## Production release

1. Confirm both projects declare the intended version and the TestPyPI candidate passed.
2. Create and push the matching stable `v<project version>` tag.
3. Approve the protected `pypi` and `pypi-library` environment deployments in sequence.

Publish a GitHub release for that tag to trigger production publication. The workflow
rejects a tag that differs from either project version. It publishes validated core
artifacts first, then library artifacts, then attaches those files to the existing
GitHub release. If one upload fails, do not reuse an already published version; fix
forward with a patch release. A `studio-v*` release skips this Python pipeline.

## Studio: one-time owner configuration

1. Own the npm `@nexufed` scope and grant the intended maintainers package access.
   If the package does not exist and npm requires an initial publication before its
   settings become available, an account owner must bootstrap it interactively with
   MFA from the exact validated tarball. Never publish a placeholder or add a token
   workaround to Actions. No package is published as part of implementing this workflow.
2. Create GitHub environment **`npm`** with required maintainer reviewers, prevent
   self-review, and restrict deployments to protected Studio release tags. Protect
   `main`, `studio-v*` tags and publishing workflow edits with repository rules/review.
3. In the package's npm settings, add a GitHub Actions Trusted Publisher:
   owner **`NexuFed`**, repository **`NexuML`**, workflow **`publish-studio.yml`**,
   environment **`npm`**. Enable **Allow npm publish** (new publishers may default to
   stage-only authorization). Match case and the filename exactly.
4. Complete the first authorized publish within npm's current **two-day** setup window;
   otherwise recreate the expired publisher. Then select **Require two-factor
   authentication and disallow tokens** in npm publishing access.

Follow the [official npm requirements](https://docs.npmjs.com/trusted-publishers)
for supported Node/npm installations and **GitHub-hosted runners**; self-hosted
runners are unsupported. The publish-only job uses OIDC
`id-token: write` and provenance in the protected environment, with no dependency
installation/build or long-lived credential. Release workflows use GitHub-hosted
`ubuntu-latest`, including the PyPI/TestPyPI jobs in `release.yml`.
Linux CPU builds/tests stay on `ika-runner`. The existing
mandatory Python GPU suites use the separately approved `ika-runner-gpu` pool.

## Studio release

1. Release a compatible Python backend first; the launcher enforces its supported
   runtime/interface contract. Package versions are independent. The API extra
   and custom libraries must be explicitly installed by the operator.
2. Merge the reviewed candidate to protected `main`, wait for complete required **CI**
   on that exact commit, and verify the `studio-distribution` artifact is unexpired.
   Studio is built/packed once; Linux project/tool jobs test that tarball and the
   separately built Python wheels. macOS/Windows jobs are temporarily disabled by
   maintainer request; their release acceptance remains unverified, not silently green.
3. Create/push `studio-v<studio/package.json version>` and publish its GitHub release
   (do not bump/tag merely to test this workflow).
4. Approve the `npm` deployment. The workflow checks the tag/version, protected-main
   ancestry, successful push CI/main required gate, artifact ID/commit, SHA-512
   integrity and packed metadata, then publishes **that tarball** with provenance.
   It never rebuilds it. Python `v*` releases skip the npm job.

If artifacts expire, rerun CI on the exact release commit. If validation fails, fix
the candidate and rerun CI rather than uploading a replacement tarball manually.
After publication, npm versions are immutable: do not overwrite/reuse the version.
For recovery, users can explicitly install a previous compatible Studio/backend;
maintainers can deprecate a bad npm version and move `latest` back to a known-good
version using interactive MFA, then ship a new patch. Do not auto-unpublish or mutate
user environments. Python core/library rollback follows the same explicit pin/fix-forward
policy; existing YAML remains usable without Studio.

## Institute runner trust boundary

CI uses trusted **repository branch pushes/manual dispatch**, not `pull_request` or
`pull_request_target`, because Linux checks must use institute runners. Fork PRs have
no automatic institute execution and remain blocked on missing required CI. Review
external code first, then bring the reviewed commit onto a trusted repository branch;
never approve arbitrary fork workflow execution to obtain a green status.

Repository fork workflow approval is set to **all external contributors** (not just
first-time contributors). Owners must also restrict the institute runner group to
approved repositories/workflows, preserve that approval setting, protect workflow edits, and
provide ephemeral, isolated runner instances without internal credentials/network access.
A workflow condition alone cannot secure a public repository's runner group against a
new malicious workflow. GPU assertions remain mandatory in the Python suite; the
`ika-runner-gpu` pool must provide its required CUDA capacity. Do not remove that assertion
or quietly skip verification to accommodate a misconfigured pool.
