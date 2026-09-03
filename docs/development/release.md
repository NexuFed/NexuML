# Publish a release

NexuML publishes `nexuml` and `nexuml-library` through `.github/workflows/release.yml`. The workflow uses PyPI Trusted Publishing; do not add API tokens to repository secrets.

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
2. Create and push the matching stable tag, for example `v0.2.0`.
3. Approve the protected `pypi` and `pypi-library` environment deployments in sequence.

The workflow rejects a tag that differs from either project version. It publishes validated core artifacts first, then library artifacts, and creates the GitHub release only after both uploads succeed. If one upload fails, do not reuse the version; fix forward with a patch release.
