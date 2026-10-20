# Studio verification

## Evidence scope

Cleanup checked on 2026-10-21 against `a3c2ca164c13add200f5b74c8f5d1facaf2f2855`
plus uncommitted `simplify-documentation-and-support-code` edits. These are local
working-tree results, not a new exact-head CI or cross-platform acceptance claim.
[PR #16](https://github.com/NexuFed/NexuML/pull/16) remains the integration record.
The concurrent archive moved this diary; no other archived acceptance gates were changed.

## Current cleanup checks

| Check | Result |
| --- | --- |
| Authored documentation audit | 72 files; 289 local/site/repository links and anchors checked; skill frontmatter unchanged |
| Strict MkDocs | Passed |
| Skill structural and documented-example scripts | Passed; 21 eval cases retained |
| Focused skill/framework/catalog/config pytest | 38 passed, 7 existing Optuna-dependent skips |
| Full API suite with backend extras and matching built-wheel startup | 29 passed, no skips |
| Ruff lint/format and ty over affected Python files | Passed |
| Studio typecheck/lint/unit tests | Passed; 42 unit tests |
| Python wheels/sdists and distribution-content test | Passed; all 92 core and 94 library Python files match source |
| Studio production build and npm pack | Passed; one tarball for both installed modes |
| Installed project package and browser workflows | Passed; all 15 browser tests, full transport/ownership checks |
| Installed tool package and browser workflows | Passed; all 15 browser tests, same tarball, full transport/ownership checks |
| Independent support-code review | No material defects found |

The initial API run had 26 passes, 1 skip and 2 failures because this worktree's
environment lacked `kubernetes` and `ray`. With explicit user approval, the declared
Ray default/train and Kubernetes dependencies were installed into the current `.venv`.
The rerun passed **all 29 API tests**, including the matching built-wheel startup check,
with no test or production-code changes. `uv pip check` passes; affected Python
lint/format/type checks were repeated successfully. Task 4.1 is complete; all ten
cleanup tasks are verified, not a claim of full-repository or exact-head remote CI.
Environment snapshots and install/test logs are retained alongside the artifact evidence.
The catalog-count regression failed with two evaluations before the edit and passes
with one evaluation while retaining both wire representations and all response fields.

Logs, artifact identities and the original diary are retained under
`/tmp/opencode/nexuml-cleanup-jPvdL0jI/`. Documentation audit details are in
`/tmp/opencode/simplify-docs-audit.log`. Local paths are evidence locations, not
requirements for normal installation.
Project package artifacts: `/tmp/opencode/nexuml packaged smoke RfcWil/`.
Tool package artifacts: `/tmp/opencode/nexuml packaged smoke 1V707U/`.
Both runs passed on their first attempt, in 4.6 and 4.2 minutes respectively. Each
completed the CPU example, save/reopen, training/reconnect, artifact/export inspection,
offline/dependency invariants, runtime selection, actual hosted HTTP/WebSocket protection
and owned/attached process cleanup. No installed package changed during either workflow.

## Reproduce

Use the [contributor environment](../../../../docs/development/install.md), not an
unrelated inherited Python environment:

```sh
.venv/bin/python skills/tests/check_skills.py
.venv/bin/python skills/tests/check_runtime.py
.venv/bin/python -m pytest tests/test_skill_checks.py tests/core/test_base_layer.py \
  tests/core/test_post_train_layer.py tests/tuning/test_optuna_tuner.py \
  tests/api/test_catalog.py tests/api/test_config.py -q -rs
.venv/bin/python -m pytest tests/api -q
.venv/bin/ruff check src/nexuml/api/worker.py skills/tests/check_*.py \
  tests/api/test_catalog.py tests/test_skill_checks.py \
  tests/core/test_post_train_layer.py tests/tuning/test_optuna_tuner.py
.venv/bin/ruff format --check src/nexuml/api/worker.py skills/tests/check_*.py \
  tests/api/test_catalog.py tests/test_skill_checks.py \
  tests/core/test_post_train_layer.py tests/tuning/test_optuna_tuner.py
.venv/bin/ty check --python .venv/bin/python src/nexuml/api/worker.py \
  skills/tests/check_skills.py skills/tests/check_runtime.py tests/api/test_catalog.py \
  tests/test_skill_checks.py tests/core/test_post_train_layer.py tests/tuning/test_optuna_tuner.py
DISABLE_MKDOCS_2_WARNING=true .venv/bin/python -m mkdocs build --strict
npm --prefix studio run typecheck
npm --prefix studio run lint
npm --prefix studio test
```

For the full API run above, install the backend extras from the project's declared
ranges into the chosen environment and set `NEXUML_API_WHEEL` to the matching built
core wheel before running pytest. This enables the opt-in built-wheel startup test.

For installed acceptance, build both Python distributions and one Studio tarball from
the same checkout. See [package commands](../../../../docs/how-to/development.md#studio-and-skill-checks).
Run `scripts/studio-test-runtime.py --help`, then prepare **new** project and tool
environments with the matching core/API and library wheels. Export each environment's
`runtime.json` values (`STUDIO_TEST_PYTHON`, `STUDIO_TEST_NEXUML`), then from `studio/`:

```sh
TMPDIR=/tmp/opencode STUDIO_KEEP_SMOKE=1 CUDA_VISIBLE_DEVICES='' \
  npm run test:package -- /path/to/the-same-built-studio.tgz
```

Run the modes sequentially so their generated browser reports do not overwrite an active
run. The harness installs the tarball into a clean directory and covers CPU training,
save/reopen, source-bound builds, reconnect, checkpoint export/artifact inspection,
offline browser assets, explicit/PATH runtime selection, unchanged installed Python/npm
metadata, hosted HTTP/WebSocket authentication, and owned versus attached shutdown.
Traces remain off because they can contain credentials. Installation may fetch packages;
normal startup never provisions Python or modifies installed dependencies.

## Scope, retained coverage and limits

- Studio README is 32 lines, down from 386. Combined Studio/API guidance is 332 lines,
  down from 711; one canonical tutorial/runtime/hosting guide retains trusted-code,
  authentication, filesystem, conflict and process-ownership limits.
- The authored-doc cleanup removes 437 lines net, excluding the pre-existing OpenChamber
  addition. Existing support implementations shrink by 100 lines, with focused regression
  tests added separately. This diary replaces 500 lines of superseded step-by-step history.
- Human-facing release/prerequisite numbers and candidate-install detours were removed.
  Skill metadata, executable component/config identities, endpoints, numerical settings,
  dates/ports, source citations and developer TestPyPI publishing remain intentional.
- [Skill coverage mapping](../../../../skills/tests/README.md) records every moved
  fitted-state/checkpoint/tuning assertion. The documented example, minimal-library
  starter, manual workflow grader, eval fixtures and ignored workspaces remain intact.
- Draft cleanup retains file conflict revisions, layout matching and semantic build
  signatures. Graph checks retain domain/order/overwritten-producer/evaluator-default
  rules without a global cache. Observation cleanup retains independent metric/terminal
  bounds, replay cursors/gaps, reconnect and terminal invalidation; disk history is unchanged.
- Original cross-platform tasks 7.2 and 8.2 remain unchecked. macOS/Windows jobs are
  disabled by the maintainer; local Linux proof does not certify those platforms or GPUs.
  No live Ray/Kubernetes submissions, new CI dispatch, publication, tag, merge or deploy.
- Other optional-dependency/type-environment and npm development-security follow-ups
  remain outside cleanup. Project dependency declarations, manifests, lockfiles and
  suppressions are unchanged. The authorized environment-only install added the backend
  dependencies and updated `packaging` transitively; existing framework packages were untouched.
- Release-owner protected environments/tags, npm scope/Trusted Publisher setup and
  institute runner isolation still require owner review; see
  [release guidance](../../../../docs/development/release.md).

## Historical evidence

Earlier local and hosted passes are historical, not passes for this cleanup. The previous
UX-polish CI is [run 39495926080](https://github.com/NexuFed/NexuML/actions/runs/39495926080);
its commit differs from the working tree above. The full superseded diary is recoverable
with `git show a3c2ca1:openspec/changes/add-local-nexuml-studio/verification.md` and in the
retained local evidence. Old screenshots, logs and installation artifacts were not deleted.
