# Non-Ray preparation

## Current instruction and ownership

The user has now approved revising this change to **rebase onto `dev` and preserve this branch's working SDK Jobs behavior within `dev`'s registered execution structure**. Register the SDK mode as `ray-jobs-sdk`, keep the existing KubeRay `ray-job` distinct, use shared Python/CLI/API dispatch and migrate to `type`/`version`/`params` without a legacy second schema. Preserve the existing setup/cache/private-venv/one-GPU defaults, optional prepared environment and asynchronous submission contract.

All acceptance jobs must use `pool_gpu_workers`, never DGX or a separate CPU-worker pool. The maximum is four in-flight jobs, limited by qualified capacity, with default whole-GPU reservations even for CPU controls. Retain the local-venv/data/evidence protections, no-local-GPU boundary and monitored 4 GiB physical free-disk stop.

This approval supersedes the earlier no-merge/no-migration ownership restriction below. It authorizes a backed-up rebase, not force-push, new implementation commits/publication, PR merge, archive, infrastructure changes or shared-environment mutation. The current revision is planning-only; the rebase, registered SDK port and remote qualification remain pending under the revised tasks.

## Historical preparation instruction (superseded for integration)

The user directed this apply to keep using this checkout's existing `.venv` and
NexuML SDK `ray_job` backend, which another agent is currently qualifying. Prepare
independent work first. Do not migrate execution schemas, merge `dev`, edit Ray or
CLI dispatch, install packages, provision infrastructure or interfere with jobs.

The base-contract discrepancy in [baseline.md](baseline.md) remains recorded,
not resolved by adding another backend or changing the user's execution choice.

## Completed tasks

- **1.1:** delivered candidate/base/import/CI baseline and reconciled all 21 RL
  recipes plus SO-101/supervised controls. No current remote gate is claimed.
- **4.2:** requested TensorBoard/DVCLive/MLflow initialization now fails visibly,
  including lazy experiment and requested MLflow system-metrics failures. Earlier
  loggers are finalized as failed; a new MLflow child is marked failed if its
  Lightning logger cannot take ownership. An explicitly supplied existing run is
  not terminated by that construction failure. Cleanup errors do not replace the
  primary error. Created run IDs survive even successful cleanup; a newly created
  auto-parent's preparation failure also attempts failure finalization, without
  touching a pre-existing parent. Disabled logging creates no writer. Partial artifact/preview
  delivery errors identify the failed destination and retain delivered outputs.

## Prepared, not marked complete

- **4.1:** tracking banners use quoted paths, loopback hosts, separate suggested
  TensorBoard/MLflow ports (6007/5000) and an actual MLflow run ID. Remote HTTP(S)
  tracking uses the existing service, not another suggested UI server. Commands
  are suggestions, not started listeners or reserved ports. Job/attempt/phase,
  remote access and completion/recovery integration remain pending with the
  execution/CLI owner.
- **4.3:** policy export records known physical `training_frames` from the owning
  live RL module/trainer or an explicit checkpoint's saved scalar counter. It
  never substitutes optimizer steps, agent counts or requested frame budgets.
  Missing counters remain null. Policy state-file and checkpoint SHA-256 hashes
  are embedded in package metadata. Actor-only policy output and supervised
  checkpoint sidecars are preserved. Final CLI receipt/hash handoff, current
  remote train/resume/replay and installed-distribution acceptance remain pending.

## Verification boundary

All checks use this checkout's `.venv/bin/python` with `CUDA_VISIBLE_DEVICES=''`.
Synthetic CPU regression fixtures are not the remote CLI/native/browser acceptance
matrix and are not learning-quality evidence. No Ray, native simulator, browser,
GPU, real-data capture or training acceptance job was launched by this apply.

Focused checks:

```text
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/tracking -q
CUDA_VISIBLE_DEVICES='' NEXUML_RUN_SIMULATOR_TESTS=0 NEXUML_RUN_SYSTEM_TESTS=0 NEXUML_RUN_HARDWARE_TESTS=0 NEXUML_RUN_DATA_TESTS=0 .venv/bin/python -m pytest tests/core/test_policy_export_provenance.py tests/tracking tests/core/test_export.py tests/core/test_policy.py tests/core/test_environment_job_scope.py tests/core/test_model_evaluation_cli.py -q -rs -k 'not export_data_module and not cifar' --tb=short
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m ty check --python .venv/bin/python src/nexuml/core/export.py src/nexuml/tracking/logger.py tests/core/test_policy_export_provenance.py tests/tracking/test_logger_initialization.py tests/tracking/test_logger.py tests/tracking/test_evaluation_media.py
ruff check src/nexuml/core/export.py src/nexuml/tracking/logger.py tests/core/test_policy_export_provenance.py tests/tracking/test_logger_initialization.py tests/tracking/test_logger.py tests/tracking/test_evaluation_media.py
ruff format --check src/nexuml/core/export.py src/nexuml/tracking/logger.py tests/core/test_policy_export_provenance.py tests/tracking/test_logger_initialization.py tests/tracking/test_logger.py tests/tracking/test_evaluation_media.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m mkdocs build --strict --site-dir /tmp/opencode/nexuml-physical-ai-preparation-site
openspec validate qualify-physical-ai-workflows-via-ray-jobs --strict --no-interactive
```

Development logs stay under `/tmp/opencode/nexuml-physical-ai-preparation-*` and
are not durable release evidence. Source changes are uncommitted. Do not claim
final-candidate CI, full-suite success or remote acceptance from these checks.

Final focused result: **242 passed, 1 GPU test skipped, 7 unrelated data-export/
CIFAR tests deselected**. Existing dependency warnings remain; no selectors or
test markers were changed to conceal them. Scoped types, Ruff lint/format, strict
documentation build, diff whitespace and strict change validation pass. The
documentation tool still prints its upstream MkDocs 2 advisory.

Independent read-only review found two MLflow ownership/identity gaps. New red
tests reproduced both; the fixes exercise the actual preparation path, original
exceptions, cleanup success/failure and pre-existing-run protection. The reviewer
confirmed both fixes with 9 targeted CPU checks and no remaining scoped findings.
This is not the full final integration review in task 8.3.

Progress: **2/36 tasks complete** (1.1 and 4.2). Tasks 4.1/4.3 remain unchecked
despite the prepared source changes above. Ray, CLI, execution types, dependency
manifests/lock and CI files have no diff from this apply; no commits, pushes,
backend migration, environment setup or external issue/PR updates occurred.

## Handoff to the execution/CLI owner

- The prior instruction to avoid execution-schema migration is superseded by the approved registered SDK port above. Preserve these prepared tracking/export changes and their staging state through the backed-up rebase; they are not permission to qualify a different backend or reuse historical GPU evidence for the migrated candidate.
- Export callers already supplying `result.lightning_module` or `result.trainer`
  can obtain actual source frames without extra learner/simulator work.
- Checkpoint export should supply the explicit source `checkpoint_path`; its
  saved physical counter takes precedence over a fresh module's zero counter.
- Existing evaluation reads `metadata['training_frames']`; retain null/unknown
  separately from backend presentation step zero.
- Preserve `policy_files_sha256` and `checkpoint` source/hash in final receipts;
  do not reconstruct model identity from the configuration hash alone.
- Requested tracking failures are now fatal by design. Provision the selected
  dependencies/stores remotely or explicitly disable that backend; do not add a
  fallback or automatic install to hide a failed selected tracker.
- Continue using the existing backend; remote lifecycle, isolation, viewer,
   learning, recovery and final release gates stay unchecked until real proof.

## Resumed registered-SDK implementation

The user subsequently authorized committing the reviewed changes and continuing
implementation until complete or genuinely blocked. Local implementation commits
are authorized; push/force-push, external status updates, PR merge, archive and
infrastructure changes are not.

- The reviewed tracking/export changes and revised plan were committed separately.
  The exact pre-rebase history is retained at
  `backup/nex-248-before-registered-rebase` (`a0140abca3af6717a98839c6c1cebc2f20f0192f`).
  Original staged/index patches and ref identities were also retained under
  `/tmp/opencode/nexuml-registered-rebase-x1mtazcp/` before committing.
- The actual fetched `dev` advanced to
  `ce31f37f56b44a42c959679f09db4668ce800b02`. The authorized rebase completed with
  merge topology preserved; that base is an ancestor of the current branch.
  Published feature history has not been rewritten remotely.
- `ray-jobs-sdk` now uses registered definitions and shared dispatch, retaining
  locked setup/private-venv/cache/GPU defaults and distinct KubeRay behavior.
  Train/resume/evaluate/export select execution before worker artifact reads;
  unsupported operations fail shared preflight instead of launching training.
- Checkpoint export now uses canonical full checkpoint restoration, including
  stateful adapters, and finalizes owned loggers. Observer initialization and
  provenance/export failures remain inside native job scope and cleanup.
- Fixed acceptance requests reuse the same SDK transport. A stock-Python guard
  wraps setup as well as execution; the test-owned driver/worker cases retain
  attempt ownership, bounded monitoring, explicit phases and revision-bound
  receipts. Unprepared distribution/SO-101/browser/recovery profiles remain
  blocked, never passed by a stub.
- Hermetic submission/lifecycle/guard/inventory regressions, scoped static checks,
  strict docs, the merged lock and source/wheel builds passed. These are source
  checks, not remote application, installed-distribution or native qualification.
  The local prepared environment is unchanged; full local typing still lacks
  `dev`'s optional Kubernetes dependency and is not claimed green.

Deployment inputs observed read-only: the approved HTTPS endpoint/CA are reachable
and report Ray 2.57.0. GPU workers use stock
`rayproject/ray:2.57.0-py312-cu128`, one-GPU scheduling and RTX-only node affinity,
not DGX. The current head plasma limit is 1,329,312,153 bytes; its shared-memory
usage was 24,576 bytes before any owned jobs, and no workers were active. Worker
shared-memory volumes are capped at 8 GiB. Initial qualification is serial, within
the aggregate 10 GiB safety budget; a four-job scheduler maximum is not proof that
four concurrent allocations satisfy that budget. Actual worker resource/profile
qualification precedes parallel claims. Durable attempts use private roots under
`/mnt/data/nexuml-qualification`, with the monitored physical 4 GiB disk floor.

Progress is **6/36**: 1.1, 2.1, 2.2, 3.1, 3.4 and 4.2. Runner tasks 3.1/3.4 were
reopened after safety review, then corrected and independently re-reviewed:
pre-setup root validation, persisted UUID submission ownership with lost-response
reconciliation, dual scratch/output monitoring and actual imported-source admission
are covered by targeted regressions. The integrated hermetic suite passed with
362 tests, two real-training cases deliberately reserved for remote execution and
82 subtests; scoped Ruff/ty and strict OpenSpec validation passed. An inert preflight
plan selects one GPU worker, a private durable attempt, locked setup and explicit
one-hour/4 GiB floor budgets, with unknown data/native/display inputs left unknown.
These checks establish runner behavior, not a qualified remote profile. All current-candidate
application/native/browser/learning/distribution gates remain unverified before
the first remote preflight. Old production smoke evidence retains its original
revision and is not reused as proof of this migration.

## Frozen candidate and first remote blocker (superseded diagnosis)

The integrated migration/guard is committed locally as
`2f51aa4f8ba7bed75b0b353dcacbf53a80372286`, without publication. Its clean source
identity is `f6233d945885513851ffa4bd5c2789cd5f5795c9767ff1c6fd4ca21eed24a3d2`.
Matching core/library wheels and sdists, strict documentation and lock checks passed;
build inputs and hashes are retained outside Git at
`/tmp/opencode/nexuml-2f51aa4-dist/candidate.json`.

The first serial registered-SDK preflight, job
`7ac39cb79f24446c8fc478d588205dcb`, terminated **BLOCKED / native FAILED / exit 2**.
Locked installation into its private environment completed and both actual source
origins matched the uploaded candidate. The worker advertised an object-store cap
of **10,000,000,000 bytes**, above the preflight's **8 GiB worker envelope** reserved
under the aggregate shared-memory budget. The head cap remains 1,329,312,153 bytes.
The worker's plasma directory was `/tmp/ray`, not `/dev/shm`; observed worker
shared-memory use was zero. This is a configuration-envelope blocker, not a claim
that 10 GB of shared memory was consumed. The preflight did not run its CUDA probe,
training, inference, native simulator or browser stages.

Do not relax that check or change the cluster/image/driver implicitly. An operator
must supply a prepared worker/profile with object-store reservations fitting the
recorded aggregate limit before qualification continues. SO-101 approved data,
prepared ROS/Gazebo world/controllers/reset/clock and private display/browser
inputs also remain absent or unqualified; required gates stay blocked/unverified.

The guard preserved the exit, private root and monitored disk floor without cleanup
errors. Physical scratch remained above 320 GB; output JuiceFS statistics were
explicitly classified as distributed quota, not backing physical storage proof.
The worker then disappeared through normal autoscaling; no worker was removed by
this session. Control-plane receipts/logs remain at
`/tmp/opencode/nexuml-registered-preflight-20261024/`. The durable attempt is
`/mnt/data/nexuml-qualification/e2c815c1130d483da0f2a50e9c95fdf3/preflight-ea5a70721a3a4ecb856736b91039df9d`.
Post-worker-removal retrieval could not be verified from the control host, which
does not mount that path; no persistence pass is claimed. No retry or subsequent
acceptance job was submitted. Progress remains **6/36**.

## Plasma-accounting correction and passing base preflight

The user asked to resolve the blocker. Investigation showed that the preflight
incorrectly charged the entire object-store cap to shared memory even when plasma
was confirmed on ext4. The previous request for an operator cap reduction is
superseded; no cluster change or larger memory allowance was necessary.

Fix `65b9d01b8e823a2d6ef71eb0292cc311d4fb1bb2` records plasma's actual filesystem
from the raylet's mount table, requiring matching mount namespaces before resolving
paths. Only confirmed disk backing is excluded from the shared-memory reservation;
missing, ambiguous, overlay or otherwise unknown backing remains blocked. The 8 GiB
worker shared-memory envelope, aggregate 10 GB limit and monitored 4 GiB disk floor
are unchanged. Review confirmed the namespace/overlay protections. The integrated
hermetic suite passed with 365 tests and 91 subtests, with the same two real-training
cases reserved for remote execution. Ruff passed. Thirteen previously present
fixture type diagnostics are unchanged, not suppressed or reported as passing.

Fresh serial GPU-worker preflight `fd56ce6a4fda4e3d9fadac41d2b188c4` succeeded
on this clean candidate, source identity
`12f5060793ffa9321cc8fcda2f2fbe10d6f72c4a8ed31a0b3acb7cb4d09136f5`.
It confirmed locked private setup, matching uploaded source origins, private writable
durable output, disk-backed plasma and a real job-visible CUDA allocation on an
RTX 4090 with driver 580.126.20. Worker shared-memory use and shared-memory-backed
plasma reservations were both zero; the separate disk-backed plasma cap remains
10,000,000,000 bytes. The unchanged head plasma cap is 1,329,312,153 bytes.

Both `case.json` and `guard.json` were retrieved and verified after the producing
process exited. Guard exit was zero with no cleanup errors. Evidence is retained
outside Git at `/tmp/opencode/nexuml-plasma-backed-preflight-20261024/`; the durable
attempt is
`/mnt/data/nexuml-qualification/a002a600c94f4619987cd3660d478140/preflight-1c7c2f7ebd084cf9802fadb7919f401f`.
Post-worker-removal persistence, full regression, installed distributions, native,
browser, learning and complete recovery gates remain unverified. SO-101 data and
prepared ROS/Gazebo/display/browser prerequisites remain unresolved. Keep actual
concurrency one; this preflight is not parallel-capacity qualification. No push,
infrastructure/image/driver or local-environment change occurred. Progress remains
**6/36**; only the base resource preflight blocker is resolved.
