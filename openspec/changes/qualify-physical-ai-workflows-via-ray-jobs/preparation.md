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
