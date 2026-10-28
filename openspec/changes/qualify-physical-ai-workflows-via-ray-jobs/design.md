# Design

## Context

See [proposal.md](proposal.md) for motivation and `specs/*/spec.md` for the four acceptance contracts. [verification-matrix.md](verification-matrix.md) defines the bounded recipe and gate inventory.

Planning baseline: source/published feature head `a812e54e1bd7ac4b52d52ce04f739064d5fe620a`, PR #15 against `dev` `f8d5490ab1446177aa26ba529124310f0b8041f2`. The PR was draft/conflicting; no checks were attached to that new head at inspection. Earlier failed CI on `a5011a0` is diagnostic history, not the new head's result. Re-resolve these identities during apply; they are not permanent pins.

Refreshed before the approved plan revision: local/published head is `851a9bc031645f98a2a33f2683304b6f65c50d89`; the fetched PR base remains `f8d5490ab1446177aa26ba529124310f0b8041f2`. The PR remains draft/conflicting with no checks on that head. The user now explicitly authorizes rebasing onto `dev` and adopting its registered execution structure while retaining this branch's working SDK Jobs behavior. The approved SDK identity is `ray-jobs-sdk`; `dev`'s KubeRay `ray-job` stays distinct. GPU workers only and at most four concurrent jobs supersede the earlier two-case deployment limit.

Relevant existing seams:

- `src/nexuml/execution/ray_job.py` serializes a resolved scenario, changes remote execution to local, optionally runs quoted setup and submits a bootstrap calling the ordinary training CLI. It returns a submission ID. It currently rejects remote trainer-checkpoint resume and has no equivalent early remote evaluation/export dispatch.
- `dev`'s `execution/definitions.py`, `execution/service.py` and registered model serialization own execution definitions, shared preflight/dispatch and the `type`/`version`/`params` representation. Its `ray-job` definition submits a KubeRay CRD through Ray Train, not this branch's SDK Jobs process. The current branch's closed `ExecutionSpec` union must be removed during the port, not retained beside the registered schema.
- `src/nexuml/cli/main.py` already owns scenario selection, overrides, native job scopes, exports and inference-only evaluation. Evaluation loads its package before considering supplied evaluation inputs, so remote selection must be moved ahead of that load without changing worker evaluation semantics.
- `src/nexuml/tracking/logger.py` owns native writers/artifact dispatch. Its banner suggests wildcard bindings and overlapping MLflow/Viser ports; logger initialization can warn and return no logger. Existing media already uses bounded buffers and native GIF previews/artifact delivery.
- `src/nexuml/core/export.py` exports complete actors without training-only state. Replay can lack `training_frames`, honestly falling back to presentation step zero; add source progress, not optimizer/checkpoint payloads, to policy provenance.
- Library Gymnasium configuration forwards native kwargs, including human rendering. Isaac already has qualified native/browser fixtures. ROS/Gazebo depends on an operator-owned world/controllers/reset bridge; fake transport is not a live proof.

The completed recorded-evaluation and focused Isaac A–D changes remain implementation dependencies, not new implementation queues. Keep their history and the umbrella's unfinished model/robot/federation work intact.

## Goals / Non-Goals

**Goals:**

- Close the user-facing release gaps using one execution transport, existing CLI commands and fixed, reviewable acceptance cases.
- Keep all application acceptance execution, native graphics and automated browser work remote. Preserve the local GPU for the user.
- Run up to four independent cases on `pool_gpu_workers` while preserving serial native lifetime and checkpoint/data dependencies within a case. Do not use DGX or a separate CPU-worker pool; CPU controls still run inside default one-GPU jobs.
- Produce current-candidate proof that distinguishes correct wiring, usable viewing, measured learning and recovery.

**Non-Goals:**

- No job scheduler service, general remote shell endpoint, new training engine, simulator renderer, dashboard or arbitrary workflow language.
- No distributed learner/GPU collector enablement, cluster provisioning, base-image/driver changes, hardware control or automatic model promotion.
- No new ACT/VLA/world-model/Meta-World/Playground implementation and no reinterpretation of existing supervised package sidecars.

## Decisions

### D1. Stabilize and identify the candidate before release qualification

Re-resolve the actual PR base, source origins and CI failures. Rebase onto `dev` under the user's explicit authorization. Before rewriting commits, retain a recoverable pre-rebase ref and an exact backup of staged/unstaged work and its index state; restore the pending tracking/export/planning changes without dropping unrelated work. Preserve `dev`'s execution infrastructure and this branch's Physical AI behavior rather than resolving conflicts wholesale to either side. Diagnose package, supported-Python and default-suite failures rather than broadly deselecting them. Existing native optional profiles need not support every published Python minor, but ordinary package support must match tested metadata; do not silently drop a minor to make CI green.

Development probes can use an identified source-tree snapshot. Release evidence must bind an immutable candidate commit, source/wheel hashes and exact runtime selection. After any relevant patch, invalidate only affected gates and retain prior results under their actual revisions. Rebasing an already published branch changes its commit identities; do not force-push. Publication requires a separate authorized non-forced route, such as a new branch/PR. A locally passing candidate is not a remotely green PR.

Alternative rejected: running expensive acceptance before integrating the base, then attributing those results to a changed merge candidate.

### D2. Extend the existing Ray Jobs transport, not the training architecture

Register `ray-jobs-sdk` as an `ExecutionBackendDefinition` within `dev`'s existing catalog. Preserve its separate `ray-job` KubeRay definition and all other backends. The SDK definition owns preflight and launch through `execution.service`, shared by Python, CLI and API; do not keep frontend-only type checks as a second dispatch system. Settings/discovery/preflight must remain lazy and must not submit jobs, install packages or allocate a local learner/native runtime.

Port this branch's `RayJobExecutionSpec` settings and validation into the registered SDK definition. Preserve default `uv sync --locked`, one GPU, worker-local cache/download settings, per-submission `{run_id}` venv/interpreter resolution without caller mutation, explicit override precedence and `setup_command=None` prepared-environment behavior. Pool tags, dependency extras, address/TLS and durable output paths remain configurable; the approved campaign selects `pool_gpu_workers`. Do not replace SDK submission with KubeRay, Ray Train or another installer/runtime.

Factor the existing small SDK submission/quoting/setup block into one shared transport inside `execution/ray_job.py`. Keep `submit_ray_job(scenario)` as a public default-training convenience returning the same asynchronous string ID, routed through the registered backend rather than duplicating transport. Add only operation-specific inputs for allowlisted normal CLI invocations: train, train-resume, export and evaluate. Preserve argument boundaries and validate mutually exclusive inputs before submission; existing worker CLI validation remains authoritative for command semantics.

Migrate scenario types, config/profile serialization, Python callers, CLI overrides, fixtures, exports and documentation together to registered `type`/`version`/`params` settings. Reject the old `execution.kind` schema; do not add a compatibility parser or keep the closed execution union. Preserve the optional execution-profile precedence, using a trusted registered SDK definition rather than the old mapping. The remote bootstrap serializes the registered local definition, removes submission credentials/settings and cannot recursively submit another job. Preserve actor-only inference and existing local/supervised behavior across this migration.

Reuse scenario selectors rather than introduce a separate execution configuration system. Add consistent `--config` selection for export/evaluate where absent; retain existing positional/package and `--scenario-file` behavior. Resolve the execution definition from those explicit inputs before loading a checkpoint/package or constructing a session on the submitter. Evaluation config supplies only its existing allowed evaluation/data/logging/environment settings, never replacement packaged weights. Remote resume requires an explicit execution scenario/config; checkpoint-only local resume stays unchanged.

Planned public journeys, after implementation, use worker paths:

```text
nexuml train --config CASE_JOB.yaml
nexuml train --config CASE_JOB.yaml --trainer-checkpoint REMOTE_CHECKPOINT
nexuml export --config CASE_JOB.yaml --checkpoint REMOTE_CHECKPOINT --output REMOTE_POLICY
nexuml evaluate REMOTE_POLICY --config EVALUATION_JOB.yaml --device cuda:0
```

Each driver command selects registered `ray-jobs-sdk` settings and submits; the worker receives the registered local definition and invokes the existing CLI. `CASE_JOB.yaml` / `EVALUATION_JOB.yaml` are inert, trusted operator configuration on the submitter; model/data/checkpoint paths inside them are explicitly remote. A checkpoint's existing restore path continues to own effective resume semantics and rejects incompatible changes. No automatic checkpoint/private-dataset upload is added. Existing historical artifacts remain intact; any unsupported old execution metadata must produce an explicit diagnostic rather than silently selecting a backend.

The acceptance runner uses this same `ray_job` transport for a fixed set of worker fixture entrypoints as well. These test-owned entrypoints launch public CLI subprocesses and supported capture/SDK scripts; they are not a new arbitrary-command scenario component. In particular, no acceptance run is dispatched with a separately implemented raw SDK/`ray job submit` shortcut and called backend qualification.

Alternatives rejected: repurposing `py_executable` or `setup_command` as a hidden test runner; exposing arbitrary shell pipelines; loading remote artifacts on the driver to discover routing; creating Ray Train placement or another trainer.

### D3. A small bounded campaign coordinates independent jobs

Use one test-owned driver, proposed `scripts/physical_ai_acceptance.py`, and fixed worker cases under existing `tests/fixtures/`. The driver resolves case definitions, calls the shared backend, follows returned IDs through native status/log/stop APIs and writes a result summary. Worker cases own their subprocess deadlines and assertions. Reuse existing native probes/oracles rather than copy simulator or algorithm logic. Do not add a daemon, database, dependency graph DSL or persistent retry queue.

Use the operator-approved maximum of four in-flight cases, explicitly recorded and further constrained by qualified capacity. All jobs use `pool_gpu_workers`, never `pool_dgx_workers` or a separate CPU-worker pool, and retain the default one-whole-GPU reservation. CPU controls may hide CUDA within that allocation. A submitted/pending/setup case occupies a slot until terminal or cancelled. Use job-visible `cuda:0`, not a node-global GPU index. Initial worker qualification is serial; then demonstrate at least two genuinely overlapping independent remote cases and record allocations. If two suitable GPUs are unavailable, the parallel-native gate is blocked; do not fake overlap with queued jobs, assume four GPUs are available or oversubscribe fractions.

Within a case, collect/fit, interrupt, resume, export and evaluation run sequentially in fresh child processes where native lifetime requires it. Separate top-level submission cases also exercise the user-facing remote resume/evaluate/export dispatch itself. Child processes remain within the parent's allocation and do not recursively submit jobs. Representative CPU process/Ray collector checks retain their existing topology restrictions and explicitly budget all remote actors; native Isaac remains direct collection.

Alternative rejected: launching all recipes unbounded or enabling distributed RL to obtain concurrency. Parallel experiments are not distributed learners.

### D4. Separate mutable state and explicitly provision durable paths

An operator supplies the existing endpoint/CA, worker-pool constraints and durable storage root. An example layout is `DURABLE_ROOT/campaign/case/attempt/`; actual mount/service values come from that deployment, not new library defaults. Allocate unique checkpoint, export, logs, local tracking database, diagnostic and evidence locations beneath the attempt. Set `NEXUML_LOGS_ROOT` for relative logging and explicit `ExportSpec.output`/checkpoint destinations for artifacts it does not redirect.

Use the ported SDK backend's default locked setup and private per-submission venv/interpreter; select required extras in project configuration. A prepared compatible environment remains an explicit opt-out with `setup_command=None`, not an implicit replacement for those defaults. Ray may cache/reuse uploaded working directories, so `.venv` in that shared directory is not a safe per-job isolation guarantee. Share only appropriate package/asset caches, with normal upstream locking, never writable interpreters or SQLite databases. Compatible existing tracking servers can be shared through distinct native run IDs; otherwise use separate per-attempt stores, not one SQLite file on a shared mount.

Assign nonconflicting viewer/UI/display ports on the actual worker host and separate ROS domain/namespace and Gazebo transport/world identities. Validate reservations before startup; fail on conflicts instead of silently moving a port. GPU reservation alone provides none of this isolation. Ports such as KubeRay's metrics 8080 are infrastructure-owned, not ours to replace.

A completed attempt must retrieve/check required hashes after its processes exit, and a representative attempt must survive worker cleanup. Missing input mounts or an ephemeral-only output root are preflight failures, not permission for local execution. Private LeRobot data is supplied through an approved remote location; never implicitly ship it with the source bundle.

Alternative rejected: treating the shared download cache, uploaded project directory or log-root variable as persistent, isolated run storage.

### D5. Preflight resources on the worker and keep the submitter GPU unused

The submitter runs with `CUDA_VISIBLE_DEVICES=''` and performs only configuration, submission, monitoring, retrieval, CPU-only static checks and review. Dispatch tests block model/checkpoint loaders, CUDA initialization and native constructors. Automated browsers run on remote CPU resources; local human review of downloaded media or a forwarded browser interface is not local simulation/training.

Each remote profile checks actual interpreter/import origins, source and wheel hashes, backend/renderer, GPU access where requested, display/codec/browser availability, approved assets/licences, data mounts and writable durable outputs. Preserve the existing monitored 4 GiB free-disk floor and declare per-case wall time, CPUs, GPU count, memory/VRAM envelope and output cap. Setup must fit the same approved budget; native capacity is not inferred from an advertised resource label.

Use bounded owned-process-group cancellation and native job IDs. Never delete another run, kill another worker, rebuild shared images, alter `/env`/drivers or provision infrastructure to pass a gate. TLS remains verified and credentials are environment/service-owned, redacted from receipts and source bundles.

Alternative rejected: probing native support on the local GPU first or falling back locally when a remote worker is unsuitable.

### D6. Inventory every existing recipe, but do not test a Cartesian product

Freeze the discovered recipes and public examples against [verification-matrix.md](verification-matrix.md). Give each supported recipe its own short CLI round trip and retain separate visual/quality/recovery dimensions. Run representative cross-feature tests for the existing supported sensor processing owners, action adapters, shared/independent actors and logging modes. A combination not natively qualified remains unqualified; CPU tensor routing cannot expand support claims.

Add a supervised synthetic control and installed-wheel actor inference as CPU workloads inside the approved GPU-worker jobs. The SO-101 row reuses its documented capture/fit/evaluate scripts and direct SDK boundary; report that route honestly, not as if every SDK call were `nexuml train`. Preserve the existing validation/physical semantics and original episode files. If approved remote data is absent or implementation provenance differs, block or perform the existing explicit requalification process; do not auto-approve a receipt or silently recapture.

ROS/Gazebo live viewing is a required new journey with an explicit prepared-worker gate. CARLA remains an inventoried external integration boundary unless its live prerequisites and scope are separately accepted. Required rows cannot be changed to optional just because infrastructure is absent. Any maintainer decision to ship a narrower set must be explicit and reflected consistently in the proposal, inventory and user claims.

Alternative rejected: rerunning every modality × backend × algorithm × logger × codec combination, or claiming every named scenario is supported from static discovery.

### D7. Make actual browsers and native viewers part of acceptance

Reuse the existing media implementation and browser fixtures. For representative cases, open real TensorBoard events, select changing GIF image previews at identified source frames, and reopen retained logs. Inspect actual MLflow run metrics/status and retrieve playable selected GIF/MP4/JSON artifacts. Compare finalized payload hashes across enabled backends. Verify TB-only, MLflow-only, both and disabled operation; do not require a particular UI version to inline MP4 when a working native download/player path is the documented behavior.

For Isaac, keep the qualified private spectator and its real camera/display/selection/reconnect/mutation-denial tests. Keep browser software rendering distinct from server sensor/RTX qualification. For native MuJoCo, use the upstream human-rendering route in the existing Gymnasium provider on a prepared remote display; use a separate compatible RGB-array evaluation definition. For Gazebo, use the real supplied world, robot, controllers, reset bridge and clock; verify that the GUI displays the same worker/world/namespace being controlled by the policy.

MuJoCo/Gazebo native GUIs are not automatically read-only. Disclose their physical controls; quantitative trials admit only validated nonintervening operation, while sessions accepting pause/reset/force edits are debug-only and excluded from quality results. Do not build another renderer or promise an isolated spectator for every upstream GUI. An existing private remote-desktop/viewer route must actually be exercised for desktop viewers; a virtual display merely launching is not proof that the user can see/interact remotely. Missing infrastructure is a blocker, not authorization to create new public viewer services.

Inspect bounded raw/processed sensor snapshots with units, labels and history order, plus existing true-final/reset oracles. Use semantic scene assertions and a human visual spot-review; dynamic pixel-exact screenshots are not stable correctness or learning oracles.

Alternative rejected: treating decodable black videos, screenshots of a different world, open sockets or headless native tests as browser/live-view acceptance.

### D8. Respect native lifetime when comparing learning stages

Gymnasium references can reuse qualified initial/periodic/final evaluation. For Isaac, retain selected early/intermediate/final checkpoints during training and replay them sequentially in fresh remote processes after training exits. Do not enable unsupported in-fit evaluation or a second native context. Every clip carries its actual checkpoint/policy hash, physical frames, reset seeds and cap/horizon labels; a live training view is not the same artifact as a frozen policy evaluation.

Freeze the existing showcase protocols before fitting: state-only Isaac Cartpole retains the documented native five-second/300-step held-out test and single-training-seed scope; SO-101 retains the exact five-action/three-goal task, episode-disjoint fit and matched expert/untrained/random/learned evaluation. Details and measured acceptance thresholds are in the matrix. Other recipes demonstrate finite real updates and replay, not convergence. No extra seed sweep or model/embodiment is silently added to finish this PR.

Alternative rejected: adjusting horizons/seeds after seeing poor results, selecting only attractive episodes, or calling a 64-frame MARL wiring run a learned solution.

### D9. Refine existing reporting and failure behavior only

Extend the existing console banner, CLI completion text and JSON artifact metadata. Connect job/case/attempt, native logger run IDs, candidate/config identity, actual source frames and durable paths. Clearly distinguish submission from training, return from task success, timeouts from action rejection, and first recorded episode from aggregate metrics. Publish exact worker resume/replay commands and private access instructions without starting extra UI services.

Correct wildcard-host suggestions and MLflow/Viser port overlap. A phase is emitted only at an observed transition; opaque Ray pending/setup state is explicitly unknown, not a fake training heartbeat or ETA. A finished fast job prints the separate viewer replay route rather than pausing/delaying physics. Track source frames from actual saved counters in policy metadata; unknown old-package values remain unknown even when the backend presentation step is zero. Never export learner optimizer/critic/history state just to add provenance.

Fail initialization of an explicitly requested logger with its original diagnostic and cleanup. Disabled logging still works. Artifact delivery failures preserve successful prior outputs and identify the failed destination; do not add silent logger fallbacks, automatic retries or cross-store rollback. Test these changes across ordinary supervised and RL paths because they share tracking.

Alternative rejected: a new dashboard, logging wrapper hierarchy, separate progress service or implicit tracking server.

### D10. Keep source, evidence and readiness separate

The bounded driver writes one per-attempt JSON receipt and per-stage logs, then a campaign summary in the existing artifact/report style. Records include source/tree/wheel/runtime identities, resolved configuration with secrets removed, remote job/allocation, actual argv/exit/status, counter and parameter-change assertions, artifact hashes, metrics, browser traces/screenshots and independent gate states. Store raw episodes, weights, run logs and media outside Git; never claim `/tmp/opencode` alone is durable release evidence.

Reuse existing hermetic tests locally with CUDA hidden and existing remote opt-in markers/CI tiers. No automatic live cluster submission on every ordinary PR test. Required static/test/package checks must be green, while required remote matrix results must exist for the final candidate; a skipped optional workflow is not that proof. An independent read-only review checks integration, secret/output handling, package boundaries and claim-to-evidence mapping before readiness. Update PR/Linear only when separately authorized; do not close the full NEX-248 roadmap.

Alternative rejected: embedding test counts in planning files as completion evidence or automatically uploading private run artifacts with the implementation commit.

## Risks / Trade-offs

- **[Insufficient remote resources]** → Explicit four-job maximum on GPU workers, full-GPU reservations and capacity-limited scheduling; missing parallel/native capacity stays blocked with no local/DGX fallback.
- **[Rebase/config migration loses behavior or user work]** → Recoverable pre-rebase ref and exact index/worktree backup; migrate every reader/writer/caller/test/doc to the registered SDK definition and verify defaults/no-local-allocation before acceptance. No force-push or legacy second schema.
- **[Native/display/ROS prerequisites differ by worker]** → Qualified existing profiles and real worker preflight; no silent backend, GUI or dependency substitution.
- **[Parallel installation or tracking corrupts shared state]** → Prepared read-only environments or unique owned venvs; separate file stores, namespaces, ports and attempt roots.
- **[Remote worker dies after training]** → Explicit durable destinations and retrieval checks; checkpoint/export ownership remains visible even when tracking delivery fails.
- **[Early remote dispatch changes local CLI semantics]** → Additive selectors, default training API retained, worker CLI authoritative, local regression and no-submit tests.
- **[Stricter logger errors break previously permissive runs]** → Document the intentional failure-behavior change and preserve explicitly disabled logging; do not add a silent compatibility mode.
- **[Mutable GUI alters the experiment]** → Label debug interactions and exclude them from scoring; preserve the existing proven Isaac spectator contract rather than generalize it untested.
- **[Long case causes scope creep]** → Bounded recipe inventory, existing showcase protocols and explicit blockers; no new algorithms, hardware or infrastructure project.

## Migration Plan

1. Record/reconcile the current base and failures, back up user/index state and pre-rebase history, then perform the authorized rebase onto `dev`.
2. Port the current SDK behavior into registered `ray-jobs-sdk`, migrate all config/caller/test/doc surfaces and remove the old execution schema. Add shared remote lifecycle dispatch and no-local-allocation tests; preserve train submission results, local commands and distinct KubeRay behavior.
3. Add fixed remote acceptance cases, isolation/persistence preflight and observability fixes; qualify a single prepared remote profile before parallel batches.
4. Run the full required CLI matrix, representative logger/native/browser cases, recovery checks and frozen showcase protocols remotely. Retest affected gates after fixes.
5. Rebuild and qualify installed distributions, review evidence, and prepare truthful final PR status. A separately authorized non-forced publication/review/ready transition then requires fresh published-head CI; do not force-push the rebased history.

The pre-rebase ref and exact index/worktree backup provide a recoverable rollback point. Do not reset the user's branch, discard pending work or force-push as an implicit rollback; stop and report any restoration conflict. Implementation rollback removes additive remote-operation selectors/acceptance code while preserving the registered SDK train transport and `dev`'s other backends. Retain all datasets, checkpoints, policies and run records. Existing local CLI behavior and ordinary supervised artifact contents remain preservation gates.

## Deployment Inputs

Before application acceptance starts, supply the approved endpoint/CA, pool resource labels, durable root/artifact route, prepared native/display/ROS profiles, allowed resource/time budget and private-data location. These are deployment inputs, not license to provision alternatives. Their absence blocks the relevant task; all source/spec/CPU-only work can continue without touching the local GPU. No jobs are submitted by this planning change.
