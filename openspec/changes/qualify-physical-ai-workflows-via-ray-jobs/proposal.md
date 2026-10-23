# Proposal

## Why

PR #15 contains substantial Physical AI functionality, but historical CLI runs, native Python fixtures and current release acceptance are not interchangeable. Finish the implemented foundation with reproducible CLI/browser journeys executed as independent remote Ray Jobs, preserving the developer's local GPU and making supported behavior, learning quality and remaining blockers explicit.

## What Changes

- Rebase the branch onto the actual `dev` base under the user's explicit authorization, preserving a recoverable pre-rebase branch and staged work. Fix required static, supported-Python, packaging and regression failures before qualifying a frozen candidate. Preserve unrelated work and existing historical evidence; publication remains separately authorized.
- Port this branch's working Ray Jobs SDK functionality into `dev`'s registered execution structure as `ray-jobs-sdk`, distinct from its existing KubeRay `ray-job`. Use the shared Python/CLI/API execution dispatch and only the registered `type`/`version`/`params` schema. Preserve locked uv setup, private per-submission venvs, cache/download settings, one-GPU defaults, configurable pool selection and asynchronous `submit_ray_job(scenario)` behavior. Extend that transport minimally for checkpoint resume/export and fresh model/policy evaluation. Never substitute Ray Train, load a model/start a simulator on the submitter or silently fall back locally.
- Execute acceptance runs only through the registered SDK backend on `pool_gpu_workers`, never DGX or a separate CPU-worker pool. Allow at most four in-flight independent jobs, further limited by qualified capacity; each job retains the default whole-GPU reservation, even for CPU control workloads. Isolate run directories, installation paths, viewer ports, ROS/Gazebo namespaces and mutable tracking stores. Keep dependent stages and native process lifetimes sequential within each case.
- Require real CLI journeys for all advertised runnable recipes; add representative sensor/MARL/logging combinations, installed-wheel inference and the existing SO-101 script/SDK workflow. Keep smoke correctness separate from frozen showcase-learning thresholds.
- Verify actual TensorBoard/MLflow browser playback, artifact retrieval and live environment viewing. Cover the qualified Isaac spectator plus native MuJoCo and Gazebo viewer journeys on prepared remote workers; missing prerequisites block the affected claim rather than authorize local GPU use or count a skip as a pass.
- Improve run summaries, private connection instructions, source-frame/checkpoint provenance, progress phases, persistence and cancellation diagnostics. **BREAKING (failure behavior):** an explicitly configured tracking backend that cannot initialize fails the job instead of being silently skipped; intentionally disabled logging remains supported.
- Publish a revision-bound CLI/visual/learning/recovery matrix and retain sanitized logs, media, receipts and artifacts outside Git. Do not represent a submission ID, a screenshot or old test counts as current-head acceptance.

## Capabilities

### New Capabilities

- `ray-job-cli-lifecycle`: The registered `ray-jobs-sdk` port of this branch's SDK submission behavior, existing CLI operations, checkpoint/package handoff, resource isolation, bounded parallelism and local-GPU exclusion.
- `physical-ai-workflow-acceptance`: Current-candidate public CLI journeys, supported-recipe inventory, quality distinctions and branch release gates.
- `physical-ai-visual-acceptance`: Real logger-browser, media and native live-view acceptance with isolated simulation roles and explicit viewer prerequisites.
- `physical-ai-run-observability`: Actionable run identity/progress/access/output reporting, physical-frame provenance and truthful tracking/failure states.

### Modified Capabilities

None of the canonical `openspec/specs/` requirements are replaced. Reuse `dev`'s `nexuml-execution-backends` registration, serialization and shared dispatch contracts without changing its KubeRay backend or retaining the old `execution.kind` schema. Also reuse `framework-test-coverage`, `logs-root-resolution`, existing package contracts, and the unarchived recorded-evaluation/native-Viser changes. The additions close remote user-journey gaps without reopening their historical completion records.

## Impact

- **Code:** existing execution types/`ray_job` adapter, CLI dispatch, export provenance and tracking summaries; narrowly demonstrated environment/viewer and documentation fixes; existing test/fixture/CI surfaces and a bounded acceptance driver, not a new service or training engine.
- **Infrastructure:** an existing operator-approved Ray Jobs endpoint, TLS trust, explicit worker resources, durable storage and prepared native/browser/display/ROS prerequisites. No cluster provisioning, shared-environment mutation, driver/container replacement, automatic image publishing or hardware actuation.
- **Compatibility:** retain local CLI behavior, asynchronous submission-only defaults, inference/checkpoint separation and optional dependencies. Migrate Python construction, configs/profiles, overrides, tests and docs to the registered SDK definition; do not maintain legacy `execution.kind` parsing. Remote paths are worker paths, never implicitly uploaded local checkpoints or private datasets. Preserve the prepared local venv and all existing outputs/data.
- **Ownership:** [NEX-248](https://linear.app/nexufed/issue/NEX-248/build-nexuml-reinforcement-learning-and-physical-ai-foundation), [PR #15](https://github.com/NexuFed/NexuML/pull/15), and the existing NEX-234 producer boundary. NexuFL integration remains separately owned.
- **Non-goals:** ACT/VLA/world models, additional embodiments, distributed RL learners, Ray Train placement, a new dashboard/renderer/scheduler, a general remote shell API, and completion of the entire Physical AI roadmap. The authorized rebase is not authorization to force-push, publish new implementation commits, merge the PR or archive the change. Creating these artifacts submits no jobs.
