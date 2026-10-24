# Tasks

All implementation tasks start unchecked. Native/shared-environment acceptance requires the user's prepared compatible interpreter, consumer lock and graphics prerequisites; proposal validation is not runtime proof. Do not edit NexuVibe/NexuFL or the concurrent Isaac change, install packages, replace `/env` or rebuild the container without separate user direction.

## 1. NexuML-owned dependencies and readiness handoff

- [ ] 1.1 Re-read current manifests, native scenario/runtime/evaluator code and pending work; start `reports/physical-ai/phase-6-mujoco.md` with commit/import origins, package versions and scoped baseline commands/results. Verify unrelated edits and baseline failures are recorded, not overwritten or repaired.
- [ ] 1.2 Add library-owned MuJoCo 3.3.0/Meta-World 3.1.1 extras, root forwarding extras and both `all` closures; update the existing repository lock. Verify metadata tests and `uv lock --check` prove transitive inclusion without base simulator dependencies or a new simulator manifest/lock.
- [ ] 1.3 Apply `>=3.12,<3.14` to both distributions and align classifiers, CI's Python matrix and packaging assertions. Verify built metadata advertises the same range, supported 3.12/3.13 checks are configured and 3.14 is rejected explicitly rather than omitting requested packages.
- [ ] 1.4 Update existing installation/development guidance for optional/forwarded extras, the temporary Python compatibility break and shared consumer ownership. Verify documented commands are shell-safe and neither duplicate simulator declarations in NexuVibe nor prescribe a separate MuJoCo environment.
- [ ] 1.5 Give the user the NexuVibe readiness checklist from D3, including interpreter/consumer-range alignment, version-specific paths/tests, correct source checkout, chosen backend and lock preparation. Verify the current setup helper's `--upgrade` discrepancy and every missing prerequisite are identified without changing or running sibling setup files.

## 2. Shared installation and native task contracts

- [ ] 2.1 After user preparation, verify joint dependency integrity and import/direct-URL origins in the single shared NexuML/NexuFL environment, recording the consumer lock digest, interpreter and backend. Confirm all requested simulator packages are present; if prerequisites or resolution fail, record the conflict and keep this gate open without overrides or a second training environment.
- [ ] 2.2 Probe native Humanoid-v5, Reacher-v5 and Meta-World MT1 reach-v3 before integration changes; record exact fields, dtypes, bounds, timestep/control rate, native task/reset behavior and asset/model provenance. Verify seeded action-sequence/reset/termination/timeout results and the module-qualified registration against the pinned release.
- [ ] 2.3 Add only missing construction-time native/declared contract checks and portable control timing/identity metadata. Verify simulator-free tests reject wrong shape, dtype, bounds, task/control meaning and period before action dispatch while native probes agree with the declared contracts.
- [ ] 2.4 Register `metaworld-reach-ppo` through the ordinary scenario path, preferring the existing Gymnasium definition; add a thin provider only with D4's failing public-composition evidence. Verify discovery, compile, YAML restoration, overrides and focused missing-extra errors with simulator imports blocked.
- [ ] 2.5 Verify direct and supported synchronous CPU process collection on the native references, covering actual frame counts, row/reset masks, terminal observations, truncation bootstrapping, copied-policy synchronization and cleanup. Retain small deterministic tests for any fixed boundary defect, without changing PPO or adding a vectorization engine.
- [ ] 2.6 Document each reference's native identity, observation settings, action ordering/scale/frame, control period and provenance. Verify Humanoid retains the selected 45/17 contract, Reacher 10/2 and Meta-World its native Cartesian/gripper semantics, with no named-hardware portability claim.

## 3. Success-only task outcomes

- [ ] 3.1 Normalize validated native `success` into row/device-correct boolean task outcomes without copying unrelated info into policy observations. Verify unit tests for booleans, exact numeric zero/one, invalid/nonfinite/non-scalar values and mixed-presence vector rows.
- [ ] 3.2 Extend the existing evaluator for stable absent, success-only and success-plus-error signatures; keep joint-error-only invalid. Verify success-only output, intermediate-step OR success, per-episode/run reset, final measured error, no fabricated metrics and strict signature/type/disappearance failures with runtime cleanup.
- [ ] 3.3 Preserve the current SO-101 paired-output regression and all no-outcome/scheduled-evaluation/RNG/mode/media behavior. Verify existing invalid-outcome tests are only changed where success-only is intentionally newly valid and provider-specific required outcomes remain enforced.
- [ ] 3.4 Document optional success/error metric semantics in the existing RL guide. Verify examples distinguish task success from return and do not imply every environment has a success or distance metric.

## 4. Complete native policy round trips

- [ ] 4.1 Prove Reacher training with changed actor weights/finite losses, partial-budget checkpoint resume, export and deterministic clean-process inference with simulator/learner/encoder imports blocked. Verify fresh native evaluation and checkpoint counters using the shared prepared environment; retain exact commands and artifacts.
- [ ] 4.2 Prove the same full Humanoid round trip with its explicit observation settings and bounded 17-action policy. Verify native contract parity and actual updates rather than equating a short run with learned locomotion.
- [ ] 4.3 Prove the same full Meta-World reaching round trip with separate task success and no fabricated joint error. Verify matched task/control/reset metadata survives export and incompatible same-shape replay overrides are rejected.
- [ ] 4.4 Exercise scheduled evaluation, process synchronization, completed-budget resume and failure/cancellation around these references. Verify existing ownership/finite-budget tests and focused native cleanup checks leave no owned simulator/collector workers running or hidden primary failures.
- [ ] 4.5 Publish tested resolve/train/resume/export/evaluate examples and actual checkpoint/package outputs in the RL guide/report. Verify each command uses ordinary APIs and the same prepared interpreter, without user source edits or a new launcher/trainer.

## 5. Headless media and native logging

- [ ] 5.1 Add focused renderer/capture tests for disabled rendering, malformed frames, missing prerequisites, terminal-frame identity and teardown. Verify state-only runs do not allocate a renderer and requested recording never silently switches backend or publishes partial media as success.
- [ ] 5.2 Run fresh-process CPU OSMesa replay for all three references in GIF, MP4 and both modes. Verify actual frames, FPS/timing, limits, terminal scenes, finalized playback/frame counts and continued scalar episodes, preserving exact evidence.
- [ ] 5.3 Run NVIDIA EGL checks and recorded replay for all three references with backend variables set before imports. Verify vendor graphics visibility plus real frames/media independently of CUDA compute; record missing graphics as a blocker rather than installing drivers or rebuilding the DevContainer.
- [ ] 5.4 Verify existing TensorBoard animation and MLflow-only/both native artifact delivery, scalar task metrics and source provenance on recorded MuJoCo/Meta-World replay. Document exact graphics/encoder prerequisites and commands; verify MLflow-only does not access a TensorBoard writer and GIF/scalar inference does not require MP4 encoders.
- [ ] 5.5 Extend existing opt-in simulator CI for the new native/recording tests and explicit headless prerequisites, without default-CI simulator execution or unapproved GPU downloads. Verify marker/collection behavior and workflow configuration; identify remote CI as unrun unless it actually executes.

## 6. Small reproducible Meta-World comparison

- [ ] 6.1 Freeze the pinned MT1 task/reward/observation/action protocol, its complete 50-goal evaluation set, ordered goal/reset identities, horizon and seeds using public upstream APIs. Verify native task enumeration/reset behavior and distinguish full protocol, development subsets and custom held-out variations in documentation.
- [ ] 6.2 Train PPO seeds 0, 1 and 2 under a declared finite budget and evaluate them plus a seeded random reference on identical frozen conditions through existing APIs. Verify actual per-condition success/return outputs and trial counts are persisted without adding another training/evaluation engine.
- [ ] 6.3 Produce machine-readable results and a concise comparison in the Phase 6 report. Verify success is counted at any step, variability distinguishes training seeds from episodes, runtime/versions/configurations are included and neither convergence nor the full Phase 4/hardware gate is claimed from this evidence.

## 7. Integrated acceptance

- [ ] 7.1 Run scoped and broader relevant CPU regression, lock, lint, formatting, typing and strict documentation gates after integration. Verify each capability's tests have already landed with its implementation group and baseline failures/diagnostics are reported separately rather than hidden or repaired outside scope.
- [ ] 7.2 Build/check wheel and sdist artifacts and run existing package/import-isolation proofs for the supported Python range. Verify published Python bounds/extras, base supervised/import behavior without simulator dependencies and inference-only policy boundaries agree with editable shared-environment behavior.
- [ ] 7.3 Reconcile the Phase 6 report with actual shared-resolution, all-three round-trip, both-renderer, media and benchmark evidence. Verify no required acceptance is satisfied by skips or static configuration; update only proved Phase 6/partial Phase 4 roadmap items, leaving unrelated live-system/Isaac work unchanged.
- [ ] 7.4 Run `openspec validate add-mujoco-reference-workflows --strict` and review the final scoped diff. Verify proposal/design/spec/task consistency, no new simulator deployment environment and no sibling-repository or unrelated changes before implementation completion is reported.

## Workflow follow-up

- Commit, review, merge and archive only when separately requested; planning completion does not perform them or imply native acceptance.
