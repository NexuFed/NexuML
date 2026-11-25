# Tasks

All tasks are new, unchecked. Existing historical evidence remains scoped to its original versions. Implement and report each gate before expanding scope. Reuse current test fixtures; avoid a new parallel test framework.

## 0. Baseline and feasibility

- [ ] 0.1 Inspect current branch, active changes and installed source versions; record precise restrictions, existing public APIs and baseline evidence. Reconcile overlapping active requirements according to design; do not edit historical completion records.
- [ ] 0.2 Probe pinned Isaac Sim/Lab/Arena, TorchRL/TensorDict/Ray APIs: native builder, reset/terminal mapping, GPU/worker placement, policy sync, app teardown, cross-image serialization. Choose maintained wrappers where correct; document deviations. Probe available cluster resources without provisioning or changing infrastructure.
- [ ] 0.3 Freeze small Cartpole, SO-101 and representative grouped scenario configurations, seeds, rollout geometry, timing and acceptance protocols. Establish direct baseline throughput, memory, outcomes and artifacts. Separate correctness smoke from multi-seed learning quality. Record blocked prerequisites.

## 1. Native environment and Arena composition

- [ ] 1.1 Refine semantic policy compatibility vs checkpoint identity and provenance using existing contracts; test compatible native mappings and rejection of mismatched units/order/control/timing/sensors/groups. Keep export/local-fit compatibility.
- [ ] 1.2 Extract shared native Isaac worker/process scope and runtime bindings as needed; preserve import order, native GPU tensors, terminal/reset behavior, read-only viewer and ordinary direct flow. Remove unnecessary robot-specific branches from generic runtime without rewriting native task implementations.
- [ ] 1.3 Implement a trusted importable Arena-composed environment definition/factory using the native builder and ordinary runtime; retain native suite evaluator/recorder, share lifecycle and bindings, keep Arena optional.
- [ ] 1.4 Add or adapt one small reward-bearing Arena training reference (SO-101 reach or maintained manipulation task with real reward and success). Prove native reset, known physical action, outcome, actual PPO updates, resume/export and fresh native replay. Also qualify a held-out Arena benchmark of the resulting compatible policy; do not label reaching as pick/place.
- [ ] 1.5 Re-run existing direct Cartpole/SO-101/MARL, supervised and NEX-234 local-fit regressions; report original benchmark behavior and artifact provenance.

## 2. Independent device placement

- [ ] 2.1 Extend existing collector config/validation for independent environment, rollout policy, storage and learner devices; preserve direct single-GPU defaults, native batches and no forced host copies.
- [ ] 2.2 Qualify same-host separate-GPU simulation/inference and learner with one synchronous collector; verify explicit device mapping, supported transfers, actual frame budgets, correct terminal/GAE semantics, policy updates and resource cleanup. Compare with same-GPU baseline; keep unsupported layouts fail-closed.
- [ ] 2.3 Test CPU/GPU transition ownership and batch boundaries; record GPU memory/utilization and copy throughput separately from physics/learning time.

## 3. Distributed GPU rollout collection

- [ ] 3.1 Refactor existing TorchRL Ray collection to accept/inherit explicit cluster target and native Ray resource/label constraints; no implicit local CPU Ray cluster when targeting existing cluster. Preserve existing CPU collection and borrowed-Ray ownership.
- [ ] 3.2 Build serializable worker-local factories and persistent worker-owned native scope. Place policy/critic replica and vectorized simulator together by default. Prove graceful close, cancellation, exception cleanup and no leaked native app/actor/object-store references.
- [ ] 3.3 Implement/verify synchronous policy-version barrier using maintained weight synchronization; preserve behavior statistics, worker seeds, valid masks, actual transition counts, environment/time axes and terminal/truncated semantics. Reject stale or incomplete batches and unsupported multi-learner/asynchronous combinations. Bound in-flight batches.
- [ ] 3.4 Qualify on an actual existing Ray cluster with two GPU simulation workers on distinct nodes and a separate GPU learner; simulator-free learner image; explicit resource selection, placement receipts, no per-step network RPC. Submit through the existing public workflow; run state reference and representative grouped contract when qualified.
- [ ] 3.5 Verify failure/cancellation and checkpoint resume with fresh native workers; verify counters, optimizer, artifact and policy versions. Benchmark direct/split/distributed modes with same frozen tasks; report throughput, latency breakdown, bytes/rollout and weights, utilization, memory/object-store and learning outcome. No promised speedup.

## 4. Independent evaluation

- [ ] 4.1 Reuse existing evaluation configuration and dispatch; add minimal placement selection and immutable actor/processor/adapter snapshot at consistent update boundary with digest and source frame/version.
- [ ] 4.2 Support separate fresh-process and Ray evaluation workers; retain native Arena suite runner, ordinary replay, package/checkpoint CLI evaluation and existing loggers. Bound queue/in-flight work; explicit failure/cancel and final-evaluation semantics.
- [ ] 4.3 Prove periodic evaluation on a separate GPU/node while learner progresses; unchanged training env/optimizer, correct snapshot identity, condition seeds, metrics, report and bounded media. Test slow evaluation, cancellation and final completion. Retain explicit rejection of unsupported in-process Isaac fit evaluation.

## 5. Second simulator and portability

- [ ] 5.1 Probe native MuJoCo/Warp or maintained Playground compatibility and tensor interop; choose smallest supported path. Retain Gymnasium MuJoCo. Coordinate with existing MuJoCo GPU and dependency-composition requirements; use upstream assets and existing image pipeline rather than a second framework.
- [ ] 5.2 Implement a native GPU-capable reference through the existing EnvironmentDefinition/Runtime boundary; prove actual device-native physics, batch/reset and TorchRL training with unchanged learner/collector code. Qualify direct and distributed placement as supported by the pinned provider, with explicit evidence of any gaps.
- [ ] 5.3 Demonstrate a matched task/contract across Isaac and MuJoCo (e.g. equivalent control/state Cartpole, not categorical vs continuous). Known-action semantic probes, same model/algorithm and exported-policy evaluation with explicit mappings; distinguish physics/quality differences from correctness.
- [ ] 5.4 Record native/CPU/GPU benchmark, package/ABI/render profiles, CI smoke and opt-in hardware evidence. Reuse existing MuJoCo/Playground acceptance tasks rather than duplicating their entire reference matrix.

## 6. Staged system-level integrations and release

- [ ] 6.1 Audit existing CARLA and ROS 2 adapters and specify concrete native Gazebo stepping/reset/time/namespace integration. Preserve native tooling and public interface. Qualify one focused simulated reference per added provider before advertising support; do not block independently qualified stages 1–5 on unrelated hardware.
- [ ] 6.2 Verify CARLA exclusive tick ownership, synchronized sensors and actor cleanup; Gazebo deterministic stepping and ROS namespaces; ROS 2 physical interaction remains safe-stop/watchdog bounded, with fake transport tests and explicitly authorized hardware validation only. No unrestricted real-robot learning.
- [ ] 6.3 Integrate qualified optional dependencies and pins into existing NexuML CI image/release workflow (including Isaac image); document worker group resource configuration, local single GPU, split GPU, Ray collection and independent evaluation recipes. Verify installed simulator-free policy inference/learner and existing CLI/SDK.
- [ ] 6.4 Publish revision-bound acceptance matrix (pass/fail/blocked/unverified) and measured evidence for every claimed topology/provider; update support docs honestly, leave unqualified work unchecked. Reconcile overlapping specs before archival. Run targeted tests, existing regression suite and `openspec validate decouple-simulation-learning-and-evaluation --strict`.

## Acceptance evidence

For each native gate record candidate SHA, immutable images/dependencies, task/asset/contract/algorithm identity, configuration, seeds, assigned nodes/GPU indices, actual commands/exit status, transitions and updates, loss/outcomes, artifact hashes, performance and cleanup evidence. GPU CI tests are opt-in where hardware is unavailable; skipped runs are not native passes. No numerical equivalence claim for different physics or seeds. No production/real-robot authorization inferred from simulated success.
