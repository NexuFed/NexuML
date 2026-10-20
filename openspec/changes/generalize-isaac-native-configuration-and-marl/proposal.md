# Proposal

## Why

Isaac currently exposes hardcoded state-only profiles rather than native task configuration, multimodal contracts or grouped learning. Generalizing this unmerged slice through existing NexuML composition allows custom native tasks, portable policies and mandatory read-only inspection without creating another ML framework.

## What Changes

- Deliver one implementation pass in order: early private/read-only Viser feasibility gate; **A** native configuration/lifecycle; **B** multimodal sensors/single-agent control; **C** complete read-only Viser; **D** native TorchRL IPPO/MAPPO.
- Add typed importable `IsaacTaskConfigSpec`/`isaac_task_config(...)`, native registered built-in/custom tasks and presets, explicit contracts/bindings and optional `ViserSpec` on `IsaacLabEnvironment`.
- Make normal CLI train/evaluate and an explicit Python whole-job scope own native startup, checkpoint/export/logger finalization and teardown.
- Support state, RGB, depth, segmentation and mixed observations; continuous/scalar categorical actions; declared processing, history, timing and physical action semantics through export/replay. Package-owned history uses a minimal opt-in caller-owned policy-state/partial-reset contract; stateless `CompiledPolicy.forward()` remains unchanged.
- Support fixed ordered heterogeneous/homogeneous groups, shared/separate actors and local/centralized critics using native TorchRL losses/GAE and existing Lightning ownership. Do not change ordinary single-agent PPO numerics.
- Keep Viser mandatory, opt-in and loopback-bound, with mutation denial at callback/request/control boundaries, separate replay activation and genuine browser acceptance. Fail closed when unqualified; if early feasibility fails, stop viewer-dependent work and seek an explicit maintainer decision before proceeding with independent headless milestones. Do not remove Viser or mark the overall change complete.
- Separate SO-101 semantic compatibility from implementation provenance without weakening physical, dataset-integrity or authorization checks; preserve NEX-234 `local_fit` and identify independent consumer requalification. K2 is a mandatory separately reported cross-repository integration gate, not a prerequisite for NexuML-only Milestone A acceptance.
- **BREAKING (unmerged Physical AI definitions only):** directly replace superseded native schemas with one current version. No legacy readers, converters, parallel representations or hypothetical checkpoint-compatibility machinery. Historical evidence/data and unrelated published interfaces remain intact.

## Capabilities

### New Capabilities

- `native-isaac-task-configuration`: trusted typed native customization, task/preset resolution, automatic lifecycle and current-schema persistence.
- `native-robot-sensor-contracts`: actual multimodal sensors, control semantics and pre-reset observation/temporal correctness.
- `read-only-isaac-visualization`: qualified native Viser construction, strict spectator capabilities, private access and cleanup.
- `multi-agent-policy-contracts`: canonical fixed-group tensors, actor visibility, sharing and per-agent/environment boundaries.
- `torchrl-multi-agent-learning`: native IPPO/MAPPO/GAE integration, validity, optimizer ownership and continuation.
- `portable-physical-policy-deployment`: complete actor-only inference, semantic/provenance separation and preserved SDK/consumer boundary.

### Modified Capabilities

None of the published `openspec/specs/` requirements changes. General package entries, dependency manifests, checkpoint sidecars, supervised and discovery contracts remain authoritative. The related Physical AI specifications are unmerged roadmap inputs, not duplicate implementation queues.

## Impact

- Preserve **ScenarioSpec → Compiler → Training/Execution → Policy → Evaluation/Export**. Core owns reusable contracts/composition/lifecycle/artifacts; Library owns Isaac configuration/bindings/Viser/reference scenarios and thin registered algorithms; Isaac owns simulation/sensors; TorchRL owns numerical algorithms; NexuFL independently owns federation/transport/aggregation.
- Target existing `core`, `interaction`, `reinforcement`, `training`, `evaluation`, CLI and library integration seams only where demonstrated requirements demand it; see [design.md](design.md). No new compiler, trainer, session, policy hierarchy, simulator registry or preprocessing framework.
- This focused change owns A–D implementation/acceptance. The [umbrella roadmap](../add-isaac-physical-ai-backend/proposal.md) retains MuJoCo, ROS 2, CARLA boundaries, ACT, Diffusion Policy, VLA, world models, robotics and federation follow-ups. Narrow ownership links do not close its tasks or rewrite historical evidence.
- Preserve the worktree `.venv`: Lab 3.0 at `70ab24fec3eabf51222817db0cdb5a4e7133bd7e`, Sim 6.2, TorchRL 0.14, existing PyTorch 2.11/CUDA stack. No installs, `/env` changes, container/driver replacement, large pretrained downloads, cloud/paid compute, public exposure or actuation; retain at least **40 GiB free**.
- Planning source: approved `/home/nexuadmin/.opencode/plan/nexuml-native-isaac-configuration-and-viser.md`, with this request's semantic-provenance and no-NexuFL-edit refinements taking precedence. This OpenSpec is the implementation/acceptance authority; it is not a second standalone plan.
- Excluded: distributed simulator execution, WebRTC/editor streaming, dynamic membership, MultiDiscrete/MultiBinary/hybrid single-actor actions, task convergence/superiority, automatic hardware promotion and NexuFL source changes. Single-agent imitation does not depend on MARL.
- Native rendering/Vulkan, task assets and safe Viser construction remain qualification gates, not assumed successes. [verification-plan.md](verification-plan.md) separates static, numerical, native, browser, distribution and SDK/consumer proof. This planning change implements no functionality and starts no native/browser jobs.
