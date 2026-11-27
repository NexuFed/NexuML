# Proposal

## Why

NexuML already has MuJoCo Humanoid/Reacher scenarios and a working Lightning/TorchRL lifecycle, but their native round trips remain unverified and benchmark success is lost at the Gymnasium boundary. Complete that foundation and one Meta-World manipulation reference without introducing another trainer or simulator environment-management stack.

## What Changes

- Verify the existing `mujoco-humanoid-ppo` and `mujoco-reacher-ppo`, then add one ordinary `metaworld-reach-ppo` scenario using the maintained Meta-World `reach-v3` task. All three must train, resume, export, reload, evaluate and record through existing NexuML APIs.
- Reuse `GymnasiumEnvironment`, the existing runtime-to-TorchRL adapter, Collector, PPO, `NexuSession`, `CompiledPolicy` and native logging/media. Add only the lazy registration, task-outcome mapping or contract checks that native probes prove necessary.
- Preserve Meta-World's validated success signal independently of reward and SO-101-specific joint error. Retain existing metrics, episode reset semantics and strict malformed-outcome validation.
- Declare MuJoCo and Meta-World dependencies in **NexuML's root/library `pyproject.toml` files**, using optional extras and root forwarding. Include the intended stack in their `all` selections so NexuVibe's existing local NexuML/NexuFL composition resolves it transitively; add no direct simulator dependencies to NexuVibe or NexuFL.
- **BREAKING:** Temporarily change both NexuML distributions' supported Python range to `>=3.12,<3.14`, updating classifiers, packaging assertions, CI and documentation. The selected Meta-World 3.1.1 requires Python `<3.14` and MuJoCo `==3.3.0`; current MuJoCo releases themselves support newer Python. Python 3.13 is the initial shared-environment target, subject to complete resolution/native verification.
- Require **one consumer-owned uv lock and one shared NexuML/NexuFL environment**, not a MuJoCo-specific venv, lock/profile or worker container. NexuVibe continues owning interpreter/backend selection; the user's compatible interpreter preparation and shared uv resolution are explicit acceptance prerequisites, not hidden sibling-repository edits.
- Verify headless NVIDIA EGL and CPU OSMesa recording separately from state-only training, preserve optional test tiers, and publish exact native/round-trip/multi-seed evidence with honest limitations.

## Capabilities

### New Capabilities

- `mujoco-reference-workflows`: Native contract checks, three ordinary PPO reference round trips, headless media, benchmark protocol and reproducible evidence.
- `rl-task-outcomes`: Validated optional success and joint-error results, independent aggregation and preservation of existing evaluation behavior.
- `physical-ai-dependency-composition`: NexuML-owned optional simulator requirements, explicit Python compatibility and transitive single-environment composition with NexuFL through NexuVibe.

### Modified Capabilities

None of the published capabilities changes its requirements. The existing unarchived RL/evaluation changes supply the lifecycle, package and media behavior reused here; this change neither rewrites their completion history nor edits the concurrent Isaac proposal.

## Impact

- NexuML packaging: root/library manifests, repository lock, Python classifiers/matrix, metadata and optional-import tests; base installations still do not require simulation dependencies.
- Library/runtime: existing Gymnasium provider and simulation scenarios, a minimal Meta-World integration only where necessary, and shared episode-outcome handling/tests.
- Documentation/evidence: RL/development installation guidance, optional simulator CI and `reports/physical-ai/phase-6-mujoco.md`. Phase 6 and the first RL portion of Phase 4 only; the whole roadmap is not completed.
- External prerequisite: inspect NexuVibe/NexuFL composition read-only and verify the user-prepared shared environment against the changed NexuML checkout. No NexuFL implementation, sibling-repository edits, environment replacement, installation, download or container rebuild is performed by proposal creation.
- Excluded: LIBERO, LeRobot/ACT/VLA, demonstrations, new robot-specific tasks, direct MuJoCo/MJX/MJWarp adapters, Isaac changes, Ray/GPU-physics expansion, generic benchmark services, hardware and unrelated baseline repair.
