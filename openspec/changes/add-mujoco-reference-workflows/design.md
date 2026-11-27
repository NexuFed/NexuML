# Design

## Context

See `proposal.md` for scope and the three capability deltas for acceptance. This design records decisions, not a claim of native compatibility or completed implementation.

The inspected branch already has:

- Static `Humanoid-v5` and `Reacher-v5` scenarios in `library/src/nexuml_library/scenarios/reinforcement/simulation.py`. Humanoid explicitly disables four additional observation groups and declares 45 observations/17 actions; Reacher declares 10/2. Preserve those settings rather than copying an obsolete default Humanoid dimension.
- `GymnasiumEnvironment` with lazy native construction, synchronous CPU/NumPy stepping, explicit reset/step/close and replay RGB support. Its `step()` currently discards native `info`.
- The execution-local TorchRL adapter/Collector, Lightning-owned PPO, frame-budget resume, complete policy exports, scheduled evaluation and native GIF/MP4 logging. Reuse them.
- `evaluate_policy()` currently couples boolean `task_success` to floating `joint_error_rad`. Existing tests reject success-only data. The shared evaluator already compares control periods and action descriptions as well as shapes/bounds.
- Optional library `mujoco` requirements, currently excluded from library `all`; no root MuJoCo forwarding extra. The root lock selects MuJoCo 3.14.0, Gymnasium 1.3.0, TorchRL 0.14.0 and TensorDict 0.14.2.

NexuVibe references local `nexuml[all,dev]`, `nexuml-library[all]`, `nexufl[all,dev]` and other local packages; it owns backend indexes and one shared environment. Its inspected manifest requires Python 3.14. NexuFL requires Python >=3.12 and Torch >=2.11. Current Meta-World 3.1.1 metadata requires Python >=3.10,<3.14 and MuJoCo ==3.3.0; that MuJoCo release supplies Linux CPython 3.12/3.13 wheels. These facts do not establish full joint resolution.

The existing Isaac change has concurrent user edits, including separate vendor stacks. Do not edit its artifacts, implement its extension, or claim that this MuJoCo composition also reconciles incompatible Isaac/Newton pins.

## Goals / Non-Goals

**Goals:** Three native references through one existing learner; honest action/task semantics; success-only benchmark evaluation; optional NexuML-owned dependencies consumed transitively in one shared NexuML/NexuFL environment; measured native/media/regression proof.

**Non-Goals:** No second session, trainer, environment manager, simulator-specific deployment venv/lock/container, generic metric registry, universal embodiment manifest, algorithm rewrite, additional benchmark suite, MJX/MJWarp, Isaac migration, NexuFL implementation or hardware operation. Disposable existing package-isolation tests are tests, not a second user training environment or resolution fallback.

## Decisions

### D1. Put concrete requirements in the library and forward them from core

Use the existing optional-extra layering:

```text
NexuVibe: local NexuML/NexuFL selections + one backend + one uv.lock
    |
    +-- nexuml[all,dev]
    |      +-- mujoco -> nexuml-library[mujoco]
    |      +-- metaworld -> nexuml-library[metaworld]
    |
    +-- nexuml-library[all]
           +-- mujoco -> reinforcement + gymnasium[mujoco] + mujoco==3.3.0
           +-- metaworld -> mujoco + metaworld==3.1.1
```

Add root forwarding extras with the existing >=0.2,<0.3 library range; extend both `all` selections. The library is the single source of concrete simulator/benchmark requirements. Keep base dependencies free of simulation and root `reinforcement` useful without MuJoCo. Existing rendering/video extras remain the encoder boundary; do not add system graphics packages to Python dependencies.

The explicit MuJoCo pin aligns standalone MuJoCo and Meta-World selections rather than allowing the same reference to silently change physics across extras. Maintain the existing repository `uv.lock`; do not add `scripts/mujoco/pyproject.toml`, a separate simulator lock or a requirements overlay. Inspect built metadata because Hatch expands self-referencing extras; current packaging tests must still verify the published concrete closure.

**Rejected:** direct simulator declarations in NexuVibe/NexuFL; `uv pip install` additions after shared sync; moving simulator packages into base dependencies; depending on `mujoco-py`.

### D2. State the temporary Python limit honestly

Set both NexuML distributions to `>=3.12,<3.14`, remove the 3.14 classifier and update CI/package expectations and install documentation. This is an explicit compatibility break, including base-package metadata, not an assertion that core Python code inherently needs the limit. Extras cannot publish their own `Requires-Python`; retaining an advertised universal range while hiding Meta-World behind omission markers would violate the requested complete `all` installation and complicate uv's universal resolution.

Start the user-owned shared environment at Python 3.13. Verify the complete composition before treating that choice as tested. Python 3.12 is within the declared range; if a concrete joint dependency blocks 3.13, record the conflict and ask the user to prepare the supported 3.12 interpreter instead. Never silently switch interpreter, substitute a second environment or loosen upstream requirements. Restoring 3.14 support is a later metadata/native-proof change once the selected upstream stack supports it.

**Rejected:** copying the linked guide's Python 3.9 example; claiming current MuJoCo generally lacks 3.14 support; broad backend/Python upgrades unrelated to the demonstrated conflict.

### D3. Treat shared-environment readiness as a real external gate

NexuVibe retains responsibility for its interpreter selection, mutually exclusive CPU/cu128/cu130/ROCm indexes, consumer lock and `/env` creation. Provide the user a concise prerequisite handoff: align its `requires-python` and `.python-version`, Python-version-specific CUDA library paths and environment smoke checks, then resolve the existing local project selections transitively. Do not edit those files or any NexuFL files under this repository-local change. The user requested to prepare the venv; proposal creation does not authorize replacing it.

After preparation, record one consumer lock digest, interpreter, backend, distribution versions, dependency-integrity result and import/direct-URL origins for NexuML core/library, NexuFL, Torch/TorchRL/TensorDict, Gymnasium, MuJoCo and Meta-World. The consumer must resolve this changed NexuML checkout. If its local paths select another worktree or an unavailable local dependency, ask the user to align them; do not count that environment as this change's proof.

Use the prepared environment's explicit interpreter for all native acceptance. Resolve first, then use locked setup/run commands; no `--no-deps`, metadata overrides or private upstream patches. The inspected NexuVibe helper invokes `uv sync --upgrade` despite README text describing locked setup. Document that discrepancy in the readiness handoff and do not execute that helper against the user's environment. Updating the consumer lock/interpreter remains user-owned; failed readiness leaves the corresponding task unchecked.

NexuML's existing standalone development lock is not a second simulator operating profile. NexuVibe's lock is authoritative for the shared proof. Keep consumer backend ownership: its cu130 runtime need not match the container toolkit's CUDA 13.3 minor version; no simulator extra selects a CUDA index. Other existing backend selections must remain resolvable or have their exact blockers reported, without claiming unexecuted AMD/GPU tests.

**Rejected:** proving only a hand-picked minimal environment; writing a hidden NexuVibe patch; overwriting its busy checkout; creating separate NexuML/NexuFL/MuJoCo training environments; mixing in the concurrent Isaac stacks.

### D4. Verify native contracts before extending the adapter

Probe pinned Gymnasium/MuJoCo Humanoid and Reacher first, then Meta-World. Use public Gymnasium registration (`metaworld:Meta-World/MT1` with `env_name=reach-v3`) through the existing definition; its explicit contract allows discovery without importing Meta-World. Confirm the pinned distribution's actual registration and reset/task-selection behavior. Only introduce a thin local provider if that public composition demonstrably cannot express the required task/protocol; retain the failed probe as justification.

At native construction validate ordered fields, dtypes, finite action bounds, native task identity/settings and control period. Derive the period from actual physics timestep times frame skip/control decimation; record normalized action scale/frame, actuator meaning and model/package identity. Put policy-relevant timing/rate and stable control identity into the existing contract fields/action description, and fuller versions/model/license provenance into existing artifact metadata. No new robot superclass or manifest service. Same-shape but different controls must fail replay compatibility rather than rewriting packaged semantics.

Meta-World's four controls are normalized end-effector displacement plus gripper, not joint torque. Preserve the native displacement scale and controller. Keep native observation dtype at the runtime boundary; use the existing policy's numeric processing rather than fabricating a different contract. For all three, compare native and NexuML reset/step outputs under identical seeded action sequences, including timeout/termination and fresh resets. Do not force autoreset/vector wrappers onto single environments or replace terminal observations with reset observations.

Start with current CPU direct collection and small synchronous batches; exercise the supported CPU process path without changing TorchRL topology. Resource factories remain execution-local. Standard MuJoCo physics is CPU work; CUDA learning and EGL rendering are independent optional devices, not evidence of GPU physics. No additional TorchRL GymEnv wrapper or new collection engine.

**Rejected:** a direct MuJoCo adapter without a failing use case; loading a robot XML and calling it a complete learning task; replacing the current Humanoid observation settings; implicit action clipping.

### D5. Preserve only the task outcomes this increment needs

At the Gymnasium boundary, normalize native scalar `info['success']` when present. Accept booleans and finite exact zero/one numerics; reject strings, arbitrary numeric truthiness, arrays with more than one element and nonfinite values. Return boolean tensor `task_success` in `InteractionStep.info`, preserving row/device identity for supported batches. Require consistent presence across rows. Continue discarding unrelated native information rather than copying task objects or every arbitrary info field into TensorDict observations.

In the existing evaluator, permit exactly three stable run signatures: no outcomes, success only, or success plus joint error. Joint error alone remains invalid. Infer the signature from the first transition and require the same keys/types on subsequent steps/episodes; do not reuse a previous value. Keep boolean scalar success, optional scalar floating finite nonnegative error, per-episode OR success and final measured error. Emit the existing success metric whenever available and the existing error metric only when actually supplied. Preserve no-outcome behavior, RNG/mode restoration, native scalar/media dispatch and failure cleanup.

SO-101 reaching still supplies both values and must retain its provider-level requirements and paired-result regression. Update the old shared test that rejects success-only data intentionally; do not weaken tests for joint-error-only, malformed types, disappearing values or cleanup. Re-read the shared evaluator before implementation because the independent Isaac work can touch this seam; integrate the same small success-only behavior rather than creating competing implementations or editing the other change's requirements.

**Rejected:** inventing zero joint error for Cartesian tasks; storing success as a policy input/reward replacement; introducing an extensible metric registry for two known fields.

### D6. Reuse renderers, encoders and native logger instances

State-only training uses no renderer. Replay requests one `rgb_array` environment and consumes the existing `render_frame()`/FPS surface, bounds, copied frames and terminal-episode capture. Test GIF, MP4 and both on all three references using existing Pillow/ImageIO/FFmpeg handling. Test native TensorBoard animation and MLflow-only artifact delivery without creating another logger/run.

For NVIDIA headless rendering set `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl` before imports. Check actual vendor EGL libraries/GLVND configuration and a real nonempty rendered frame; CUDA visibility alone is insufficient. For CPU rendering set both variables to `osmesa` in a fresh process and verify the needed system runtime separately. Use the existing DevContainer and only document missing graphics libraries; no driver install, global backend mutation, automatic fallback or container rebuild. Legacy guide apt names are not assumed valid on the selected Ubuntu release. SDL dummy/Xvfb does not prove MuJoCo EGL.

**Rejected:** requiring cameras for state policies; a separate video writer; applying EGL globally to CPU jobs; treating rendered replay as visual-policy training.

### D7. Keep benchmark evidence small and comparable

The initial Meta-World task is MT1 `reach-v3`, not ML1 adaptation, a full suite or hardware proof. Freeze the exact task/reward version, native task-goal identities, reset selection, horizon (initially 500 native policy steps), seed, observation/action contract and success rule. Use public upstream task-selection/evaluation helpers where compatible; keep actual NexuML policy episode execution in the existing evaluator. A minimal offline experiment script/test may sequence existing APIs and aggregate results; it is not another evaluator service or training engine.

Run PPO training seeds 0, 1 and 2 under a declared finite budget, then compare against a seeded random-action native reference on the same frozen evaluation cases. Target the pinned MT1 protocol's complete 50-goal set with one episode per goal; verify its actual task-selection/reset behavior rather than assuming seed-plus-episode enumerates goals. Persist ordered goal/reset identities or equivalent reproducible native identifiers and per-run outputs. If a smaller subset is necessary during development, label it a subset, never official complete MT1 evidence. The completion benchmark uses the frozen complete set.

Use the native rule that any-step success counts once. Retain per-training-seed success/returns, episode counts, variation and elapsed time; report uncertainty without treating episodes from one trained policy as independent training replicates. Scripted success checks can validate the task/outcome calculation, but are not learned-policy success. Held-out goal variations are an explicitly named custom robustness check, not a relabeled MT1/ML1 score. Native helpers with incompatible vector/autoreset assumptions must not be forced into the current collector; document a protocol-preserving public-API composition instead.

**Rejected:** comparing incompatible tasks/controls, a new multi-suite registry, stochastic quality thresholds in ordinary CI, claims that a tiny PPO update converged, or silently dropping Meta-World when its gate fails.

### D8. Bound the implementation and completion claims

Expected ownership:

| Area | Existing or proposed edit homes |
| --- | --- |
| Dependencies/Python | `pyproject.toml`, `library/pyproject.toml`, existing `uv.lock`, CI matrix and packaging metadata tests |
| Native task boundary | `library/src/nexuml_library/environments/gymnasium.py`; a thin provider file only if D4 proves it necessary |
| Scenarios | `library/src/nexuml_library/scenarios/reinforcement/simulation.py` and simulator-free contract tests |
| Outcomes | `src/nexuml/training/reinforcement.py`, `tests/core/test_rl_evaluation.py`, focused native/provider regressions |
| Evidence/tooling | Existing simulator CI, new opt-in MuJoCo integration tests, existing RL/install docs, `reports/physical-ai/phase-6-mujoco.md` |

Test native before integration; prove actual actor updates, partial-budget resume, complete policy export and clean-process deterministic inference for all three. Reuse scheduled evaluation, policy-sync and cleanup tests. Exercise same-process/fresh-process render cleanup and failure/cancellation; verify both headless backends, actual selected formats/playback, native logger behavior and truthful statuses. Default CPU CI remains simulator-free, with native simulator/graphics checks explicitly opted in. Installed-wheel metadata/import isolation must agree with editable development behavior.

The report records code/import origins, full shared lock/digest, package/interpreter/backend/system libraries, native model/task identities/licenses, all commands, seeds, horizons, frame budgets/counts, outcomes, media and failures. Record existing unrelated baseline diagnostics separately; do not repair or install them away merely to obtain green totals. No shared-composition or native completion from resolver metadata, fake tests, skips or CI configuration alone. Update only the roadmap's actually proved Phase 6/partial Phase 4 items after implementation; do not rewrite its broader program or other OpenSpec histories.

## Risks / Trade-offs

- **[Python support is narrowed package-wide]** -> Mark the break in release/install guidance and wheel metadata; preserve earlier artifacts, and require a later explicit proof before lifting the upper bound.
- **[Shared composition is blocked by another package/backend]** -> Retain the exact resolver error and pause for user-owned preparation; no new environment or unsupported override. In-scope NexuML metadata may be corrected without loosening upstream contracts.
- **[The consumer selects the wrong checkout or setup upgrades unexpectedly]** -> Verify source identities and authoritative lock before native work; do not run the discrepant setup helper or modify sibling files.
- **[MuJoCo changes from the current 3.14.0 lock to 3.3.0]** -> Re-run all native references and record physics/task version; old and new benchmark scores are not assumed identical.
- **[Concurrent Isaac work changes shared outcome handling]** -> Re-read, preserve its paired outcomes and contract validation, then add only missing shared behavior/regressions.
- **[Benchmark scheduling or EGL produces plausible but incorrect results]** -> Frozen native goal/reset identities, transient-success tests, real graphics/media checks and explicit insufficient-evidence outcomes.

## Migration Plan

1. Record the existing baseline and pending changes; add the temporary Python bounds, NexuML extras/forwarding and metadata/lock tests.
2. Hand off compatible interpreter/consumer-lock preparation to the user; verify the single shared environment and correct checkout before native acceptance. Metadata/unit work may proceed while this external gate is open, but completion may not.
3. Probe native Humanoid/Reacher/Meta-World, add only proven boundary/outcome gaps, then prove all three round trips and recording paths.
4. Run the frozen small benchmark and scoped regression/package/docs/static gates; publish evidence and update only proved roadmap items.

Rollback removes only this change's additions or restores its changed metadata/configuration under normal version control. Never delete/recreate `/env`, consumer locks, simulator caches, recorded experience, policy artifacts or unrelated work automatically. Existing policy artifacts remain usable for supported portable inference; do not silently reinterpret their task or physics identity.

## Sources checked during planning

- [TorchRL MuJoCo installation guide](https://docs.pytorch.org/rl/main/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html): modern bindings, EGL/OSMesa and graphics diagnostics; legacy instructions are not the installation target.
- [Meta-World 3.1.1 metadata](https://pypi.org/pypi/metaworld/3.1.1/json) and [MuJoCo 3.3.0 metadata](https://pypi.org/pypi/mujoco/3.3.0/json): exact Python/physics constraints and wheel availability.
- [Gymnasium registration](https://gymnasium.farama.org/api/registry/) and [MuJoCo environments](https://gymnasium.farama.org/environments/mujoco/): lazy module registration, v5 settings, reproducibility and rendering.
- [Meta-World evaluation](https://metaworld.farama.org/evaluation/evaluation/) and [action contract](https://metaworld.farama.org/benchmark/action_space/): any-step success, MT1 goal protocol and Cartesian/gripper controls; pinned runtime probes remain authoritative.
- [uv resolution](https://docs.astral.sh/uv/concepts/resolution/): transitive requirements and universal Python resolution. Context7 was unavailable due to quota; primary documentation/metadata were read directly.

Planning performed no package resolution/install, native simulation, GPU work, system change or container rebuild.
