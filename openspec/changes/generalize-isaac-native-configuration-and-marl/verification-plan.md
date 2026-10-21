# Verification plan

This is an authored acceptance protocol, not execution evidence. Requirement IDs refer to the six specs; task IDs refer to [tasks.md](tasks.md). Do not mark tests run or milestones complete during this planning conversion.

## Gate rules and environment

Before native/server work, verify approved versions/source identities and more than **4 GiB free**, monitor space throughout execution, account for existing jobs and bound environment/camera counts, duration, VRAM and output sizes. The maintainer explicitly replaced the original 40 GiB floor with this 4 GiB stop threshold during implementation. Use only the worktree `.venv` (Lab 3.0/Sim 6.2/TorchRL 0.14/PyTorch 2.11+cu130 selections in design D5). No installs, upgrades, `/env`/driver/container changes, large pretrained downloads, public endpoints, paid/cloud compute or actuation. Missing graphics/Vulkan/assets/API prerequisites block only the corresponding proofs; do not claim completion or provision replacements.

All native/browser checks are explicitly opt-in. Pure CPU tests require no simulator/assets/hardware; native constructor qualification requires only its explicitly stated installed prerequisites. Do not skip a required gate into a pass. A failure preserves actual status and owned cleanup. Record exact commands, revision, seed, device, task/preset/assets, budgets, counters and output locations in fresh approved local evidence **outside Git**. Keep originals unchanged, never copy run evidence into these planning files.

| Milestone | Implementation present | Unit/API passed | Actual native/browser passed | Blocked/unverified at planning time |
| --- | --- | --- | --- | --- |
| Early gate | No implementation claimed | Source constraints only | Not executed | Safe private/no-global-patch seam needs F0 |
| A | Not implemented | Not executed | Not executed | Native/reference and SDK/consumer gates |
| B | Not implemented | Not executed | Not executed | Sensor/renderer/final-frame qualification |
| C | Not implemented | Not executed | Not executed | F0 plus actual browser/read-only acceptance |
| D | Not implemented | Source-plan API probes only | Not executed | Native groups, optimizers, export/continuation |

This table is initial disposition, not a results ledger. During implementation keep detailed outcomes externally and update tracked tasks only on proof. A **NexuML producer** milestone's acceptance task closes after its own listed gates pass. Mandatory **K2** is a separately tracked end-to-end NexuML–NexuFL integration gate: an older consumer pin does not make a passing producer milestone fail, but cross-repository compatibility cannot be claimed until K2 passes. Native/package data structures compiling do not imply acceptance. Small verified commits may describe implementation-only slices accurately, but may not claim an unpassed milestone complete.

## 0. Front-loaded feasibility

### F0 — Private/read-only native construction viability

**Owner:** task 0.1, design D6; **required before substantial implementation**.

Audit exact pinned viewer constructor/sidebar and supported callback/HTTP/WebSocket/native handoff paths. Identify supported read-only/private-host extension or the smallest isolated instance-local adapter that reuses native geometry while bypassing `NewtonViewerViser`'s global `ViserServer` reassignment. Qualify bind-before-listen and mutation denial through bounded actual constructor/dispatcher tests; no vendor edits, global patch, public-first transient socket or second scene renderer. Assert global class identity unchanged and actual listener private at the requested port; close proof resources on every outcome.

**Pass:** a concrete safe seam and small reproducible tests prove viability. Full browser/training acceptance is still B2. **Fail/blocked:** unsupported seam, unsafe writes or missing prerequisites; stop viewer-dependent work, preserve the exact blocker and request a maintainer decision before continuing independent headless A/B work (or D after prerequisites). Do not proceed automatically, remove Viser or claim C/overall completion. Explicitly authorized headless work may pass its own gates without pretending F0 passed. Do not implement other milestones first in the hope that viewer safety will later work.

## 1. Static, API and schema verification

### S0 — Immutable baseline and resource controls

Record installed distribution/source identities, effective interpreter and available approved resources read-only; do not launch/import native runtime just to discover a scenario. Re-resolve NEX-235 consumer branch/revision before K2; distinguish its SDK pin from current producer. Verify the current repository's standard test commands and baseline failures without fixing unrelated functionality. Preflight and monitor disk during each native/server test; stop owned work when free space reaches 4 GiB.

### S1 — Authoring, trust, persistence and exact APIs

Tests accompany existing factory/config/discovery/compiler suites and new focused library tests. Verify function annotations/signatures, JSON arguments, native mutate/replace result, explicit bindings/contracts, unknown fields/task/preset errors, protected count/device/seed/final-observation ownership and native type classification. Block native imports/CUDA allocations for discovery, description, YAML round-trip, static compilation and package metadata load. No import of custom task/config package for actor-only load; trusted task installation needed only at simulation execution. Reject remote target/processor/model/data-selection attempts using existing SDK trust boundaries.

Check one current native schema and direct active fixture migration; no legacy native readers, converters or alternate representations. Preserve ordinary factory/CLI override/general schema behavior. Verify the D5 exact public imports/keys/estimator class/layout through CPU tests against the pinned release, not a later documentation API. No package/stack upgrade is a remedy.

### S2 — Whole-job lifecycle and general regressions

Tests cover normal CLI train and selected evaluation definitions, resumed scenarios, no-op nonnative job scope, safe same-job nesting, conflicting owners and launcher-derived Kit lifetime. Failure injection at startup, allocation, collection, save/export and logging verifies native close happens last, original exceptions/SIGTERM status survive and partial resources close. Exhausted resume/discovery/inference allocates zero simulator. Unsupported scheduled/in-fit evaluation rejects before allocation. Preserve existing supervised/self-supervised/industrial and Gymnasium behavior; ordinary single-agent conventions remain unchanged.

## 2. CPU numerical, gradient and regression tests

### U1 — Sensors, processing and actions

Author deterministic fixtures for RGB/depth/segmentation/mixed observations, layout/dtype/units, normalization, invalid depth/sanitization masks, wrong label vocabulary, missing/undeclared fields and nonfinite tensors. Test the optional caller-owned `CompiledPolicy` state lifecycle for package-owned processing: deterministic initialization, `(observation, state) -> (actions, next_state)`, targeted per-environment/group reset masks, no implicit global history, stable ordering/rates, interleaved independent environments and stateless `forward()` regression. Exercise external-runtime preprocessing/history with exact sequence/reset parity and rate/ordering metadata. Scalar categorical class mapping/event dimensions and continuous bounds/physical decoding require explicit numerical tests. Simulated cached-reset tests verify no duplicate reset and continuing-stream history; actual frames remain N2's responsibility.

### U2 — Single-agent PPO preservation

Run existing numerical/loss/update/collector/evaluation regressions with unchanged PPO/GAE defaults, inputs and seeded outcomes. Assert no MARL defaults or normalization behavior leaks into ordinary PPO. Offline imitation/SDK tests run without constructing RL algorithms.

### U3 — Canonical grouped routing and identity

Assert root live `[N]`, rollout `[N,T]`, children `[N,T,A_g]`, including `N=1`, unequal widths and continuous/categorical groups. Validate explicit order persistence, incompatible sharing rejection, saved per-agent log probabilities (only event reduction), value-like trailing singleton and genuine root team reward broadcasting. Native GAE runs before flattening to root `[M]` / child `[M,A_g]`; counters equal `N*T`. Unequal independent rewards cannot be averaged into learner/team rewards or global success. Cover grouped image/vector and differing adapters in CPU routing tests without claiming native combinations.

### U4 — Native GAE/loss masks and numerical oracles

Use actual native `IPPOLoss`/`MAPPOLoss`/`MultiAgentGAE` and independent small **test-only** numerical oracles. Cover no/partial/all endings, simultaneous flags, real termination versus time-limit truncation, per-agent endings versus global reset and actual final values. Compare native values/advantages/targets/losses and finite gradients at declared tolerances.

Explicit regressions:

- `loss_mask_key=(g,"valid")` masks invalid entries; perturb finite invalid unused rewards/targets and require unchanged valid targets/gradients.
- Real terminal samples stay valid with no bootstrap; timeouts bootstrap actual final actor/critic inputs, never reset values. Reject unsupported separate privileged final-state binding.
- Reproduce the pinned default-normalization custom-mask limitation and `shifted_valid` terminal-counterexample (about `1.0 → 2.8` in the qualified fixture). Assert the new integration instead uses distinct learning validity plus native genuine boundaries and defaults `normalize_advantage=False`, `average_gae=False`, `value_norm=None`.
- If optional normalization is implemented, require valid-target statistics, defined update frequency and U6 continuation for the exact supported layout. Otherwise reject it; optional work is not required to ship default IPPO/MAPPO.
- Do not perturb legitimate physical state while asserting unused-sample invariance. A finished agent may still affect a continuing agent's centralized value.

### U5 — Graph dependencies, gradients and unique optimizer steps

Perturb another agent/group observation: each local actor and IPPO value is independent; declared MAPPO values depend on qualified centralized features. Check homogeneous native network sharing/value-head semantics and heterogeneous `CrossGroupCritic` 4D/3D encoders/trunk/head gradients without padding. Equivalent existing compiled graphs need the same dependency proof.

Assert actual shared parameter object identity (`functional=False` integration), unique optimizer slots, no duplicate shared critic step, cleared stale gradients and correct collection-policy sync. Warm independent actor optimizers with nonzero momentum/decay, then make one actor/group all-invalid: its parameters **and optimizer state** must remain unchanged while valid peers move. Entirely invalid batches produce no nonfinite reductions/steps. Packed unshared modules are not assumed safe merely because gradients are zero; use proven separate ownership. Shared actors can move from valid peers. Check real actor **and critic** updates, not only finite scalar loss.

### U6 — Current checkpoint numerical continuation

Under controlled data, RNG and deterministic kernels compare uninterrupted updates to save/restore continuation: actor/critic weights, unique optimizer states, shared aliases, group ordering, counters and qualified normalization buffers match within explicitly justified tolerances (exact CPU equality where supported). Native resume separately proves lifecycle/environment semantics; do not require bitwise uncheckpointed simulator state. Current save/resume required; no historical-checkpoint test matrix.

## 3. Real native Isaac simulation acceptance

### N1 — Native configuration/lifecycle reference matrix

Run ordinary NexuML commands (not hidden upstream trainers) for:

- Manager `Isaac-Cartpole`, direct `Isaac-Cartpole-Direct`, `Isaac-Reach-Franka`; verify canonical native reset/final tensors, physical action mapping and preserved recipe semantics.
- `NexuML-SO101-Joint-Reach`; preserve K1's existing numerical/physical/learning proof, not the six-control vial task.
- One genuinely installed registered custom task with an importable typed configuration function. Verify actual changed task values and declared bindings; deliberately wrong spaces/controls/presets/count/device reject before native stepping.
- Representative task-native physics/preset/renderer selections from the approved plan: PhysX, Newton and standalone OvPhysX where supported. Reject incompatibilities; no universal backend Cartesian-product claim or hidden fallback.

For each claimed reference/path: true native transitions, bounded real NexuML updates, accurate physical-frame budget, partial/completed resume, full export, X1 clean actor inference and fresh-process replay. Exercise success/failure/SIGTERM startup/cleanup and record actual launcher ownership. Do not borrow old EA/native-profile evidence for the current path.

### N2 — Native multimodal and categorical matrix

Use camera Cartpole or an equivalently small maintained registered task to produce **real RGB, depth and semantic segmentation**, plus mixed state/image and scalar categorical control. Verify nonempty **scene-correct** values, units/labels/invalid pixels and actual native control effects; nonblack images alone do not pass. Verify rendered terminal scene/final-buffer identity before autoreset, history isolation and sensor/control timing. Sensor generation remains independent of Viser and no viewer screenshot substitutes for these checks.

Each family needs actual finite updates/changed actors, current checkpoint/resume, X1 parity and native replay under its exact qualified renderer/assets/preset. Missing graphics/Vulkan leaves that combination blocked; never silently substitute a renderer to pass a selected gate. No pretrained weights or convergence run needed.

### N3 — Native grouped IPPO/MAPPO matrix

- `Isaac-Pendulum-MARL-Direct`: verify actual groups/source spaces (approved source inventory: observation widths 4/3), run **both IPPO and MAPPO**, test independent actors/local versus cross-group critic dependencies, actual per-group rewards/boundaries and meaningful actor/critic updates.
- `Isaac-Shadow-Handover-Direct`: source/asset/backend-qualify the actual homogeneous groups (approved inventory: 157/157 observations, 20/20 actions); demonstrate compatible shared-policy identity and actual grouped learning, resume, export and replay. Missing assets is an open gate, not permission to claim Handover support from Pendulum.
- CPU tests cover partial-agent endings even if a chosen native reference ends only globally. Record the native task's actual behavior; do not fabricate agent endings or team success.
- For grouped image/vector and differing per-group action adapters, exercise CPU contracts and require an actual native combined smoke for **each advertised combination**. Until then list it unverified, not supported merely by tensor routing.

All claimed grouped families require native losses/GAE, real actor/critic updates, unique optimizer ownership, correct `N*T` accounting, complete current checkpoint/resume and decentralized actor-only X1 reload/fresh replay without privileged critic data. Numerical correctness and finite updates suffice; convergence/superiority is not required.

## 4. Actual Viser browser acceptance

### B1 — Actual dispatch, private binding and failure tests

After F0, test the implemented pinned adapter's supported native callback/HTTP/WebSocket dispatch and final mutation handoffs, not a fabricated immutable mock. Invoke direct callbacks and forged/unknown/stale message IDs attempting pause/resume/reset/config/action/force/control/timing/budget writes; instrument native mutators and require zero calls/state effects. Prove absence of global server reassignment/vendor changes. Check exact actual localhost listener/port; reject occupied/silently-reassigned/public settings, startup failures and lost qualification; close partial resources. Headless no-viewer paths must work without importing Viser.

### B2 — Genuine training and separate replay browsers

Through localhost and SSH/VS Code forwarding, connect an actual browser to a running native NexuML training job and **separately** to explicitly enabled policy replay. Confirm actual native geometry, camera movement, environment selection, markers and supported live plots. Replay without opt-in opens no listener.

Perform display pause, camera movement, connect/disconnect/reconnect and supported-interface mutation attempts. Compare native/task/policy sensor/control/history/RNG state and counters at equivalent logical steps with a controlled nonmutating baseline; different wall-clock throughput is allowed. Learning must continue without client presence or resetting/pausing. Verify actual bound sockets stay private and no owned listener/thread/simulator survives completion, injected failure or cancellation. This is required even if B1 or headless simulation passes. Do not expose public networking for a test.

## 5. Export/reload and installed-wheel verification

### X1 — Complete portable computation and general format regression

Run existing export/policy/packaging regressions and clean isolated import tests for each new sensor/action/group family. Use existing complete inference packaging on PyTorch 2.11, not an assumed native unshared `torch.export` graph. Verify that a package-owned processor's **definition and initial/reset rules** are portable, that a fresh caller-owned state can be initialized after reload and partially reset without changing other streams, and that ephemeral runtime history is not mistaken for persisted model weights. Preserve the exact existing stateless `CompiledPolicy.forward()` interface. Block imports of `isaaclab`, `isaacsim`, native support modules, `omni`, `pxr`, TorchRL, Viser and the custom task/config package; remove workspace source paths. The artifact executes from declared actor inputs with no critic/optimizer/session construction, preserving sharing, distributions, decoded actions, preprocessing ownership and episode-reset sequences.

Verify dependency metadata reflects actual external inference dependencies and contains no hidden simulator/learner/custom task requirement. For native training networks, any portable topology/weight transfer has action/processing/sharing parity. Test normal current export/reload and fresh native replay separately; inference load must not execute serialized factories. Existing general supervised train packages retain stable entries, sidecars, dependency manifests and continued-training behavior.

Use existing isolated wheel/sdist checks, including `tests/packaging/test_local_fit_installed.py`, through the repository's existing packaging workflow; do not install new packages or replace the pinned active environment to run them. Missing a pre-existing isolated test environment is a reported blocker until separately authorized.

## 6. NEX-234 SDK and NexuFL consumer regressions

### P1 — Semantic/provenance decision tests

In the current receipt schema, compare explicit task semantics separately from original capture/source/package identities. Author fixtures for semantic mismatch with equal dimensions, tampered provenance/data/splits/asset hashes, unchanged-semantic viewer refactor and potentially behavior-affecting runtime change. Preserve dataset bytes/checksums and origin attribution. Changed action/order/units/rates/assets/processors/behavior rejects; same semantics plus changed implementation requires named affected qualification, not blanket recapture. Unknown equivalence blocks reuse until proven or new matching capture/evaluation exists. No legacy reader/hash bypass or historical-data rewriting.

### K1 — Producer SDK, native SO-101 and installed distributions

Run `tests/core/test_local_fit.py`, SO-101 contract/demonstration validation tests, `tests/fixtures/so101_local_fit_contract.json` comparisons and existing SDK scripts/docs/installed-wheel checks. Keep `local_fit`/manifest signatures, CPU/reset defaults, deterministic repeated fits with different initial weights, explicit approved CUDA, exact trainable scopes/frozen tensors, named roles/shapes/dtypes, base/schema identity, counts and finite metrics. Negative requests cannot import executable targets, authorize local paths or expose raw data/optimizer state. Repeated fits revalidate changed private content.

Preserve native SO-101 `[15]` state and five absolute `0.5*action`-rad controls, fixed gripper, 120/30 Hz timing, asset-layer/calibration validation, complete episodes and identity processor. Re-run affected expert/control/learning/held-out complete export/replay quality/safety checks from the existing reference protocol. Expert/offline success alone cannot establish learned closed-loop success; this preserves the existing proof, not a new MARL convergence requirement. Use new capture only when P1's semantic/integrity/capture analysis requires it, retaining originals and fresh current implementation evaluation evidence.

### K2 — Exact read-only consumer contract check

Re-resolve the actual NEX-235 branch/revision; initial target is `feature/nex-235-so101-federated-imitation` at `38cf25c04434899d788521be6bfa960724c2aaae` in design D9's worktree. It pins NexuML `6516822` and exact SDK/scenario file SHA256s, not current `7e60bad`. Respect CPU-only worker/CUDA-hidden requirements; NexuML CUDA acceptance is K1, not a demand to change that consumer.

Compare reviewed files/digests and manifests **before** executing consumer tests. Inspect `clients/python/src/nexufl_client/so101.py`; preserve its source-pin, `so101/<recipe_hash>`, safetensors named schema/roles, reset/full-weight result, approved metrics/counts, private binding/checksum and no continuation assumptions. On changed source pins do **not** monkeypatch them: record a failed/blocked compatibility gate and exact independent NexuFL review delta. On a compatible independently reviewed revision, run focused `tests/federation/test_so101_program.py` and existing installed probe/handoff checks under its already-approved test setup without sibling writes.

Report old/new producer commit/source digests, receipt keys/version, semantic and physical/recipe hashes, any tensor schema changes (normally none), request/result parity and affected consumer fixtures/docs/tests. Actual new values are generated by implementation. Source/schema repinning or fixture updates belong to an independent NexuFL follow-up; no such edits/tasks are implemented here. A synthetic adapter test is not native federation/four-site evidence. Claim K2 pass only after unmodified reviewed consumer checks pass; otherwise preserve the exact **cross-repository integration** blocker. K2 must always be attempted and reported, but a consumer's intentionally older source pin is not a NexuML Milestone A **producer** acceptance gate; no claim of successful NexuFL integration is permitted while K2 is blocked.

## Coverage and completion mapping

Each normative requirement has WHEN/THEN scenarios in its spec. This table connects those requirements to implementation and gate protocols; it is not a substitute for the scenarios or execution evidence.

| Requirements | Implementing tasks | Required verification |
| --- | --- | --- |
| NC1 | 1.1, 1.9 | S1 |
| NC2 | 1.1–1.3, 1.9 | S1, N1 |
| NC3 | 1.4, 1.5, 1.9 | S2, N1 |
| NC4 | 1.3, 1.6, 1.11, 5.1 | S1, S2, P1, X1 |
| NC5 | 1.2, 1.10, 1.12 | N1, X1, K1 |
| NC6 | 0.1, 0.2, 2.5, 2.7 | S0 and every native/server preflight |
| SC1 | 2.1, 2.5, 4.11 | S1, U1, N2 |
| SC2 | 2.2, 2.5, 2.6 | U1, N2, X1 |
| SC3 | 2.3, 2.6 | U1, N2, X1 |
| SC4 | 2.4–2.7, 4.11 | U1, N2, X1; N3 before combined claims |
| V1 | 0.1, 3.1, 3.3, 3.6 | F0, B1, B2 |
| V2 | 0.1, 3.2, 3.5 | F0, B1, B2 |
| V3 | 3.1, 3.2, 3.4, 3.5 | S1, B1, B2 |
| V4 | 0.1, 3.1, 3.3–3.5 | F0, B1, B2 |
| V5 | 3.5, 3.6 | B2 |
| MG1 | 4.1, 4.4, 4.10, 4.11 | S1, U3, U5, X1, N3 |
| MG2 | 4.2, 4.11 | U3, N3 |
| MG3 | 4.3 | U4, N3 |
| MG4 | 4.1, 4.3–4.5 | U3–U5, X1, N3 |
| MG5 | 4.2 | U3, N3 |
| TL1 | 0.2, 2.6, 4.6, 4.10 | S1, U2, U4, N3 |
| TL2 | 4.4, 4.5, 4.10 | U5, N3 |
| TL3 | 4.3, 4.6 | U4, U6 if normalization enabled |
| TL4 | 4.7 | U5, N3 |
| TL5 | 4.8, 4.9 | U6, N3, X1 |
| TL6 | 4.10–4.12 | U3–U6, N3, X1 |
| PD1 | 1.11, 2.2, 2.6, 4.9 | X1 plus N1/N2/N3 replay |
| PD2 | 1.6 | P1, K1 |
| PD3 | 1.6 | P1, K1 native requalification |
| PD4 | 1.7, 1.10, 1.11 | K1, X1 |
| PD5 | 0.2, 1.8, 1.12, 5.2 | Mandatory separately reported K2 integration check; not an A producer gate |
| PD6 | milestone reviews, 5.1, 5.2 | coverage/status/evidence audit and all required gates |

Acceptance aggregation for **NexuML producer milestones**: **A = S0–S2 + N1 + P1 + K1 + X1**; **B = S1 + U1 + U2 + N2 + X1**, with K1 repeated if its contract changes; **C = F0 + B1 + B2**, preserving headless regressions; **D = S1 + U2–U6 + N3 + X1**, preserving earlier gates. Separately report **cross-repository integration = K2**, attempted during A and repeated after any affected producer change or independently reviewed consumer requalification. Failed source pins block integration but **not** otherwise passing A/B producer acceptance. Early feasibility is not C completion; a failed F0 needs explicit maintainer authorization before unrelated headless work continues. Report exact failed/unexecuted gates rather than a blended “tests pass.”

## Planning validation and integrated audit

Use installed tooling, no dependency installation:

```text
openspec validate generalize-isaac-native-configuration-and-marl --type change --no-interactive
openspec validate generalize-isaac-native-configuration-and-marl --type change --strict --no-interactive
openspec status --change generalize-isaac-native-configuration-and-marl
```

Check relative links, capability inventory, unique task/requirement IDs, every requirement's scenarios, every task's requirement/gate mapping and this coverage table. Review native prerequisites versus completion, Viser/IPPO/MAPPO scope, source-plan refinements, general export/SDK preservation and umbrella ownership. Compare historical report hashes and umbrella checkbox sequences before/after; inspect staged planning-only content. Commit explicit authored planning paths only with `NEX-212`; do not include local reports or executable/runtime changes. Conditional push must honor the user's clean-branch condition without force or unrelated-work cleanup.
