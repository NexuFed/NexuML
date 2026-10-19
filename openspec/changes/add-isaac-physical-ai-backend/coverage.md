# Scope, roadmap coverage and acceptance ledger

## Snapshot and reading rules

Source: `.openchamber/plans/nexuml-physical-ai-roadmap.md`, inspected 2026-10-19.

Source SHA256: `d1d08bf4f2a3cbe2d7c40a6ace9d431e1cc16fdbeeb57688246e779747d8a0f2`.

The snapshot contains **236 checkboxes in 36 contiguous checklist groups**. `Gnn.i` means the one-based checkbox ordinal within the heading identified below; an inclusive range assigns every item in that range, not only its first or last item. Line numbers locate the snapshot, not a permanent API. Summary and shared-definition-of-done checkboxes are counted too; this is not a claim of 236 independent features.

No source checkbox is edited or marked complete by this document. Historical checked items retain their original evidence and limits. Every new requirement/task is still unimplemented. If the source changes, task 1.2 reindexes the changed snapshot and accounts for new/removed items explicitly before using this ledger as completion evidence.

Disposition:

- **Inherited:** retain the existing proof; rerun when a new environment or changed shared code invalidates its scope.
- **Included:** required program work, with exact requirement/test/evidence links populated as implementation proceeds.
- **External:** independently owned non-Isaac work; reference rather than duplicate it. It can keep the original whole-roadmap gate open even if an Isaac milestone passes.
- **Manual/resource gate:** included, but requires approved hardware or infrastructure; unexecuted means incomplete.
- **Conditional:** the source itself makes the action contingent on a measured need, compatible target or device. Deliver its explicit compatibility/feasibility assessment; do not advertise unsupported execution.

## Capability keys

| Key | Spec path under `specs/` |
| --- | --- |
| C | `physical-ai-environment-composition/spec.md` |
| T | `physical-policy-contracts/spec.md` |
| S | `isaac-simulation-workflows/spec.md` |
| L | `isaac-torchrl-learning/spec.md` |
| D | `robot-experience-workflows/spec.md` |
| I | `robot-imitation-learning/spec.md` |
| V | `vision-language-action-workflows/spec.md` |
| W | `physical-world-model-learning/spec.md` |
| E | `physical-policy-evaluation/spec.md` |
| R | `ros-physical-ai-runtime/spec.md` |
| B | `isaac-embodiment-workflows/spec.md` |
| P | `physical-policy-deployment/spec.md` |
| X | `isaac-execution-and-inspection/spec.md` |

## Exhaustive source-checklist assignment

| Source IDs | Count / snapshot lines | Heading / obligations | Disposition and acceptance destination |
| --- | --- | --- | --- |
| G01.1–12 | 12 / 10–21 | Coarse Steps, including Phase 0 follow-up and Phases 1–10 | Summary indicators only: derive from G02–G35 evidence and task 17.4; never independently check them from a lower-level smoke. |
| G02.1–8 | 8 / 81–88 | Original Phase 0 audit | Inherited audit; C/T/L/E/X, tasks 1.1–1.4 and 17.1 preserve old failures and audit changed seams. |
| G03.1–5 | 5 / 93–97 | Five-contract, manifest, multimodal, provenance and non-robot audit | Included: T/D/E/B, tasks 3.1–3.7, 7.2, 13.1 and 14.6. |
| G04.1–15 | 15 / 107–121 | Phase 1 Gymnasium, checkpoints, evaluation/media | Inherited; regression obligation in tasks 3.7, 4.7, 11.1–11.2 and 17.1–17.2. No Gymnasium rewrite. |
| G05.1–10 | 10 / 131–140 | Original Phase 2 Isaac core | Inherited EA proof plus required canonical migration: C/S/L/X/B, tasks 2, 4 and 5. Do not reuse old GPU-image results as new-stack evidence. |
| G06.1–4 | 4 / 145–148 | Observable inputs, randomization, held-out robustness, system identification | Included: S/T/I/E/P, tasks 5.7, 8.5, 11.3, 14.5 and 16.4. |
| G07.1–10 | 10 / 158–167 | LeRobot data, ACT, temporal contracts, full round trip | Included: D/I/T/E, tasks 7.1–7.2 and 8.1–8.8, including the user-selected Diffusion comparison, with package/state parity in 3.6 and explicit non-RL evaluation configuration/CLI acceptance in 3.8–3.9. |
| G08.1–6 | 6 / 172–177 | Dataset v3, leader/follower, corrections, quality, provenance, offline replay | Included: D/I/P, tasks 7.1–7.7, 8.1–8.3 and manual capture 16.3–16.4. Prerecorded data does not establish live teleoperation. |
| G09.1 | 1 / 185 | Inspect actual LeRobot benchmark tooling | Included: E/I/V, tasks 1.4 and 11.7; no generic benchmark service. |
| G09.2 | 1 / 186 | Native Meta-World reference | External: `add-mujoco-reference-workflows`; compatibility conflict recorded under C, task 1.5 and design D3. Not silently installed/dropped or replaced by an Isaac approximation. |
| G09.3 | 1 / 187 | Native LIBERO subset | External compatible-benchmark follow-up, not an existing completed change or an Isaac implementation. Task 11.7 records its unresolved owner/compatibility; original Phase 4 remains open until its exact suite is actually delivered. |
| G09.4–9 | 6 / 188–193 | Protocols, outcomes, two-method comparison, variation, results and tests | Included common evaluation: E/I/V, tasks 8.4, 8.8, 9.7, 9.11 and 11.1–11.7; native benchmark-specific proofs remain with the external suites. |
| G10.1–5 | 5 / 198–202 | Frozen regression/release suite and candidate decisions | Included: E/P, tasks 11.3–11.7 and 16.6–16.7. |
| G11.1–10 | 10 / 210–219 | Pretrained VLA, processing, adaptation, metrics, export and later families | Included: V/T/E, tasks 9.1–9.11 and non-RL package CLI replay in 3.8–3.9. SmolVLA, GR00T and user-selected resource-gated π0.5 are required references, not an unbounded model catalogue. |
| G12.1–3 | 3 / 224–226 | Full/parameter-efficient adaptation, forgetting and runtime limitations | Included: V/E/P, tasks 9.5–9.11 and 16.6. Supported deployment targets remain parity/resource gated. |
| G13.1–7 | 7 / 234–240 | Existing MuJoCo references and justified alternative adapters | External: `add-mujoco-reference-workflows`; C/E/L regression and compatibility ownership under 1.3–1.5. Newton MJWarp does not satisfy native Gymnasium MuJoCo acceptance. |
| G14.1–10 | 10 / 248–258 | Arena compatibility, suite execution, variations, failures and reports | Included: E, tasks 11.4–11.7. Published legacy installation is not shared-stack proof. |
| G15.1–6 | 6 / 272–277 | ROS transport audit and live Gazebo/ros2_control | Included: R, tasks 12.1–12.4 and 12.8. Original RL 9.5 stays open until actual Gazebo proof. |
| G16.1–6 | 6 / 281–286 | ROS inference, sensor mapping, controller bounds, MCAP and metadata | Included: R/T/D/P, tasks 12.2–12.3, 12.7 and 16.2–16.4. |
| G17.1–10 | 10 / 290–299 | Isaac ROS 5.0 perception, planning, deployment and limitations | Included: R/X, tasks 12.1 and 12.5–12.8. Converter/hardware-specific actions remain conditionally gated by actual supported contracts; the generic runner must work. |
| G18.1–4 | 4 / 303–306 | OPC UA and conditional MQTT/industrial ingestion | Included read-only reference and conditional gateway assessment: D/B/R, tasks 13.8–13.9. MQTT is added only for a concrete reference; no broker/IoT platform or actuator writes. |
| G19.1–5 | 5 / 342–346 | Machine index, native baseline, first-arm choice, licensing and classical baselines | Included: B/T/C, tasks 1.4, 13.1 and per-machine native proofs. |
| G20.1–5 | 5 / 350–354 | Shared embodiment contracts, compatibility, workflow, metrics and toolchains | Included: B/T/E/X, tasks 3, 11, 13 and 15. User-approved one primary environment replaces isolation by default; incompatible native processes remain separated and conflicts explicit. |
| G21.1–5 | 5 / 358–362 | SO-101/OpenArm RL/IL/VLA, bimanual and runtime export | Included: B/I/V/R, tasks 5, 8, 9, 12.2–12.3 and 13.2. |
| G22.1–5 | 5 / 366–370 | Exact G1 locomotion and bounded hardware scope | Included: B/L/E, task 13.3 and shared export/evaluation. Real humanoid hardware remains excluded; optional Humanoid Lite is a distinct reviewed reference, not a substitute. |
| G23.1–4 | 4 / 374–377 | Go2 training, robustness and artifact | Included: B/L/E, task 13.4. This expanded program requires the fifth embodiment rather than borrowing another robot's proof. |
| G24.1–5 | 5 / 381–385 | Crazyflie, separate PX4 SITL, high-level commands and failsafes | Included: B/R/L, tasks 13.5 and 13.10. PX4 requires actual policy offboard/trajectory commands, injected offboard loss and matched classical tracking/failsafe comparison. No physical flight or false Crazyflie/PX4 equivalence. |
| G25.1–5 | 5 / 389–393 | TurtleBot3, Nav2/perception, learned policy, evaluation and ROS | Included: B/R/E, tasks 12.6 and 13.6. Unsupported exact assets/tasks remain blocked, not silently renamed. |
| G26.1–5 | 5 / 397–401 | UR5e, LumenPnP offline utility, simulator feasibility, OPC UA and industrial metrics | Included/conditional: B/D/R, tasks 13.7–13.9 and 14.6; maintained gear insertion is additional, not a UR5e substitution. |
| G27.1–4 | 4 / 415–418 | Hardware identity, independent safety and operator criteria | Manual gate: P, task 16.1. |
| G28.1–5 | 5 / 422–426 | Artifact parity, low-speed trial, sim-real gap and fallback | Manual gate: P/R/T, tasks 16.2–16.4 and 16.7. |
| G29.1–6 | 6 / 430–435 | Local recording, validated conversion, offline fine-tuning, regressions, promotion and report | Included plus manual gate: P/D/I/E, tasks 16.3–16.8. |
| G30.1–5 | 5 / 439–443 | Curate/select/correct/retrain loop, forgetting and bounded adaptation operation | Included: P/D/E, tasks 16.5–16.8; no federation or fleet implementation. |
| G31.1–6 | 6 / 455–460 | X1 episode/stream contracts, provenance, interchange and quality | Included: T/D, tasks 3.1–3.3, 7.1–7.7 and 13.8. |
| G32.1–6 | 6 / 464–469 | X2 complete artifacts, clean parity, processors, reference runtime and trust | Included: T/P, tasks 3.4–3.7, per-family export proofs, 12.7 and 16.2. |
| G33.1–5 | 5 / 473–477 | X3 frozen suites, robustness, shadow evaluation and safety boundaries | Included: E/P, tasks 11.1–11.7 and 16.6–16.8. SARM assessment 11.8 explicitly preserves measured-success and safety boundaries; it is not a substitute for release evaluation. |
| G34.1–6 | 6 / 481–486 | X4 corrections, selection, continual and self-supervised learning, offline RL and site-local operation | Included: D/I/L/P, tasks 6.3, 7.3, 14.6 and 16.5–16.8. No on-hardware learning. |
| G35.1 | 1 / 490 | Hierarchical skill reference | Included: W/P, independent task 10.9 after atomic policy contracts; requires a high-level selector over at least two low-level skills. Task 10.8's learned dynamics alone cannot close it; classical planning remains upstream. |
| G35.2 | 1 / 491 | World models and learned dynamics | Required DreamerV3 workstream: W, tasks 10.1–10.8, after a conventional baseline. User-selected VLA-JEPA/FastWAM assessments 10.10–10.11 are additional feasibility-only deliverables and cannot close the DreamerV3 gate. |
| G35.3 | 1 / 492 | Separately gated research families | Multi-agent learning and measured local/skill adaptation included in tasks 6.6, 10.8–10.9 and 16.6; prototypical-class research and foundation-model pretraining remain external non-goals, not advertised Isaac capabilities. |
| G35.4 | 1 / 493 | General industrial learning compatibility | Included: T/D/B, tasks 3.7, 13.8 and 14.6. |
| G36.1–10 | 10 / 510–519 | Shared per-phase definition of done | Applies to every implementation group: tests/docs with each slice; integrated evidence and scope audit in tasks 17.1–17.5. |

Ranges above cover 236 source checkboxes exactly once. Task 17.4 must replace group-only evidence with specific tests/results for implemented items; the grouped plan is not completion evidence.

## Isaac ecosystem families beyond the original first increment

The default support matrix is for the pinned upstream and selected references, not moving `develop`. For every family record task/preset, source, prerequisite, license, tested device/topology, command, result and evidence location. Mark unsupported individual combinations honestly. **Required** families cannot be removed or marked complete merely because their upstream integration is difficult.

| Workflow family | Required reference / scope | Capability / tasks |
| --- | --- | --- |
| Manager and direct task authoring | One native validated task per workflow, normal scenario overrides and static discovery | S / 4.3–4.4 |
| Rigid multi-backend physics | Supported Newton, Kit PhysX and standalone OvPhysX task presets; no mixed incompatible process | S / 4.5, 14.2 |
| Asset import and articulation identity | Maintained URDF/MJCF/USD conversion on a concrete required asset | S/T/B / 4.6, 13.1 |
| Cameras and other sensors | Policy-consumed image/depth, range/ray, IMU, contact/wrench and tactile data where selected upstream supports them | S/T / 14.1; missing required families stay blocked |
| Randomization, curricula, system identification | Recorded sampled parameters, complete-task play and held-out distributions | S/E/P / 5.7, 14.5, 16.4 |
| On/off-policy and offline RL | PPO, SAC and IQL with their own data/update semantics | L / 5, 6.1–6.3 |
| Recurrent and asymmetric learning | Per-stream state/masks and privileged critic only | T/L / 3.4, 6.4–6.5 |
| Multi-agent learning | One maintained direct multi-agent task with actual updates and correct agent semantics | L / 6.6 |
| Teleoperation and human corrections | Maintained input/retarget/capture tools, headless prerecorded route and manual actual arm capture | D/P / 7.3–7.4, 16.3 |
| Mimic and synthetic sensor data | Native generation and parity-checked episode conversion, not a copied generator | D / 7.5–7.6 |
| ACT, Diffusion and visuomotor distillation | Offline ACT and Diffusion round trips/comparison plus SO-101 state-teacher/camera-student proof | I / 8.1–8.8 |
| Pretrained VLA | Actual SmolVLA, GR00T and resource-gated π0.5 inference/adaptation/export/evaluation; not every foundation model family | V / 9.1–9.11 |
| VLA RL post-training | One actual pretrained model with a valid action-head objective; RLinf remains a baseline | V/L / 9.6–9.7 |
| World models | Actual maintained TorchRL DreamerV3, state then vision, imagination, complete latent inference and baseline comparison | W / 10.1–10.8 |
| World/action-model research candidates | Separate VLA-JEPA and FastWAM source/API/data/license/resource feasibility verdicts; no implied integration or DreamerV3 substitution | W / 10.10–10.11 |
| Learned progress/reward research candidate | SARM feasibility and validation protocol, keeping learned progress distinct from physical success and safety | E / 11.8 |
| Hierarchical skills | Independent high-level selector, at least two compatible low-level skills, actual transitions and controller/reset checks | W/P / 10.9 |
| Generalization/Arena | Compatible maintained Arena suite, frozen scenes/objects, complete episodes and distributions | E / 11.4–11.7 |
| Multi-embodiment | Exact arm, humanoid locomotion/manipulation, Go2, Crazyflie, mobile and industrial references | B / 5, 8, 13 |
| Advanced contact/material simulation | Supported rigid/soft/particle presets plus a maintained deformable/coupled learning reference | S / 14.2–14.3 |
| Real-to-sim reconstruction | One maintained reconstructed scene with collision and sensor provenance and policy evaluation | S/E / 14.4 |
| ROS, perception and planning | Actual bridges, current Isaac ROS 5.0 components, standard messages and independent Gazebo proof | R / 12.1–12.8 |
| Deployment and local improvement | Reference/optimized parity, approved real-arm trial, offline candidate evaluation and human rollback | P / 16.1–16.8 |
| Remote visualization/media | SSH browser replay, private WebRTC with UDP/NVENC proof and existing bounded media | X/E / 4.7, 15.1–15.2 |
| Scale, HPC/cloud, sweeps/PBT | Existing placement/tuning seams; actual multi-GPU/node and population-update proof where claimed | X / 15.3–15.8 |
| Optimized execution | Reference action/gradient/state parity and measured benefit before optimized kernels/capture | X/W / 10.8, 15.7 |

Other upstream model families, teleoperation devices, hardware variants and backend/renderer combinations are **catalogue extensions**, not an implicit promise that every Cartesian product is mandatory or supported. Foundation-model pretraining, simulator/driver rewrites, unsafe actuation and fleet/federation remain excluded. This boundary does not remove any required family listed above.

## Earlier changes and cross-repository ownership

| Existing work | Retained responsibility | Coordination gate |
| --- | --- | --- |
| `add-reinforcement-learning` | Core RL foundation; live Gazebo 9.5 and CARLA 10.4 remain independently tracked | Task 12.4 can supply real evidence for 9.5; never claim CARLA completion here. |
| `verify-gymnasium-rl-and-record-evaluation` | Completed scoped CPU/Gymnasium/media behavior | Preserve regression and source-step/logger semantics; no repeated implementation. |
| `add-isaaclab-gpu-robotics` | Original 42-task/15-native-test EA baseline and its open 23-item extension | New canonical design is not the old two-profile acceptance. Review/adopt/reconcile forward obligations explicitly; do not rewrite counts or archive silently. |
| `add-mujoco-reference-workflows` | Native MuJoCo/Meta-World and shared success-only outcome behavior | Designate one outcome owner; resolve mandatory-Meta-World conflict through reviewed coordination, not hidden edits. |
| NexuVibe | Python/backend, consumer lock, Docker/native packages, automatic setup and interpreter settings | Reviewed delivery in its owning repository; canonical clean-container acceptance depends on it. |
| NexuFL | Existing consumer dependency/import compatibility only | No federation implementation or repository edits; local adaptation is only a future interface. |

## Milestones and completion meaning

1. **Environment and existing references:** canonical automatic setup, original reference migration, physical/temporal contracts and real SO-101 manipulation. This is useful, not the full backend.
2. **Learning and policy stack:** data/ACT/Diffusion/distillation, additional RL families, real VLA including resource-gated π0.5, DreamerV3 and frozen evaluation. VLA and world models are required, not omitted later work. VLA-JEPA/FastWAM/SARM add bounded assessments, not unapproved full integrations.
3. **Ecosystem execution:** Arena, ROS/Gazebo/perception, exact multi-embodiment, advanced simulation, remote access and scale. Missing native/distributed prerequisites remain explicit.
4. **Full-program acceptance:** integrated packaging/regression evidence plus the approved physical-arm/local-adaptation cycle. No full-completion claim while a required compatibility, distributed or manual gate is open.

Original roadmap exit criteria 1–9 map respectively to L/T, I/S/D/E, V, T/P, R, B, P, C and B/D/P. Independent Meta-World/LIBERO/MuJoCo obligations can keep the **original whole roadmap** open after an Isaac milestone; neither this proposal nor a new-stack install marks them complete.

## Evidence records to populate during implementation

Use existing report/artifact conventions, with a backend report and per-slice manifests rather than a new evidence service. Every result records the consumer lock/source commit, resolved scenario, task/assets/models/data hashes, device/resource budget, command, exit status, sample/frame counts, native outputs and known limits. Retain durable retrievable log/artifact locations and checksums; a scratch `/tmp/opencode` directory alone is not a permanent release record.

Planning status: **all new tasks unchecked; no new installation or native capability established**. Exact blocker and unsupported-target decisions must remain visible. Before closing, audit every group/family against its requirements and evidence and show unresolved external/manual work separately.

### Apply checkpoint — 2026-10-19

Tasks 1.1–1.3 are verified; the planning status above records the original proposal,
not current progress. [Implementation evidence](../../../reports/physical-ai/isaac-backend.md)
retains the fresh baseline, exact 23-item legacy mapping, shared-code owners and
source-derived 236-checkbox coverage assertion. Task 1.4 remains partial and task
1.5 is blocked by independently reproduced candidate dependency conflicts. No new
native capability, sibling change or legacy-task completion is claimed.

### User-confirmed LeRobot model scope — 2026-10-19

The user selected complete Diffusion Policy and resource-gated π0.5 examples,
plus feasibility-only assessments for VLA-JEPA, FastWAM and SARM. Eight new
unchecked tasks (8.7–8.8, 9.9–9.11, 10.10–10.11 and 11.8) bring the plan to
**3/136 verified**. All thirteen capabilities and the 236-checkbox source
fingerprint remain unchanged. Model families are mapped above to existing
capabilities, with concrete requirements in their corresponding specs.

Assessment completion is not model-support completion. Unavailable compute
keeps the complete π0.5 reference open; it does not authorize a cloud job,
weight download or alternate incompatible environment. This scope confirmation
neither approves dependency overrides nor replaces or closes Meta-World work.
