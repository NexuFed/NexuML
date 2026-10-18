# NexuML Physical AI Roadmap

> **Status:** Planning / implementation TODO. **Scope:** NexuML only.  
> **Target branch:** `feature/nex-212-add-reinforcement-learning`  
> **Goal:** Extend NexuML's existing RL and model infrastructure into a reusable framework for developing, training, benchmarking, exporting, and deploying Physical AI policies, beginning with simulation.  
> **Explicit exclusion:** No NexuFL implementation, federation algorithms, fleet orchestration, aggregation, or NexuFL repository changes in this roadmap.

## Why this plan starts with an audit

The target branch **already implements much of the RL foundation**. The existing `openspec/changes/add-reinforcement-learning/` tasks identify as completed: `InteractionContract`, `EnvironmentDefinition`/`EnvironmentRuntime`, `CompiledPolicy`, `NexuSession` + Lightning/TorchRL training, PPO, Gymnasium CartPole/Pendulum, a MuJoCo Humanoid and Reacher reference, ROS 2 transport/fake tests, policy exports, and test-tier infrastructure. **Do not rebuild these components.** Verify their actual functionality and focus on demonstrated gaps.

Two items remain unchecked in that existing OpenSpec change: live headless ROS 2 + Gazebo + `ros2_control` integration (9.5), and live CARLA smoke testing (10.4). ROS/Gazebo is in scope later below; CARLA is not on the critical path of this plan.

The branch currently has no `.openchamber/plans/` folder. This file establishes that plan location.

## Architectural principles (apply to every phase)

1. **Reuse the repository, not a parallel framework.** Keep `ScenarioSpec`, `NexuSession`, the existing compiler, `CompiledPipeline`, `CompiledPolicy`, `InteractionContract`, the `data`/`interaction` split, and the typed registry/serialization model. Do not introduce a second `RobotSession`, `PhysicalAISession`, or training engine without a proven necessity.
2. **Keep training, policy inference, and environment execution independent.** `CompiledPolicy` remains deployable without a simulator or an RL learner. An RL critic/optimizer belongs to a training checkpoint, not a normal inference artifact. IL/VLA training may use different training logic, but should share existing session/configuration/artifact conventions.
3. **Prefer thin adapters over copies.** Integrate Gymnasium, TorchRL, Isaac Lab, LeRobot, MuJoCo, benchmarks, ROS 2, and Gazebo through their maintained public APIs. Do not copy their algorithms, simulators, robot drivers, or entire training stacks into NexuML.
4. **Do not force every policy into PPO semantics.** IL uses offline trajectories/demonstrations; VLA can involve image, proprioception, language, and temporal action chunks. Use the existing supervised training path where it fits. Add a minimal explicit training mode only when a concrete example proves it necessary.
5. **Preserve action/observation meaning.** Record shapes, dtypes, units, frames, action ranges, control rate, modality, temporal context, sensor preprocessing, normalization, action decoding/chunking, and robot embodiment. Fail clearly for incompatible contracts; never silently reinterpret controls.
6. **Build stateful resources at the execution site.** Keep environment/simulator/ROS handles out of serializable definitions and checkpoints. Own lifecycle/teardown explicitly. Do not serialize live envs across Ray/Kubernetes workers.
7. **Keep heavy dependencies truly optional.** Base `import nexuml` and ordinary supervised workflows must work without RL/Isaac/LeRobot/MuJoCo/ROS/Gazebo installed. Pin compatible integrations in dedicated environments where ABI/CUDA/Python versions conflict; do not put system middleware into base extras.
8. **Every integration needs a small runnable example, a tested round trip, and honest prerequisites.** Distinguish CPU unit tests, optional simulator tests, GPU tests, ROS/system tests, and operator-supervised hardware tests. Heavy downloads and system installations must never happen during ordinary CI.
9. **Prefer one end-to-end capability per increment.** Make small, reviewable changes. No broad rewrite, speculative abstraction hierarchy, major dependency expansion, or multi-simulator rollout in one PR.
10. **Safety is external and mandatory on hardware.** NexuML can produce bounded high-level commands, but cannot replace real-time control, collision protection, hardware interlocks, or emergency stops. No autonomous online exploratory RL on real robots as part of this roadmap.

## Sequence and acceptance gates

| Order | Phase | Priority | Main proof / gate |
|---|---|---|---|
| 0 | Audit existing Physical AI/RL foundation | P0 | Verified baseline with failures and gaps documented |
| 1 | Gymnasium RL reference workflow | P0 | CartPole + Pendulum round trips reproducible through NexuML |
| 2 | NVIDIA Isaac Lab | P0 | NexuML trains/evaluates a batched simulated articulated-robot policy |
| 3 | LeRobot + imitation learning | P0 | Train/evaluate a manipulation policy from demonstrations |
| 4 | LeRobot manipulation benchmarks | P0 | Comparable multi-seed results on Meta-World or LIBERO |
| 5 | Pretrained VLA models | P1 | Inference and fine-tuning/evaluation of one VLA policy |
| 6 | MuJoCo deeper coverage | P1 | Existing humanoid/reacher validated; direct adapter only if justified |
| 7 | Isaac Lab-Arena | P1 | Evaluation on varied scenes with standardized results |
| 8 | ROS 2, Gazebo, robot deployment | P1/P2 | Live simulation round trip; later safe supervised hardware test |

**Dependencies:** 0 -> 1 -> 2. Phase 3 follows the verified core and can proceed alongside Phase 2. Phase 4 requires Phase 3 and an evaluable manipulation task. Phase 5 requires Phases 3–4. Phase 6 may progress in parallel after Phase 1. Phase 7 requires Isaac Lab + standardized evaluation (2 + 4). Phase 8 reuses the existing ROS adapter and should finish its real Gazebo proof before attempting physical hardware.

---

## Phase 0 — Audit and stabilize the existing foundation (P0)

**Objective:** Establish what works now; avoid implementing functionality already present in the RL OpenSpec.

- [ ] Read `openspec/changes/add-reinforcement-learning/{proposal,design,tasks}.md` and `docs/how-to/reinforcement-learning.md`. Compare their status claims with code and executable tests.
- [ ] Review `src/nexuml/core/policy.py`, `src/nexuml/interaction/`, `src/nexuml/reinforcement/`, `src/nexuml/training/reinforcement.py`, and relevant library scenario definitions.
- [ ] Confirm registry/serialization, `ScenarioSpec.data` + `interaction` coexistence, training modes, contract validation, `CompiledPolicy` export/load, and inference without a training algorithm.
- [ ] Check the current PPO/TorchRL collector and Lightning ownership, including teardown, evaluation, checkpoint restore, and policy synchronization. Do **not** rewrite this functioning architecture just to introduce robotics.
- [ ] Verify the current library's optional extras for `reinforcement`, `mujoco`, and other simulation paths; inspect the root and library `pyproject.toml` and existing test markers.
- [ ] Run a baseline CPU unit/regression suite and existing documentation/lint/type gates; record exact commands, results, and unresolved environment-dependent tests.
- [ ] Create a small capability/coverage matrix: **present & tested**, **present but not system-tested**, **missing**, **deferred** for Gymnasium, MuJoCo, ROS 2/Gazebo, datasets, IL, VLA, Isaac, benchmarks, and export.
- [ ] Identify only essential, evidence-driven changes to the model/policy contract for temporal action chunks and multimodal inputs. Defer implementation until Phase 3/5 provides concrete needs.

**Gate:** Verified baseline with a gap report. No new generic abstractions solely because another integration is planned.

## Phase 1 — Gymnasium RL: end-to-end proof (P0)

**Objective:** Demonstrate the existing NexuML RL workflow, not implement a competing one.

- [ ] Run the **existing** `cartpole-ppo` and `pendulum-ppo` library scenarios using `nexuml[reinforcement]` and the normal NexuML CLI/session path.
- [ ] Exercise both discrete/categorical and bounded continuous action adapters; verify finite frame budgets, reset/step semantics, termination vs truncation, and reproducible seeds.
- [ ] Confirm at least one real PPO optimizer update, rollout collection, episodic evaluation, and stable return/length metrics on both examples.
- [ ] Confirm YAML resolve/serialize/restore and CLI overrides; do not introduce a special RL-only configuration path.
- [ ] Save/resume a training checkpoint and export/reload a deployable `CompiledPolicy`; verify deterministic inference without loading PPO/collector/environment state.
- [ ] Validate direct/process collector smoke paths; test supported copied-policy synchronization, device restrictions, and graceful cleanup.
- [ ] Add/fix only missing tests for these scenarios; separate fast structural/integration tests from long reward-convergence experiments.
- [ ] Provide concise getting-started docs with exact commands, dependency installation, expected artifacts, and how to evaluate/replay a model.

**Gate:** Both scenarios run, evaluate, resume, export, and reload without manual source changes. CPU smoke tests pass. No unjustified stochastic reward threshold as a regular CI gate.

## Phase 2 — NVIDIA Isaac Lab: GPU robotics simulation (P0)

**Objective:** Use one of the established robot-learning platforms while retaining NexuML policy/training conventions.

- [ ] Choose and document a tested Isaac Lab/Isaac Sim/PyTorch/CUDA/Python combination. Confirm current supported release and compatibility before coding. Maintain a separate, reproducible Isaac environment/container.
- [ ] Run native Isaac Lab Cartpole and one articulated robot reaching/manipulation task **outside** NexuML; record commands and native baseline metrics.
- [ ] Determine the smallest adapter between an Isaac Lab task and the existing `EnvironmentDefinition`, `EnvironmentRuntime`, `InteractionContract`, and TorchRL collector path. Reuse its Gymnasium-compatible API where possible.
- [ ] Correctly handle vectorized GPU simulation: number of environments, batched observations/actions/rewards/dones, device placement, partial resets, episode boundaries, and policy control frequency.
- [ ] Test whether native Isaac Lab vectorization and TorchRL collectors should be composed or kept as distinct execution paths. **Do not** accidentally double-vectorize or serialize live simulation contexts into workers.
- [ ] Run a NexuML-managed Cartpole smoke experiment; compare outputs against the native execution.
- [ ] Run a minimal reaching/manipulation experiment using an articulated robot (SO-101 if a compatible maintained task exists; otherwise use an officially supported arm task).
- [ ] Evaluate a saved policy in a fresh simulator instance; export it and load it for simulation-only inference without bringing up the training learner.
- [ ] Add dependency diagnostics, simulator lifecycle/shutdown tests, static contract tests and opt-in GPU/simulator integration tests.
- [ ] Document headless training and how to replay a trained policy, with explicit VRAM and backend limitations.

**Gate:** One reproducible Isaac Lab robot-control policy trained/evaluated through NexuML; regular CPU CI remains unaffected.

## Phase 3 — LeRobot + imitation learning (P0)

**Objective:** Add the first offline robotics learning workflow using real demonstration formats.

- [ ] Inspect LeRobot's maintained data, policy, processor, trainer and simulation APIs; identify the minimal necessary NexuML adapters and pin compatible versions.
- [ ] Load a small LeRobot demonstration dataset (e.g. PushT or an SO-101 dataset). Preserve episode boundaries, timestamps, train/validation splits, image data, action vectors and dataset metadata.
- [ ] Map named observations/actions into the existing TensorDict/`InteractionContract` conventions. Capture image layout, normalization, robot state, action units, camera calibration assumptions, and timing.
- [ ] Reuse NexuML's existing data source, training/session, evaluation, registry and experiment handling. Design a minimal IL training mode only if the current supervised path cannot express the workflow cleanly.
- [ ] Wrap one maintained imitation-learning policy: **ACT first**. Do not rewrite ACT or introduce a parallel pipeline implementation for external models.
- [ ] Support temporal observation history, action chunks, inference state reset, action selection, preprocessing and checkpoint/normalization round trips.
- [ ] Fine-tune/train the policy on a small demonstration dataset and evaluate it in a matching simulator. Prefer a controlled manipulation task that is reproducible across machines.
- [ ] Save and load both a trainer-resume checkpoint and a deployable policy artifact. Validate inference without the full training stack.
- [ ] Add tiny synthetic/unit dataset tests, episode/chunk and processor contract tests, and an opt-in dataset-backed training smoke. No implicit large dataset downloads in routine tests.
- [ ] Document end-to-end IL usage and compare learning from demonstrations to a simple scripted or RL reference where task semantics match.

**Gate:** A complete NexuML demonstration -> ACT training -> simulator evaluation -> portable policy workflow, with normalized multimodal inputs and temporal action semantics preserved.

## Phase 4 — LeRobot benchmarks: Meta-World and LIBERO (P0)

**Objective:** Make NexuML useful for scientifically comparable robotics-policy experiments.

- [ ] Inspect existing LeRobot benchmarking/evaluation integrations before writing any wrappers. Resolve the intended “LeBot benchmark” label to the actual available LeRobot benchmark tooling.
- [ ] Integrate one **Meta-World** manipulation task as the first fast benchmark (especially useful for RL), using native maintained wrappers when practical.
- [ ] Integrate a small **LIBERO** benchmark subset for demonstration/IL and later VLA policy comparisons, without duplicating its environment or dataset machinery.
- [ ] Define task identity, suite/version, episodes, seed, demonstration split, observation/action format, reset policy, maximum horizon, and success criteria in reproducible configuration.
- [ ] Record task success rates **separately** from episode returns; include completion/failure breakdown, mean/std or uncertainty across seeds, runtime and relevant inference latency.
- [ ] Evaluate at least two methods on the same task/data/observation interface (e.g. maintained reference baseline vs NexuML-run ACT). Avoid invalid comparisons between incompatible environments.
- [ ] Add held-out variations (objects, starting positions, lighting/cameras where supported) and report robustness rather than one training-environment score.
- [ ] Save versioned machine-readable results alongside policy artifacts and the resolved experiment configuration; create a small comparison report.
- [ ] Add fast evaluator/aggregation tests and opt-in simulator/data integration suites.

**Gate:** Reproducible multi-seed manipulation evaluation using a real published benchmark and transparent policy/benchmark provenance.

## Phase 5 — Vision-Language-Action policies (P1)

**Objective:** Fine-tune and evaluate a pretrained multimodal robot policy rather than pretraining a foundation model from scratch.

- [ ] Review actual API/model support in the pinned LeRobot release and choose **SmolVLA first** (or a comparably supported open VLA if compatibility prevents this).
- [ ] Demonstrate pretrained inference on image + proprioception + language observations, including tokenization/text processing and action decoding.
- [ ] Define only the missing extension points for language inputs, multi-camera synchronization, temporal context, action chunk semantics, and stateful vs stateless inference.
- [ ] Reuse LeRobot's pretrained model and fine-tuning mechanics via a thin NexuML adapter; do not fork the VLA model/trainer unnecessarily.
- [ ] Fine-tune on a small, compatible robot manipulation dataset and record frozen vs trainable components, optimizer/precision, dataset identity, and compute usage.
- [ ] Compare pretrained vs fine-tuned success on the **same** held-out manipulation tasks; compare with ACT where the action/observation/task definitions are compatible.
- [ ] Evaluate policy memory requirements, inference latency, action output frequency, and feasibility of edge execution.
- [ ] Support export/load with required processors, normalization, tokenization/model metadata, action decoder, embodiment, and inference contract. Do not claim ONNX deployability for models that do not export faithfully.
- [ ] Investigate LoRA/parameter-efficient fine-tuning as a **measured follow-up**, not a requirement for first inference.
- [ ] Add later VLA families (OpenVLA, GR00T, etc.) only after one reusable integration has demonstrated a real need.

**Gate:** One pretrained VLA can be loaded, optionally fine-tuned, evaluated, and reloaded through NexuML with comparable metrics and no hidden training-only requirements during inference.

## Phase 6 — MuJoCo coverage and alternative simulation (P1)

**Objective:** Validate the existing MuJoCo path; implement new integration only for proven gaps.

- [ ] Run and review the **existing** `mujoco-humanoid-ppo` and `mujoco-reacher-ppo` scenarios and their optional-dependency diagnostics.
- [ ] Confirm high-dimensional proprioception, continuous joint control, action limits, episode reset and reproducible model/environment versions.
- [ ] Validate simulator-specific cleanup and headless smoke tests without pulling MuJoCo into core dependencies.
- [ ] Assess what Gymnasium's MuJoCo environments and Meta-World/LIBERO already provide before building a direct MuJoCo adapter.
- [ ] Implement direct `mujoco`/MJX/MuJoCo Warp support **only if** an identified robot, sensor, vectorization, contact or model-loading requirement cannot be expressed using existing maintained wrappers.
- [ ] If direct integration is justified, keep it small and test contract parity against the existing Gymnasium route.
- [ ] Document which simulator is recommended for lightweight algorithm iteration (MuJoCo) versus GPU-scale training and sensor-rich scene simulation (Isaac).

**Gate:** Existing MuJoCo examples are validated; any additional adapter is justified by a concrete failing use case.

## Phase 7 — Isaac Lab-Arena and generalization evaluation (P1)

**Objective:** Go beyond single-task success toward repeatable physical-policy evaluation across scenes and conditions.

- [ ] Confirm current Isaac Lab-Arena compatibility with pinned Isaac Lab, and inspect existing LeRobot integration to avoid a redundant wrapper.
- [ ] Select one manipulation policy/task that can already be evaluated through Phase 2 or Phase 4.
- [ ] Connect Arena's task execution to the common NexuML evaluation/reporting path, using external task setup and metrics where possible.
- [ ] Test configurable scenes/objects, held-out variations, seeds, reproducible environment versions, and evaluation-run provenance.
- [ ] Produce success-rate distributions and failure categories across variations, rather than only a single average score.
- [ ] Verify policy action/observation semantics match Arena's expected robot task; refuse unsupported action formats instead of silently translating.
- [ ] Add a small opt-in GPU evaluation workflow and tests for metadata/metric aggregation without launching Isaac in default CI.
- [ ] Document how this differs from Meta-World/LIBERO and when developers should use each.

**Gate:** One NexuML policy evaluated on a reproducible Arena task suite, with machine-readable and human-readable generalization results.

## Phase 8 — ROS 2, Gazebo and eventual real robots (P1/P2)

**Objective:** Connect policy artifacts to complete robot systems, beginning with simulation, without turning NexuML into robot middleware.

### 8A. Validate the already implemented ROS 2 reference path (P1)

- [ ] Audit existing `Ros2JointReachEnvironment`, `Ros2Transport`, `RclpyTransport`, fake transport tests and discoverable ROS control scenario before modifying them.
- [ ] Complete **OpenSpec item 9.5**: establish a real headless ROS 2 + Gazebo + `ros2_control` fixture with a known articulated robot and explicit launch/prerequisite documentation.
- [ ] Verify actual joint-state subscription, action command publishing, timeouts, ROS simulation clock, simulator reset, clean shutdown, and closed-loop motion on the production ROS transport—not only fake tests.
- [ ] Load an exported NexuML policy and run it through the ROS 2 simulation adapter. Test model I/O, deterministic evaluation, control period, and bounds.
- [ ] Make middleware version, Python ABI, DDS domain, controller interfaces, and required ROS packages explicit. Keep `rclpy`, Gazebo and robot packages out of base Python installation.
- [ ] Reject stale sensor samples, unknown joint mappings, invalid timestamp/frame conventions, unsupported action semantics and transport failure with focused diagnostics.

### 8B. Data capture and deployment boundary (P1)

- [ ] Demonstrate a **non-training** ROS 2 policy runner or reference application that loads the existing NexuML inference package (reuse `CompiledPolicy`), subscribes to observations, and sends validated high-level commands.
- [ ] Add an optional `rosbag2` or robot-log ingestion adapter that converts recorded episodes into the same typed demonstration dataset path used in Phase 3. Test clocks, missing samples, resets, and episode boundaries.
- [ ] Include deployment metadata for robot identity/embodiment, calibration requirements, sensors, coordinate frames, control frequency, action limits and processor versions.
- [ ] Support recording policy/experiment versions and local run metrics; never require central data upload to run inference.
- [ ] Keep model deployment / active-policy swapping distinct from the robot's real-time motor control; require deliberate operator-driven validation for changes.

### 8C. Specialized embodiments and real hardware (P2)

- [ ] Add a simple **PX4 + Gazebo + ROS 2** drone software-in-the-loop example only after the arm integration is stable. Use existing PX4 flight controllers; NexuML supplies policy-level high-level autonomy rather than safety-critical flight stabilization.
- [ ] Document other robot embodiments (humanoid/mobile platform) as extensions of the same policy/data/inference contracts, not as entirely new NexuML runtime stacks.
- [ ] Select a low-cost robot arm for a supervised hardware trial; verify calibration, action semantics, sensor synchronization, joint/collision limits, physical workspace, emergency stop and manual recovery independently of NexuML.
- [ ] Run a supervised low-speed replay of a previously simulation-validated policy, with explicit go/no-go criteria and safe fallback. No automatic hardware CI or unsupervised exploratory training.
- [ ] Record sim-to-real observations, performance differences and required model/preprocessing changes as input to a later implementation plan.

**Gate:** A tested ROS 2 policy-inference round trip in Gazebo; later, one operator-supervised physical-arm execution under separately verified hardware safety controls.

---

## Shared definition of done for each phase

- [ ] **Functionality:** A minimal example works using publicly maintained upstream APIs and the existing NexuML user-facing workflow.
- [ ] **Architecture:** No duplicate session/compiler/policy hierarchy; no simulator-specific imports in core; registry and resolved configuration remain serializable.
- [ ] **Artifacts:** Every trained policy and evaluation report can be traced to code/config, algorithm, environment/task, dependencies, data/robot identity where applicable, and seeds.
- [ ] **Inference:** Export/load or resume behavior is explicitly tested; unsupported formats and missing preprocessors fail loudly.
- [ ] **Testing:** Unit/contract tests are fast; heavy GPU/simulator/ROS tests are opt-in and actually exercised before their capability is claimed complete.
- [ ] **Documentation:** Install prerequisites, one command to run, one command to evaluate, expected outputs and limitations are documented.
- [ ] **Reviewability:** Each integration is a focused change; retain previous workflows and supervised regression coverage.
- [ ] **Scope:** No NexuFL code, no speculative fleet/federation services, and no unsafe autonomous real-hardware RL.

## What this roadmap deliberately does NOT include

- Implementing NexuFL integration, federated aggregation/personalization, a fleet rollout service, or distributed robot learning. These are separate future plans.
- Writing a physics simulator, duplicating LeRobot training code, or replacing ROS 2/robot control middleware.
- Training a new VLA/world foundation model from scratch.
- Full custom robotics hardware, drone flight stabilization, safety-critical motor-control software, or production robot certification.
- CARLA expansion, world models, diffusion policies, multi-agent RL and additional foundation-model families before the corresponding core examples work. These may become separate, evidence-driven plans.

## Exit criteria for the NexuML-only program

NexuML is ready for a **subsequent, separately scoped** Physical AI integration effort when it can:

1. Train and evaluate standard RL tasks using the existing TorchRL/PPO architecture.
2. Train/evaluate at least one articulated robotic policy in simulation through Isaac Lab.
3. Train an imitation-learning manipulation policy from LeRobot demonstrations and benchmark it reproducibly.
4. Load/evaluate (and where practical fine-tune) at least one pretrained VLA policy without redefining the core training framework.
5. Export a complete, versioned policy inference artifact with accurate observations, actions, preprocessing and temporal behavior.
6. Demonstrate a safe, opt-in ROS 2/Gazebo inference path for such an artifact.
7. Preserve compatibility with existing supervised workflows and clearly separate mandatory, optional and hardware-only dependencies.

**Working principle:** First make one end-to-end task excellent; then add adapters and embodiments without redesigning NexuML each time.
