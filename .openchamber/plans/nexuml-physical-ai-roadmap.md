# NexuML Physical AI Roadmap

> **Status:** Planning / implementation TODO. **Scope:** NexuML only.  
> **Target branch:** `feature/nex-212-add-reinforcement-learning`  
> **Goal:** Extend NexuML's existing RL and model infrastructure into a reusable framework for developing, training, benchmarking, exporting, and deploying Physical AI policies, beginning with simulation.  
> **Explicit exclusion:** No NexuFL implementation, federation algorithms, fleet orchestration, aggregation, or NexuFL repository changes in this roadmap.

## Why this plan starts with an audit

The target branch **already implements much of the RL foundation**. The existing `openspec/changes/add-reinforcement-learning/` tasks identify as completed: `InteractionContract`, `EnvironmentDefinition`/`EnvironmentRuntime`, `CompiledPolicy`, `NexuSession` + Lightning/TorchRL training, PPO, Gymnasium CartPole/Pendulum, a MuJoCo Humanoid and Reacher reference, ROS 2 transport/fake tests, policy exports, and test-tier infrastructure. **Do not rebuild these components.** Verify their actual functionality and focus on demonstrated gaps.

Two items remain unchecked in that existing OpenSpec change: live headless ROS 2 + Gazebo + `ros2_control` integration (9.5), and live CARLA smoke testing (10.4). ROS/Gazebo is in scope later below; CARLA is not on the critical path of this plan.

This roadmap is stored in `.openchamber/plans/` and intentionally describes NexuML work only.

## Architectural principles (apply to every phase)

1. **Reuse the repository, not a parallel framework.** Keep `ScenarioSpec`, `NexuSession`, the existing compiler, `CompiledPipeline`, `CompiledPolicy`, `InteractionContract`, the `data`/`interaction` split, and the typed registry/serialization model. Do not introduce a second `RobotSession`, `PhysicalAISession`, or training engine without a proven necessity.
2. **Keep training, policy inference, and environment execution independent.** `CompiledPolicy` remains deployable without a simulator or an RL learner. An RL critic/optimizer belongs to a training checkpoint, not a normal inference artifact. IL/VLA training may use different training logic, but should share existing session/configuration/artifact conventions.
3. **Prefer thin adapters over copies.** Integrate Gymnasium, TorchRL, Isaac Lab, LeRobot, MuJoCo, benchmarks, ROS 2, Gazebo, and optional Isaac ROS 5.0 through their maintained public APIs. Do not copy their algorithms, simulators, robot drivers, or entire training stacks into NexuML.
4. **Do not force every policy into PPO semantics.** IL uses offline trajectories/demonstrations; VLA can involve image, proprioception, language, and temporal action chunks. Use the existing supervised training path where it fits. Add a minimal explicit training mode only when a concrete example proves it necessary.
5. **Preserve action/observation meaning.** Record shapes, dtypes, units, frames, action ranges, control rate, modality, temporal context, sensor preprocessing, normalization, action decoding/chunking, and robot embodiment. Fail clearly for incompatible contracts; never silently reinterpret controls.
6. **Build stateful resources at the execution site.** Keep environment/simulator/ROS handles out of serializable definitions and checkpoints. Own lifecycle/teardown explicitly. Do not serialize live envs across Ray/Kubernetes workers.
7. **Keep heavy dependencies truly optional.** Base `import nexuml` and ordinary supervised workflows must work without RL/Isaac/LeRobot/MuJoCo/ROS/Gazebo/Isaac ROS installed. Pin compatible integrations in dedicated environments where ABI/CUDA/Python versions conflict; do not put system middleware into base extras.
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
| 8 | ROS 2 Lyrical, Isaac ROS 5.0, Gazebo and runtime integration | P1 | Live ROS simulation and optional accelerated perception/deployment proof |
| 9 | Multi-embodiment training examples | P1/P2 | Reproducibly train and evaluate policies for arm, humanoid, drone and mobile robot; quadruped optional |
| 10 | Sim-to-real validation and field data | P2 | Controlled hardware deployment, demonstrations and offline retraining with measured sim/real gap |

**Dependencies:** 0 -> 1 -> 2. Phase 3 follows the verified core and can proceed alongside Phase 2. Phase 4 requires Phase 3 and an evaluable manipulation task. Phase 5 requires Phases 3–4. Phase 6 may progress in parallel after Phase 1. Phase 7 requires Isaac Lab + standardized evaluation (2 + 4). Phase 8 reuses the existing ROS adapter, establishes a Lyrical/Isaac ROS 5.0-compatible runtime path, and closes the live Gazebo proof. Phase 9 consolidates the trained policy examples from Phases 2–8, independently adding each new robot task; drone/mobile workflows depend on Phase 8. Phase 10 requires a proven Phase 8 runtime and at least one validated policy from Phase 9. No simulator or embodiment expansion blocks the first working arm implementation.

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

## Phase 8 — ROS 2 Lyrical, Isaac ROS 5.0, Gazebo and deployment integration (P1)

**Objective:** Connect NexuML policy artifacts to real robot software stacks using current ROS 2 interfaces, with optional NVIDIA GPU-accelerated perception and deployment. Start in simulation; physical hardware validation is reserved for Phase 10. Isaac ROS is **not** a simulator or a policy-training framework.

**Version decision (October 2026):** Target **Isaac ROS 5.0.0 + ROS 2 Lyrical Luth** for all **new Isaac ROS integration work**. Isaac ROS 5.0 migrated from Jazzy to Lyrical; do not introduce a new Isaac ROS 4.x/Jazzy dependency to simplify initial setup. Its officially tested x86 platform is Ubuntu 24.04 with an Ampere-or-newer NVIDIA GPU, CUDA 13.2+ and driver 595+; Jetson support requires a compatible JetPack 7.2 platform (Thor/Orin). Use NVIDIA's Lyrical packages/buildfarm and pinned Isaac ROS development environment. **Do not assume Isaac Lab, Isaac Sim, Gazebo, PX4, Nav2, ROS 2, ROS control and Isaac ROS all support the same version combinations:** establish a tested compatibility matrix, isolate incompatible environments and bridge only through explicitly tested supported interfaces. Existing Jazzy-based reference tests may remain independently functional while Lyrical support is added; do not silently break them.

**Isaac ROS 5.0 migration note:** Do not code against the removed \`isaac_ros_nitros*\` package APIs. Isaac ROS 5.0 uses ROS 2 Lyrical's standard \`rosidl::Buffer\` + CUDA buffer backend for accelerated message memory and interoperability. For the visual-SLAM integration, use the renamed \`isaac_ros_cuvslam\` package. Consult the version-5.0 release notes rather than older tutorials.

### 8A. Verify the maintained ROS 2 environment and Gazebo path

- [ ] Audit the existing \`Ros2JointReachEnvironment\`, \`Ros2Transport\`, \`RclpyTransport\`, fake transport tests and discoverable joint-control scenario; reuse instead of replacing them.
- [ ] Complete **OpenSpec item 9.5**: build a live headless ROS 2 + Gazebo + \`ros2_control\` simulation fixture with a concrete robot, controller, reset bridge and repeatable start/stop commands.
- [ ] Prefer a tested ROS 2 **Lyrical** Gazebo reference stack if compatible; otherwise document the exact independently supported ROS distro for this fixture, and test interoperability with the Lyrical Isaac ROS workspace separately.
- [ ] Validate joint-state topics, bounded command publication, timestamps, TF2/frames, ROS simulation clock, timeout and stale-data handling, reset, teardown and actual closed-loop simulated movement.
- [ ] Load an exported \`CompiledPolicy\` through a ROS 2 **inference-only** reference runner; verify determinism, observation preprocessing, action decoding, command-rate matching and controller boundaries.
- [ ] Keep \`rclpy\`, ROS middleware, Gazebo, \`ros2_control\` and external robot packages optional, not base Python dependencies. Include system-level tests and exact prerequisites.

### 8B. Recorded robot data and portable inference

- [ ] Use the existing NexuML inference package: implement the thinnest ROS 2 node/process adapter for policy execution; do not create a new training/session architecture.
- [ ] Map images, depth/perception outputs, joint state and optionally IMU/LiDAR to named \`InteractionContract\` observations with typed shapes, units, coordinate frames, timestamps, calibration and preprocessing metadata.
- [ ] Convert output actions into validated high-level robot-controller inputs; explicitly handle action bounds, stale commands, watchdogs, and safe controller fallback outside NexuML.
- [ ] Support robot data recording and offline ingestion. Prefer existing \`rosbag2\`/MCAP and **Isaac ROS Data Tools MCAP-to-LeRobot** conversion rather than writing a second generic recorder.
- [ ] Validate recorded episode boundaries, action/observation synchronization, calibration, timestamps, resets, missing frames and reproduction of preprocessing between training and inference.
- [ ] Store artifact version, embodiment identity, robot firmware/controller interface, preprocessing, inference timing and local metrics. No automatic upload or hardware updates.

### 8C. Optional Isaac ROS 5.0 accelerated perception and deployment

- [ ] Establish a **separate, pinned** Isaac ROS 5.0/Lyrical environment on supported hardware and document the official install path, GPU/driver/CUDA/JetPack requirements, ROS package availability, and tested compatibility with the chosen simulation pipeline.
- [ ] Follow a maintained official Isaac ROS 5.0 perception example, first running it independently and then consuming its **standard ROS messages** as NexuML policy observations.
- [ ] For the **robot-arm demonstration**, connect \`isaac_ros_foundationpose\` (or a maintained comparable pose-estimation example) to a manipulation policy; measure validity of pose units, coordinate frames, uncertainty/availability and latency.
- [ ] For the **mobile-robot demonstration**, test \`isaac_ros_cuvslam\` for visual localization and \`isaac_ros_nvblox\` for mapping/obstacle representation, with Nav2 where compatible. Confirm that this perception/navigation path is not itself a trained policy.
- [ ] Investigate \`isaac_ros_cumotion\` / MoveIt for acceleration of classical arm planning, keeping it optional and independent of learned policy training.
- [ ] Benchmark sensor-to-policy-to-command timing, GPU memory use, processing throughput, dropped frames, cold startup and failure recovery against a plain ROS 2 baseline. Avoid claiming that using Isaac ROS automatically improves policy learning.
- [ ] Explore **Isaac ROS Deploy** for packaged policies: it can run appropriate **LEAPP-bundled** policies using ONNX/Triton in ROS nodes or \`ros2_control\`, subject to supported converters. Implement an exporter/converter **only if** an actual NexuML model matches the supported LEAPP contracts; otherwise keep the generic ROS 2 \`CompiledPolicy\` runner. Never assume an arbitrary NexuML, ACT or VLA checkpoint can be exported directly.
- [ ] Verify the deployed model's outputs and any supported runtime safety controller against the reference inference result; do not treat NVIDIA's optional safety controller as certification or a replacement for robot safety systems.
- [ ] Check the version-5.0 release limitations before using Jetson, RealSense, cuMotion/vendor packages or specific deployment examples; avoid stale Isaac ROS 4.x NITROS instructions in documentation.
- [ ] Provide standalone, opt-in accelerated-runtime integration tests; all core/NexuML CPU tests still run without Isaac ROS.

**Phase 8 gate:** A version-pinned ROS 2/Gazebo simulated robot accepts an exported NexuML policy and produces validated high-level commands; a second, optional Isaac ROS **5.0/Lyrical** workflow successfully supplies perception data to the same observation contract. Hardware execution is **not** claimed yet.

**Official version-specific references:**
- [Isaac ROS 5.0 release notes](https://nvidia-isaac-ros.github.io/v/release-5.0/releases/index.html)
- [Isaac ROS 5.0 getting started and supported platforms](https://nvidia-isaac-ros.github.io/v/release-5.0/getting_started/index.html)
- [Isaac ROS Deploy](https://nvidia-isaac-ros.github.io/v/release-5.0/repositories_and_packages/isaac_ros_deploy/index.html)
- [Isaac ROS Data Tools](https://nvidia-isaac-ros.github.io/v/release-5.0/repositories_and_packages/isaac_ros_data_tools/index.html)

---

## Phase 9 — Multi-embodiment training and validation (P1/P2)

**Objective:** Prove that the **same NexuML policy, task, training, artifact and evaluation architecture** supports learning for fundamentally different physical systems. An adapter that loads a robot scene without successfully training/evaluating a policy is **not** sufficient evidence.

**Design rule:** Common **interfaces**, not shared trained weights or identical action spaces. Robot embodiments require separate task definitions, actuator/control interfaces, safe control frequencies and policy/data contracts. Reuse maintained Isaac Lab, Gymnasium/MuJoCo, LeRobot, Gazebo/PX4, and ROS 2/Nav2 examples rather than writing new robotics engines.

### 9A. Shared cross-embodiment contracts and evaluations

- [ ] Define a minimal per-task manifest: embodiment and version, robot description identifier (USD/URDF/MJCF as applicable), required controller/actuator semantics, sensors, observation/action spaces, frames, units, control rates, task/reward or demonstrations, benchmark split and supported simulation backend.
- [ ] Validate incompatible policy/robot pairings before execution, especially mismatched joint order, dimensions, units, limits, sampling rates and required sensors. No implicit policy portability across embodiments.
- [ ] Standardize experiment invocation, checkpointing, evaluation and export using the existing \`ScenarioSpec\`/\`NexuSession\`/\`CompiledPolicy\` workflow and trainable model pipeline; add only adapter-specific code.
- [ ] Report task success, average returns where appropriate, seed variability, constraint violations, wall-clock/frames, inference latency and training resource usage. Require task-appropriate reference baselines.
- [ ] Keep each robot's full simulation/ROS/GPU toolchain isolated and opt-in; maintain fast simulator-free contract/registration tests.

### 9B. Articulated robot arm — RL and IL/VLA reference (P1)

- [ ] Consolidate the existing Phase 2 Isaac Lab reaching/lifting and Phase 3–5 manipulation examples into one documented arm workflow; use a supported Franka/UR arm scene if an SO-101 Isaac Lab task is not maintained.
- [ ] **Train** a reaching or cube-manipulation RL policy, and **train** an ACT manipulation policy from demonstrations; optionally evaluate fine-tuned SmolVLA on an equivalent task where action/observation definitions match.
- [ ] Measure pick/reach success on held-out object placements/camera conditions and compare to scripted motion-planning or upstream reference baselines.
- [ ] Export at least one policy for Phase 8 ROS/Gazebo simulated inference, preserving preprocessing and calibration requirements.

### 9C. Humanoid locomotion (P1)

- [ ] Select a maintained **Unitree G1** locomotion task in Isaac Lab (or a maintained equivalent). Record exact robot/task/actuator versions and pin the environment.
- [ ] **Train** a PPO standing/walking or command-velocity-tracking policy, with observations including proprioception/IMU as required by that task.
- [ ] Evaluate on new velocity commands and terrain/friction/domain-randomization settings; check fall rates, episode duration, body stability and action/actuator bounds.
- [ ] Export and reload policy inference without the simulator learner and demonstrate inference on the same embodiment; optionally explore motion imitation later.
- [ ] Keep whole-body control, fall recovery and real-humanoid deployment outside the first proof unless supported upstream components and independent safety validation exist.

### 9D. Quadruped locomotion (P2, recommended)

- [ ] Select an existing ANYmal/Unitree Go2 or similarly maintained quadruped task rather than implementing a locomotion environment from scratch.
- [ ] **Train** a velocity-tracking or simple terrain locomotion policy with PPO and compare to a known upstream baseline.
- [ ] Evaluate tracking, balance/falls and generalization over contact/friction/terrain variations.
- [ ] Export the trained policy using the same artifact conventions as the arm and humanoid.

### 9E. Drone / UAV (P1/P2)

- [ ] **Train** a small quadrotor hover or goal-position-tracking policy in a supported simulation task (for example a maintained Isaac Lab Crazyflie environment).
- [ ] Validate episode resets, 3D motion, target tracking, control step timing, thrust/angular-rate or higher-level action semantics, and disturbances.
- [ ] Separately demonstrate **PX4 + Gazebo software-in-the-loop** integration using the maintained flight controller and ROS 2 bridge when a compatible Lyrical stack is available; do not invent direct motor-command control.
- [ ] Integrate only high-level policy commands within PX4's supported offboard/trajectory interface and validate failsafe behavior and offboard loss in simulation.
- [ ] Compare tracking error, stability and collision/failsafe metrics to the supported PX4 classical-controller baseline; do not claim flight-ready hardware deployment.

### 9F. Wheeled autonomous mobile robot / AMR (P1/P2)

- [ ] Select a maintained ROS 2/Gazebo mobile platform (e.g. a TurtleBot-like differential-drive robot) with a reproducible navigation testbed.
- [ ] Establish a **Nav2 classical baseline** for mapping/localization/path following; use Isaac ROS 5.0 cuVSLAM/nvBlox only where the supported Lyrical toolchain permits.
- [ ] **Train** a lightweight learned waypoint-following, local obstacle-avoidance or goal-reaching policy in simulation, with explicit action limits, and compare on the same tasks to the classical navigation baseline.
- [ ] Evaluate unseen obstacle layouts, localization error/delays, goal success, collisions, intervention counts and inference timing.
- [ ] Export the policy and run the simulation-only ROS 2 runtime integration; never interpret classical Nav2 alone as evidence of learned robot behavior.

**Phase 9 gate:** At minimum **four distinct policy-training proofs** through NexuML (arm, humanoid, drone, mobile robot), each with repeatable evaluation and a policy artifact. Quadruped is a recommended fifth validation. If a required simulator/ROS combination is incompatible, record the blocker and use an independently supported equivalent rather than fabricating a working integration.

---

## Phase 10 — Sim-to-real policy validation and real-world data loop (P2)

**Objective:** Demonstrate that an already trained NexuML policy can be safely evaluated on a **real** robot and that newly recorded local experience can support **offline** model improvement through NexuML. This is a NexuML-only workflow; no NexuFL aggregation or fleet features.

### 10A. Choose hardware and establish safety

- [ ] Start with **one low-cost robot arm**, such as SO-101 with a known supported control stack, or another arm where the controller, documentation and emergency-stop provisions can be validated. Do not start sim-to-real with a humanoid or drone.
- [ ] Record real hardware/firmware, URDF or other model, sensor and camera calibration, frames, joint ordering, limits, actuator command type, action rate, latencies and physical workspace.
- [ ] Independently validate the low-level control stack, collision and actuator limits, physical emergency stop, human intervention path, protective workspace limits, and ability to return the arm to a safe state. The NexuML action bounds alone are not a safety mechanism.
- [ ] Create operator-reviewed go/no-go criteria and fail-safe behavior. No normal CI, unattended run, or exploratory online RL is permitted to actuate real hardware.

### 10B. Controlled deployment and evaluation

- [ ] Load a previously simulation-validated \`CompiledPolicy\` or a demonstrably equivalent supported policy export using the real ROS 2/vendor interface (Isaac ROS Deploy optional if artifact-compatible).
- [ ] Verify the full observation -> preprocessing -> inference -> command path against offline recorded samples **before enabling motion**.
- [ ] Start with low-speed, limited-workspace, supervised motions and known object/target configurations; measure success, intervention rate, tracking errors, inference/control latency and safety-trigger behavior.
- [ ] Run matched simulated and real scenarios to characterize domain gaps in observations, camera calibration, robot dynamics, delays, contacts and task distribution.
- [ ] Retain a well-tested fallback policy or classical controller and a manual rollback procedure; no hot-swapping safety-critical actions mid-execution.

### 10C. Collect and learn from real experience (NexuML only)

- [ ] Collect operator-supervised demonstrations, task outcomes, error/failure cases and relevant camera/joint recordings locally, with consent/data-handling appropriate to the facility.
- [ ] Convert to the same validated LeRobot/episode data format from Phase 3, preserving timestamps, calibration, policy version, action/observation units and train/validation separation.
- [ ] Use NexuML to **fine-tune offline** an ACT or other already integrated manipulation policy on captured real-world experience.
- [ ] Re-evaluate the new policy in simulation and on held-out recorded data, check forgetting/regressions and compare to both the initial model and local-only baseline.
- [ ] Promote the fine-tuned policy to another **operator-approved** low-speed hardware test only if review gates pass; otherwise retain the previous validated policy.
- [ ] Publish a reproducible sim-to-real report including limitations, measurements, artifact versions and failures, not just a demo video.

**Phase 10 gate:** One physical arm safely executes a validated NexuML policy under supervision; its locally recorded experience is used to train a second offline model; both are evaluated with an explicit human-controlled promotion/rollback process. No NexuFL or autonomous fleet updates.

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
- Full custom robotics hardware, drone flight stabilization, safety-critical motor-control software, or production robot certification. Physical validation is restricted to an operator-supervised arm demonstrator in Phase 10.
- CARLA expansion, world models, diffusion policies, multi-agent RL and additional foundation-model families before the corresponding core examples work. These may become separate, evidence-driven plans.

## Exit criteria for the NexuML-only program

NexuML is ready for a **subsequent, separately scoped** Physical AI / NexuFL integration effort when it can:

1. Train and evaluate standard RL tasks using its existing TorchRL/PPO architecture and preserve supervised compatibility.
2. Train/evaluate an articulated robot manipulation policy in Isaac Lab and an IL manipulation policy from LeRobot demonstrations, with real benchmark evidence.
3. Load/evaluate and, where supported, fine-tune a pretrained VLA without reimplementing its foundation model/training stack.
4. Export and reload complete versioned inference artifacts with robot/action/observation contracts, image and language preprocessing and temporal semantics as applicable.
5. Demonstrate an opt-in ROS 2 Lyrical/Gazebo policy runner and a separately validated **Isaac ROS 5.0** perception pathway (and only use Isaac ROS Deploy for supported policy exports).
6. **Actually train** reference policies for an arm, humanoid, drone and mobile robot; validate a quadruped as an optional fifth embodiment. Never confuse simulation compatibility with convergence or robot-hardware portability.
7. Run one operator-supervised real-arm sim-to-real evaluation, capture local real experience, offline fine-tune an integrated NexuML policy, and validate the result before a manually approved re-deployment.
8. Preserve clean optional dependency/environment boundaries; no robotics simulator, ROS middleware or Isaac ROS becomes a mandatory dependency of ordinary NexuML use.
9. Keep federation, fleet learning and NexuFL development out of this roadmap and reserve them for a separate change after reliable single-site learning works.

**Working principle:** First make one end-to-end task excellent; then add adapters and embodiments without redesigning NexuML each time.
