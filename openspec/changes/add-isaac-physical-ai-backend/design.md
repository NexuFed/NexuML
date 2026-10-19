# Design

## Context

See `proposal.md` for motivation and the thirteen capability deltas for observable behavior. This is a program-sized, staged change: planning readiness is not package, native, learning-quality or hardware readiness.

### Inspected foundation

| Existing seam | Current behavior and consequence |
| --- | --- |
| `core/components.py` | Typed immutable definitions and execution-local build contexts already separate environment, action adapter and learner ownership. |
| `core/types.py` | `ScenarioSpec` composes `data`, `interaction`, `pipeline`, `policy`, training and execution. RL currently rejects multi-device/Ray placement. Tensor contracts do not yet encode full temporal/physical semantics. |
| `core/compiler.py` | One compiler has dataset and interaction input contexts. Keep it; do not introduce an Isaac compiler or select simulation through `execution.kind`. |
| `core/policy.py` | `CompiledPolicy` validates numeric leaves and flattens inference batch axes. This is insufficient for language processing, action chunks and recurrent/latent streams without explicit sequence/state handling. |
| `training/reinforcement.py` | `NexuRLLightningModule` already delegates optimizers and `update(rollout, ...)` to an algorithm-owned module. Reuse this seam for replay/model-based logic, rather than inserting a second trainer. |
| `training/lightning.py` | Dataset training currently produces a pipeline but not the policy assembled in the RL path. Offline IL needs the same deployable-policy assembly from dataset-derived training and a separately validated interaction contract. |
| `cli/main.py` and `evaluate_policy()` | Replay currently requires an embedded `ReinforcementLearningSpec` and reads `training.evaluation`. Offline ACT/VLA export therefore also needs an explicit training-independent evaluation configuration and CLI handoff, not only state/batch extensions. |
| `library/environments/isaaclab.py` and `_isaaclab_runtime.py` | The verified three-task EA provider is intentionally pinned to one source revision, cu128, PhysX, direct single-GPU collection and fresh-process scalar replay. It is not a generic backend or proof for a new release. |
| Existing data, exports and tracking | Reuse typed registration, resolved configuration, datasets, source/dependency manifests, complete policy export and native TensorBoard/MLflow recording. |

`add-isaaclab-gpu-robotics` has 42 original completed tasks and 23 open extensions. Its retained 15 native-image test passes are historical acceptance only. `add-mujoco-reference-workflows` is planned, owns alternative-simulator references and success-only outcome work, but its Meta-World dependency plan conflicts with the selected Newton family. The full roadmap snapshot is indexed in `coverage.md`; no source roadmap or sibling proposal is changed here.

The inspected running DevContainer is Ubuntu 26.04.1 with `compute,utility`, although its prepared NVIDIA configuration selects Ubuntu 24.04 plus graphics. NexuVibe currently declares Python 3.14 and uses `/env`; the user explicitly accepts changing that primary environment to Python 3.12 and CUDA 13. Prepared files are not effective runtime evidence.

## Goals / Non-Goals

**Goals:** One Isaac-first developer experience and one NexuML learning/artifact architecture; broad explicit workflow coverage with native-first, numerical and end-to-end proofs; one automatically provisioned primary Python environment; full inference semantics; secure remote development; staged multi-embodiment and supervised field evidence.

**Non-Goals:** No second session/compiler/trainer, universal robotics superclass, custom physics engine or driver, copied foundation model, compulsory robotics dependency for all users, unmanaged remote code execution, robot-safety certification, autonomous hardware exploration, fleet operation or NexuFL implementation. "Everything" means all catalogued workflow families, not every upstream preset or unsupported Cartesian product. Missing prerequisites leave required gates open; they are not permission to narrow the request silently.

## Decisions

### D1. One umbrella, explicit ownership, no rewritten history

This change owns the forward Isaac-first architecture and full-program acceptance. Preserve the older 42-task result and existing policy packages. Before shared implementation, record which open legacy extension items are adopted here, retained as legacy-profile proof, or require an explicitly authorized artifact amendment. Do not check off the older two-environment extension from a one-environment migration, and do not archive either change by implication.

`coverage.md` is the source-to-capability map. Implementation expands it to exact requirement/test/evidence identifiers and compares it with the source snapshot before completion. The MuJoCo change retains ownership of its native Humanoid/Reacher/Meta-World work; one designated owner implements shared success-only evaluator behavior, with both suites consuming it. Original live Gazebo and CARLA statuses remain unchanged except where this program actually supplies the required Gazebo proof; CARLA expansion remains external.

**Rejected:** duplicate all existing specs under a new name, count old tests as new-release proof, or let two agents edit the same evaluator/contract concurrently.

### D2. One canonical consumer-owned Python environment

Select a compatible release-3.0 family rather than mixing old EA packages with a current tutorial. The following is a source-inspected **candidate**, not a validated combined lock:

| Component | Candidate / authority |
| --- | --- |
| OS / Python | Ubuntu 24.04, CPython 3.12; exact patch recorded by the consumer lock/setup |
| Isaac Lab | Release 3.0 source `d061bf35a58273ef0cc8aceae76392166f9b7d15`, subject to its native and complete-composition gates |
| SO-101 tutorial | `9a4286b4bc8a5720ac97f87e5e0140f7542c22d9`; its lock records the Lab revision above |
| Torch family | 2.11.0+cu130 / torchvision 0.26.0+cu130, with matching audio/codec dependencies selected by the consumer |
| Newton / Warp / MuJoCo | Newton 1.6.1, Warp 1.17.0, MuJoCo/MJWarp 3.12 family from the selected upstream recipe |
| Isaac Sim | 6.2.0.0 candidate for this Lab revision; separate compatibility proof from historical Sim 6.1 |
| TorchRL / TensorDict | Begin at the existing 0.14 family; verify exact installed DreamerV3/VLA/multi-agent APIs and freeze a compatible released pair, not moving documentation or unapproved nightlies |
| ROS accelerated runtime | Isaac ROS 5.0/Lyrical on NVIDIA's Noble buildfarm; select and record Sim's independently supported bridge |
| LeRobot / Arena / GR00T | Select compatible maintained releases and immutable model identities during the first joint-resolution gate; legacy Python-3.11 Arena recipes are not installation targets |

Concrete integration requirements belong in NexuML library extras, with root forwarding where appropriate; ordinary base packages remain simulator-free. NexuVibe selects the backend indexes and authoritative combined `uv.lock`, including its existing NexuFL dependency closure. Keep NexuML's standalone development lock for its own repository checks, not as a second deployed simulator profile. No automatic narrowing of all NexuML users' Python range merely because the configured developer environment selects 3.12.

The old EA/cu128 image remains historical reproducibility material, not a required second daily-use environment. Port/revalidate the references on the canonical stack and version changed contracts; do not erase original artifacts or silently run them against new physics. Any additional compatibility environment requires a concrete conflict and explicit user approval and cannot substitute for canonical acceptance.

**Rejected:** separate venv per framework, making bare `all` mean silently incompatible packages, or treating CUDA toolkit and wheel-runtime minor versions as identical concepts.

### D3. Treat dependency and installation conflicts as first-class gates

Published metadata establishes a real contradiction: Meta-World 3.1.1 requires `mujoco==3.3.0`; Newton 1.6.1 `[sim]` requires `mujoco~=3.12.0`, and MJWarp 3.12.0 requires `mujoco>=3.11.0`. These requirements cannot inhabit one interpreter. Changing Python does not fix them.

Keep canonical Isaac authoritative. Inspect maintained Meta-World alternatives first. If none is compatible, propose an explicit coordinated amendment to the MuJoCo change's mandatory `all` closure or optional mutually exclusive selection; do not alter that proposal or drop the benchmark here without review. Merely removing an extra from `all` can still leave uv universal resolution conflicting: represent real exclusivity in the consumer/repository resolver configuration where needed. The independent roadmap benchmark remains blocked until its own accepted route works. Never patch upstream metadata, use `--no-deps`, pin an older Newton silently or claim full roadmap completion.

Apply the same gate to LeRobot, Arena, GR00T, ROS and every newly selected package. Freeze source/model/asset hashes, licenses and the full transitive closure. Separate dependency presence from actual public API availability and native operation. Install one selected code stack automatically, but fetch large datasets/weights only from the explicitly configured approved reference set, not every upstream download.

**Rejected:** repairing a successful minimal environment after the fact with untracked pip additions or claiming that a lock excluding requested capabilities proves the complete composition.

### D4. Coordinate DevContainer delivery without hidden sibling writes

NexuVibe owns reviewed updates to its `pyproject.toml`, `.python-version`, consumer lock, NVIDIA Docker stage/config, hooks, versioned library paths, VS Code interpreter settings and setup tests. Deliver these in that repository under an explicitly scoped implementation session/change; this proposal's repo-local scope does not authorize hidden sibling edits. Canonical clean-container acceptance depends on the delivered consumer revision.

The creation hook synchronizes the locked shared environment and runs bounded readiness checks; container build/setup supplies system packages and ROS. It verifies source/asset manifests and matching caches, diagnoses partial failures and never overwrites unrelated upstream work. Replace the existing unconditional `uv sync --upgrade` behavior for reproducible creation through the owning change. Do not retrofit OS/graphics or replace `/env` while active jobs use it; recreation timing remains operator-approved.

Check effective CPU/RAM/VRAM/storage quotas, CUDA tensors, Vulkan/EGL, NVENC, system libraries and disk/cache expansion separately. The old 60-GiB installation cap cannot be assumed sufficient for the expanded stack: measure a staged budget and obtain approval before exceeding it. Preserve standing Isaac EULA acceptance, no optional telemetry, and separately reviewed dataset/model terms. Make no host-driver installation part of Python setup.

**Rejected:** import-time installers, full asset-pack downloads, silent failed setup, global library-path hacks across ROS/Kit processes or a mandatory worker container for everyday development.

### D5. Extend existing contracts only where a runnable slice requires them

Map Task, Experience, Learning, Policy Artifact and Evaluation to existing types before adding fields. Extend `InteractionContract`/`TensorFieldContract` with versioned physical/temporal metadata needed by concrete references: embodiment and asset/calibration identity, units/frames/joint order, control mode/scale/rate, sensor identity, required versus privileged groups and validity/timing. Reuse existing model serialization/registry and default absent additions to current stateless meaning. Test old YAML/checkpoint restore and reject incompatible semantic overrides before action.

Keep model tensors numeric. Raw language and image preprocessing enter through a registered processor/layer adapter with versioned assets, producing token/mask/image/state tensors in the agreed TensorDict schema; do not teach the numerical compiler to guess string coercions. External models remain actual layers/components of the existing graph, not another pipeline implementation.

Preserve environment, time and agent axes explicitly. The current `CompiledPolicy.forward_parameters()` flattening cannot indiscriminately flatten sequence time or agent groups. Reuse `PipelineCompileContext` to carry concrete sequence/input metadata and require a sequence-aware layer test before changing flatten behavior. Static discovery and config resolution must not download weights or allocate native sensors; shape contracts come from validated metadata, with actual model construction at the normal explicit runtime boundary.

For stateful inference, extend **the existing** `CompiledPolicy` with an explicit initial-state and step/reset surface carrying state in/out as TensorDict values. Stateless `forward(observation, deterministic=...)` remains unchanged. A scalar/batched executor holds pending action chunks, hidden/RSSM state and previous actions per stream; it is resettable runtime state, not an immutable definition or process-global singleton. Use maintained TorchRL execution/primer machinery where it matches; do not add `RobotPolicy` or `IsaacPolicy` hierarchies.

**Rejected:** every future robotics field in one superclass, arbitrary unchecked metadata as the only action contract, hidden state surviving episode resets, or retraining a tokenizer because text is inconvenient.

### D6. Expand the library provider, not the execution backend taxonomy

Simulation remains `ScenarioSpec.interaction.environment`; `execution` still means placement. Evolve the existing library-owned Isaac provider with explicit task/physics/renderer configuration and allowlisted task parameters resolved from normal NexuML overrides into public upstream task configuration. Keep the native app, task configuration objects, USD stages and ROS handles execution-local. Avoid a general simulator plugin service or a duplicate Hydra configuration truth.

Support manager and direct tasks only after native probes establish their actual reset/final-state APIs. Retain the canonical runtime-to-TorchRL adapter fallback unless maintained `IsaacLabWrapper` public composition passes the complete transition oracle. Verify no-done/partial/all-done, termination/truncation/simultaneous flags, explicit masked resets, mutable buffers and numerical value targets. Never zero-fill unavailable terminal observations, bootstrap from reset states or double-vectorize native batches.

Physics, camera production and interactive visualization are separate choices. Cover Newton MJWarp/Kamino/VBD/MPM, Kit PhysX and standalone OvPhysX through **supported native presets**, with task-specific feature evidence. OvPhysX and Kit cannot share a process. Same task semantics across backends require explicit articulation-order/control/physics validation, not a backend-name override on an old artifact. Reuse upstream URDF/MJCF/USD conversion and retain converted content hashes, gains, collisions, limits and license.

Sensors must be actual input sources when used by a learner: image/depth, contact/wrench, IMU, range/ray and visuo-tactile each have native shape/timing/validity probes. A replay camera remains inspection-only unless included in the trained input contract. Expose upstream randomization and curriculum as versioned task parameters, including delays/noise; record physics-identification values from later traces. Distinguish train, nominal play, held-out and canonical-home reset distributions.

**Rejected:** CPU Gymnasium loops for GPU Isaac tensors, silent backend fallback, generic direct-MuJoCo work inside this provider or sensor support established from fabricated observations.

### D7. Reuse algorithm-owned updates for multiple learning families

Keep `RLAlgorithmDefinition.build(RLAlgorithmBuildContext)` and the algorithm module's optimizer/update ownership. PPO retains ordered collect -> advantage estimation over N/T -> optimize -> synchronize -> collect. Register maintained TorchRL SAC as the first off-policy learner and IQL as the first offline reference; target-network/replay/warm-up/update-ratio state belongs to those algorithm modules, not the environment or a second training loop.

Offline replay learning needs a bounded data-driven iteration path rather than `train_dataloader()` unconditionally building an interaction collector. Add the minimal explicit data-source/budget choice to the existing RL lifecycle and serialization, demonstrated by IQL and Dreamer tests. Do not introduce a top-level session per algorithm. Reuse `TrainingSpec` for supervised IL/VLA losses; add a new training mode only if a concrete upstream model cannot fit the existing optimizer/data/loss interfaces, with a failing test and reviewed boundary first.

Add recurrent PPO and separate actor/critic observation mapping with TorchRL's supported primers/masks/value APIs. A privileged critic is a training module; it cannot add hidden required actor fields. For one maintained direct multi-agent task, preserve agent groups and per-agent/global termination; use maintained IPPO/MAPPO-compatible objectives rather than flattening agents into an unrelated single-agent batch.

Functional proofs require changed parameters, valid gradients/losses, finite and actual counters, remaining-budget resume and portable policy replay. Quality experiments use declared budgets and multiple seeds; do not make stochastic success thresholds default CI requirements. Native RSL-RL/RL Games/RLinf examples remain control/task/quality baselines, not claimed NexuML training.

**Rejected:** force ACT, diffusion/flow policies or world models through Gaussian PPO, run a hidden external trainer and wrap its checkpoint, or add a general learner orchestration framework before the concrete families need it.

### D8. Use maintained episode formats and processors

Implement LeRobotDataset v3 ingestion as a registered data source using existing datasets/loaders. Preserve source episode, timestamp, camera and action identities, masks, normalization and train-only statistics. Validate older supported formats and conversions explicitly. Use native LeRobot leader/follower recording for SO-101 and maintained Isaac Capture/teleoperation for keyboard/gamepad/spacemouse/XR/Haply modes where available. Physical device evidence is opt-in; approved prerecorded data supports headless offline acceptance.

Use public Mimic annotation/generation and synthetic sensor tooling rather than copying their generators. Verify HDF5-to-LeRobot mappings with sampled state/action/frame parity and source-group splits, since descendants of one demonstration are not independent held-out examples. MCAP conversion reuses Isaac ROS Data Tools where compatible; test episode/clock/calibration boundaries across sessions.

Keep simulator, real, scripted, policy and human-correction origins distinct. Track intervention intervals and task outcomes without unauthorized identifiers. Provide a dataset quality/consumption report. Failure/novelty selection and unlabeled robot/industrial representation learning reuse dataset windows/sampling; no universal experience storage engine or custom teleoperation UI.

**Rejected:** copy arbitrary native `info` into observations, normalize with held-out data, implicit large downloads, or invent robot episodes for an industrial stream.

### D9. IL and VLA are models in the existing graph

ACT is the first real demonstration round trip. Wrap maintained policy/processor and masked losses in normal library layer/data definitions, using TorchRL `ACTLoss`/`BCLoss` and LeRobot adapters where their **installed** APIs match. Model training consumes future action chunks; exported execution converts predictions into correctly timed physical-step actions through the same `CompiledPolicy` state surface. Factor policy assembly so dataset training can produce a deployable policy without fabricating PPO or a critic.

Distill the SO-101 vial teacher to camera-plus-joint observations as a second concrete use of sequence/processor contracts. The source state teacher can use training-only information; held-out student inference must succeed with that information blocked. Do not claim hardware readiness from the teacher's hidden object/contact state.

Require actual pretrained SmolVLA and a maintained GR00T release, not only `TinyVLA`. Use maintained LeRobot/GR00T model and preprocessing APIs, with exact model weights/code/processor hashes. Compare full fine-tuning and supported LoRA/adapters on at least one model, preserve composed export identity and test forgetting. Other VLA families remain catalogued extensions, not mandatory foundation-model pretraining.

Use the TorchRL VLA schema/adapters when available: `LeRobotPolicyWrapper`, `LeRobotExperienceReplay`, `ActionChunkTransform`, `ActionScaling`, `MultiStepActorWrapper` and masked `BCLoss`. Verify actual package symbols and behavior rather than infer availability from `main` docs. VLA RL post-training is required on one real pretrained model; select an objective valid for its action-head distribution. RLinf's published GR00T/Isaac mapping informs a native baseline, but do not import its complete parallel training stack to bypass NexuML. If no compatible maintained objective is available, record a required integration blocker rather than inventing Gaussian likelihoods or counting a synthetic token update as the real-model proof.

**Rejected:** fork ACT/SmolVLA/GR00T, add a remote model microservice by default, promise faithful ONNX export for every processor/head, or label an untested family supported because its checkpoint downloads.

### D10. DreamerV3 uses maintained TorchRL components

Versioned TorchRL 0.14 documentation describes actual V3 components: `RSSMPriorV3`, `RSSMPosteriorV3`, `RSSMRolloutV3`, `RSSMStateEstimatorV3`, `DreamerV3ModelLoss`, `DreamerV3ActorLoss`, `DreamerV3ValueLoss`, `DreamerV3Loss` and supporting update/normalization tools. Verify installed source/API behavior and pin it before model integration. This is stronger evidence than the user's generic world-model link, but still not installed/runtime proof.

Implement the library algorithm using the existing session and algorithm-owned update seam. Start with a small state-based Isaac task, then a bounded visual arm task so the backend proves both dynamics and sensor representation. Replay carries contiguous N/T streams, episode-first/reset masks, real final observations, optional burn-in and generation-checked posterior context. Use explicit paper-compatible versus author-recipe settings: categorical RSSM, KL/free-nats/unimix, vector/image reconstruction, reward/continuation heads, actor/value targets and slow-critic update behavior. Do not confuse current defaults with reproduced paper settings.

Lightning must have one effective owner for optimizer stepping, precision and checkpoint state. Prefer maintained compositional losses with Lightning-owned stepping; use `DreamerV3OptimizationStepper` only through a public integration that exposes its state and prevents duplicate optimizer/scaler/target updates. Verify with step counts and state round trips. The target updater, percentile/normalization state, policy RNG and fractional update ratio are checkpointed. When replay is omitted on resume, close old stream tails and explicitly refill/reset owed updates; restarted physics is not bitwise trajectory continuation.

Export encoder + observation-conditioned latent filter + actor + necessary action/normalization/state initialization. A Dreamer actor alone is not a deployable policy. Keep training decoder, critic, imagination losses and replay out unless a particular deployed planner genuinely needs them. TorchRL inference components can remain declared runtime dependencies; "inference-only" forbids training-only construction, not every library that also offers training. Preserve the old PPO tests that genuinely require no TorchRL at inference.

Compare predictive errors by horizon and task success/sample efficiency against the existing PPO reference with matched observation and physical-frame budgets. Separate real collected frames, replay samples and imagined transitions. A negative comparative result is valid evidence, not grounds to hide it. Add a bounded learned-dynamics planning reference and an independent hierarchical selector-plus-controller reference after the atomic policies are stable. The hierarchy selects at least two compatible existing skills, records transitions and task completion, rejects invalid targets and resets pending state at episode boundaries. It can reuse upstream planning tools and does not require a new foundation model. A dynamics-only demonstration cannot close the hierarchical gate; uncertainty and imagination never replace physical safety.

**Rejected:** replace V3 with older Dreamer losses, embed the author JAX runner as a second NexuML trainer, count generic future-state prediction as full Dreamer support, or enable optimized scan/Triton/capture modes before reference parity.

### D11. One release evaluator, with Arena and task-aware outcomes

Keep the current `evaluate_policy` scalar path and native loggers. Extend for explicit policy state and optional native GPU batches without counting partial episodes, leaking row state or resetting training. The success-only evaluator extension is shared with the MuJoCo change; preserve existing SO-101 paired error semantics. Task predicates decide whether transient/terminal/held success counts; do not universally reinterpret physical placement as distance or return.

Add an optional trainer-independent `ScenarioSpec.policy_evaluation` block reusing the existing episode/media option schema; do not overload the existing dataset `evaluation` block. It is authoritative when present. Existing RL packages without it continue reading `training.evaluation` and retaining their documented override path. For offline ACT/VLA, export the original `TrainingSpec`, complete policy, declared interaction/evaluation environment and this block; do not fabricate an RL algorithm or mutate the training mode to make replay pass.

Factor episode execution to consume the validated policy, environment definition and episode/media options independently of training. Preserve the existing scenario-based call through normalization. Episode seed is explicit, with the historical RL training-seed fallback only for old RL configurations. The ordinary CLI accepts complete offline-trained policy packages, permits `policy_evaluation` overrides and retains legacy `training.evaluation` overrides only on the legacy configuration path; ambiguous mixed paths fail clearly. Contract compatibility, trusted loading, device placement, logger finalization and media constraints remain unchanged. Require clean-package ACT and VLA CLI tests that block session/learner/optimizer/collector construction, exercise scalar and recorded replay and reject missing evaluation inputs; retain old RL CLI/override regressions.

Freeze suite/task/asset versions, sensor/processor/control contracts, episode/reset policy, seeds, held-out conditions, baseline identity, thresholds and sample sufficiency. Emit per-trial machine-readable results and a small pass/fail/insufficient-evidence comparison with uncertainty and worst-condition evidence. Include latency tails, interventions, dropped frames and constraint violations where measurable; statistical success is not safety certification.

Use maintained Arena/LeRobot EnvHub composition for one actual adapted policy and scene/embodiment/object variations. The published Python-3.11/Lab-2.3/Sim-5.1 recipe is incompatible with this target; select a maintained compatible release and prove it natively. Do not write an Arena replacement or port a legacy suite silently. Native Meta-World/LIBERO comparisons retain their own task protocols and compatibility gate, not reconstructed approximations advertised as the same benchmark.

For inspection media, reuse bounded GIF/MP4/both, TensorBoard animation, MLflow-only handling and final-frame-before-reset capture. For shadow/offline evaluation report action/state agreement, not unobservable counterfactual success. Keep task quality separate from bounded integration tests.

**Rejected:** generic benchmark services, duplicate video/logger runs, higher reward as promotion permission, or using simulator scores to certify hardware safety.

### D12. ROS is an external hosting and sensing boundary

Reuse `Ros2JointReachEnvironment`, its fake/live transport seams and the existing complete-policy loader for an inference-only node/process. Standard messages carry images, state, poses and high-level commands; retain frame/units/clock/calibration, QoS, freshness and rate checks. Controllers own watchdogs, safe fallback, collision limits and emergency stop. Do not make `rclpy` or native middleware a core pip dependency.

Isaac ROS 5.0 targets Lyrical on Ubuntu 24.04 through NVIDIA's Noble buildfarm, CUDA 13.2+ and driver 595+. The selected Sim 6.2 documentation recommends Jazzy, so do not claim Lyrical bridge support without a native compatibility test. Separate process environments/workspaces prevent ROS Python/ABI/library contamination; use the same canonical Python where supported, but do not force differently generated messages or native libraries into Kit. Default to standard interfaces; custom messages require a recorded Python-3.12 build and wire proof. Any distro boundary must prove discovery, QoS, timestamps, reset, delivery and error behavior. Cross-distro DDS is a gate, not a supported assumption or automatic fallback to older Isaac ROS.

First prove Sim observations -> policy -> simulated arm commands -> MCAP -> validated episodes. Independently close the original live Gazebo/ros2_control fixture; Sim success does not satisfy it. Establish FoundationPose input for manipulation, cuVSLAM/nvBlox for mobile perception, and optional compatible cuMotion/MoveIt planning. Test against plain ROS for latency, memory, frame loss and failures. Use current standard buffer APIs rather than removed NITROS packages and retain known release limitations, including RealSense and vendor package constraints.

Isaac ROS Deploy is an optimized host only for actual compatible LEAPP/ONNX/Triton contracts. Add a converter only after the model satisfies those contracts and parity passes. Otherwise the reference PyTorch `CompiledPolicy` runner is the supported runtime, not a false universal conversion claim. Read-only OPC UA can supply industrial observations independently; protect credentials and never implement automatic actuator writes.

**Rejected:** ROS as a training framework, safety implemented solely by action clamping, arbitrary checkpoint-to-LEAPP conversion, or sourcing multiple distro libraries globally into every process.

### D13. Scale and inspect without replacing placement infrastructure

Start with one learner/GPU/native batch. Enable recurrence/asymmetry before distributed scale, and only lift current `ScenarioSpec`/provider device/topology guards as corresponding tests land. Multi-GPU uses one owned simulator per rank with explicit CUDA remapping, rank seeds, global N*T accounting, supported gradient/target synchronization and rank-zero artifact publication. Multi-node adds the existing placement/rendezvous boundary, not serialized live contexts. Test both two-GPU and cross-node failure/cancellation; one-GPU inventory cannot satisfy these gates.

Reuse existing Ray/execution and Optuna/search-space conventions for supported worker placement and sweeps. Population-based updates need a separate verified state/update boundary within those controls; do not call configuration sampling PBT. HPC/cloud instructions identify exact environment/image/source and scheduler resources; installing every cloud platform is not an acceptance requirement. Native upstream distributed runners are baselines only.

For Newton replay, wire a pinned Viser viewer bound to loopback with public sharing off and document an SSH TCP tunnel. For Kit streaming, select Lab's supported experience, expose NVENC through `video` plus graphics, and prove signaling TCP 49100 and UDP media 47998 (or recorded configured ports). Use the existing private NetBird path where available; ordinary SSH `-L` is not a UDP tunnel. Streaming endpoints lack authentication/encryption by default. Do not recommend the standalone `isaacsim.exp.full` experience when the pinned Lab guard rejects it. Interactive resets are debug-only or invalidate a measured trial.

Preserve whole-job fast-close semantics only for a profile that demonstrably needs them; use normal owned cleanup for kit-less runtimes. Verify source-justified Torch-before-app exceptions against the chosen source; the old EA launcher explicitly handles preloaded Torch. Optimize sequence kernels, compile, CUDA graphs or batched cameras only after reference gradients/actions and stateful resets match, with explicit warm-up and end-to-end benchmarks.

**Rejected:** every native environment as a Ray actor, exposing unauthenticated viewers publicly, enabling multi-device by removing a guard alone, or adopting upstream logging/trainer stacks just to get scale.

### D14. Train exact references and keep industrial intelligence first-class

Reference order: Cartpole -> Franka -> SO-101 joint reach -> SO-101 contact vial -> ACT and visual student -> VLA -> G1. After shared interfaces stabilize, independently add OpenArm unimanual/bimanual, Go2, verified Crazyflie, TurtleBot3/mobile and industrial references. Each needs native-first provenance, real updates, resume, export/replay and a separate quality experiment. Preserve five-action fixed-gripper versus six-action relative-joint SO-101 semantics; canonical-home vial placement needs grasp/lift/transport/seating/release/hold, not attachment or proximity.

G1 is an exact 29-DoF locomotion **training** task, not a DDS simulator; humanoid manipulation is an independent IL/Mimic proof. Go2 uses its own actuator model. Crazyflie needs its actual physics/control model; PX4 SITL uses another compatible airframe and is not a Bitcraze deployment path. Its separate acceptance runs an exported NexuML policy through PX4's supported offboard/trajectory interface, validates frame/rate/bounds/heartbeat, injects offboard loss and measures the configured native failsafe. Compare tracking, stability and failsafe outcomes with the classical controller on matched cases; native SITL startup is not this proof. TurtleBot3 requires matching assets and Nav2 baseline; lack of a current Isaac task is a visible integration blocker, not permission to label another mobile robot TurtleBot3. A reviewed exact alternative must record the roadmap decision.

Use the maintained UR10e/Rizon gear-insertion reference with its already-grasped start and measured perception inputs. Keep UR5e/URSim validation separate. A LumenPnP offline vision/offset/anomaly example reuses ordinary datasets and model training; missing turnkey dynamics gets a costed feasibility report rather than a new simulator requirement. Include compatible read-only industrial data gateways and self-/semi-supervised representation learning on approved unlabeled streams, without forcing robot/ROS concepts into all models.

**Rejected:** substitute generic simulation toys for obtainable machines, treat model loading as learning, make classical Nav2/MoveIt itself the learned policy, or require physical humanoid/drone trials before the first arm is useful.

### D15. Full program includes a manual real-arm and local adaptation gate

Software milestones are independently useful, but the full program does not close until the stated manual arm workflow is demonstrated. Start with exact SO-101 or an explicitly approved arm, calibrated sensing/controller and independently validated protective workspace/stop/intervention. Offline recorded-sequence parity comes before motion. Operator enables low-speed bounded trials; no unattended real-world exploration, online training or hot swaps.

Collect authorized local demonstrations/corrections/failures, validate and split them, fine-tune offline and run shadow/simulator/old-new regression. The local adaptation operation accepts parent artifact, authorized dataset reference, explicit trainable scope/budget and frozen evaluation suite; outputs candidate, provenance and decision. Compare naïve adaptation with replay/regularization and targeted failure selection with equal-budget random selection. Human approval and a retained known-good artifact control promotion and rollback.

No NexuFL implementation follows from this operation. Preserve its future interface only: artifact/data/config in, candidate/trainable-scope/evidence out. All site permissions and embodiment/parameter compatibility remain explicit. Missing hardware or operator approval keeps the manual gate open without blocking unrelated software work or being reported as a pass.

**Rejected:** learned safety certification, automatic command enablement, hardware CI, automatic uploading, fleet updates or unaudited control setpoint writes to industrial machines.

## Risks / Trade-offs

- **[Program breadth]** -> Staged tasks and per-family proofs; do not turn the first arm milestone into a claim of full support or grow a generic framework preemptively.
- **[Unsatisfiable unified dependencies]** -> Resolve the entire requested closure first, preserve exact conflicts and require reviewed compatibility decisions; no hidden environments/overrides.
- **[Upstream docs differ from installed/pinned code]** -> Immutable sources, explicit API inventory and native/numerical tests before adapters.
- **[Sequence or processor changes break old policies]** -> Stateless default behavior, old-package/config round trips and exact new sequence/reset parity.
- **[Two optimizer owners]** -> One Lightning-compatible stepping owner, serialized target/ratio/normalization state and exact update-count tests.
- **[ROS distro/native-library mismatch]** -> Separate validated processes/workspaces, standard interfaces and explicit unavailable-capability status.
- **[VRAM/disk/resource contention]** -> Staged approved budgets, bounded N/cameras/models, effective-limit checks and no unrelated-job interruption.
- **[Model/asset licensing or remote code]** -> Source/content manifests and operator-approved code/model trust; no copied arbitrary code or presumed redistribution rights.
- **[Passing smokes imply quality]** -> Independent multi-seed task experiments and pass/fail/insufficient-evidence release decisions.
- **[Incomplete hardware/distributed infrastructure]** -> Visible blocked manual/resource gates; full completion remains unavailable rather than manufacturing a narrower pass.

## Migration Plan

1. Freeze source/roadmap/baseline identities, ownership and old-change coordination; create the compatibility/API/license inventory before shared code changes.
2. Deliver the coordinated NexuVibe Python-3.12/cu130/system setup in its owning repository, obtain recreation timing and prove the canonical clean installation. Keep old images/checkpoints and rollback inputs.
3. Add minimal physical/temporal contract and inference-state extensions with old-workflow regression tests, then port and revalidate the original references and vial task on the canonical stack.
4. Implement data/IL/distillation and common evaluation; integrate real VLA and DreamerV3 through existing learner/model seams. Keep each slice runnable and tested before the next family.
5. Add Arena/robustness, ROS/Gazebo/perception and exact embodiment references. Extend advanced physics, scaling and remote inspection only with their independent proofs.
6. Run full regression/distribution/install evidence and the separately approved real-arm/local-adaptation cycle. Reconcile all coverage rows and required gates before claiming full backend completion.

Rollback is per slice through normal reviewed version control and the retained consumer lock/image/config. Do not delete or recreate user environments, caches, datasets, checkpoints or staged work automatically. Keep base imports and previous approved inference artifacts usable; an incompatible target gets an explicit migration/rejection, not changed physics under the same identity. Commit, merge and archive remain separately requested operations.

## Source and evidence policy

Inspected 2026-10-19. These sources inform architecture, not completed compatibility:

- Local roadmap and the three existing RL/Isaac/MuJoCo changes; `reports/physical-ai/{phase-0-audit,phase-2-isaaclab}.md` for historical proof boundaries.
- [Isaac ecosystem](https://isaac-sim.github.io/IsaacLab/develop/source/isaaclab/ecosystem.html), [installation](https://isaac-sim.github.io/IsaacLab/develop/source/isaaclab/installation.html), [physics](https://isaac-sim.github.io/IsaacLab/develop/source/concepts/physics_backends.html) and [visualization](https://isaac-sim.github.io/IsaacLab/develop/source/concepts/visualization.html).
- [Pinned tutorial](https://github.com/isaac-sim/IsaacLabTutorial/blob/9a4286b4bc8a5720ac97f87e5e0140f7542c22d9/pyproject.toml) and its `uv.lock`; [candidate Lab metadata](https://github.com/isaac-sim/IsaacLab/blob/d061bf35a58273ef0cc8aceae76392166f9b7d15/pyproject.toml).
- [TorchRL Isaac guide](https://docs.pytorch.org/rl/main/reference/generated/knowledge_base/ISAACLAB.html), [VLA tutorial](https://docs.pytorch.org/rl/main/tutorials/vla.html), [versioned DreamerV3](https://docs.pytorch.org/rl/0.14/reference/dreamer_v3.html) and [objectives](https://docs.pytorch.org/rl/main/reference/objectives.html).
- [Humanoid imitation](https://isaac-sim.github.io/IsaacLab/develop/source/features/imitation-learning/humanoids_imitation.html), [SO-101 learning path](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/train-a-robot-arm/02-so101-robot.html), [gear insertion](https://isaac-sim.github.io/IsaacLab/develop/source/policy_deployment/02_gear_assembly/gear_assembly_policy.html), [RLinf VLA reference](https://isaac-sim.github.io/IsaacLab/develop/source/experimental-features/rlinf_vla_posttraining.html) and [multi-GPU reference](https://isaac-sim.github.io/IsaacLab/develop/source/features/multi_gpu.html).
- [Arena/LeRobot recipe](https://huggingface.co/docs/lerobot/en/envhub_isaaclab_arena), [Isaac ROS 5.0 requirements](https://nvidia-isaac-ros.github.io/v/release-5.0/getting_started/index.html), [release limitations](https://nvidia-isaac-ros.github.io/v/release-5.0/releases/index.html), [Noble buildfarm](https://nvidia-isaac-ros.github.io/v/release-5.0/getting_started/isaac_ros_buildfarm_cdn.html), [Sim ROS bridge](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/install_ros.html) and [streaming security/connectivity](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/manual_livestream_clients.html).
- Published [Meta-World 3.1.1](https://pypi.org/pypi/metaworld/3.1.1/json), [Newton 1.6.1](https://pypi.org/pypi/newton/1.6.1/json) and [MJWarp 3.12.0](https://pypi.org/pypi/mujoco-warp/3.12.0/json) metadata for the demonstrated dependency contradiction.

Context7 was unavailable due to quota; official documentation and published/pinned sources were used. No installer, resolver, simulator, GPU workload or hardware operation ran during proposal creation. Native package verification, final model/release selection and measured budgets are explicit first implementation gates, not unresolved architectural choices.
