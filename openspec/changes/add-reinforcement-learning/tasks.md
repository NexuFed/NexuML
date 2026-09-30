## 1. Core Interaction And Policy Contracts

- [ ] 1.1 Add portable tensor-field/action-space/`InteractionContract` models covering shape, dtype, modality/semantics, optional bounds/rate metadata, vectorization, and optional control period.
- [ ] 1.2 Add optional `ScenarioSpec.interaction: InteractionSpec | None` and `ScenarioSpec.policy: PolicySpec | None` without changing current supervised construction/YAML.
- [ ] 1.3 Add `EnvironmentBuildContext`, `EnvironmentDefinition`, framework-neutral `EnvironmentRuntime` protocol/base, `ActionAdapterDefinition`, `RLAlgorithmBuildContext`, and `RLAlgorithmDefinition` without importing TorchRL at core import time.
- [ ] 1.4 Define `EnvironmentDefinition.describe()` separately from `build()`; enforce that description returns portable metadata and does not retain long-lived runtime resources or connect to real hardware.
- [ ] 1.5 Add `@environment`, `@action_adapter`, and `@rl_algorithm` decorators and extend the existing component registry with exactly those new component kinds.
- [ ] 1.6 Add registry/serialization tests proving interaction components persist by stable kind/name/version and contain no live simulator/ROS/collector objects.
- [ ] 1.7 Validate that `data` and `interaction` may coexist so future offline demonstrations + online interaction are not structurally forbidden.

## 2. Environment-Aware Pipeline And Policy Compilation

- [ ] 2.1 Extract `PipelineCompileContext` from the current data-specific compiler inputs and keep `compile(scenario)` as the ordinary supervised wrapper.
- [ ] 2.2 Add interaction-contract-to-compile-context translation for supported observation fields, carrying shape plus relevant dtype/modality metadata.
- [ ] 2.3 Fail loudly for unsupported required observation leaves rather than fabricating shapes/defaults.
- [ ] 2.4 Add reusable pipeline compilation for the actor/policy and algorithm-owned critic/value pipelines without introducing a second pipeline implementation.
- [ ] 2.5 Add `CompiledPolicy` that composes `CompiledPipeline` + action-adapter runtime + portable interaction contract while leaving `CompiledPipeline.forward()` unchanged.
- [ ] 2.6 Keep supervised loss/metric/optimizer behavior unchanged; RL-compiled pipelines must not accidentally use supervised optimizer/loss paths.
- [ ] 2.7 Add tests compiling from both `DataSpec` and `InteractionContract` and validating multimodal observation keys/shapes.

## 3. Reinforcement Configuration And Dependencies

- [ ] 3.1 Add `CollectorSpec`, `RLEvaluationSpec`, and `ReinforcementLearningSpec`; make `ScenarioSpec.training` accept the existing `TrainingSpec` or the new RL spec.
- [ ] 3.2 Remove environment ownership from `ReinforcementLearningSpec`; require RL scenarios to use `scenario.interaction`.
- [ ] 3.3 Add `policy_sync_interval_batches` or equivalent semantic collector setting for stale-policy prevention on copied-policy backends.
- [ ] 3.4 Add root `nexuml[reinforcement]` optional dependencies with lockfile-compatible TorchRL/Gymnasium bounds and include the pip-safe extra in `all`.
- [ ] 3.5 Add matching `nexuml-library[reinforcement]` support; keep simulator-specific Python dependencies in separate extras where practical.
- [ ] 3.6 Do not add `rclpy`, Gazebo, Isaac, or CARLA as mandatory Python/base reinforcement dependencies.
- [ ] 3.7 Prove `import nexuml` and supervised scenarios work without reinforcement dependencies installed.

## 4. Framework-Neutral Environment Runtime And TorchRL Adapter

- [ ] 4.1 Implement the NexuML environment runtime reset/step/close contract using TensorDict observations/actions and explicit reward/terminated/truncated semantics.
- [ ] 4.2 Add a private reinforcement adapter from NexuML environment runtime to the installed TorchRL environment/collector API.
- [ ] 4.3 Support environment build contexts with `purpose`, seed, device, worker/rank identity, and requested `num_envs`.
- [ ] 4.4 Validate vectorized environment behavior and batch dimensions independently of unbatched policy field shapes.
- [ ] 4.5 Ensure direct environments are built in the Lightning worker, not in driver-side `NexuSession` construction.
- [ ] 4.6 Ensure process/Ray collectors receive immutable factories/definitions and materialize environments in collector workers rather than pickling live environments.
- [ ] 4.7 Add deterministic teardown for collectors/environments on normal completion and failure/cancellation paths.

## 5. Lightning + TorchRL Training Runtime

- [ ] 5.1 Add collector materialization through TorchRL's current `Collector` front door and an iterable/DataLoader bridge with `batch_size=None` and `num_workers=0` for direct learner consumption.
- [ ] 5.2 Add `NexuRLLightningModule` with `automatic_optimization=False`, algorithm delegation, frame/reward logging, and no collection hidden in epoch-start hooks.
- [ ] 5.3 Build/register the RL algorithm runtime as checkpointable Lightning module state before state restoration/optimizer creation; critic/value/target modules must not be invisible Python objects.
- [ ] 5.4 Return RL optimizers through Lightning `configure_optimizers()` and use wrapped optimizers, `manual_backward()`, clipping, and Lightning precision/accelerator semantics.
- [ ] 5.5 Update `NexuSession` runtime/fit/run paths to select supervised versus RL semantics while retaining one public session and `L.Trainer`.
- [ ] 5.6 Make RL runtime artifacts independent of `NexuDataModule`; supervised runtime artifacts/properties remain backward compatible.
- [ ] 5.7 Treat `total_frames` as the finite RL budget and avoid fake user-visible RL epochs.
- [ ] 5.8 After configured learner updates, synchronize current policy weights to process/Ray collectors using the supported TorchRL weight-update mechanism; direct shared-policy collection may no-op.
- [ ] 5.9 Add tests that deliberately use a copied fake collector policy and verify it changes after synchronization.

## 6. PPO And Policy Adapters

- [ ] 6.1 Add `library/src/nexuml_library/reinforcement/algorithms/ppo.py` with typed `PPO` definition and a private checkpointable runtime using TorchRL `GAE` and `ClipPPOLoss`.
- [ ] 6.2 Keep the critic/value network owned by PPO configuration/runtime and compile it from the same interaction-derived context.
- [ ] 6.3 Implement PPO minibatch/update epochs inside the algorithm runtime; do not add PPO conditionals to the generic RL Lightning module.
- [ ] 6.4 Add direct, bounded-Gaussian, and categorical action-adapter definitions/runtimes.
- [ ] 6.5 Add continuous/discrete policy helpers that emit the adapter parameter keys without embedding environment-specific code.
- [ ] 6.6 Verify deterministic policy inference uses adapter mean/mode/direct behavior without constructing the PPO learner.
- [ ] 6.7 Log actor/value/entropy/clip/reward/frame metrics with stable NexuML prefixes.

## 7. Fast Gymnasium Reference Scenarios

- [ ] 7.1 Add typed `GymnasiumEnvironment` under the library with lazy Gymnasium/TorchRL imports and interaction-contract description.
- [ ] 7.2 Add `pendulum-ppo` for continuous control and `cartpole-ppo` for discrete control.
- [ ] 7.3 Add tiny CPU smoke configurations that complete collection, at least one optimizer update, evaluation, result construction, and policy export without stochastic reward-threshold assertions.
- [ ] 7.4 Verify both scenarios discover, resolve to YAML, restore, train, evaluate, checkpoint, export, and reload through the ordinary CLI/session path.

## 8. Visual And Physical-AI Simulation Scenarios

- [ ] 8.1 Add a lightweight visual-observation control scenario whose policy consumes an image tensor directly; mark image modality/shape/dtype in the interaction contract.
- [ ] 8.2 Add an opt-in MuJoCo (or equivalently maintained) humanoid PPO scenario exercising high-dimensional proprioception and continuous control.
- [ ] 8.3 Add an opt-in articulated-robot manipulation/control scenario separate from humanoid locomotion.
- [ ] 8.4 Keep heavy simulator scenarios discoverable/documented when dependencies are missing, but fail execution with focused prerequisite diagnostics.
- [ ] 8.5 Add configuration/contract tests for heavy scenarios without booting their simulators in normal CI.
- [ ] 8.6 Add opt-in simulator integration markers/jobs for actual environment reset/step/training smoke validation.

## 9. Maintained ROS 2 Adapter And Simulation Path

- [ ] 9.1 Implement a maintained ROS 2 reference transport/environment boundary with lazy `rclpy` imports, explicit interaction contract, bounded high-level joint control, configurable control period/timeouts, and deterministic shutdown.
- [ ] 9.2 Separate transport from task/reward logic so the same task environment can use production `rclpy` transport or a fake transport.
- [ ] 9.3 Implement fake transport tests for joint-state reception, command publication/action calls, reset behavior, timeouts, termination, and close/shutdown without requiring ROS middleware.
- [ ] 9.4 Add a discoverable ROS joint-reach/control scenario using the maintained adapter.
- [ ] 9.5 Add an opt-in headless ROS 2 + Gazebo + `ros2_control` integration fixture/test proving the production ROS transport can control a simulated robot.
- [ ] 9.6 Document how the same high-level adapter maps to real hardware while low-level real-time control, drivers, limits, safety interlocks, and emergency stops stay outside NexuML.
- [ ] 9.7 Do not silently fall back from missing real ROS transport to the fake transport in user runs.

## 10. Autonomous-Driving Simulation Scenario

- [ ] 10.1 Add an opt-in CARLA-style driving environment/scenario using camera plus ego/vehicle state and bounded control or trajectory-level actions.
- [ ] 10.2 Keep CARLA/system dependencies outside base `reinforcement` installation.
- [ ] 10.3 Add configuration/contract tests that do not require a live CARLA server.
- [ ] 10.4 Add an opt-in integration smoke path that connects to a simulator, performs reset/step, collects rollouts, and evaluates/exports the policy.
- [ ] 10.5 Document how additional lidar/radar/GNSS/IMU fields extend the same interaction contract without changing the training architecture.

## 11. Episodic Evaluation

- [ ] 11.1 Implement `RLEvaluationSpec` with episode count, deterministic/stochastic mode, seed, and optional max steps.
- [ ] 11.2 Support `interaction.evaluation_environment` override; otherwise materialize a fresh runtime from the training environment definition.
- [ ] 11.3 Report mean/std episode return, mean episode length, and useful termination/truncation statistics through existing `TrainResult`/logger surfaces.
- [ ] 11.4 Ensure evaluation uses `CompiledPolicy` and its action adapter, not algorithm-specific PPO inference code.
- [ ] 11.5 Ensure RL evaluation does not invoke dataset post-train-fit semantics or reuse stale training environment state.

## 12. Export, Checkpoint, And NexuFL-Ready Boundary

- [ ] 12.1 Export the complete `CompiledPolicy` contract: pipeline weights + action-adapter definition/state + resolved portable interaction/policy I/O contract.
- [ ] 12.2 Preserve an identifiable trainable `scenario.pipeline` state subset for future federated update selection.
- [ ] 12.3 Record RL training mode, algorithm identity/version, environment identity/version, action-adapter identity/version, interaction-contract hash, and resolved provenance in export metadata.
- [ ] 12.4 Exclude trajectories, replay buffers, live environment/ROS/simulator objects, collectors, optimizer state, and training-only critic/target networks from the normal portable policy artifact.
- [ ] 12.5 Preserve actor/critic/optimizer/counter/checkpointable algorithm state in Lightning checkpoints where required for local resume.
- [ ] 12.6 Add export/load tests that run deterministic policy inference on TensorDict observations in a clean runtime without constructing an environment or PPO learner.
- [ ] 12.7 Document that later NexuFL interactive learning should exchange selected learned policy state while local raw experience/runtime state remains local by default.

## 13. Documentation, Test Tiers, And Validation

- [ ] 13.1 Add a Physical-AI/RL mental model showing `pipeline`, `CompiledPolicy`, `interaction`, environment runtime, TorchRL collector, algorithm runtime, and Lightning ownership.
- [ ] 13.2 Document fast versus optional simulator scenarios and exact prerequisite boundaries.
- [ ] 13.3 Document the ROS fake-transport test path, Gazebo simulation path, and later real-hardware path.
- [ ] 13.4 Document why `data` and `interaction` are separate and how that prepares later imitation/VLA/world-model training without implementing those modes now.
- [ ] 13.5 Add test markers/CI tiers for ordinary CPU, optional simulator, optional GPU/special-system, and manual real-hardware validation.
- [ ] 13.6 Run the existing supervised suite and prove no existing scenario needs source changes merely to remain valid.
- [ ] 13.7 Run strict docs/build/lint/type checks used by the repository.
- [ ] 13.8 Run `openspec validate add-reinforcement-learning --strict` and resolve every proposal/design/spec/task inconsistency before implementation is considered complete.
