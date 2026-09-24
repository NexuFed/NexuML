## 1. Core Typed RL Contracts

- [ ] 1.1 Add `EnvironmentBuildContext`, `EnvironmentDefinition`, `RLAlgorithmBuildContext`, and `RLAlgorithmDefinition` to the typed component system without importing TorchRL at core import time.
- [ ] 1.2 Add `@environment` and `@rl_algorithm` decorators and extend the existing component registry scan with exactly those two new kinds.
- [ ] 1.3 Add `CollectorSpec` and `ReinforcementLearningSpec`; change `ScenarioSpec.training` to accept the existing `TrainingSpec` or the new RL spec while keeping current supervised construction and YAML valid.
- [ ] 1.4 Add focused registry/serialization tests proving RL definitions persist by stable kind/name/version and contain no live runtime objects.

## 2. Environment-Aware Pipeline Compilation

- [ ] 2.1 Extract `PipelineCompileContext` from the current data-specific compiler inputs and keep `compile(scenario)` as the ordinary supervised wrapper.
- [ ] 2.2 Add environment-spec-to-input-shape translation for tensor observation leaves and fail loudly for unsupported observation structures.
- [ ] 2.3 Add a reusable pipeline compilation seam for the actor and training-only critic without introducing a second pipeline implementation.
- [ ] 2.4 Keep supervised loss/metric/optimizer behavior unchanged; ensure RL-compiled pipelines do not accidentally call supervised optimizer/loss paths.
- [ ] 2.5 Add one regression test compiling the same simple pipeline from both data-derived and environment-derived shape contexts.

## 3. Lightning + TorchRL Runtime

- [ ] 3.1 Add the `nexuml[rl]` extra with a lockfile-compatible TorchRL/Gymnasium range, include it in `all`, and prove base `import nexuml` still works without RL dependencies.
- [ ] 3.2 Add the private CompiledPipeline-to-TorchRL TensorDict adapter without changing `CompiledPipeline.forward()`.
- [ ] 3.3 Add collector materialization through TorchRL's `Collector` front door and an `IterableDataset`/DataLoader bridge with `batch_size=None` and `num_workers=0`.
- [ ] 3.4 Add `NexuRLLightningModule` with `automatic_optimization=False`, algorithm delegation, frame/reward metric logging, and no collection work hidden in epoch-start hooks.
- [ ] 3.5 Make RL algorithm runtimes return optimizers through Lightning `configure_optimizers()`; use `self.optimizers()`, `manual_backward()`, clipping, and wrapped optimizer steps.
- [ ] 3.6 Update `NexuSession` runtime construction/fit/run paths to select the RL module while retaining the same public session and `L.Trainer`.
- [ ] 3.7 Treat `total_frames` as the finite training budget and avoid converting RL frames/collections into fake user-visible epochs.

## 4. PPO Reference Implementation

- [ ] 4.1 Add `library/src/nexuml_library/reinforcement/algorithms/ppo.py` with a typed `PPO` definition and private mutable runtime using TorchRL `GAE` and `ClipPPOLoss`.
- [ ] 4.2 Add a small reusable continuous-policy head/helper that emits explicit location/scale keys for TorchRL probabilistic action construction.
- [ ] 4.3 Keep the critic/value network owned by PPO configuration/runtime and separate from `scenario.pipeline`; compile it from the same environment-derived input context.
- [ ] 4.4 Implement PPO minibatch/update epochs inside the algorithm runtime; do not add PPO-specific branches to `NexuRLLightningModule`.
- [ ] 4.5 Log actor/value/entropy/clip metrics exposed by TorchRL plus reward/frame metrics using stable NexuML prefixes.

## 5. Gymnasium Environment And Example Scenario

- [ ] 5.1 Add a typed `GymnasiumEnvironment` definition under `library/src/nexuml_library/environments/gymnasium.py` with lazy TorchRL/Gymnasium imports.
- [ ] 5.2 Add `library/src/nexuml_library/scenarios/reinforcement/pendulum_ppo.py` using a small policy MLP, PPO value network, `Pendulum-v1`, finite frames, normal logging, and normal exports.
- [ ] 5.3 Add a tiny CPU smoke configuration for tests that completes quickly without asserting stochastic convergence.
- [ ] 5.4 Verify the scenario can be discovered, resolved to YAML, restored, trained, and evaluated through the same CLI/session path as existing scenarios.

## 6. RL Evaluation

- [ ] 6.1 Build a fresh evaluation environment and deterministic policy from the same immutable environment definition after training.
- [ ] 6.2 Evaluate the configured number of episodes and return mean/std episode return and mean episode length through the existing `TrainResult` result fields.
- [ ] 6.3 Ensure RL evaluation does not invoke dataset post-train-fit semantics or silently reuse stale training environment state.

## 7. Export And NexuFL-Ready Boundary

- [ ] 7.1 Keep the normal exported primary model equal to `scenario.pipeline` / the actor policy and verify it loads without constructing an environment.
- [ ] 7.2 Record RL training mode, algorithm stable identity/version, environment stable identity/version, and relevant config hashes in export metadata.
- [ ] 7.3 Exclude trajectories, replay buffers, environment/ROS runtime objects, and training-only critic/target networks from the normal portable policy artifact.
- [ ] 7.4 Preserve resumable RL state in Lightning checkpoints where supported without changing the deployable policy contract.
- [ ] 7.5 Add an export/load smoke test that runs the trained policy on an observation TensorDict in a clean runtime.
- [ ] 7.6 Document that later NexuFL federated RL should exchange selected policy weights/updates while keeping experience and environment state local by default; do not implement that client procedure here.

## 8. ROS 2 / Robotics Extension Boundary

- [ ] 8.1 Document how a custom `EnvironmentDefinition` wraps a ROS 2/Gazebo or real-robot runtime while keeping reward, reset, observation/action mapping, and termination task-specific.
- [ ] 8.2 Include a concrete `Ros2JointControl`-style code example using topics/services and bounded high-level actions, clearly separating NexuML from `ros2_control`/hard-real-time safety.
- [ ] 8.3 Do not add `rclpy` to the base or `rl` extras; if a helper package is added later, keep ROS imports optional and test its transport boundary with fakes in normal CI.
- [ ] 8.4 State explicitly that ROS is an environment integration, not a NexuML training backend.

## 9. Collection Placement And Ray Guardrails

- [ ] 9.1 Map `CollectorSpec` to TorchRL direct collection first and keep process/Ray fields serializable for staged validation.
- [ ] 9.2 Reject `scenario.execution.kind="ray"` combined with `collector.backend="ray"` until nested Ray placement is explicitly supported and tested.
- [ ] 9.3 Add a focused config/runtime test for the nested-Ray rejection; do not require a live Ray cluster for the core RL smoke path.
- [ ] 9.4 Leave multi-learner distributed RL outside this change.

## 10. Documentation And Validation

- [ ] 10.1 Add a concise RL mental model showing dataset learning vs environment learning under the same Lightning trainer.
- [ ] 10.2 Document the Pendulum PPO scenario and the distinction between deployable policy and training-only critic/replay state.
- [ ] 10.3 Document the ROS extension example and future NexuFL local-experience/federated-policy boundary.
- [ ] 10.4 Run existing supervised tests plus the focused RL tests and ensure no existing scenario needs source changes just to remain valid.
- [ ] 10.5 Run strict docs/build/lint/type checks used by the repository.
- [ ] 10.6 Run `openspec validate add-reinforcement-learning --strict` and resolve every proposal/spec inconsistency before implementation is considered complete.
