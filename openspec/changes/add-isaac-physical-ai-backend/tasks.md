# Tasks

## Linear issue and PR linkage

- **Implementation issue:** [NEX-234](https://linear.app/nexufed/issue/NEX-234) — SO-101 local imitation learning and NexuML Python SDK handoff (Gate 0 / F01).
- **PR branch (recommended from Linear):** `feature/nex-234-so-101-local-imitation-learning-and-nexuml-python-sdk`. PR title example: `NEX-234: SO-101 local behavior cloning and Python SDK handoff`.
- **PR body:** include `Closes NEX-234` **only** on the PR that actually satisfies verified tasks 0.1–0.8. Planning-only PRs should say `Related to NEX-234`; do not close the issue.
- Keep the PR small: do not merge the entire unfinished Physical AI umbrella to deliver the handoff. Later FL implementation is tracked by [NEX-235](https://linear.app/nexufed/issue/NEX-235).
- **Status automation:** linking alone is not a guarantee of transitions; the Linear↔GitHub integration and team's workflow automations must be configured to move the issue on PR creation/review/merge.


Source of truth: proposal.md, design.md, all capability specs (including the new local-learning-handoff spec), reference-catalogue.md and coverage.md. The first three verified planning tasks retain their historical checkmarks; every new consolidation task is unchecked. This is the single implementation queue for Isaac, ordinary MuJoCo/Meta-World, GPU MuJoCo Playground and Physical AI model/robot examples. Deliver tests and runnable documentation with each slice. Native, dataset, ROS, distributed and hardware checks require the declared prerequisites; static validation, skipped tests and old-profile evidence do not satisfy them.

This is one staged program, not one unattended command or giant PR. Keep existing completed work intact. Cross-repository delivery must occur in its owning scope; no hidden NexuVibe/NexuFL edits, active-environment replacement, rebuild, unapproved download or hardware actuation is authorized by this task file.

**Current campaign authorization:** the user permits direct task-related NexuVibe edits preserving other work, plus installs/native experiments in one isolated `.venv` in this worktree. Never target `/env`, rebuild the active DevContainer, change drivers or disturb other jobs; retain at least40GiB free. Prioritize Isaac; defer large-model weight/training work for now while retaining its unchecked full-program gates. Evaluate further dependency selections/overrides only with recorded source/API rationale and tests, not blind native/physics incompatibility. Use the ten-hour target as a resumable work budget; no paid compute or hardware actuation is authorized.

The user-selected LeRobot additions require complete Diffusion Policy and resource-gated π0.5 examples, but only bounded feasibility assessments for VLA-JEPA, FastWAM and SARM. Large-model work is deferred in the current campaign, not removed from full-program acceptance. D3 records packaging, Transformers, NumPy, urllib3 and isolated-importer usd-exchange as initial exception cases; further documented evaluations are authorized with security/ABI/behavior checks. Independent implementation may proceed while unrelated compatibility/provisioning remains open. Meta-World remains required, not an implicit physics override.

**Implementation supersession:** the previous add-mujoco-reference-workflows proposal remains available for historical audit, but its native task, outcome, dependency, media and benchmark requirements have been imported under this change. Do not execute its tasks.md separately. The original 42/42 scoped Isaac and 23 open extension items remain historical evidence, not rewritten completion.

## 0. PRIORITY INTEGRATION GATE — local SO-101 imitation + Python SDK PR

**Complete this narrow milestone first**, using only the necessary upstream prerequisites from later sections. This is NOT permission to mark the broad ISAAC/ACT/VLA/ROS/world-model work as completed. `federation-handoff.md` is the cross-repo contract. The acceptance of this gate is a **STOP for a narrow NexuML PR** before starting NexuFL integration.

- [ ] **0.1 Freeze real task**: qualify the actual maintained SO-101 reach task/asset, actor observations, five arm action semantics, controller/clock/reset rules, pinned simulator release, available hardware, dependency/compute/permission prerequisites and fresh-process native exercise; do not rely solely on old EA evidence.
- [ ] **0.2 Record real expert episodes**: collect scripted/IK or demonstrator trajectories with native success, train-only observation fields, action units, timestamps, provenance, resets and seed; reject invalid/incomplete episodes. Use a documented reusable episode format compatible with the planned LeRobot conversion.
- [ ] **0.3 Genuine state behavior cloning**: train a small actor from distinct valid SO-101 reaching episodes via the existing NexuML session/optimizer, not a new IL trainer or copied upstream trainer. Freeze disjoint train/validation/test episodes; measure actual optimizer/weight changes and stable first-run behavior.
- [ ] **0.4 Real closed-loop check**: export a complete policy, clean-reload in a new process and run fresh seeded SO-101 simulator episodes. Report trained policy success, expert success, untrained/random reference, action/limit violations and failures; offline loss or expert scripted success alone cannot pass.
- [ ] **0.5 Generic bounded Python local fit**: expose and test an SDK operation reusing `NexuSession`, trusted local recipe, externally provided validated initial tensor state, authorized local dataset, deterministic step/epoch budget, seed, explicitly reset per-round optimizer and shared trainable scope. Two repeated invocations with different initial tensors produce correctly attributable updates. No REST or NexuFL imports.
- [ ] **0.6 Interoperability and privacy**: version exact model/processor/action semantics, trainable names/shapes/dtypes/finiteness, base model identity and measured eligible training windows/episodes; return validated numeric weights and approved scalar metrics, never raw episodes, private paths, optimizer buffers or executable packages. Exercise invalid tensor schema, wrong SO-101 control contract and unauthorized data/source negative paths.
- [ ] **0.7 Package/SDK regression and reproducibility**: a clean installed wheel runs SDK fit without FastAPI, gRPC, simulator in the training process or optional REST; document Python call and deterministic CLI/script demo, clean export/replay and exact dependency/asset/seed receipts. Normal industrial/supervised uses and export behavior remain intact.
- [ ] **0.8 HANDOFF STOP / PR**: pass targeted native learning quality + SDK/negative/regression tests, review minimal public surface, artifact schema and source/LOC; pin the exact consumer wheel/commit and publish a contract fixture; **pause to prepare a narrowly scoped NexuML PR**, rather than continuing all remaining roadmap tasks or opening/migrating a NexuFL PR automatically.

**Gate accepted only with real learned SO-101 closed-loop success and externally repeated fit over portable weights; no FL, gRPC, Kubernetes, ACT or hardware proof is implied.** All new checkboxes initially remain open.

## 1. Ownership, scope and compatibility baseline

- [x] 1.1 Record current code/import origins, effective configuration and existing CPU/static failures in the backend evidence report; verify exact commands and preserve unrelated work instead of repairing or hiding baseline failures.
- [x] 1.2 Reconcile the roadmap snapshot with all coverage groups and legacy change obligations; verify every source checkbox and program capability has an owner, acceptance destination and explicit existing/pending/blocked disposition.
- [x] 1.3 Assign one owner for shared outcome, temporal-policy and learner changes; verify the dependency map prevents duplicate implementations across this program and the Isaac/MuJoCo changes without changing their historical completion checks.
- [ ] 1.4 Audit immutable source/API/asset/model/license/compute targets across Isaac, current LeRobot, classic MuJoCo, Meta-World, Playground/MJWarp/MJX, JAX, ROS, DreamerV3 and benchmarks. Freeze actual task IDs and native APIs, wheel/driver/source/operator constraints before dependent implementation; simulator-independent slices use their own focused source/contracts audit under D3.
- [ ] 1.5 Resolve the complete Python-3.12/CUDA-13 **primary Isaac** stack and optional compatibility matrix. Prefer maintained compatible releases; evaluate documented D3 exceptions under campaign authority, preserving original bounds, exact versions/lock, security/ABI checks and provisional status. Reproduce Meta-World/Newton conflicts; prioritize Isaac and record blocked/deferred optional selections. No global Python <3.14 bound, conflicting all union or implied native/model proof.
- [ ] 1.6 Publish slice ownership, resource/license needs and compatibility/exception records with original/replacement bounds, source/upstream reference, owner, lock, affected tests, removal condition and rollback inputs; reconcile original obligations/exclusions, distinguishing policy approval, resolution and executed acceptance. Full native readiness remains blocked; independent slices may proceed under D3 with their source/contracts audit.

## 2. Canonical shared environment and automatic setup

- [ ] 2.1 Declare concrete optional robotics dependencies in library robots with root forwarding; exclude the new robotics stack from both all closures. Use genuine Python/API minimums and evidence-backed incompatibility bounds, not development Python/CUDA/Torch/index/lock pins. Verify built metadata, base/all import isolation, extra selection and consumer version/backend choice with negative tests; preserve unrelated extras and real upstream restrictions.
- [ ] 2.2 Pin exact primary Isaac Lab/Sim/Torch/TorchRL/TensorDict/Newton/LeRobot sources and approved native lanes in the consuming root uv project, not downstream package requirements. Explicitly select all/dev/robots and backend there; declare D3 exceptions via root tool.uv.override-dependencies without metadata edits. Verify complete consumer-lock resolution/provenance, retain checker discrepancies and block unexpected conflicts; affected behavior still requires proof.
- [ ] 2.3 Deliver or coordinate reviewed NexuVibe changes to Python selection, consumer lock, Docker/system libraries, hooks, versioned paths, interpreter settings and setup tests; verify the owning repository revision and locked, idempotent automatic provisioning without hidden sibling edits.
- [ ] 2.4 Inspect effective resources and propose a staged disk/RAM/VRAM/cache budget before approved installation/recreation; verify driver/CUDA, Vulkan/EGL, NVENC and ROS prerequisites independently and preserve active experiments.
- [ ] 2.5 Exercise clean approved container creation plus repeated setup with minimum reference assets; verify automatic installation, hashes, failure propagation, no unrequested upgrades and one canonical interpreter. For declared exceptions, retain checker output and exact known mismatches, verify all other dependencies and affected native startup/version/task behavior; real-model gates in section 9 remain independently required.
- [ ] 2.6 Test wrong interpreter, CUDA family, source revision, missing LFS, bad hash, partial download and inherited library paths. Add exception regressions for undeclared/widened bounds, changed versions, unexpected checker mismatches, failed native/model behavior and upstream-fix removal; require requalification and ordinary integrity after removal. Verify actionable failures preserve unrelated installations, never trigger an import-time/hidden installer and never widen overrides automatically.
- [ ] 2.7 Publish exact create/sync/readiness/rollback commands and consumer lock/source identities; verify commands as written and distinguish installed packages, native readiness and model/dataset download approvals.

## 3. Physical, multimodal and temporal policy contracts

- [ ] 3.1 Map the five foundation contracts to existing types and add only concrete physical metadata gaps; verify serialization and negative tests for joint order, units, frames, control mode/rate, sensors, bounds and calibration identity.
- [ ] 3.2 Extend existing compile input metadata for multimodal/sequence axes without flattening time or agent identity incorrectly; verify static shape tests and unchanged ordinary supervised/stateless policy behavior.
- [ ] 3.3 Add registered image/language/normalization processor adapters producing explicit tensor inputs; verify camera ordering, token/mask behavior, processor identity and no weight download during discovery/config resolution.
- [ ] 3.4 Extend the existing complete policy with explicit initial-state and step/reset execution; verify caller-owned recurrent/latent/chunk state, batch isolation, partial resets, action stride and unchanged stateless calls.
- [ ] 3.5 Reuse policy assembly for offline-trained graphs with an independently validated interaction contract; verify a dataset-trained model can export a complete policy without constructing an RL learner or simulator.
- [ ] 3.6 Extend complete-policy packaging for required processors, composed adapters and inference state metadata; verify clean-process sequence/action parity, integrity/trust failures and exclusion of critics, optimizers, replay and raw recordings.
- [ ] 3.7 Document the contract/state/processor surfaces and migration rules; verify old YAML/checkpoint/package regressions and non-robot training/export without robotics dependencies.
- [ ] 3.8 Add the trainer-independent policy-evaluation block and normalize it into the existing episode runner before offline closed-loop work; verify configuration/export round trips, missing-input errors, explicit seed handling and unchanged legacy RL configuration precedence.
- [ ] 3.9 Extend the ordinary evaluation CLI for original non-RL ACT/VLA package scenarios; verify clean-package scalar/media tests with session/learner/optimizer/collector construction blocked, explicit new/legacy override rules and documented commands without fabricated RL training settings.

## 4. Isaac runtime, tasks and native transition correctness

- [ ] 4.1 Adapt existing launchers/provider to canonical interpreter and checkout locations; verify module/script execution, lazy launch, actual exit codes and no hardcoded worker paths.
- [ ] 4.2 Validate selected release launch/import order and resource ownership; verify native startup/failure/SIGTERM cleanup, profile-specific fast or normal close and zero simulator allocation for portable or completed-budget operations.
- [ ] 4.3 Support one manager and one direct task through public native configuration; verify task discovery, native N/device/control contracts and no outer simulator vectorization or CPU/NumPy control round trips.
- [ ] 4.4 Probe maintained wrapper composition and canonical fallback for each supported task workflow; verify exact reset counts, final/reset identity, mutable buffers and numerical GAE for no/partial/all done, truncation and simultaneous flags.
- [ ] 4.5 Add explicit Newton, Kit PhysX and standalone OvPhysX selections for supported reference presets; verify native construction and incompatible process/backend/renderer rejection without silent fallback.
- [ ] 4.6 Reuse native URDF/MJCF/USD import for one concrete required asset; verify conversion/source/content/license identity, joint/actuator/collision metadata and runtime contract rejection on altered assets.
- [ ] 4.7 Connect replay camera/frame caching to existing media/loggers; verify real terminal-scene identity, GIF/MP4/both, animated TensorBoard, MLflow-only behavior and truthful cancellation on the canonical stack.
- [ ] 4.8 Publish canonical runtime/task/backend commands and diagnostics; verify fake/static tests land with the provider changes and native tests remain opt-in with prerequisites stated.

## 5. Canonical PPO and SO-101 manipulation references

- [ ] 5.1 Run native Cartpole and Franka control/transition/training baselines on the canonical stack; verify resolved versions, matching physical mappings, finite counters and owned shutdown before adapter claims.
- [ ] 5.2 Port ordinary Cartpole and Franka NexuML scenarios; verify real PPO updates, partial/completed-budget resume, complete export, clean inference and fresh scalar/recorded replay independently of old EA evidence.
- [ ] 5.3 Port SO-101 five-joint reaching with fixed gripper; verify exact asset/control identities, scripted signs/limits, error/hold/timeout rules and the complete learned-policy round trip.
- [ ] 5.4 Validate the pinned upstream six-control vial task natively; verify contact model, reset dataset hashes, observation ordering, relative action timing, nominal curriculum and canonical-home play settings.
- [ ] 5.5 Prove a canonical-home physical-contact placement reference; verify grasp/lift/transport/seating/release/hold and negative held/airborne/proximity cases without attachment, post-reset teleportation or borrowed learned-success claims.
- [ ] 5.6 Integrate vial placement through existing NexuML PPO; verify changed actor weights, checkpoint/resume, complete inference export, home-start scalar/media replay and task-specific cached outcomes.
- [ ] 5.7 Expose upstream randomization/curriculum and physics-identification parameters; verify seeded held-out variations, recorded sampled values and separation of privileged versus deployable inputs.
- [ ] 5.8 Publish all four canonical round trips and a separately budgeted multi-seed quality comparison; verify actual learned home-start success is reported independently from functional smokes or scripted feasibility.

## 6. Additional TorchRL learning families

- [ ] 6.1 Add algorithm-owned replay/warm-up/update-ratio support to the existing RL lifecycle; verify finite data/physical-frame budgets, target-update counts and unchanged PPO collection/synchronization semantics.
- [ ] 6.2 Integrate maintained SAC on a canonical continuous-control task; verify numerical loss/target tests, real updates, replay checkpoint/resume, export and fresh evaluation.
- [ ] 6.3 Add an explicit offline-data iteration path and maintained IQL reference; verify valid sequence sampling, actual updates and zero simulator/hardware collection during offline training.
- [ ] 6.4 Integrate recurrent PPO with explicit sequence masks/primers; verify partial-reset hidden-state isolation, temporal advantages and fresh-process recurrent inference parity.
- [ ] 6.5 Integrate an asymmetric-critic reference; verify privileged inputs reach only training modules and actor export/replay works with them blocked.
- [ ] 6.6 Validate one maintained direct multi-agent task with supported IPPO/MAPPO objectives; verify agent/time/environment axes, done/reward semantics, changed parameters, resume and portable policy replay.
- [ ] 6.7 Publish algorithm-specific setup/run/resume/evaluate examples and acceptance evidence; verify each family has its own focused tests and no support is inferred from another family's passing smoke.

## 7. Experience ingestion, capture and synthetic data

- [ ] 7.1 Add a LeRobotDataset v3 data source through existing loaders; verify tiny episode/video/state fixtures, schema/version diagnostics, timestamps, split identity and offline operation.
- [ ] 7.2 Add time-alignment, quality and consumption reports; verify missing/stale frames, bounds, rates, calibration changes, duplicates, padding and source-group leakage failures before training.
- [ ] 7.3 Ingest intervention/correction segments with producer and policy provenance; verify deterministic selection, no unauthorized identifiers and preserved old/new dataset splits.
- [ ] 7.4 Document and exercise upstream simulation teleoperation/recording plus approved prerecorded SO-101 leader/follower data; verify controls, reset/record boundaries and no custom capture UI or implicit physical actuation.
- [ ] 7.5 Run a bounded native Mimic annotation/generation workflow and validated HDF5-to-LeRobot conversion; verify source/annotation/seeds, episode and sampled action/frame parity with derivative-aware splits.
- [ ] 7.6 Validate supported synthetic sensor-data generation and conversion through maintained upstream tooling; verify renderer/asset/label provenance and no synthetic samples leaking into real-data held-out evaluation.
- [ ] 7.7 Publish data/capture/conversion commands and trust/license requirements; verify negative fixtures, metadata-only discovery and no routine-test large downloads or simulator requirements.

## 8. ACT and teacher/student imitation learning

- [ ] 8.1 Wrap maintained ACT/model processors and masked losses in existing library pipeline/data components; verify a tiny offline optimizer update without ROS, simulation or a replacement trainer.
- [ ] 8.2 Validate train-only normalization, chunk targets and observation histories; verify padded boundary losses, action decoding, replanning and per-row reset tests.
- [ ] 8.3 Complete ACT dataset training, intermediate resume, export and clean sequence inference; verify full checkpoint/normalization/processor restoration and deterministic action parity.
- [ ] 8.4 Evaluate ACT in the matching Isaac manipulation task against a compatible scripted/RL reference; verify held-out task success, exact contracts and a separately labeled multi-seed quality report.
- [ ] 8.5 Distill the vial state teacher to wrist-camera-plus-joint inputs; verify real student updates, teacher/dataset provenance, no privileged inference inputs and complete export/replay/held-out comparison.
- [ ] 8.6 Publish offline and simulator imitation round trips; verify documented commands, tiny synthetic regressions, optional data/native tests and no requirement for a new IL session or policy hierarchy.
- [ ] 8.7 Integrate maintained Diffusion Policy and its actual masked training loss/sampler through existing LeRobot data/model/policy seams; verify exact code/processor identities, training-only statistics, sampling RNG, inference scheduler/denoising settings, chunk/reset semantics and a small real optimizer update without an upstream trainer.
- [ ] 8.8 Complete Diffusion training, intermediate resume, complete export/reload and fresh compatible Isaac evaluation; verify seeded sequence/action parity, inference-component packaging and matched-data/task/interface comparison with ACT, retaining held-out outcomes, measured resources and runnable tests/docs.

## 9. Real pretrained VLA policies and post-training

- [ ] 9.1 Pin and verify supported **GR00T N1.7** and SmolVLA model/code/processor releases, asset licenses/trust, source dependencies and embodiment mapping. Do not select obsolete GR00T N1.5 as a new implementation target.
- [ ] 9.2 Integrate SmolVLA image/state/language inference through the existing graph; verify processor/token/action parity on approved observations and real chunk execution/reset behavior.
- [ ] 9.3 Fine-tune SmolVLA on a compatible task dataset and complete resume/export/Isaac evaluation; verify changed trainable parameters, pretrained-versus-adapted results, latency/memory and old-task regression.
- [ ] 9.4 Integrate the selected GR00T model and embodiment transforms; verify pretrained inference, task-specific supervised adaptation, resume/export and compatible Isaac closed-loop evaluation independently of SmolVLA.
- [ ] 9.5 Compare full and supported parameter-efficient adaptation on one actual model; verify trainable scope, exact base-plus-adapter export and matched-budget old/new-task evidence.
- [ ] 9.6 Integrate simulated RL post-training for one real pretrained VLA with a valid head-specific objective; verify likelihood/action semantics, actual updates and no hidden RLinf trainer or fabricated Gaussian/token probabilities.
- [ ] 9.7 Compare pretrained, supervised and RL-adapted variants on identical frozen conditions; verify sample counts, uncertainty, forgetting, effective command rate and latency tails rather than a synthetic-policy-only proof.
- [ ] 9.8 Publish model-family and runtime support documentation with focused processor/state/loss tests; verify unsupported optimization/export targets fail explicitly and inference requires no training runner.
- [ ] 9.9 Freeze a maintained π0.5 model/backbone/tokenizer/configuration and a compatible Isaac task/dataset mapping; verify licenses/trust, dependency/API compatibility, normalization statistics and action conventions plus measured/approved compute needs, without automatic weight downloads, resource rental or a second incompatible stack.
- [ ] 9.10 Execute actual pretrained π0.5 inference and task-specific NexuML adaptation/resume/export/reload/closed-loop evaluation on approved resources; verify changed parameters, complete processor/config restoration, correct training-split statistics and action parity, leaving unavailable-resource gates incomplete rather than using a tiny substitute.
- [ ] 9.11 Publish matched-condition π0.5 comparisons with the established VLA/ACT/Diffusion references where interfaces genuinely match; verify old/new-task regression, sample counts, memory/latency/control-rate evidence, negative prerequisite/parity tests and runnable documentation without promising universal embodiment transfer.

## 10. Learned dynamics and DreamerV3

- [ ] 10.1 Verify and pin actual TorchRL V3 RSSM/model/actor/value/state-estimation APIs against the chosen protocol; verify documented symbols and explicit paper-versus-author-recipe settings without relabeling older Dreamer code.
- [ ] 10.2 Implement contiguous sequence replay and posterior context through existing algorithm/data seams; verify reset masks, burn-in/padding, stream tails, generation checks and physical/replay/imagined counters.
- [ ] 10.3 Compose maintained V3 world-model objectives on a small state-based Isaac task; verify reconstruction/reward/continuation/KL behavior, valid gradients and actual model updates against controlled numerical cases.
- [ ] 10.4 Add imagination actor/value learning with one effective optimizer/precision owner; verify exact step/slow-target/update-ratio/normalization counters and actual actor/value parameter changes.
- [ ] 10.5 Complete interrupted resume with and without recorded replay; verify networks/optimizers/RNG/ratio restoration, explicit refill semantics, remaining physical budget and no claim of continuing old physics trajectories.
- [ ] 10.6 Export encoder/filter/actor and inference-state initialization; verify clean-process latent/action parity across resets with training decoders/critics/replay/collectors blocked.
- [ ] 10.7 Extend the reference to a bounded visual arm task and compare against matched PPO; verify horizon-wise held-out prediction errors, continuation/reward metrics, control quality, physical sample efficiency and compute cost without presumed improvement.
- [ ] 10.8 Add a bounded learned-dynamics planning example and publish full Dreamer commands/results; verify controller-contract preservation, model-error attribution and reference-versus-optimized execution parity before optional kernel acceleration.
- [ ] 10.9 Deliver an independent hierarchical example selecting at least two compatible low-level skills; verify actual selector-driven transitions/task outcomes, invalid-target rejection, episode-state resets and preserved controller limits with an executable acceptance test and runnable documentation.
- [ ] 10.10 Produce a bounded VLA-JEPA feasibility report from maintained source/model contracts; verify exact component/API/license identities, dependency constraints, camera/state/action and dataset/embodiment fit, auxiliary predictive-training versus action-inference semantics, resource uncertainty and a proposed validation test, with an explicit verdict and no DreamerV3 or implemented-support claim.
- [ ] 10.11 Produce the independent FastWAM feasibility report; verify Wan/model/processor identities, dependency and license/trust conditions, actual training/inference component boundaries, action/chunk/data mapping and conservative resource needs with a bounded proposed test and explicit verdict, without unapproved model work or equating the published large-GPU example with a minimum requirement.

## 11. Common release evaluation and Arena

- [ ] 11.1 Integrate the single-owned success-only evaluator extension and task-specific predicates; verify no fabricated joint error, preserved SO-101 paired metrics, transient/held/terminal success semantics and per-episode isolation.
- [ ] 11.2 Extend existing evaluation for stateful policies and supported native batches; verify exact completed-episode budgets, partial resets, terminal observations and unchanged scalar/scheduled non-Isaac behavior.
- [ ] 11.3 Define frozen evaluation suites and candidate decisions; verify hash/version/reset/seed/sample metadata, pass/fail/insufficient-evidence outputs, uncertainty and worst-condition reports with focused fixtures.
- [ ] 11.4 Validate a compatible maintained Arena release natively in the canonical composition; verify the old Python-3.11 recipe is not reused as proof and retain any unresolved dependency/task mismatch.
- [ ] 11.5 Evaluate one actual adapted NexuML policy over pinned Arena scene/object/embodiment variations; verify policy mapping, held-out distributions, failure categories and reproducible machine-readable trials.
- [ ] 11.6 Exercise offline shadow, forgetting and reference comparisons plus real GIF/MP4/both delivery; verify counterfactual limits, terminal-frame identity, animated TensorBoard/MLflow behavior and truthful cancellation.
- [ ] 11.7 Verify maintained Isaac Arena, compatible LeRobot eval/EnvHub and actual native LIBERO/LIBERO-plus interfaces, with optional separately gated RoboTwin subset. Freeze task/suite/split/processor versions and collect real native evaluation evidence; do not copy or fake unavailable benchmark runs.
- [ ] 11.8 Produce a bounded SARM feasibility report through existing data/layer/evaluation seams; verify source/model/processor identities, dependency/license/resource constraints, annotation/temporal-prior provenance and a leakage-aware progress/weighted-imitation validation protocol covering failed/stalled/reversed/interrupted episodes, with an explicit verdict and measured-success/safety separation rather than a trained-support claim.

## 12. ROS, Gazebo and Isaac ROS runtime

- [ ] 12.1 Validate selected Sim bridge and Isaac ROS 5.0/Lyrical installation/workspaces; verify native distro/Python ABI/library identities and an actual standard-message/QoS/clock wire test rather than assumed cross-distro compatibility.
- [ ] 12.2 Add the thin exported-policy ROS inference runner using existing transport seams; verify fake and live observation/frame/unit/action/rate mapping with no session, optimizer or collector construction.
- [ ] 12.3 Exercise Isaac simulation observations -> policy -> simulated arm commands -> recording; verify reset, timestamps/TF/simulation clock, stale-data watchdog/fallback, failure/cancellation and actual movement.
- [ ] 12.4 Complete the independent live headless Gazebo/ros2_control fixture from original RL item 9.5; verify repeated startup/reset/closed-loop movement/shutdown and retain its separate ROS-distribution evidence.
- [ ] 12.5 Supply manipulation observations from FoundationPose or a reviewed maintained equivalent; verify calibration/frame/units/availability and compare sensor-to-command latency, memory and dropped data against plain ROS.
- [ ] 12.6 Exercise compatible cuVSLAM/nvBlox/Nav2 mobile perception and cuMotion/MoveIt planning; verify current 5.0 APIs, vendor/release limitations and separation of classical robotics from learned-policy claims.
- [ ] 12.7 Validate MCAP-to-LeRobot conversion and one actually compatible Isaac ROS Deploy/LEAPP target or explicit target-specific incompatibility; verify episode/sample parity, independent runtime output parity and continued reference PyTorch inference.
- [ ] 12.8 Publish system setup/launch/record/convert/run diagnostics with focused fake/system tests; verify optional import boundaries, no removed NITROS APIs, protected credentials and no hardware enablement in ordinary commands.

## 13. Exact multi-embodiment and industrial references

- [ ] 13.1 Publish machine manifests for the required reference inventory; verify obtainable hardware, exact asset commits/licenses, native task availability and separate asset/rollout/learning/evaluation/deployment statuses.
- [ ] 13.2 Validate maintained OpenArm unimanual and then bimanual tasks; verify native-first contracts and actual NexuML training/resume/export/replay with held-out manipulation evidence for each.
- [ ] 13.3 Train exact G1 29-DoF locomotion and a separate maintained humanoid manipulation/imitation reference; verify actuator/joint controls, fall/tracking outcomes, demonstrations where applicable and complete policy artifacts.
- [ ] 13.4 Train and evaluate exact Go2 velocity/terrain control; verify native baseline, real updates, checkpoint/export/replay, falls and held-out contact/terrain results.
- [ ] 13.5 Train a verified Crazyflie 2.1+ model for hover/goal tracking; verify its exact control contract, learning/resume/export/replay and held-out disturbance results, keeping the separate PX4 command/failsafe proof in task 13.10.
- [ ] 13.6 Train the TurtleBot3 Waffle Pi or an explicitly reviewed exact supported alternative; verify matching Isaac assets/task, held-out learned navigation, Nav2 comparison and separate ROS/Gazebo runtime evidence.
- [ ] 13.7 Validate maintained UR10e/Rizon gear insertion and independent UR5e/URSim control; verify pre-grasp and perception/controller assumptions, real insertion-policy learning and complete artifacts without cross-model substitution.
- [ ] 13.8 Deliver a non-actuating LumenPnP or approved offline industrial-data experiment and read-only OPC UA reference where available; verify task utility, units/timing/security, no actuator writes and a costed simulator feasibility report rather than an invented maintained asset.
- [ ] 13.9 Publish per-embodiment setup/train/resume/evaluate docs and focused contract/native tests; verify named-hardware compatibility, quality evidence and no physical humanoid/drone/mobile deployment claims.
- [ ] 13.10 Execute an exported NexuML policy through a supported PX4 offboard/trajectory interface on a separate compatible SITL airframe; verify bounded framed/rate-correct commands, actual motion, injected heartbeat/command loss and native failsafe timing against a matched classical-controller baseline, with reproducible tests/docs and no Crazyflie or hardware-flight claim.

## 14. Advanced simulation and representation workflows

- [ ] 14.1 Validate native image/depth, range/ray, IMU, contact/wrench and visuo-tactile sensor inputs; verify shapes/frames/rates/freshness and an actual policy-consuming example for each supported family, not an inspection-camera proxy.
- [ ] 14.2 Exercise maintained Newton rigid/soft/particle solver and standalone PhysX references with recorded presets; verify physical/contact validity, native task compatibility and renderer/runtime separation for each claimed path.
- [ ] 14.3 Integrate a maintained deformable or coupled-solver learning reference; verify native stability, observable task/control semantics, actual learning and replay without replacing deformables with rigid proxies.
- [ ] 14.4 Validate one reconstructed real-to-sim scene and compatible policy evaluation; verify capture/geometry/collision/sensor/license provenance and held-out results without claiming visual realism proves dynamics fidelity.
- [ ] 14.5 Exercise explicit domain shifts and trace-driven physics identification on applicable references; verify sampled values, held-out robustness and retained calibration/dynamics provenance across simulation backends.
- [ ] 14.6 Add one self-/semi-supervised representation experiment on approved unlabeled robot or industrial streams; verify dataset/split identity, changed representations and matched downstream utility without foundation-model pretraining.
- [ ] 14.7 Publish a pinned feature-support matrix and native probe commands; verify each upstream workflow family is supported with evidence, explicitly blocked, or an identified optional extension rather than silently advertised.

## 15. Remote inspection and scaled execution

- [ ] 15.1 Add optional pinned browser-viewer replay through the existing runtime configuration; verify actual loopback SSH viewing, share-off defaults, disconnect/cleanup and debug resets invalidating measured trials.
- [ ] 15.2 Validate supported Lab WebRTC experience and private-network access; verify NVENC/graphics, actual TCP signaling and UDP media, truthful missing-network diagnostics and no public unauthenticated exposure.
- [ ] 15.3 Introduce one-simulator-per-rank multi-GPU support through existing placement/learning seams; verify unit tests for rank/device/seed/global-frame/log ownership before removing current topology guards.
- [ ] 15.4 Execute actual two-GPU learning/resume/export and injected-rank failure/cancellation; verify synchronized updates, truthful global counters, stopped owned ranks and no false complete artifact.
- [ ] 15.5 Exercise multi-node placement/rendezvous using the selected existing execution path; verify native construction at workers, no serialized live handles and actual cross-node cleanup/failure evidence.
- [ ] 15.6 Integrate supported scenario sweeps and bounded population-based updates through existing tuning controls; verify parameter/state transfer, synchronization, checkpoint identity and no claim that ordinary search alone is PBT.
- [ ] 15.7 Benchmark startup, simulation/rendering, end-to-end learning and inference separately; verify synchronized timings, warm-up/resource identities and action/gradient/state parity before enabling compile/capture/optimized sequence backends.
- [ ] 15.8 Publish SSH/WebRTC, multi-device and representative HPC/cloud launch instructions with opt-in tests; verify unavailable infrastructure leaves corresponding gates open rather than counted from fake tests or CI wiring.

## 16. Operator-supervised hardware and local improvement

- [ ] 16.1 Select an approved exact arm and record firmware, calibration, controller and workspace; verify independent stop/intervention/fallback provisions and operator go/no-go criteria before motion.
- [ ] 16.2 Validate complete policy and optimized-target parity on recorded observations; verify processor/state/action equivalence, rate/memory/sensor readiness and zero commands while hardware enablement is absent.
- [ ] 16.3 Perform separately approved low-speed arm trials and leader/follower or corrective capture; verify actual task/intervention/latency/safety-trigger evidence and immediate operator-stop behavior, never unattended exploration.
- [ ] 16.4 Compare matched simulated and physical trials and validate captured local experience; verify sensing/dynamics/delay gaps, calibration/episode provenance, authorized retention and held-out splits.
- [ ] 16.5 Expose the local offline adaptation operation through existing scenario/session APIs; verify approved parent/data/config inputs, trainable scope/budget and deterministic candidate/evidence/error outputs without NexuFL or automatic upload.
- [ ] 16.6 Compare naïve fine-tuning against replay/regularization and targeted failure selection against equal-budget random selection; verify old/new-task results, forgetting, selected episode identities and no contaminated evaluation.
- [ ] 16.7 Apply frozen shadow/simulator/regression decisions before a separately approved candidate trial; verify pass/fail/insufficient-evidence handling, retained known-good artifact, manual rollback and no mid-episode hot swap.
- [ ] 16.8 Publish the supervised sim-real/local-improvement report and reproducible commands; verify hardware evidence remains explicitly blocked when resources/approval are absent and no software smoke is counted as a physical pass.

## 17. Integrated acceptance and full-program closeout

- [ ] 17.1 Run relevant broad CPU/coverage, lint/format/type, lock and strict docs/OpenSpec gates after integration; verify baseline failures are separately retained and all per-slice tests/docs already exist.
- [ ] 17.2 Build and inspect wheels/sdists plus clean installed-package inference/training proofs; verify base optionality, actual dependency manifests, processors/state/adapter composition and old supervised/Gymnasium/package behavior.
- [ ] 17.3 Run compatible primary setup, Isaac/ROS/VLA/world-model and **MuJoCo/Meta-World/MJWarp/correction/reward/model/benchmark** native acceptance, including separately approved distributed/hardware gates. Preserve task/source/lock hashes, exact commands, unexecuted/skipped statuses and recorded failures.
- [ ] 17.4 Reconcile all 236 original roadmap checkboxes, all capability specs (including the new local-learning-handoff spec), reference-catalogue.md and unified tasks 18–22 against exact tests/artifacts. No acceptance may be borrowed from upstream training, an old Isaac profile, another robot/backend or the historical MuJoCo task queue.
- [ ] 17.5 Publish milestone and full-program status, runnable user documentation and retained rollback inputs; verify full completion requires every mandatory software and manual gate, all owned jobs are stopped and no merge/archive/deployment is implied.

## 18. MuJoCo, Meta-World and task-outcome integration

- [x] 18.1 Reconcile all 32 tasks of the historical MuJoCo proposal against this single task queue. Inspect existing GymnasiumEnvironment, PPO, evaluator, scenarios, packages and tests; preserve verified work, record unresolved prerequisites and choose one shared task-outcome code owner.
- [ ] 18.2 Declare genuine MuJoCo/Meta-World compatibility requirements with library extras/root forwarding and built metadata; freeze reference versions in consumer locks rather than impose development pins on all users. Validate an approved native lane where actual dependencies conflict with the primary. No --no-deps, global Python <3.14, silent omission or hidden trainer.
- [ ] 18.3 Probe native Humanoid-v5 (existing 45/17 observation/action contract), Reacher-v5 (10/2) and Meta-World reach-v3 (normalized Cartesian displacement+gripper); verify upstream identity, bounds, units, control period, reset, termination, truncation and final-observation semantics.
- [ ] 18.4 Register metaworld-reach-ppo lazily through existing GymnasiumEnvironment and ordinary configuration/compile/YAML/override/CLI discovery. Only add a thin provider after an actual failed public upstream registration path; base installs must not import simulators.
- [x] 18.5 Implement missing success-only outcome support with exact validated bool or finite native zero/one, batch/device and stable absent/success-only/success+nonnegative-error signatures; transient any-step success is counted once. Preserve SO-101 paired outcome regressions and no fabricated joint error.
- [ ] 18.6 Verify real CPU direct/process collection, TorchRL policy sync, reset/timeout targets, precise frame budgets, interrupted/completed resume and failure cleanup independently for all three references.
- [ ] 18.7 Train all three via NexuML-owned updates; validate actor changes, finite losses, interrupted checkpoint state/remaining-budget resume, complete export, clean inference with training/simulator imports blocked and fresh native evaluation. A smoke alone is not convergence.
- [ ] 18.8 Exercise true CPU OSMesa and NVIDIA EGL headless capture on all three with nonempty RGB, correct final-frame identity, GIF/MP4/both, FPS/limits, TensorBoard and MLflow-only logging, truthful cancellation and no renderer for state-only training.
- [ ] 18.9 Freeze actual Meta-World MT1 50-goal identities, horizon/reward/success/protocol and three PPO training seeds (0,1,2); compare to matched native seeded random actions with per-goal any-step success, returns, uncertainty/runtime and separately labeled held-out variations.
- [ ] 18.10 Publish the exact native lock/interpreter/system/graphics/task/asset evidence, package/CI negative tests, scenario commands and Phase 4/6 progress in the unified report; do not edit earlier MuJoCo OpenSpec history or mark its old tasks complete.

## 19. GPU MuJoCo Playground and MJWarp

- [ ] 19.1 Inventory maintained MuJoCo Playground release, MJX-JAX/MJWarp native API, JAX/Torch backend and version/asset/license requirements. Pin small classic control, articulated locomotion/manipulation and visual reference tasks to real native task IDs.
- [ ] 19.2 Expose lazily discovered optional NexuML library task definitions, renderer and physics selection with portable control/task contracts; don't allocate JAX GPU context, download Menagerie assets or add a new session when merely discovering a scenario.
- [ ] 19.3 Native-probe GPU batched transition/final/auto-reset, device and time/environment axis, action units, episode accounting, truncation numerical targets and cleanup against a real pinned native oracle; no fake TensorDict pass as native GPU proof.
- [ ] 19.4 Run maintained upstream train-jax-ppo or train-rsl-ppo as explicitly labeled external quality baselines, not as a substitute for NexuML's optimizer/resume/policy/export evidence.
- [ ] 19.5 For a genuinely supported MJWarp/Torch task, implement NexuML-owned learner updates, checkpoint/resume, complete policy export/reload and native evaluation with no hidden upstream trainer. If public APIs prevent this, record an explicit non-completion/interop blocker and design evidence.
- [ ] 19.6 Validate one exact articulated model and one true image-observation/rendered policy example with provenance, held-out metrics and measured render/native/training throughput versus upstream on matched control semantics.
- [ ] 19.7 Publish separate locked optional profiles, executable commands, native GPU/render opt-in test tiers, unsupported hardware/JAX combinations and reproducible benchmark artifacts without degrading CPU Gymnasium/supervised NexuML.

## 20. Corrections, reward learning, flow policies and timed inference

- [ ] 20.1 Audit maintained LeRobot v0.6.1 rollout/DAgger, reward model, dataset-depth/language/time, model processor and benchmark APIs (including changed imports), recording exact code/model license/resources before use.
- [ ] 20.2 Ingest real/simulated corrections with proposed versus executed action, intervention interval, operator policy checkpoint, sensor timestamps, depth/language/subtask labels, masks and episode-disjoint train/held-out provenance.
- [ ] 20.3 Demonstrate simulation-first SO-101 DAgger-style correction collection → NexuML offline candidate training/resume/export → frozen old/new/failure evaluation, using upstream teleop tools rather than another custom GUI.
- [ ] 20.4 Implement one bounded RLPD-style offline demonstration + new simulation replay workflow on existing SAC/algorithm-owned replay interfaces, with mixture, policy staleness, target/optimizer accounting, seed and true interrupted resume.
- [ ] 20.5 Integrate simulated HIL-SERL interventions and one trained reward classifier, with negative reward-versus-native-task-success tests and maintained task-safety boundaries; no automatic online real-robot learning.
- [ ] 20.6 Assess Robometer and TOPReward processors, versions, licensing, compute and held-out failure/reversed/stalled episodes; optionally implement small approved scoring while keeping learned progress/reward distinct from frozen task outcomes.
- [ ] 20.7 Add one maintained small flow-matching/Diffusion Transformer action-policy training/resume/export/clean-inference/Isaac round trip with time masks, iterative sampler, RNG, chunk/reset parity and matched SO-101 ACT/Diffusion comparisons.
- [ ] 20.8 Add optional asynchronous local/remote chunk inference with per-stream timestamps/state, stale observation and missed-deadline rejection, bounded buffering, controller-delegated fallback and recorded action/latency parity. Local stateless inference remains unaffected.
- [ ] 20.9 Produce independent actual update/quality/forgetting/intervention/latency/sample-efficiency evidence for corrections, offline-to-online, learned reward and flow methods; freeze comparable controls and disallow reward-only quality claims.

## 21. Reference catalogue and benchmark acceptance

- [ ] 21.1 Port upstream SO-101 six-action physical-contact vial placement with exact home-reset/success; train state PPO and teacher-to-wrist-camera distillation using actual student on-policy trajectories and deployment-only student inputs.
- [ ] 21.2 Implement SO-101 camera PPO **from scratch**, asymmetric critic where supported and same camera/control/reset/sim2real randomization tests as distillation, proving NexuML rather than upstream RSL-RL training.
- [ ] 21.3 Complete matched task/contract round trips for SO-101 state PPO, camera PPO, distillation, ACT, Diffusion, flow, SmolVLA, GR00T N1.7 and approved resource-gated pi0.5; report privileged-input and modality differences rather than pretending identical observations.
- [ ] 21.4 Preserve exact OpenArm unimanual/bimanual (+ optional drawer task), G1 exact 29-DoF locomotion/Mimic locomanipulation, Go2, Crazyflie/PX4, TurtleBot3, UR10e/Rizon, UR5e/URSim and LumenPnP source identity, control and separate learning/evaluation status from the original plan.
- [ ] 21.5 Run compatible native LIBERO/LIBERO-plus through maintained LeRobot evaluation (in addition to Arena) with exact matching policy/task protocol, and assess an optional RoboTwin 2.0 subset without fabricating incompatible suite results.
- [ ] 21.6 Feasibility-gate optional dexterous Allegro/Shadow/DexSuite object reorientation and GR-1 Mimic manipulation; promote only actual maintained feasible examples with independent policy training/evaluation and recorded approvals.
- [ ] 21.7 For each I01–I22, M01–M06 and B01–B03 reference ID record source/release/model/assets, sensor/control conventions, provenance and separate stages: native, NexuML training, resume, export, fresh evaluation/media, quality and manual hardware. Never count another embodiment or old release.
- [ ] 21.8 Publish per-embodiment held-out outcome and failure injection protocol (falls, grasp/lift/release, occlusion/latency, collision, flight failsafe and industrial outliers), preserving matching classical/upstream baselines where relevant.

## 22. Single programme ownership and closeout

- [ ] 22.1 Reconcile all capability specs, new reference catalogue, imported MuJoCo requirements, original 236-checklist roadmap and historical 42 completed/23 open Isaac/32 MuJoCo tasks; assign exact test/evidence owner and forbid a parallel MuJoCo execution queue.
- [ ] 22.2 Run one staged campaign in dependency order: compatible setup+Isaac/CPU MuJoCo, SO-101 control/manipulation/camera/data, native Meta-World/MJWarp, modern imitation/VLA/reward/correction, frozen benchmarks, additional embodiments/ROS, approved manual proof. Implement a usable vertical slice, not one unchecked monolith.
- [ ] 22.3 After each slice verify applicable tests, native action/transition oracle, learning, resume, export/replay/media, hashes, quality and dependency negative paths; retain failures locally. After a verified milestone create a focused local commit under D20: explicitly stage owned source/tests/docs/configuration/locks/specs/task changes and inspect staged paths/content; exclude all execution evidence, including reports, logs, generated results/media, checkpoints/weights and mixed-file evidence. Do not publish the branch automatically.
- [ ] 22.4 Run final strict OpenSpec and broad regression/wheel/setup/native CPU/GPU/VLA/benchmark acceptance. Unsupported licensed assets, ROS/distributed resources or manual hardware remain explicit blockers for corresponding required completion.
- [ ] 22.5 Reconcile every required reference independently against evidence, promote P0 milestones without falsely closing P1/P2 or hardware; document exact commands, limitations and user-approved resource steps.
- [ ] 22.6 Reconcile verified local milestone commits and provide truthful consolidated capability/evidence status for review outside Git. Do not commit evidence reports/results, modify old task counts, silently edit siblings or start downloads/rebuilds/hardware actions. Keep push, external review publication, merge and archive separately authorized; preserve prior committed history and local evidence.

## Workflow follow-up

- Implement the historical MuJoCo obligations **only through this unified change**; coordinate any NexuVibe setup and operator-gated actions in their own scopes without overwriting busy checkouts.
- Obtain specific operator approval for container recreation, host/system changes, newly licensed assets/models, expanded resource budgets and physical trials when those actions become necessary.
- Create one focused local commit after each verified implementation milestone under D20, excluding execution evidence and unrelated work; check pending verified milestones at the next apply run. Push, external review publication, merge and archive only when separately requested. Full completion remains unavailable while a required hardware, distributed or compatibility gate is unverified.
