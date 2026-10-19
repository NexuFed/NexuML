# Design

## Context

See `proposal.md` for motivation and the declared capability deltas for observable behavior. This is a program-sized, staged change: planning readiness is not package, native, learning-quality or hardware readiness.

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

`add-isaaclab-gpu-robotics` has 42 original completed tasks and 23 open extensions. Its retained 15 native-image test passes are historical acceptance only. `add-mujoco-reference-workflows` is a historical plan whose alternative-simulator references and success-only outcome requirements are now imported into this umbrella. Its original Meta-World pin conflicts with the prior selected Newton family, and its impossible combined dependency closure is no longer authoritative. The full roadmap snapshot is indexed in `coverage.md`; no source roadmap or sibling proposal is changed here.

The inspected running DevContainer is Ubuntu 26.04.1 with `compute,utility`, although its prepared NVIDIA configuration selects Ubuntu 24.04 plus graphics. NexuVibe currently declares Python 3.14 and uses `/env`; the user explicitly accepts changing that primary environment to Python 3.12 and CUDA 13. Prepared files are not effective runtime evidence.

## Goals / Non-Goals

**Goals:** One Isaac-first developer experience and one NexuML learning/artifact architecture; broad explicit workflow coverage with native-first, numerical and end-to-end proofs; one automatically provisioned primary Python environment; full inference semantics; secure remote development; staged multi-embodiment and supervised field evidence.

**Non-Goals:** No second session/compiler/trainer, universal robotics superclass, custom physics engine or driver, copied foundation model, compulsory robotics dependency for all users, unmanaged remote code execution, robot-safety certification, autonomous hardware exploration, fleet operation or NexuFL implementation. "Everything" means all catalogued workflow families, not every upstream preset or unsupported Cartesian product. Missing prerequisites leave required gates open; they are not permission to narrow the request silently.

## Decisions

### D0. Local SO-101 behavior cloning and Python SDK handoff before the broad roadmap

The first integration milestone uses **the real, source-pinned SO-101 reaching task**, not a generic Reacher or inferred success from earlier PPO tests. Generate reproducible expert trajectories on the simulator, with correct arm joints/action units, goal/reset identity, timestamps and source provenance. The expert may use privileged information, but the student must see only its declared deployable **state-based actor observations**. Train a small behavior-cloning policy using existing NexuML data/pipeline/session/Lightning machinery, then export/reload a complete policy and verify fresh, held-out **closed-loop** simulator success separately from expert success and training loss. Simulation need only produce episodes and evaluate policies: it need not run inside each later FL fit.

For an external caller, add **only** the minimum stable Python operation composed over existing `NexuSession`, dataset and export APIs. Semantically it accepts a trusted installed recipe/model, exact compatible initial **named numeric tensors**, authorized local on-disk episodes, bounded step/epoch budget, deterministic seed and explicit optimizer semantics; it returns updated tensors, approved finite metrics, number of eligible training windows/episodes and model/data/physical-schema fingerprints. The first stateless FedAvg consumer SHALL use **optimizer reset per round**; persistent per-site optimizer continuation is a later independently verified capability, never silently promised. Second-call initial weight override and error cases must be tested. No duplicate trainer/graph, no `FederatedTrainer`, no network orchestration.

Export is separated into (1) operator-trusted, pinned executable recipe/`torch.package` if required locally; (2) portable Safetensors-compatible numeric weights with architecture, parameter names, dtypes, shapes, base version, tensor roles and exact SO-101 observation/action/processor/joint/order/control-rate identity; (3) **private** local episodes and optimizer state. Never treat matching shapes as physical compatibility or ship Python pickle/executable packages as remote model updates. Preserve train-only normalization, episode-disjoint splits, honest consumed-window counts, simulator-free ordinary imports, and existing non-robot workflows.

**Python SDK is authoritative; REST is explicitly out of scope** for this milestone. NexuFL's locally installed `ClientProgram` calls the SDK in process and its NodeAgent independently owns gRPC and Kubernetes. When the native expert → learned policy → clean export/reload → held-out closed-loop reference and repeated bounded local-fit contract pass, **stop and prepare a reviewable narrow NexuML PR**. Do not merge the entire unfinished feature branch as the dependency for FL. The NexuFL integration proposal owns subsequent local FedAvg, external gRPC, Kubernetes, ACT, personalization and active learning.

### D1. One umbrella, explicit ownership, no rewritten history

This change owns the forward Isaac-first architecture and full-program acceptance. Preserve the older 42-task result and existing policy packages. Before shared implementation, record which open legacy extension items are adopted here, retained as legacy-profile proof, or require an explicitly authorized artifact amendment. Do not check off the older two-environment extension from a one-environment migration, and do not archive either change by implication.

`coverage.md` is the source-to-capability map. Implementation expands it to exact requirement/test/evidence identifiers and compares it with the source snapshot before completion. This umbrella owns the native MuJoCo Humanoid/Reacher/Meta-World workflows and GPU Playground tasks, with a single shared outcome evaluator/adapter owner. The historical MuJoCo plan shall not be executed as a second queue. Original live Gazebo and CARLA statuses remain unchanged except where this program actually supplies the required Gazebo proof; CARLA expansion remains external.

**Rejected:** duplicate all existing specs under a new name, count old tests as new-release proof, or let two agents edit the same evaluator/contract concurrently.

### D2. One primary consumer-owned Python environment and explicit compatibility lanes

Maintain one automatically provisioned, pinned Python 3.12/CUDA-13 Isaac-first NexuVibe development environment where the currently tested Isaac/LeRobot/TorchRL/NexuML stack works. Treat Isaac Lab, Isaac Sim, Newton, Warp, MuJoCo, PyTorch/TorchRL and model versions in the old candidate table as **unverified release candidates**, not an immutable compatible set. Reinspect the pinned upstream tutorial, modern source APIs, CUDA wheels, licenses, driver requirements, resource budget and actual complete uv resolution. No default-weight download or unrelated package upgrade during discovery. Preserve original EA/cu128 evidence.

NexuML library declares concrete named optional extras and core forwards only compatible groups. Plain NexuML remains simulator-free. Do not force all available native simulators, ROS/CUDA families, JAX and old Meta-World into one undifferentiated all extra, or restrict the Python range of all NexuML users merely because one benchmark requires an older Python/MuJoCo version. Record a tested compatibility matrix for primary and optional components including CPU, CUDA, EGL and OSMesa renderers.

#### Published compatibility versus consumer environment selection

The user confirmed that downstream users should have as few restrictions as
genuine compatibility permits. Library metadata declares required packages and
API-backed minimums, with exclusions/upper bounds only for demonstrated
incompatibility. Do not copy the development lock, CUDA family, exact Torch
version, wheel index or optional simulator's Python restriction into general
NexuML requirements. Retain the real core Python minimum (currently >=3.12)
unless a separately tested compatibility change lowers it; do not invent a
package-wide upper bound merely because a reference targets Python 3.12.

The consuming project's root `pyproject.toml` selects Python, backend/indexes,
concrete package/source versions and approved `[tool.uv].override-dependencies`;
its `uv.lock` records the complete reproducible environment. uv ignores member/
installed-library override declarations during root project resolution. Our
Python-3.12/cu130 pins and repository lock are validation/provisioning choices,
not universal consumer requirements. Overrides neither install missing packages
by themselves nor prove native compatibility. Existing API/ABI/security and
physical-contract checks remain; an otherwise compatible selection is not
rejected solely for differing from the reference environment. Untested
combinations remain unverified, not automatically supported.

Publish `nexuml-library[robots]` as the concrete optional robotics group and
`nexuml[robots]` as its forwarding extra. Exclude that group and its new native/
model stack from both `all` closures; retain unrelated general-purpose extras.
The development consumer explicitly selects `nexuml[all,dev,robots]` and its
backend so the implementation environment contains the requested stack without
burdening ordinary installations. Keep incompatible native providers selectable
through explicit named extras/lane rules rather than hiding impossible unions
inside `robots`. This does not omit any required reference or authorize an
unapproved environment. Selecting `robots` may still expose real upstream
Python/wheel/ABI constraints; do not use markers to silently skip a requested
integration. uv also resolves optional extras when locking, so the group name
alone does not fix conflicts.

Acceptance must inspect wheels/sdists and root forwarding, ensure base/`all`
imports do not force the new robotics stack, audit every retained exact pin/
upper bound against a concrete compatibility reason, and verify that consumer
backend/index/version choices remain authoritative within genuine requirements.
Record tested/blocked/untested versions separately from installable ranges.

When *native* compatibility truly cannot be attained via maintained upstream releases, an explicitly reviewed, opt-in pinned process/worker or separately locked execution profile may host the incompatible benchmark. It must preserve normal NexuML scenario, policy, data and metrics semantics at an explicitly tested process boundary, while never serializing active physics handles. It remains blocked until real native and policy interoperability evidence exists, not a hidden fallback. This exception does not create an environment per algorithm or replace the primary setup.

### D3. Meta-World, Newton, MuJoCo and JAX cannot simply be combined by Python pinning

The previously documented Meta-World 3.1.1/MuJoCo==3.3.0 requirement and Newton/MJWarp MuJoCo-3.12-family candidate conflict is substantive; an interpreter change alone cannot resolve it. Re-evaluate actual published metadata and maintained Meta-World alternatives; publish compatible selections with exact release and lock identities. The sibling MuJoCo proposal's all-extras closure and global >=3.12,<3.14 bound are **superseded in this umbrella design**; preserve sibling files for history. Avoid --no-deps, unsupported upstream patches, hidden pip fixes or silent benchmark removal.

Maintain three independent MuJoCo paths: (a) Gymnasium native CPU Humanoid-v5/Reacher-v5 reference PPO and headless replay; (b) native Meta-World MT1 reach-v3 with exact goal/task/action rules and success-only outcomes; (c) GPU MuJoCo Playground using its maintained MJWarp/MJX/renderer and real task assets. A JAX-native upstream PPO baseline stays labeled **external baseline**, not a NexuML Lightning/TorchRL learner. Only claim NexuML-owned GPU training when an actual supported Torch/MJWarp integration executes NexuML optimizer, checkpoint/resume, policy export and native evaluation. A future JAX learner requires an independent reviewed optimizer/serialization contract, not an informal TorchRL bridge.

Shared evaluator success-only changes have one code owner, regardless of simulator. Native task/asset/control contracts, terminal/reset transition parity, Python package integrity, CPU OSMesa/NVIDIA EGL and policy exports are independently tested in the chosen lane. Missing native prerequisites leave that capability visibly incomplete; other reference slices may still pass.

#### Temporary constraint exceptions — policy revision approved 2026-10-19

Prefer a maintained upstream-compatible selection. When the recorded primary
Isaac/LeRobot/OvPhysX bounds still conflict, permit explicit consumer-root
constraint overrides for the initial **packaging, Transformers, NumPy, urllib3
and isolated-importer usd-exchange conflicts**, with further documented cases
permitted under the user's campaign authorization. This approves a
bounded evaluation/validation route, not a particular replacement version or a
claim that the constraints are merely stale metadata. Select an Isaac-supported
combination rather than upgrade every dependency independently. Other conflicts
need a documented source/API rationale and focused compatibility tests under the
user's subsequent campaign authorization; no blind Meta-World/MuJoCo physics
override, native ABI, Python/CUDA compatibility or driver requirement is waived. Keep required families
and the existing approved-lane rules intact.

Use the existing backend report for each exception record, not a new registry:
affected package/source and original requirement; exact replacement constraint
and resolved versions; reason and upstream issue/source reference; consumer-lock
identity; owning setup slice; affected workflows; approved resource/license
prerequisites; test commands/results; removal condition and rollback inputs.
An upstream issue may be not yet filed; record that honestly rather than invent
one. Consumer-owned configuration makes the exception explicit without editing
source/wheel metadata, vendoring a fork, adding hidden pip fixes or using
`--no-deps`. NexuML package metadata remains truthful about published optional
dependencies; consumer override policy is not silently exported to every user.

Keep evidence stages separate. Task 1.5 may prove complete **resolution with
declared exceptions** and immutable provenance; this does not prove native or
model compatibility. Before claiming an affected workflow ready, verify:

- **Packaging:** actual selected native startup, version-checking paths,
  task construction/stepping and owned shutdown for OvPhysX/Isaac, plus affected
  packaging callers; an import alone is insufficient.
- **Transformers:** actual maintained SmolVLA/GR00T N1.7 processors/tokenizers,
  pretrained inference, trainable-parameter updates and the required
  resume/export/reload/closed-loop references; include π0.5 when selected in
  the same environment. Tiny substitute models do not qualify real models.
- **NumPy:** test the recorded Sim 2.3.1/LeRobot <2.3 boundary with the selected
  OpenCV build, dataset conversions and real model/native paths, including
  NumPy/native-extension ABI and numerical parity; a resolver override is not
  evidence that either caller works.
- **urllib3:** preserve Lab's >=2.8.0 security floor, not Sim's older pin; verify
  the selected HTTP/S3/download paths and security review before provisioning.
- **USD:** keep Lab's matched usd-exchange 3.0.1 selection provisional until
  isolated importers, OpenUSD/schema/native-library identity, converted-asset
  parity and simulator startup/stepping/shutdown pass. Never co-install
  competing pxr providers or silently extend this to other native libraries.
- **Integrity:** clean locked provisioning and import/source identities,
  preserved ordinary NexuML/TorchRL behavior and all relevant native/action
  contracts. Required assets/resources without approval keep their execution
  gates open; provisional resolution never closes them.

Preserve the normal upstream dependency-check output. A consumer override may
leave `pip check` reporting its original bound: match and disclose only the
exact recorded discrepancies, verify all other dependencies and fail on any
unexpected mismatch. Never patch metadata or suppress the checker to claim
an unqualified clean installation. If behavior fails, retain the failure and
block/roll back that selection rather than broaden exceptions automatically.

On every affected version/source change, requalify the exact exception and
rerun its relevant tests; old evidence does not carry forward. Once a maintained
upstream set resolves the conflict, remove the exception and prove ordinary
resolution/integrity plus behavior again before updating the primary lock.
This policy revision supersedes earlier no-overrides statements in historical
coverage/report checkpoints while preserving their conflict evidence. It
authorizes no installs, active-container replacement, sibling-repository writes,
downloads, paid compute, driver changes or physical actuation by itself.

The user's subsequent approval also starts simulator-independent implementation
(including shared outcome task 18.5) against focused source/contracts and
regression tests. This is a scheduling exception, not completion of 1.4–1.6 or
permission for native-dependent slices to skip full resolution, source/license,
resource, clean-provisioning and runtime proof. Preserve all blocked states.

### D4. Coordinate DevContainer delivery without hidden sibling writes

The user now explicitly authorizes direct task-related edits in NexuVibe,
preserving its staged/uncommitted unrelated work. This is the owning consumer
delivery scope, not permission to sweep other changes into our commits. For the
current campaign reuse the single isolated `.venv` in this NexuML worktree and
override inherited VIRTUAL_ENV/UV_PROJECT_ENVIRONMENT explicitly on all package
commands. Package/source installs, bounded native tests and dependency
experiments are authorized there; `/env` and the active DevContainer are not
replacement targets. Keep at least 40 GiB user-available disk and avoid
interfering with other jobs. No paid compute, driver changes, container rebuild,
gated/custom-license acceptance or physical actuation follows from this scope.

Isaac has priority over incompatible optional libraries in this campaign.
Record blocked integrations and continue independent work rather than stop the
campaign or quietly force a physics mismatch. Defer large pretrained/foundation
model weight downloads and training for now, including GR00T/pi0.5 and large
research models. Their full-program requirements stay open; this is a staging
decision, not advertised support or permanent scope deletion. Small native PPO,
data/contract and bounded small-model workflows remain in scope. The ten-hour
target is a work budget with resumable milestone checkpoints, not a guarantee of
session lifetime or full-program completion within that window.

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

Add maintained LeRobot **Diffusion Policy** as the second complete IL example after ACT. Reuse its actual noise-prediction loss and inference sampler, recording training/inference scheduler identities, denoising steps, normalization, camera/state layout, chunk/stride and randomness. Do not replace the objective with Gaussian PPO or assume every head is plain action MSE. Check actual updates, intermediate resume, complete inference export and fresh Isaac replay; compare with ACT on the same approved data/splits and physical interface. Seeded reference sequences must reproduce sampled actions and reset behavior within recorded numerical tolerances. Required inference schedulers/denoising components are part of the policy, not discarded as training-only code.

Distill the SO-101 vial teacher to camera-plus-joint observations as a second concrete use of sequence/processor contracts. The source state teacher can use training-only information; held-out student inference must succeed with that information blocked. Do not claim hardware readiness from the teacher's hidden object/contact state.

Require actual pretrained SmolVLA, a maintained GR00T release and **π0.5**, not only `TinyVLA`. Use maintained LeRobot/GR00T model and preprocessing APIs, with exact model weights/code/processor hashes. Compare full fine-tuning and supported LoRA/adapters on at least one model, preserve composed export identity and test forgetting. Other VLA families remain catalogued extensions or the explicitly scoped research assessments below, not mandatory foundation-model pretraining.

For π0.5, pin the real base/adapted model composition, tokenizer and any separately gated backbone terms. Preserve complete checkpoint configuration, camera/state/action mapping, absolute/relative action conventions, chunk/replanning settings and the selected normalization mode. Quantile normalization needs training-split `q01`/`q99` statistics; do not replace them with held-out statistics or silently switch to mean/std. Upstream weight-only versus full-config load paths differ, so prove the exported NexuML package restores the actual chosen computation. Require pretrained inference, task-specific adaptation, intermediate resume, export/reload, matching Isaac closed-loop evaluation and old/new-task regression. No claim that LIBERO-trained action heads transfer unchanged to SO-101, G1 or another control space.

The inspected LeRobot catalogue is [the user-linked PyPI page](https://pypi.org/project/lerobot/) (0.6.1 at inspection), not a request to install every extra or freeze against moving docs. Upstream's indicative training guide estimates roughly 8–14 GiB for Diffusion and 24–40 GiB for large VLAs at batch 8; its concrete π0.5 full-fine-tuning recipe uses an 80-GB GPU. These are sizing hints, not measured local requirements or guarantees. The last inspected local GPU is 16 GB: stage offline training and simulation separately, measure actual limits and obtain approved larger resources when necessary. Missing resources keep the π0.5 round-trip gate open; do not count a feasibility report as that required example or submit a cloud job automatically. Larger compute does not authorize a second incompatible Python stack.

Use the TorchRL VLA schema/adapters when available: `LeRobotPolicyWrapper`, `LeRobotExperienceReplay`, `ActionChunkTransform`, `ActionScaling`, `MultiStepActorWrapper` and masked `BCLoss`. Verify actual package symbols and behavior rather than infer availability from `main` docs. VLA RL post-training is required on one real pretrained model; select an objective valid for its action-head distribution. RLinf's published GR00T/Isaac mapping informs a native baseline, but do not import its complete parallel training stack to bypass NexuML. If no compatible maintained objective is available, record a required integration blocker rather than inventing Gaussian likelihoods or counting a synthetic token update as the real-model proof.

**Rejected:** fork ACT/Diffusion/SmolVLA/GR00T/π0.5, add a remote model microservice by default, promise faithful ONNX export for every processor/head, or label an untested family supported because its checkpoint downloads.

### D10. DreamerV3 uses maintained TorchRL components

Versioned TorchRL 0.14 documentation describes actual V3 components: `RSSMPriorV3`, `RSSMPosteriorV3`, `RSSMRolloutV3`, `RSSMStateEstimatorV3`, `DreamerV3ModelLoss`, `DreamerV3ActorLoss`, `DreamerV3ValueLoss`, `DreamerV3Loss` and supporting update/normalization tools. Verify installed source/API behavior and pin it before model integration. This is stronger evidence than the user's generic world-model link, but still not installed/runtime proof.

Implement the library algorithm using the existing session and algorithm-owned update seam. Start with a small state-based Isaac task, then a bounded visual arm task so the backend proves both dynamics and sensor representation. Replay carries contiguous N/T streams, episode-first/reset masks, real final observations, optional burn-in and generation-checked posterior context. Use explicit paper-compatible versus author-recipe settings: categorical RSSM, KL/free-nats/unimix, vector/image reconstruction, reward/continuation heads, actor/value targets and slow-critic update behavior. Do not confuse current defaults with reproduced paper settings.

Lightning must have one effective owner for optimizer stepping, precision and checkpoint state. Prefer maintained compositional losses with Lightning-owned stepping; use `DreamerV3OptimizationStepper` only through a public integration that exposes its state and prevents duplicate optimizer/scaler/target updates. Verify with step counts and state round trips. The target updater, percentile/normalization state, policy RNG and fractional update ratio are checkpointed. When replay is omitted on resume, close old stream tails and explicitly refill/reset owed updates; restarted physics is not bitwise trajectory continuation.

Export encoder + observation-conditioned latent filter + actor + necessary action/normalization/state initialization. A Dreamer actor alone is not a deployable policy. Keep training decoder, critic, imagination losses and replay out unless a particular deployed planner genuinely needs them. TorchRL inference components can remain declared runtime dependencies; "inference-only" forbids training-only construction, not every library that also offers training. Preserve the old PPO tests that genuinely require no TorchRL at inference.

Compare predictive errors by horizon and task success/sample efficiency against the existing PPO reference with matched observation and physical-frame budgets. Separate real collected frames, replay samples and imagined transitions. A negative comparative result is valid evidence, not grounds to hide it. Add a bounded learned-dynamics planning reference and an independent hierarchical selector-plus-controller reference after the atomic policies are stable. The hierarchy selects at least two compatible existing skills, records transitions and task completion, rejects invalid targets and resets pending state at episode boundaries. It can reuse upstream planning tools and does not require a new foundation model. A dynamics-only demonstration cannot close the hierarchical gate; uncertainty and imagination never replace physical safety.

#### Bounded LeRobot world/action-model research assessments

Assess **VLA-JEPA** and **FastWAM** separately; neither replaces the mandatory TorchRL DreamerV3 workflow or gains advertised NexuML support from this assessment. For each record exact source/model/processor and component identities, licenses/trust, declared dependency conflicts, actual public API availability, expected inputs/actions/state, dataset/embodiment compatibility, training/inference composition and conservative resource requirements. Produce a feasible/blocked/insufficient-evidence verdict with source evidence, unknowns and a bounded proposed test. Metadata/source inspection may finish the assessment; it is not native or model-execution proof. Integration, large approved-weight experiments or unsupported dependency exceptions require separately accepted follow-up, not automatic implementation from a positive assessment.

VLA-JEPA combines a language/vision backbone and flow-matching action head with a V-JEPA2 predictive training signal; its world-model component is not required for the documented action-only inference path. Check latent prediction and action semantics separately, including cross-embodiment head reinitialization and camera ordering. It is not an imagination-trained Dreamer actor merely because the catalogue groups it with world models.

FastWAM uses video modeling during training, initializes from the documented Wan2.2 TI2V 5B components and predicts action chunks directly at inference. Identify which exact Wan/processor components its chosen checkpoint needs at inference rather than stripping everything named world model or exporting the full training system by default. Its documentation's H20 140-GB evaluation example is a reference configuration, not a demonstrated minimum. Do not promise it fits the local 16-GB GPU or substitute a small synthetic model as the real-model proof. Existing LIBERO checkpoint examples do not establish compatibility with the canonical Newton stack or an Isaac embodiment.

**Rejected:** replace V3 with older Dreamer losses, embed the author JAX runner as a second NexuML trainer, count generic future-state prediction as full Dreamer support, or enable optimized scan/Triton/capture modes before reference parity.

### D11. One release evaluator, with Arena and task-aware outcomes

Keep the current `evaluate_policy` scalar path and native loggers. Extend for explicit policy state and optional native GPU batches without counting partial episodes, leaking row state or resetting training. The success-only evaluator extension is shared with the MuJoCo change; preserve existing SO-101 paired error semantics. Task predicates decide whether transient/terminal/held success counts; do not universally reinterpret physical placement as distance or return.

Add an optional trainer-independent `ScenarioSpec.policy_evaluation` block reusing the existing episode/media option schema; do not overload the existing dataset `evaluation` block. It is authoritative when present. Existing RL packages without it continue reading `training.evaluation` and retaining their documented override path. For offline ACT/VLA, export the original `TrainingSpec`, complete policy, declared interaction/evaluation environment and this block; do not fabricate an RL algorithm or mutate the training mode to make replay pass.

Factor episode execution to consume the validated policy, environment definition and episode/media options independently of training. Preserve the existing scenario-based call through normalization. Episode seed is explicit, with the historical RL training-seed fallback only for old RL configurations. The ordinary CLI accepts complete offline-trained policy packages, permits `policy_evaluation` overrides and retains legacy `training.evaluation` overrides only on the legacy configuration path; ambiguous mixed paths fail clearly. Contract compatibility, trusted loading, device placement, logger finalization and media constraints remain unchanged. Require clean-package ACT and VLA CLI tests that block session/learner/optimizer/collector construction, exercise scalar and recorded replay and reject missing evaluation inputs; retain old RL CLI/override regressions.

Freeze suite/task/asset versions, sensor/processor/control contracts, episode/reset policy, seeds, held-out conditions, baseline identity, thresholds and sample sufficiency. Emit per-trial machine-readable results and a small pass/fail/insufficient-evidence comparison with uncertainty and worst-condition evidence. Include latency tails, interventions, dropped frames and constraint violations where measurable; statistical success is not safety certification.

Use maintained Arena/LeRobot EnvHub composition for one actual adapted policy and scene/embodiment/object variations. The published Python-3.11/Lab-2.3/Sim-5.1 recipe is incompatible with this target; select a maintained compatible release and prove it natively. Do not write an Arena replacement or port a legacy suite silently. Native Meta-World/LIBERO comparisons retain their own task protocols and compatibility gate, not reconstructed approximations advertised as the same benchmark.

For inspection media, reuse bounded GIF/MP4/both, TensorBoard animation, MLflow-only handling and final-frame-before-reset capture. For shadow/offline evaluation report action/state agreement, not unobservable counterfactual success. Keep task quality separate from bounded integration tests.

Assess **SARM** as a research-only video/text/state progress/reward model through existing data/layer/evaluation seams. Record source/model/CLIP processor identity, licenses, dependency/resource constraints and annotation modes/target provenance; distinguish temporal or annotation-derived progress from measured physical task completion. The feasibility report must consider failed, stalled, reversed and interrupted episodes, train-only temporal priors, leakage-aware splits and a held-out progress-quality/RA-BC comparison protocol. Uniform progress over episode duration cannot be assumed correct for failed demonstrations. Do not substitute learned progress for the frozen success predicate, promotion evidence or controller safety, and do not launch external VLM annotation, upload data or add a new reward service. A feasible assessment is not trained SARM or policy-improvement acceptance; further integration needs separately accepted work.

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

### D16. Treat the reference catalogue as an executable set of tests

reference-catalogue.md is the embodiment/method matrix for Isaac, MuJoCo and LeRobot; tasks.md is the single implementation queue. Every mandatory row has independent build/native stepping, actual NexuML learning, interruption/resume, complete package export, fresh-process inference and closed-loop/scalar/media testing. Quality acceptance separately requires a pinned native success predicate and held-out/multi-seed outcomes, not a smoke. Exact SO-101 vial placement remains the anchor: six relative controls, correct home-reset evaluation, physical grasp/lift/insert/release and no hidden object attachment. Include direct camera PPO **and** teacher-to-camera distillation with identical deployed observations, and compare matched ACT, Diffusion and small flow policies on the same valid source splits. Privileged state-policy results are not misrepresented as observation-matched camera results.

Keep the already listed OpenArm, G1, Go2, Crazyflie/PX4, TurtleBot3, UR10e/Rizon, UR5e and LumenPnP; require exact asset/task identities, not nearby substitutes. Include G1 Mimic locomanipulation, an OpenArm drawer/bimanual feasibility option, and bounded optional DexSuite/GR-1 references only after source/license/runtime verification. Full-program completion remains open for each mandatory asset and manual gate; optional exploratory examples are not quietly promoted to hard requirements.

### D17. Human corrections, reward models and simulation-first offline-to-online improvement

Augment the existing LeRobot v3 episode/data/capture ownership with intervention semantics, timestamped sensor/subtask/language/depth data, action-executed-versus-proposed differences, train/held-out split protection and dataset quality. Integrate DAgger-style correction capture (using upstream operators), one genuine SAC-based offline-to-online RLPD mix and simulated HIL-SERL with a trainable reward classifier. Optionally assess Robometer/TOPReward source/model/processor requirements before large downloads. Run matched old/new/failure suites and always distinguish learned reward/progress from measured task success. No autonomous hardware exploration, controller bypass or new teleoperation UI.

### D18. Modern action models and asynchronous inference

Use the existing compiled graph, processors and complete stateful-policy package for an actual small flow-matching/Diffusion Transformer reference alongside ACT and Diffusion. Preserve masked chunks, iterative sampling, RNG/seed, normalization, observation history, per-stream chunk/state buffers, reset and timeouts. Build an optional asynchronous policy-serving bridge only after ordinary local inference works; reject stale sensor frames and late chunks, with action fallback enforced by the external controller, not invented runtime safety certification. Select GR00T N1.7 versus legacy N1.5 explicitly from the maintained LeRobot release; SmolVLA and resource-gated pi0.5 remain required. Other new VLAs remain feasibility candidates until source/asset/compute checks pass.

### D19. Benchmark parity and cross-backend evidence

Retain Isaac Lab-Arena, exact Meta-World MT1 scoring and matching task/sensors/actions. Add one compatible LeRobot LIBERO/LIBERO-plus native suite and an optional RoboTwin 2.0 subset; do not rebuild evaluation engines. Pin suite/task/reset/object/seed/processor versions and record per-task success, failures, uncertainty, latency and resources. Native simulator policy success, offline shadow agreement and manually supervised hardware success are different evidence classes. Multi-run performance is not a substitute for required interoperability or license approval.

### D20. One campaign, many independently reviewable slices

Implement tasks 1–22 in dependency order; every slice lands focused tests, runnable user documentation, version/asset provenance and an actual native round trip before the next dependent slice claims readiness. A single agent/campaign can coordinate this programme, but it shall not duplicate the previous MuJoCo plan, rewrite existing session/compiler/framework architecture, wait for every P2 embodiment before releasing useful P0 milestones, or mark all required gates complete from CPU-only CI. Preserve tested checkpoint/report histories; follow ordinary approved cross-repository/container/hardware gates.

#### Verified milestone commits, without execution evidence

The user approved one focused **local commit after each verified implementation
milestone**, not a single end-of-program commit or a commit for every checkbox.
The milestone's applicable tests/build checks must pass; distinguish recorded
unrelated baseline failures without claiming an all-green repository. Include
its source, focused tests, durable user documentation, authored configuration,
dependency locks and relevant specification/task updates. Follow repository
commit-message rules. Planning edits alone do not retroactively stage prior
implementation work; check pending verified milestones at the next apply run.

**Evidence must not be committed.** Keep execution reports, raw command/test
logs, machine-specific run inventories, benchmark/result tables, generated
recordings/screenshots, checkpoints, weights and other run outputs out of the
index. This includes evidence content embedded in planning/docs, not just files
under reports/. Commit authored behavior/protocol documentation and task status,
not copied run evidence. Dependency locks and small authored deterministic test
fixtures are source inputs, not execution evidence. Existing evidence-only
coverage checkpoints must not be swept into a planning commit.

Stage only explicit milestone-owned files/hunks and inspect the staged paths
**and content** before committing; never use a blanket git add to sweep up
evidence, secrets or unrelated user changes. Retain evidence in the user's local
reports or an explicitly approved artifact location. References may identify
where retained evidence can be reviewed without embedding it in the commit.
"Publish evidence" elsewhere in this change means make it available for review
outside Git, not add it to source control or upload it automatically. Do not
delete/untrack previously committed evidence or rewrite historical commits.

If verification, staging isolation or commit creation fails, retain the work
and report the blocker; do not conceal the failure, include evidence as a
shortcut, amend history or force-push. This standing approval covers local
milestone commits only. Push, external review publication, merge and archive
remain separately requested operations.

## Risks / Trade-offs

- **[Program breadth]** -> Staged tasks and per-family proofs; do not turn the first arm milestone into a claim of full support or grow a generic framework preemptively.
- **[Conflicting optional simulator/native dependencies]** -> Resolve the primary compatible graph first, maintain an explicit named profile matrix for incompatible Meta-World/JAX cases, and test a reviewed opt-in execution boundary; no hidden environments/overrides.
- **[A relaxed bound hides a real API break]** -> D3 enumerates five exception names, pins exact selections, preserves security/ABI checks and checker discrepancies, and requires affected native/real-model proof and an upstream removal condition. Failed or unexecuted tests leave support incomplete.
- **[Upstream docs differ from installed/pinned code]** -> Immutable sources, explicit API inventory and native/numerical tests before adapters.
- **[Sequence or processor changes break old policies]** -> Stateless default behavior, old-package/config round trips and exact new sequence/reset parity.
- **[Two optimizer owners]** -> One Lightning-compatible stepping owner, serialized target/ratio/normalization state and exact update-count tests.
- **[ROS distro/native-library mismatch]** -> Separate validated processes/workspaces, standard interfaces and explicit unavailable-capability status.
- **[VRAM/disk/resource contention]** -> Staged approved budgets, bounded N/cameras/models, effective-limit checks and no unrelated-job interruption.
- **[Model/asset licensing or remote code]** -> Source/content manifests and operator-approved code/model trust; no copied arbitrary code or presumed redistribution rights.
- **[Passing smokes imply quality]** -> Independent multi-seed task experiments and pass/fail/insufficient-evidence release decisions.
- **[Incomplete hardware/distributed infrastructure]** -> Visible blocked manual/resource gates; full completion remains unavailable rather than manufacturing a narrower pass.

## Migration Plan

1. Freeze source/roadmap/baseline identities, reference-catalogue.md, historic ownership and each slice's source/contracts before code changes. Simulator-independent slices may proceed under D3 while the full primary inventory/resolution remains open; native-dependent implementation still requires its complete prerequisites.
2. Deliver the coordinated NexuVibe Python-3.12/cu130/system setup in its owning repository, obtain recreation timing and prove the canonical clean installation. Keep old images/checkpoints and rollback inputs.
3. Add minimal physical/temporal contract and inference-state extensions with old-workflow regression tests, then port and revalidate the original references and vial task on the canonical stack.
4. Validate classic MuJoCo/Meta-World and Playground native routes alongside canonical Isaac, then implement SO-101 data/ACT/Diffusion/flow/camera workflows, correction/reward learning and common evaluation; integrate pretrained VLA and DreamerV3 through existing seams. Keep each slice runnable and tested before the next family.
5. Add Arena, compatible LIBERO and optional RoboTwin, robustness, ROS/Gazebo/perception and exact embodiment references. Extend advanced physics, scaling and remote inspection only with their independent proofs.
6. Run full regression/distribution/install evidence and the separately approved real-arm/local-adaptation cycle. Reconcile all coverage rows and required gates before claiming full backend completion.

Rollback is per slice through normal reviewed version control and the retained consumer lock/image/config. Do not delete or recreate user environments, caches, datasets, checkpoints or staged work automatically. Keep base imports and previous approved inference artifacts usable; an incompatible target gets an explicit migration/rejection, not changed physics under the same identity. D20 authorizes verified local milestone commits excluding evidence; push, external review publication, merge and archive remain separately requested operations.

## New source constraints for integrated change

- https://github.com/isaac-sim/IsaacLabTutorial (maintained SO-101 state, camera, distillation and sim2real references; freeze actual release-specific task IDs)
- https://github.com/google-deepmind/mujoco_playground (GPU MJWarp/MJX, upstream JAX and supported RSL PPO baseline paths; no implied TorchRL interop)
- https://github.com/huggingface/lerobot/releases and https://huggingface.co/blog/lerobot-release-v060 (LeRobot v0.6+ model/processors, DAgger, reward API, depth and benchmarks; v0.6.1 import break must be audited)
- https://metaworld.farama.org/evaluation/evaluation/ (native protocol)
- The previous MuJoCo proposal is a source of requirements, not a second implementation plan.

## Source and evidence policy

Inspected 2026-10-19. These sources inform architecture, not completed compatibility:

- Local roadmap and the three existing RL/Isaac/MuJoCo changes; `reports/physical-ai/{phase-0-audit,phase-2-isaaclab}.md` for historical proof boundaries.
- [Isaac ecosystem](https://isaac-sim.github.io/IsaacLab/develop/source/isaaclab/ecosystem.html), [installation](https://isaac-sim.github.io/IsaacLab/develop/source/isaaclab/installation.html), [physics](https://isaac-sim.github.io/IsaacLab/develop/source/concepts/physics_backends.html) and [visualization](https://isaac-sim.github.io/IsaacLab/develop/source/concepts/visualization.html).
- [Pinned tutorial](https://github.com/isaac-sim/IsaacLabTutorial/blob/9a4286b4bc8a5720ac97f87e5e0140f7542c22d9/pyproject.toml) and its `uv.lock`; [candidate Lab metadata](https://github.com/isaac-sim/IsaacLab/blob/d061bf35a58273ef0cc8aceae76392166f9b7d15/pyproject.toml).
- [TorchRL Isaac guide](https://docs.pytorch.org/rl/main/reference/generated/knowledge_base/ISAACLAB.html), [VLA tutorial](https://docs.pytorch.org/rl/main/tutorials/vla.html), [versioned DreamerV3](https://docs.pytorch.org/rl/0.14/reference/dreamer_v3.html) and [objectives](https://docs.pytorch.org/rl/main/reference/objectives.html).
- User-selected LeRobot model scope: [PyPI catalogue](https://pypi.org/project/lerobot/), [versioned Diffusion reference](https://github.com/huggingface/lerobot/blob/v0.6.1/docs/source/policy_diffusion_README.md), [π0.5](https://huggingface.co/docs/lerobot/pi05), [indicative compute guide](https://huggingface.co/docs/lerobot/hardware_guide), [VLA-JEPA](https://huggingface.co/docs/lerobot/vla_jepa), [FastWAM](https://huggingface.co/docs/lerobot/fastwam) and [SARM](https://huggingface.co/docs/lerobot/sarm). Source documentation was inspected, not executed; freeze actual package/API/model revisions during implementation.
- [Humanoid imitation](https://isaac-sim.github.io/IsaacLab/develop/source/features/imitation-learning/humanoids_imitation.html), [SO-101 learning path](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/train-a-robot-arm/02-so101-robot.html), [gear insertion](https://isaac-sim.github.io/IsaacLab/develop/source/policy_deployment/02_gear_assembly/gear_assembly_policy.html), [RLinf VLA reference](https://isaac-sim.github.io/IsaacLab/develop/source/experimental-features/rlinf_vla_posttraining.html) and [multi-GPU reference](https://isaac-sim.github.io/IsaacLab/develop/source/features/multi_gpu.html).
- [Arena/LeRobot recipe](https://huggingface.co/docs/lerobot/en/envhub_isaaclab_arena), [Isaac ROS 5.0 requirements](https://nvidia-isaac-ros.github.io/v/release-5.0/getting_started/index.html), [release limitations](https://nvidia-isaac-ros.github.io/v/release-5.0/releases/index.html), [Noble buildfarm](https://nvidia-isaac-ros.github.io/v/release-5.0/getting_started/isaac_ros_buildfarm_cdn.html), [Sim ROS bridge](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/install_ros.html) and [streaming security/connectivity](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/manual_livestream_clients.html).
- Published [Meta-World 3.1.1](https://pypi.org/pypi/metaworld/3.1.1/json), [Newton 1.6.1](https://pypi.org/pypi/newton/1.6.1/json) and [MJWarp 3.12.0](https://pypi.org/pypi/mujoco-warp/3.12.0/json) metadata for the demonstrated dependency contradiction.

Context7 was unavailable due to quota; official documentation and published/pinned sources were used. No installer, resolver, simulator, GPU workload or hardware operation ran during proposal creation. Native package verification, final model/release selection and measured budgets are explicit first implementation gates, not unresolved architectural choices.
