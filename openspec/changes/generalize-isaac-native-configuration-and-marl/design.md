# Design

## Context

See [proposal.md](proposal.md) for scope and ownership. The approved source plan's decisions are retained except for two explicit refinements: unrelated provider changes do not automatically require recapturing demonstrations, and NexuFL source adjustments are independent follow-ups, never tasks in this change.

### Existing seams and concrete limitations

| Current seam | Reuse / required change |
| --- | --- |
| `library/src/nexuml_library/environments/isaaclab.py` | Hardcoded task/physics/revision literals and state-only profiles become native registered identities plus explicit contracts; retain reference-specific physical checks. |
| `_isaaclab_runtime.py`, `_isaaclab_rgb.py` | Reuse native launch/reset/render ownership; remove task-name workflow classification and unconditional noise/history changes. Distinguish spectator geometry from policy sensors. |
| `src/nexuml/core/factory.py`, serialization/discovery | Reuse importable typed-function capture and JSON-safe arguments; no executable factory resolution on static YAML restoration. |
| `core/components.py`, `core/types.py`, `interaction/runtime.py` | Add only the default job-scope seam and portable sensor/group/transition meaning required here. Keep native bindings in Library. |
| `core/compiler.py`, `core/policy.py`, action adapters | Compose existing pipelines and group views; current flattened observation routing needs explicit group/event axes, not another compiler/policy hierarchy. |
| `reinforcement/torchrl_env.py`, `collector.py` | Current flat single-agent specs and scalar transition signals need nested group specs and correct native reset conversion. |
| `training/reinforcement.py` | Keep `NexuRLLightningModule`, `algorithm.configure_optimizers()`/`update(...)`, policy sync and checkpoint owner. Root `rollout.numel()` remains physical frames; root reward reporting must not invent team rewards. |
| Library `reinforcement/algorithms/ppo.py` | Existing single-actor `ClipPPOLoss`/`GAE` path remains numerically unchanged. New IPPO/MAPPO definitions use native multi-agent classes. |
| `evaluation/policy.py`, CLI | Extend actual grouped outcomes and selected evaluation environment lifecycle; retain explicit rejection of unsupported in-fit/conflicting evaluation. |
| `training/local_fit.py`, imitation `so101.py` | Preserve validated local SDK; separate the receipt's semantic task projection from implementation-file hashes currently compared to the running provider. |
| `core/export.py` and packaging tests | Keep complete-policy packaging and general train-package formats distinct; actor inference does not reconstruct the simulator or learner. |

### Qualification baseline, not completion evidence

The approved plan records pinned API/source qualification for Lab `3.0.0` at `70ab24fec3eabf51222817db0cdb5a4e7133bd7e`, Sim `6.2.0.0`, TorchRL `0.14.0`, TensorDict `0.14.2`, PyTorch `2.11.0+cu130`, Viser `1.1.1`. Preserve this worktree `.venv`; the broader roadmap's other environment lanes do not authorize provisioning here. API probes are not native/browser acceptance. Graphics/Vulkan/NVENC passthrough is a known possible blocker; recheck actual prerequisites without changing infrastructure.

## Goals / Non-Goals

**Goals:** implementation-ready contracts for one composition/lifecycle; exact native numerical integration; mandatory private/read-only Viser; portable physical meaning; preserved SO-101 and general workflows. All A–D stay in this pass with separately reported implementation, unit/API, native/browser and blocked states.

**Non-Goals:** a universal robotics framework, remote code loader, new provenance service, environment manager, vendor fork, separate numerical trainer, federation/transport, distributed simulator launch or legacy Physical AI compatibility layer. Non-goals of this focused change do not delete broader roadmap obligations.

## Decisions

### D1. Native configuration stays a typed library component

`IsaacLabEnvironment` retains task identity and explicit contract/bindings. Add `IsaacTaskConfigSpec`, `isaac_task_config(fn, **json_args)` and optional `ViserSpec` using existing factory conventions. The importable function receives the resolved native config as a runtime argument; it may mutate it and return `None`, or return its replacement. Reject lambdas, closures, `__main__`, invalid signatures and nonportable arguments. Native objects never enter YAML.

Runtime order:

1. Load trusted installed task registrations using Isaac's supported mechanism; resolve actual task and native preset IDs.
2. Set NexuML-owned count, device, seed and required pre-reset observations. Count comes only from `training.collector.num_envs_per_collector` for training, not a competing environment option.
3. Invoke customization in the owned native job, resolve returned preset wrappers with upstream resolution, and verify requested selections.
4. Validate protected settings, actual workflow/type/task identity, spaces/bindings, sensors, physical controls, agents, rates and source/config identity before stepping. Native imports follow the resolved launch/import contract; no user factory allocates an app/environment.
5. Construct and validate native resources under that same job owner. Unknown presets or incompatible task/backend/sensor combinations fail explicitly; no silent fallback.

Native solver/material/reset/reward/randomization/curriculum/sensor options stay in the typed native function, not generic string-path dictionaries or another Hydra tree. Preserve current reference noise/history explicitly; do not disable them generically. Replay does not silently invoke `play_mode()` to change the saved distribution. Deliberately different evaluation semantics use an explicit evaluation definition.

`ViserSpec` covers purpose activation (configured defaults to training only), loopback address/port, eye/look-at, visible environment selection, markers and supported plots. Absent means no viewer. Replay requires separate explicit activation and cannot replace actors or task/control semantics.

**Trust:** static describe/discovery/YAML restoration does not resolve or execute factories, import native runtime or allocate CUDA. Trusted local execution may import approved operator-installed functions/packages; portability is not a sandbox. Remote SDK fields cannot choose recipe/model/factory/processor classes, archives or local dataset paths. Reuse existing trust checks.

**Rejected:** automatic native contract inference, arbitrary native-object serialization, simulator registry and Isaac-specific compiler. They duplicate upstream mechanisms or lose physical meaning.

### D2. Whole-job ownership uses the existing environment boundary

Add a default no-op `EnvironmentDefinition.job_scope()`; Isaac supplies a lazy process-owned scope. Normal `nexuml train` wraps the effective loaded/resumed scenario, session, final checkpoint/export/provenance and logger cleanup. `nexuml evaluate` scopes the selected evaluation definition, not always the training definition. Core CLI does not import Isaac directly.

Python retains `with environment.job_scope(): ...` around the entire job, including outputs after `session.run()`. Same-job nested scopes join the owner without a second app or premature outer shutdown; conflicting active runtimes reject. Resolve Kit/process-terminating ownership from native launcher/runtime state, not a physics-name comparison: sensors can require Kit independently.

Keep one GPU and one synchronous direct collector. Native handles remain worker-local. Fresh-process replay uses one physical environment with its declared agents. Completed-budget resume, discovery and actor inference allocate no simulator. Preserve failures and signal-derived exit status; finalize outputs/loggers before native fast-close. Browser disconnect is not job termination. Scheduled in-fit evaluation stays rejected unless an already-supported isolated path is proven; this change does not add a second simulation context.

**Rejected:** manual CLI launch prefixes, another session hierarchy, unconditional Kit ownership and distributed launch inferred from MARL support.

### D3. One portable sensor/control meaning, explicit processing ownership

Extend existing tensor/physical contracts only as needed for actor-visible and training-only fields. Library bindings map typed native source path components to declared tensor fields/group/agent identities; they are not general config overrides. Native workflow classification uses validated native types/contracts, not special-case task strings.

State, RGB, depth, segmentation and mixed inputs retain layout, dtype, units, normalization parameters, depth representation and invalid-value/sanitization mask, segmentation vocabulary/label identity, processor version/assets, frame history/order/reset initialization, sensor/action rates and timing/latency convention. Finite policy tensors are required after declared processing; no hidden coercion of NaN/Inf or unknown segmentation IDs.

For every operation, persist one owner: portable package processor/layer, or external runtime providing an identically defined tensor interface. External stacking explicitly owns buffering/reset; package-owned state uses the current resettable policy interface. No second preprocessing framework and no assumption that native MDP preprocessing accompanies an actor outside Isaac. Test both supported ownership modes on representative sequences.

Continuous actions preserve event dimensions, bounds and physical scale/offset/joint order/decoding. Scalar categorical actions are integer class indices with an explicit class-to-control mapping. If a new categorical reference is necessary, register it under a distinct identity rather than relabel continuous Cartpole. MultiDiscrete/MultiBinary/hybrid actions within one actor are not promised.

True pre-reset observations include sensor freshness/rendering and history. Validate same-step autoreset once, reuse cached reset observations without another reset, and never use a next-episode frame as the terminal frame. Spectator cameras/ground truth remain separate from actor sensors.

**Rejected:** shape-only compatibility, silent `play_mode` transformations and built-in preprocessing guessed by modality.

### D4. Canonical fixed-group TensorDict boundary

`N` is environments, `T` control transitions, `A_g` fixed ordered compatible agents. Root live batch is `[N]` (including `N=1`), root rollout `[N,T]` with explicit time dimension; group children are `[N,A_g]` / `[N,T,A_g]`. Do not change unrelated single-agent scalar-batch conventions.

| Rollout key | Tensor shape / meaning |
| --- | --- |
| `(g, observation_field)` | `[N,T,A_g,*S_field]`, actor-visible leaves |
| `(g, action)` continuous / categorical | `[N,T,A_g,D_g]` / int64 `[N,T,A_g]` |
| `(g, action_log_prob)` | `[N,T,A_g]`, reduce action events only, never agents |
| `(g, state_value/advantage/value_target)` | `[N,T,A_g,1]` |
| `("next",g,reward)` | `[N,T,A_g,1]`, actual agent rewards |
| `("next",g,terminated/truncated/done)` | bool `[N,T,A_g,1]`, explicit distinct facts |
| `(g, valid)` | bool `[N,T,A_g,1]`, collector padding plus learning validity |
| Root `("next",reward)` | optional `[N,T,1]`, only genuine native team-shared reward |
| Root `("next",terminated/truncated/done)` | bool `[N,T,1]`, actual environment boundary |
| Declared root/group critic-only fields | explicit shapes, training only |

Different roles stay in separate groups with distinct observation widths/action spaces/distributions. Group/agent ordering is persisted, not dictionary-order inference. Sharing requires identical compatible semantics as well as shapes. Use existing compiled per-group/actor graphs and small nested-key views, not a second policy hierarchy. Allow compatible shared actors and truly independent actor modules.

The small native transition adapter preserves raw agent endings and native environment reset separately, deriving effective GAE boundaries from both according to the task. Do not invent `any`/`all` reset rules. A terminal transition is valid; subsequent inactive suffixes are invalid until native reset. Continuing agents retain their trajectories. Inactive learning samples do not erase physically valid centralized world state: a finished body may still affect another agent's value.

Use real pre-reset values for time-limit bootstrapping. `compute_final_obs` alone does not prove a separate privileged `state()` was captured. A critic derived from final agent observations is supported; independent privileged-state bindings require a qualified public pre-reset source, otherwise reject them. Never bootstrap from reset state.

Run GAE with `[N,T]` before flattening environment/time to `[M]`; group children then remain `[M,A_g]`. Frame budgets/logging/resume count `N*T`, never agent samples or optimizer steps. Route genuine team signals through native broadcasting/keys; reporting means/sums are explicitly labeled and never become fabricated learning reward or success.

**Rejected:** heterogeneous padding, agent-as-environment accounting, dynamic membership and using unchecked `info` for learner semantics.

### D5. Native TorchRL 0.14 supplies numerical algorithms

Preserve the qualified public API inventory; implementation must not rediscover a different algorithm design:

| Public API | Qualified interface / limitation |
| --- | --- |
| `torchrl.objectives.multiagent.IPPOLoss`, `MAPPOLoss` | `actor_network`, `critic_network`, `value_norm=None`, `entropy_coeff`, `normalize_advantage`, `normalize_advantage_exclude_dims=(-2,)`, plus `ClipPPOLoss` kwargs. IPPO's independence is enforced by critic inputs, not its class name. |
| Loss key/estimator setup | `set_keys(...)`; `make_value_estimator(gamma=..., lmbda=..., time_dim=1, shifted=False, auto_reset_env=False, average_gae=False)`. Default `ValueEstimators.MAGAE`; assert actual estimator is native `MultiAgentGAE`. |
| `torchrl.objectives.value.MultiAgentGAE` | `agent_dim=-2`, other GAE kwargs. Returns `[...,T,A,1]`; broadcasts signals missing exactly one agent dimension, e.g. `[N,T,1]`. Validate layouts; it is not heterogeneous orchestration. |
| `torchrl.modules.MultiAgentMLP` | `(n_agent_inputs,n_agent_outputs,n_agents,*,centralized,share_params,use_td_params=True,...)`; `[...,A,F] → [...,A,O]`. Common width within a group. Centralizes within that group only; centralized/shared configuration repeats value output across its agents. |
| `torchrl.modules.models.cross_group_critic.CrossGroupCritic` | `(group_map,*,d_model=64,trunk_depth=2,trunk_cells=256,share_params=False,detach_groups=None,device=None,...)`; ordered `CrossCriticGroupSpec(obs_dim,n_agents,obs_key,value_key)` maps vector groups of differing widths to `[...,A_g,1]`. Native group encoders/shared trunk; `share_params` controls value heads, not actors. Images need existing encoder features first. |
| Optional `torchrl.modules.PopArtValueNorm` | `shape=1,beta=0.99999,epsilon=1e-5,device=None`. Native loss updates statistics in critic-loss evaluation; the name does not prove validity-aware statistics or output-preserving critic-weight updates. |

Pinned source anchors beneath installed `torchrl/`: `objectives/multiagent/mappo.py:28–241`, `objectives/value/advantages.py:2482–2594`, `modules/models/multiagent.py:348–519`, `modules/models/cross_group_critic.py:25–245`, `modules/value_norm.py:120–197`, `objectives/common.py:286–377` and PPO normalization handling. The source plan's CPU qualification constrains the integration; no raw probe output belongs in this planning commit.

Register thin Library `IPPO`/`MAPPO` `RLAlgorithmDefinition`s. Compile actor/critic graphs through current NexuML components, adapt keys to native `ProbabilisticActor`/TensorDict modules, and use one native loss/estimator per group/policy binding. Start with `functional=False` to preserve actual parameter identities. Reuse current Lightning update/minibatch/backward/synchronization seams; a small group-iteration adapter is justified, a copied PPO loop is not.

- **IPPO:** critic and actor for an agent depend on its own local declared inputs. Shared local weights do not centralize observations.
- **MAPPO:** actors remain local. Homogeneous centralized critics use suitable native `MultiAgentMLP` or an existing compiled graph reading declared state. Heterogeneous critics use suitable `CrossGroupCritic` or equivalent existing graph outputs, never a new custom centralized-network implementation. Do not force repeated shared values when roles/rewards need distinct heads.
- Native losses/GAE own clipping, entropy, critic regression and advantage recursion. Ordinary single-agent PPO and its defaults stay unchanged. No task convergence or algorithm superiority is required.

#### Masking and optimizer semantics

Native `loss.loss_mask_key=(g,"valid")` masks reduction but does not automatically exclude inactive values from advantage-normalization statistics. Native PPO standardization recognizes `shifted_valid`; this key has estimator-specific trajectory-shift meaning and is not an agent mask. The qualified counterexample changes a true-terminal advantage from `1.0` to about `2.8` if it is misused.

New MARL defaults are `normalize_advantage=False`, `average_gae=False`, `value_norm=None`. Preserve genuine done/terminated/truncated boundaries and finite inactive-suffix views for native GAE. Invalid unused rewards/targets must not change valid targets or gradients. This does not require invariance to a real change in valid critic-visible physical state. Any optional normalization requires separate public-API proof of validity-aware statistics, update frequency and checkpoint continuation for the exact layout; reject unqualified combinations, do not add a custom normalizer or private numerical workaround.

Own each parameter exactly once. Aggregate native group losses, backpropagate before stepping shared parameters, step a shared critic/trunk once, and clear stale gradients to `None`. Skip inactive independent actor optimizer steps, including after warm-up momentum/weight decay. Packed `share_params=False` tensors with one Adam do not guarantee per-agent slice isolation; prefer existing separate compiled actor modules/optimizers where needed. Shared actors legitimately move from other compatible agents' valid samples.

Checkpoint actual actor/critic identities, unique optimizer slots, group order, counters and qualified normalizer buffers. Resume must restore sharing and match deterministic continuation under controlled data/seeds; changing count/semantics without declared support rejects before native action.

### D6. Front-loaded Viser feasibility; full delivery in C

First task `0.1` qualifies private/read-only construction on the exact pin before substantial implementation. Audit native constructor, sidebar and all supported callback/request/mutation handoffs. Prefer a supported native read-only/server-injection hook; otherwise qualify a narrow instance-local adapter in Library reusing native scene/geometry methods.

Known constraints: `ViserVisualizerCfg` lacks a read-only flag and defaults to all interfaces; `_setup_isaaclab_sidebar` registers pause/reset callbacks; `NewtonViewerViser.__init__` temporarily reassigns global `viser.ViserServer`; underlying Newton `ViewerViser` has no public host/server injection argument. Its `apply_forces` no-op is not proof of interface immutability. Source locations are installed `isaaclab_visualizers/viser/{viser_visualizer_cfg,viser_visualizer}.py` and `newton/_src/viewer/viewer_viser.py`.

The adapter must bypass, not invoke or reproduce, global reassignment. Private binding is established before listening. Vendor edits, process-global monkeypatches, another renderer or a broad proxy framework are prohibited. Qualification checks source/versions, construction, actual bind and request denial with bounded tests; all proof execution occurs during implementation, not this planning task.

Register only camera/display/selected-environment/marker/supported-plot capabilities. Viewer-facing objects do not receive mutable physics/task/actor/optimizer/control/timing/budget handles. Omit mutating native callbacks and deny mutation again at native handoff methods: hidden widgets, stale IDs, forged HTTP/WebSocket messages and direct callback invocations cannot reach simulation writes. Read-only covers supported viewer interfaces, not arbitrary privileged host Python.

Camera movement changes browser or isolated spectator camera state, never policy sensor pose/timing, actor history, USD/physics transforms or RNG. Display pause is not simulation pause. Compare physical/control trajectories at equivalent logical steps; wall-clock throughput need not match. Environment selection never changes `N` or budgets. Browser absence/disconnect/reconnect never stops/resets/pauses learning.

Default localhost with no public share link; use SSH/VS Code TCP tunnel. Verify real listener address and exact requested port. Occupied ports or silent port reassignment, startup errors and lost qualification fail the viewer-requested job and unwind partial resources; never silently run headless as if viewing succeeded. No-viewer headless jobs remain independent. Success/failure/cancellation closes listeners/threads before job termination. If feasibility cannot meet all restrictions, stop dependent work, record the precise blocker and leave C/overall acceptance open; Viser stays required in this pass.

**Rejected:** visual-only control removal, launching publicly then rebinding, native global patch “temporarily,” and replacing Viser with another scene renderer.

### D7. Complete current-schema policy artifacts, not simulator reconstruction

Reuse `CompiledPolicy`, existing processors/action adapters and complete inference packaging. Persist actor topology/weights/sharing, processing ownership, physical/temporal contracts and integrity metadata. Simulation replay separately requires the matching trusted task/config package; inference loads without importing that package, Isaac, TorchRL or Viser and without critic/privileged inputs. Configuration metadata remains inert data during load.

Prefer existing portable NexuML actor modules and native multi-agent networks for training-only critics. Pinned `MultiAgentNetBase` warns unshared `torch.export` needs PyTorch ≥2.12; current stack is 2.11. Do not upgrade or confuse this with existing `torch.package` packaging. `get_stateful_net()` does not prove TorchRL-free inference. If a native training actor is selected, explicitly qualify topology/weight transfer to ordinary portable actor modules with sharing, distribution, processing and decoded-action parity; otherwise use existing portable actors directly.

Training checkpoints keep critics/optimizers locally; inference-only payloads exclude them. Preserve unrelated general train-export entries, sidecars, dependency manifests and continued-training semantics. No general export-format bump is justified by the native schema change.

### D8. Minimal SO-101 semantic/provenance split

Keep the existing SO-101 receipt/validation path, not a new provenance subsystem. New active receipts have one current version with two explicit projections:

- **Task semantics:** task/robot/asset/calibration identity; physics and behavior-relevant reset/reward/success/goal parameters; joint order/units/limits; action meaning/scale/offset/control period; observation/processor/sensor/history meanings; dataset schema, episode/time/split rules and expert behavior. Exclude viewer settings, file locations and implementation-only code identity. Relevant randomness/task distributions remain semantic.
- **Implementation provenance:** capture Git/source digests, package versions, native release/revision, resolved implementation/config identity and capture environment. Capture provenance is verified against its retained manifest/content identities, not overwritten with current-runtime hashes. A hash proves integrity only relative to a trusted record, not authentic native execution by itself.

Project current `IsaacLabEnvironment` into those semantics rather than serializing spectator options into compatibility. Preserve asset-layer hashes, physical validation, fixed splits, timing, complete transitions, native/expert quality checks and dataset content checksum. `_task_identity()`/`_source_hashes()` currently couple these concerns; adjust only this existing contract and its authored fixtures/tests/capture documentation.

| Change | Admission and required evidence |
| --- | --- |
| Equal width but changed units/order/asset/control/sensor/processor/task/dataset meaning | Strict compatibility rejection before fit/replay/action. Explicit new task/data contract needs new matching capture and closed-loop evaluation; never relabel old samples. |
| Missing/tampered capture provenance, asset or dataset checksum/splits | Reject integrity failure; no implementation qualification waives it. |
| Viewer initialization/refactor with unchanged task/data semantics | Preserve capture provenance and original bytes. Requalify affected runtime/SDK/native behavior; no automatic demonstration regeneration. |
| Simulator/dynamics/capture/processor change potentially affecting behavior | Require source-delta and bounded native qualification. If semantic equivalence is not established, fail closed for reuse; collect new matching demonstrations and evaluation evidence. |
| Implementation-only change proven equivalent | Record new local qualification tied to original dataset checksum, capture identity, semantic digest, current revision and named checks; prior native evidence remains attributed to its original revision, never recast as current proof. |

Do not edit historical datasets/reports or add legacy readers, converters, arbitrary hash allowlists or a generalized equivalence registry. Directly update authored active schema/fixtures since this is unmerged. Unsupported old receipt schemas stay untouched and unsupported; ordinary current-schema validation suffices. The no-recapture rule applies to semantically unchanged current-schema data after implementation changes, not an obligation to add a reader for every historical receipt. Preserve the working SO-101 proof through matching current-schema native regression/evaluation, making new data only when semantics/integrity actually require it.

### D9. Preserve SDK behavior and identify the real NexuFL delta

Preserve `local_fit`/`local_fit_manifest`, validated trusted requests, CPU/reset defaults, explicit approved `cuda:N`, deterministic bounded fits, named tensor roles/scopes, finite update checks, physical/schema hashes and private data/optimizer-state boundaries. Retain SO-101 float32 observation `[15]`, action `[5]`, ordered five arm joints, absolute `0.5*action` radians without home offsets, fixed gripper `0.2555162024919699`, 120 Hz physics / 30 Hz control and identity-v1 processor. No MARL/Viser dependency for offline imitation.

Actual read-only-inspected consumer, not `/workspaces/NexuFL`'s unrelated checkout:

- Branch `feature/nex-235-so101-federated-imitation`, revision `38cf25c04434899d788521be6bfa960724c2aaae`.
- Worktree `/home/nexuadmin/.local/share/opencode/worktree/dea2a4ce9f097800955a98d578f9909addbffa87/nex-235-so101-federation`.
- `clients/python/src/nexufl_client/so101.py` pins NexuML `6516822a23437c1686d5da573f0e07fb6688b392` (not this branch's `7e60bad` SDK), plus SHA256 for `nexuml.training.local_fit` and `nexuml_library.scenarios.imitation.so101`; it verifies those files before import and fit.
- SDK source digests there: `36e1d119bf243327c8849178d3cfdb7d19ff35cea75f50638d7c84e15ffa4e71` and `3f60a724f72b54fd41275a159b1d0100fdc55ffa67230b7f540bcdbfd3ff5741`, respectively. These are consumer trust constraints, not permission to accept arbitrary changed files.
- Consumer builds `architecture_id="so101/"+manifest["recipe_hash"]`, checks named tensor dtype/shape/role, full safetensors/reset state, exact counts/approved metrics and local binding/checksum. Its worker is intentionally CPU-only; NexuML's approved CUDA remains a separate producer gate.
- Matching checks: `tests/federation/test_so101_program.py`, `tests/robotics/installed_so101_probe.py`, `tests/robotics/test_so101_handoff_runner.py`, `scripts/robotics/verify_so101_handoff.py` and `docs/so101-federation.md`. Tests include authored synthetic receipt construction; synthetic data is not native/four-site proof.

Expected delta: new serialized native/semantic contract may change `physical_schema_hash` and `recipe_hash` without changing names/physical numerics; receipt fixture construction changes; modified SDK/scenario files invalidate exact source pins even before numerical checks. Preserve request/result schema and numeric tensor names unless a concrete need proves otherwise. Compute exact new digests/hashes from implementation, do not invent them in planning.

NexuML tasks update only its own active fixture/docs/tests and emit an exact old/new contract delta. Read-only consumer checks must never patch pins to manufacture a pass. A pin/schema mismatch is a reported failed compatibility gate requiring an **independent NexuFL review/update/requalification**, not NexuFL edits here or relaxed validation. Re-resolve consumer revision before running checks; no claim it already accepts current SDK. No federation or transport enters NexuML.

### D10. Focused authority, milestone gates and bounded execution

The six capability specs own the focused behavior; [tasks.md](tasks.md) owns implementation order, [verification-plan.md](verification-plan.md) owns gate protocols and coverage. Umbrella proposal/design/tasks receive narrow ownership links only. Its generic compatibility/debug-view allowances cannot authorize legacy Physical AI machinery or mutating Viser here. Keep its broader requirements, historical evidence and completion checkboxes unchanged; no archive/whole-program completion.

Order is feasibility → A → B → C → D, with small verified milestone commits, never one giant final commit. Stop at an unresolved early viability blocker rather than silently deleting C. Each milestone reports implementation present, unit/API passed, real native/browser acceptance passed, blocked/unverified separately. A checkbox cannot close from shapes, mocks, generated files or skipped opt-in tests. An external consumer mismatch remains explicitly blocked even if producer-only work passes.

Native acceptance keeps Cartpole manager/direct, Franka, SO-101, one genuinely registered custom task, RGB/depth/segmentation/mixed/categorical cases, `Isaac-Pendulum-MARL-Direct` (qualified source widths 4/3) and `Isaac-Shadow-Handover-Direct` (157/157 observations, 20/20 actions, subject to assets/backend proof). Qualify exact supported combinations; no Cartesian-product claim or substitution of upstream trainer success for NexuML updates.

Use bounded small batches/cameras/budgets and at least **40 GiB free before and throughout native work**. No environment replacement, `/env`, driver/container changes, new dependencies/pretrained downloads, public ports, paid/cloud compute, physical actuation or interference with other jobs/worktrees. Missing prerequisites are blockers, not authority for provisioning. Keep raw execution evidence outside Git; preserve existing user data/reports. This planning commit contains authored protocols and API constraints only.

## Risks / Trade-offs

- **Pinned Viser seam cannot be safe** → early task `0.1`; fail closed and leave C/overall incomplete, not a renderer fork or global patch.
- **Sensor or graphics prerequisite absent** → separate native sensor and browser gates, exact renderer/preset prerequisites; no tensor-only completion or infrastructure workaround.
- **Final privileged state unavailable** → critic-from-final-observations path or explicit unsupported binding; never reset-state bootstrap.
- **Masked numerics / optimizer drift** → native nonnormalizing defaults, partial-terminal oracles, real gradients/steps and warmed momentum/decay isolation tests.
- **Export restriction on PyTorch 2.11** → portable NexuML actors and clean-import/parity tests; no 2.12 upgrade or unsupported graph-export assumption.
- **Source pins mistaken for semantic incompatibility** → two receipt projections and explicit requalification, retaining integrity. Unknown equivalence fails closed, not universal recapture or blanket acceptance.
- **Consumer source-pin drift** → exact independent NexuFL delta and blocked compatibility status; no sibling edits here.

## Migration Plan

Directly replace unmerged native definitions/examples/fixtures/scenarios with one current version; remove obsolete revision/config branches. No Physical AI checkpoints need migration: add no old-reader/converter/rejection subsystem. Continue validating newly produced current-schema checkpoints/resume/export.

Preserve general supervised/export interfaces with existing regression/wheel tests. Rollback uses ordinary reviewed commits and retained original data/artifacts, never automatic environment resets or data rewriting. Keep milestone implementation and native acceptance status distinct. OpenSpec readiness authorizes no installs, simulations, server launch, sibling changes, PR creation or merge during this planning task.
