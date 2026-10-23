# Remote Physical AI acceptance matrix

This is a planned inventory, not a result ledger. Reconcile it with the candidate's discovered scenarios before execution. All application, native, browser and SDK executions use this branch's SDK Jobs transport ported into `dev`'s registered `ray-jobs-sdk` backend, only on `pool_gpu_workers`. Never use DGX or a separate CPU-worker pool; CPU control workloads still execute inside default one-GPU jobs. Local work is CPU-only control-plane/static validation. Each attempt records candidate/source/wheel identity, actual job and allocation, commands, budgets, exits and durable artifact locations.

## Gate labels

- **C — CLI:** real resolve/train/partial-resume/checkpoint-export/fresh-evaluate, changed parameters, finite metrics and correct physical-frame counters. Dedicated top-level remote dispatch cases prove resume/export/evaluate do not read their worker inputs locally.
- **V — Visual:** actual viewer/browser interaction and downloaded media playback for the claimed worker/backend/renderer combination. Sensor oracles and spectator viewing remain separate.
- **L — Learning:** existing frozen showcase outcome protocol, not finite-update smoke.
- **R — Recovery:** cancellation/deadline/startup/delivery failure, truthful status, current checkpoint recovery and owned-resource cleanup.
- **D — Distribution:** candidate installed-wheel/source exclusion, actor-only import isolation and preserved ordinary supervised behavior.

Every gate records pass, fail, blocked or unverified. A case's successful submission or API fixture result does not imply C. A skipped native job does not imply V. Candidate-affecting fixes invalidate affected gates, not historical records.

## Recipe inventory

| Case | Existing recipes/workflow | Required coverage |
| --- | --- | --- |
| G1 | `cartpole-ppo`, `pendulum-ppo` | C for each; start/progress/final evaluation, selected recording formats, logger combinations, R/D representatives |
| G2 | `visual-cartpole-ppo` | C; actual policy image input, shape/dtype and raw/processed visual checks; no vision-convergence claim |
| M1 | `mujoco-humanoid-ppo`, `mujoco-reacher-ppo` | C for each on qualified native MuJoCo; native human viewer/private remote access on Reacher plus separate RGB-recording replay; do not infer Humanoid GUI acceptance from it |
| I1 | `isaac-cartpole-ppo`, `isaac-cartpole-learning-ppo`, `isaac-franka-reach-ppo`, `isaac-so101-joint-reach-ppo`, `isaac-custom-cartpole-ppo` | C for each supported declared backend; current native final/reset checks and R; Cartpole live spectator and learning showcase; no universal task/backend product |
| I2 | `isaac-rgb-cartpole-ppo`, `isaac-depth-cartpole-ppo`, `isaac-segmentation-cartpole-ppo`, `isaac-mixed-cartpole-ppo`, `isaac-categorical-cartpole-ppo` | C for each; actual sensor/action meanings, terminal scenes and history; existing package/external processing-owner coverage, D; qualified mixed Cartpole viewer as representative spectator |
| A1 | `isaac-pendulum-ippo`, `isaac-pendulum-mappo`, `isaac-handover-ippo`, `isaac-handover-mappo` | C for all four; actual actor/critic changes, unique optimizer/sharing/counters and clean decentralized replay; bounded two-env/64-frame native fixture values labelled smoke-only, not learning or native grouped-sensor support |
| R1 | `ros2-gazebo-joint-reach-ppo` | Required real CLI/native-GUI journey with prepared world, controllers, reset hook and clock; dedicated simulation-only domain and private display route; blocked without these, no fake-transport substitute |
| C1 | `carla-lane-following-ppo` | Inventoried external live-integration boundary; existing contract/fake-server checks are not live acceptance. Keep experimental/unqualified claim unless a separately approved real server journey is added |
| S1 | Existing SO-101 demonstration collection, `scripts/so101_local_fit.py`, `local_fit`/manifest and native exported-policy replay | Real remote script/SDK workflow, original episode/physical validation, repeated fits, L/R/D. This is explicitly a script/SDK route rather than every operation being a `nexuml` subcommand |
| B1 | Existing synthetic supervised scenario and exported-model test-split evaluation | Real remote CLI control plus D; no native/model/recording dependency introduced into ordinary usage |

The registered RL inventory above contains 21 recipes: 19 core Gymnasium/MuJoCo/Isaac recipes, the required ROS/Gazebo journey, and the explicit CARLA external boundary. Any scope change needs an explicit maintainer decision and consistent documentation; do not drop a discovered recipe or convert a required row to optional during execution.

## Bounded execution and dependency order

1. Back up pre-rebase history and exact user/index state, perform the authorized rebase onto `dev`, and migrate to its registered execution structure while preserving SDK behavior and distinct KubeRay semantics. Freeze one candidate and publish/build immutable inputs to an approved remote-accessible location only under the relevant authorization. No force-push or implicit private-data/checkpoint upload.
2. Preflight one remote worker/profile: source origins, runtime/native/display/browser/codec versions, real requested devices, assets/licences, durable paths and budgets.
3. Use at most four in-flight jobs, further limited by qualified capacity, all on `pool_gpu_workers` with the default whole-GPU reservation. Pending/setup consumes a slot. Qualify the first worker/profile serially, then demonstrate genuine overlap for at least two independent remote GPU jobs on distinct full allocations; insufficient capacity blocks the parallel-native proof, not permission to use DGX, another pool or the local GPU.
4. Keep each native case's training, interrupted continuation, export, evaluation and replay stages sequential. No second unsupported Isaac runtime during fit.
5. Use separate per-attempt mutable environments, databases, ports and ROS/Gazebo identities. Record actual allocations rather than assume GPU reservations isolate them.
6. Retain logs, receipts, checkpoints, complete policies and selected bounded media at explicit durable destinations. Validate retrieval after processes finish and one representative worker is cleaned up.

## Representative visual/logging coverage

- G1 covers TensorBoard-only/MLflow-only/both/disabled logging and GIF/MP4/both without multiplying every native recipe by those settings. Use an initial/intermediate/final sequence with changing actual preview frames.
- A qualified I2 camera case repeats native media/logging retrieval and terminal-scene checks; declared renderer combinations remain explicit. Viser software browser graphics do not qualify server RTX.
- Open TensorBoard Images and actual scalars in a real remote browser, select frame-labelled stages and reopen retained logs. A GIF beside event files is not an event-summary proof; MP4 remains a downloadable artifact.
- Open actual MLflow runs, inspect status/metrics/config and retrieve selected media/JSON. Verify playback, shared payload identities for two loggers and no unconfigured writer. Inline MP4 capability is not assumed.
- Exercise the actual Isaac spectator during CLI train and separate replay: camera/display/selection/markers/plots/reconnect plus mutation denial and cleanup. Retain browser traces/screenshots and visual spot-review, not pixel-exact screenshots as an oracle.
- For M1, qualify Reacher's live upstream native viewer on a prepared remote display and its actual private access path. Human viewing and `rgb_array` recording use separate compatible environment definitions; do not change policy inputs.
- For R1, qualify the actual native Gazebo GUI with world/robot/clock/namespace matched to the training process. Controller motion and real policy updates need independent evidence. A virtual display or GUI startup alone is insufficient.
- Native MuJoCo/Gazebo GUI interventions are debug-only and excluded from quantitative trials. Do not advertise them as a general read-only spectator without corresponding proof.
- Retain small labelled raw/processed RGB/depth/segmentation/history diagnostics. Bounded recordings label first-episode selection and cap status while full scalar evaluation continues.

## Frozen showcase protocols

### L1 — State-only Isaac Cartpole

Use the existing `isaac-cartpole-learning-ppo` architecture/task/reset/reward semantics and a recorded single training seed. Existing reference budget: 2,097,152 physical frames, 512 environments and 8,192 frames/batch; any resource-driven change is declared before running and requalified rather than borrowed from the old result. Bound job duration and memory explicitly for the approved worker.

Compare the same untrained architecture and trained export on 32 held-out reset seeds 10000–10031, native five-second/300-step horizon, without an evaluator step cap. Retain the existing criteria: trained mean return at least 4.0, mean length at least 285 and at most one native early termination, with improvement over the matched untrained policy. Report one-training-seed scope; no multiseed, camera-policy or cross-backend equivalence claim.

Select early/intermediate/final checkpoints before fitting. Replay them after the training native process has exited, in fresh remote processes with true saved source-frame identities. Do not fabricate intermediate policies by changing a final export's labels.

### L2 — SO-101 local behavior cloning

Keep the documented five-action fixed-gripper/three-goal joint-reach semantics, 72 successful complete expert episodes and fixed 48/12/12 episode split when that approved dataset is used. Preserve original episode files and capture identity; require the existing explicit current-runtime qualification instead of bypassing it or automatically recapturing.

Repeat the existing two initial-state/500-optimizer-step fits and clean installed-payload parity checks. Evaluate expert, untrained, random and learned references on the same 60 fresh seeds 10000–10059, native horizon and predicate. Require learned/expert success at least 95%, learned improvement at least 30 percentage points over untrained and zero learned/expert action or physical joint-limit violations. Report untrained contract rejections separately from native timeouts. Do not tune on these held-out seeds or claim unseen-goal/general-hardware support.

## Failure, isolation and UX coverage

- Remote input missing, invalid selector/override, setup nonzero/141, absent codec/renderer/browser/display, occupied port and tracking initialization/delivery/finalization errors.
- Controlled interruption after a valid checkpoint, exhausted resume, native startup failure, viewer disconnect, deadline and monitored disk-floor cancellation. Do not promise recovery from an unsaved abrupt kill or delete other jobs to free resources.
- No local model/checkpoint load or CUDA/native initialization on remote dispatch; no implicit local fallback. Remote failures include stage, job ID and original exit/status.
- Registered SDK catalog/config/profile round trips and shared Python/CLI/API dispatch; preserved uv setup, one-GPU defaults, fresh per-submission venvs, explicit overrides and prepared opt-out. Reject old `execution.kind` settings without maintaining a second schema; retain `dev`'s KubeRay `ray-job` and other execution backends.
- Independent concurrent allocations, distinct mutable state and private connections; two successive owned attempts can reuse a released port without overwriting evidence.
- Accurate run summaries, setup/initialization versus learning, physical frames versus optimizer/agent counts, source-frame unknown versus true zero, checkpoint versus inference package and exact recovery commands.
- Explicitly requested logger failure fails the run. Disabled logging performs no writer/server work. Partial delivery stays visible without cross-store rollback promises.
- Installed wheels with editable roots removed, paths containing spaces and another working directory. Actor-only inference with forbidden native/learner/encoder imports; ordinary supervised control and existing local CLI regressions.
- Bounded viewer/recording-on versus off measurements for throughput, startup, memory/VRAM and disk. Record the profile and observed costs; agree a regression budget before enforcing a universal percentage threshold.

## Readiness rule

The final candidate needs green required static/test/package/docs checks, all required matrix gates complete on its actual identity, retrievable reviewed evidence and an independent code/claim review. Preserve inherited OpenSpec history and distinguish scoped strict validation from any pre-existing unrelated warnings. A required blocked native/viewer row keeps this change incomplete unless the maintainer explicitly approves a narrower scope; source edits, push/PR-ready/merge/archive and infrastructure actions retain their own authorization boundaries.
