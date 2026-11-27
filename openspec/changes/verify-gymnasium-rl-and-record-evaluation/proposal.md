# Proposal

## Why

NexuML already implements the RL foundation, but roadmap Phases 0–1 need current, reproducible evidence rather than historical completion claims. Users also need to inspect policy behavior during training through bounded GIF/MP4 recordings and the existing logging system, including MLflow-only operation, without replacing the working trainer or rebuilding its components.

## What Changes

- Audit the existing foundation first and publish a baseline/gap report with commands, import origins, dependencies, results, and honest capability classifications.
- Verify and stabilize the existing `cartpole-ppo` and `pendulum-ppo` CLI/session workflows: real updates, evaluation, YAML/overrides, interrupted resume, complete policy export/reload, and direct/process collector cleanup and synchronization.
- Extend `RLEvaluationSpec` with opt-in initial and frame-based periodic evaluation; preserve current final-only defaults and the scalar result surface. Isolate evaluation environments, policy modes, and RNG state from learning.
- Add optional evaluation RGB rendering without changing vector policy observations or requiring every environment to render.
- Add bounded first-episode recording with selectable GIF, silent H.264 MP4, or both, plus compact provenance JSON. Reuse one captured rollout for all formats.
- Route metrics and artifacts through native `LoggingSpec`, existing logger instances, and `log_artifact`. MLflow-only runs save media in their existing run; configured TensorBoard loggers additionally receive animated previews. No custom logger, standalone writer, or duplicate tracking run is introduced.
- Add a thin inference-only `nexuml evaluate <policy-package>` command using existing package loading, scenario restoration, overrides, logging, and the same evaluator.
- Add scoped optional rendering/video extras, focused hermetic tests, opt-in real rendering/local-tracking proofs, CI wiring, and exact getting-started/replay commands.

No breaking change is intended. The two reference scenarios will configure the existing TensorBoard logger by default, but periodic evaluation and rendering remain opt-in. Supervised behavior, ordinary RL training without encoders, and portable policy inference remain supported.

## Capabilities

### New Capabilities

- `gymnasium-rl-reference-workflow`: Audited, reproducible discrete/continuous reference workflows, artifact boundaries, and documented verification evidence.
- `scheduled-policy-evaluation`: Typed initial/periodic/final evaluation, collected-frame cadence, isolated episodic execution, resume, and preserved default lifecycle behavior.
- `policy-evaluation-media`: Optional RGB capture, GIF/MP4 formats, bounded encoding, native backend artifact routing, TensorBoard previews, and recording provenance.
- `policy-package-evaluation`: Inference-only evaluation/replay of exported policy packages through the ordinary CLI configuration and logging conventions.

### Modified Capabilities

None. Existing `logs-root-resolution`, `cli-override-system`, and `framework-test-coverage` contracts are reused unchanged. The RL capabilities are still defined by the unarchived `add-reinforcement-learning` change rather than canonical `openspec/specs/`; this follow-up adds distinct capabilities instead of duplicating or silently rewriting those pending specs.

## Impact

- **Baseline dependency:** The current `feature/nex-212-add-reinforcement-learning` implementation and `openspec/changes/add-reinforcement-learning/` remain prerequisites. Do not reset its completed tasks, archive it, or mark live ROS/Gazebo/CARLA proofs complete.
- **Primary surfaces:** Core evaluation/build-context types; optional interaction rendering; Gymnasium adapter and existing reference scenarios; RL Lightning/session lifecycle; native tracking helpers; CLI evaluation command; tests, docs, and opt-in CI.
- **Dependencies:** Optional core/library `rendering` support for Pillow and `video` support for ImageIO/FFmpeg, with compatible bounds proved before integration. No MoviePy, mandatory system installation, implicit runtime download, or broad dependency upgrade.
- **Artifacts:** Native run-owned GIF/MP4/JSON outputs and TensorBoard events; resumable checkpoints remain distinct from portable policy artifacts. No encoders, environment handles, evaluation buffers, or learner-only state enter inference requirements.
- **Scope:** The supplied Phase 0–1 plan and `.openchamber/plans/nexuml-physical-ai-roadmap.md` are preserved. No Phase 2+, IL/VLA, simulator expansion, hardware, NexuFL, Ray expansion, asynchronous evaluation, new algorithm, or convergence tuning is included.
- **Delivery:** Baseline/evidence-driven fixes → periodic scalar evaluation → GIF/MP4/native logging/replay and documentation. This proposal creates planning artifacts only; execution and compatibility evidence are implementation gates, not claimed results.
