# Spec Delta

## Purpose

Record bounded evaluation behavior as portable GIF/MP4 artifacts through native experiment logging, with optional TensorBoard previews and traceable recording metadata.

## ADDED Requirements

### Requirement: Recording configuration selects explicit export modes

`training.evaluation.video` SHALL be a nested typed configuration supporting `enabled=False`, `format="gif"`, `max_frames=500`, `max_size=256`, and `fps=None`. Format SHALL accept only `gif`, `mp4`, or `both`; numeric limits SHALL be positive and any explicit FPS finite/positive. Recording with zero evaluation episodes SHALL be rejected.

#### Scenario: Ordinary overrides enable both formats
- **WHEN** users set `training.evaluation.video.enabled=true` and `training.evaluation.video.format=both`
- **THEN** the settings survive ordinary overrides, YAML restoration, and checkpoint scenario restoration
- **AND** each selected evaluation records one episode for GIF and MP4 export

#### Scenario: Recording is disabled or invalid
- **WHEN** recording is disabled, or its format/limits/FPS/episode combination is invalid
- **THEN** disabled recording performs no rendering/encoding, while invalid configuration fails with a focused diagnostic before a long run

### Requirement: Rendering is optional and separate from policy observations

Evaluation recording SHALL request RGB rendering during evaluation-environment construction and consume an optional CPU uint8 H×W×3 frame capability with an available FPS. Environments without rendering SHALL remain usable for scalar evaluation. Recording SHALL NOT replace state observations with image observations or mutate the training environment definition.

#### Scenario: A vector-observation reference is recorded
- **WHEN** CartPole or Pendulum evaluates with recording enabled
- **THEN** its evaluation runtime is created for RGB-array rendering while the policy continues receiving its original vector observations/actions
- **AND** training/collector runtimes are not rendered or reconfigured

#### Scenario: Rendering prerequisites are unavailable
- **WHEN** recording requires unsupported rendering, invalid/empty RGB frames, incompatible render modes, or missing FPS metadata without an explicit FPS
- **THEN** a focused prerequisite/frame diagnostic is reported
- **AND** scalar-only evaluation does not acquire that rendering requirement

### Requirement: Recording captures one bounded rollout buffer

Only the first evaluation episode SHALL be recorded. Capture SHALL copy reusable frame buffers, resize with preserved aspect ratio before buffering, and keep at most `max_frames` frames whose maximum edge is `max_size`. All requested formats SHALL reuse this buffer without rerunning the environment or retaining a second rollout buffer.

#### Scenario: A recording spans an episode boundary
- **WHEN** a recorded episode begins and later terminates within the frame cap
- **THEN** capture includes its reset frame and subsequent transition frames, including the terminal frame
- **AND** it does not concatenate subsequent episodes into that recording

#### Scenario: A long episode reaches the cap
- **WHEN** captured frames reach `max_frames` before the episode ends
- **THEN** no additional frames are retained, while scalar evaluation continues
- **AND** metadata records that capture was capped

### Requirement: Selected files faithfully encode the same episode

GIF export SHALL produce a decodable timed image sequence. MP4 export SHALL produce silent H.264 video using yuv420p at the selected FPS. MP4 SHALL pad odd dimensions to even dimensions without cropping, distortion, or implicit macroblock scaling; padding SHALL be recorded. Each selected format SHALL be encoded once per evaluation point.

#### Scenario: Both exports are requested
- **WHEN** `format=both` records one episode
- **THEN** matching frame-step GIF and MP4 files represent that same captured sequence and timing, allowing documented GIF timing quantization and lossy MP4 color differences
- **AND** source and encoded dimensions identify any at-most-one-pixel MP4 padding per axis

#### Scenario: One format is selected
- **WHEN** `format=gif` or `format=mp4` is used
- **THEN** only that downloadable media format is published
- **AND** a configured TensorBoard preview does not cause an extra downloadable GIF in MP4-only mode

### Requirement: Metrics and artifacts use native configured loggers

Recording SHALL use the run's ordinary logging configuration, existing logger instances, and native artifact dispatch. Every selected media file and matching JSON SHALL reach each configured artifact-capable backend once. Scalar metrics SHALL use native metric logging at the actual evaluation frame step. Recording SHALL NOT create a custom logger, second tracking run, or standalone writer.

#### Scenario: Only MLflow is configured
- **WHEN** evaluation recording runs with MLflow enabled and TensorBoard absent
- **THEN** selected GIF/MP4 files, evaluation JSON, and scalar metrics are saved in the existing MLflow run
- **AND** no TensorBoard writer is constructed, accessed, or required

#### Scenario: MLflow and TensorBoard are configured together
- **WHEN** evaluation publishes selected recordings through both native loggers
- **THEN** the same finalized encoded files and JSON reach each backend exactly once under that backend's existing run identity
- **AND** TensorBoard additionally receives its preview without duplicating artifact uploads
- **AND** unconfigured backends receive nothing

#### Scenario: Recording has no artifact-capable target
- **WHEN** recording is explicitly enabled but no configured backend/output target can persist its files
- **THEN** a focused output-target diagnostic is reported rather than claiming successful recording
- **AND** unsupported backends retain native explicit diagnostics

### Requirement: TensorBoard previews use the active native writer

Only a configured TensorBoard logger SHALL receive an animated image summary under the stable `eval/rl/episode` tag and actual collected-frame step. The preview SHALL be written and flushed before logger finalization through its existing writer. Copying a GIF/MP4 beside event files SHALL NOT substitute for a summary event.

#### Scenario: Users compare training stages
- **WHEN** TensorBoard reads evaluation events at multiple training frame counts
- **THEN** its Images view presents animated previews selectable by those frame steps
- **AND** saved event bytes decode as the expected bounded GIF sequence, not merely a static first-frame substitute

#### Scenario: MP4-only export still has a TensorBoard preview
- **WHEN** `format=mp4` is selected and TensorBoard is configured
- **THEN** TensorBoard receives an animated-image preview while the MP4 remains a downloadable artifact
- **AND** the application does not claim that the native preview embeds an MP4 player

### Requirement: Recordings retain machine-readable provenance

Each recording SHALL have a matching compact JSON artifact with aggregate metrics, evaluation frame step/seeds/count/horizon/mode, environment identity, config/contract provenance, FPS, captured count/cap status, selected formats, and encoder/codec/dimension/padding metadata. It SHALL NOT include raw training trajectories or live runtime state.

#### Scenario: A recorded evaluation is inspected
- **WHEN** users retrieve a frame-step media file and its JSON from any supported configured backend
- **THEN** they can identify its source evaluation configuration, metrics, environment, seed sequence, timing, and exported representations
- **AND** metadata distinguishes captured source dimensions from any padded MP4 dimensions

### Requirement: Encoding failures do not publish incomplete success

Requested encoders/targets SHALL be checked before long training. Encoding SHALL finalize and close owned resources before a file is published. Failure/cancellation SHALL clean staged partial files and remain visible; requested MP4 SHALL NOT silently fall back to GIF. Backend delivery failures SHALL NOT be reported as completed delivery.

#### Scenario: MP4 encoding fails or is interrupted
- **WHEN** an encoder fails after writing partial bytes
- **THEN** its writer closes and no incomplete MP4 is published as a completed artifact
- **AND** the failure is visible rather than being replaced by a GIF-only success claim

#### Scenario: Artifact upload fails
- **WHEN** a configured backend rejects a finalized file
- **THEN** recording reports the delivery failure
- **AND** it does not promise transactional rollback of artifacts already delivered to another backend

### Requirement: Encoding dependencies remain explicit and optional

GIF-only export SHALL NOT require an MP4 encoder. Scalar training/evaluation and portable policy inference SHALL NOT import recording encoders. Optional rendering/video installation SHALL declare its dependencies directly and diagnose missing renderers, encoders, executables, or H.264 support. Runtime SHALL NOT install system packages or download executables implicitly.

#### Scenario: GIF-only users omit the video extra
- **WHEN** GIF recording runs with rendering prerequisites installed and the MP4 backend unavailable
- **THEN** GIF artifacts and configured TensorBoard previews remain usable
- **AND** no MP4 encoder is imported or launched

#### Scenario: MP4 prerequisites are missing
- **WHEN** `format=mp4` or `format=both` is requested without a compatible encoder/executable/codec
- **THEN** preflight names the missing video prerequisite before long training
- **AND** no implicit binary download, package installation, or format substitution occurs
