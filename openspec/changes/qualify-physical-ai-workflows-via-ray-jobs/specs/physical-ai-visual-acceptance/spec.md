# Spec Delta

## Purpose

Verify that users can inspect the correct live simulation and recorded policy behavior through real remote viewing and tracking interfaces, without confusing scene display, actor observations or learning quality.

## ADDED Requirements

### Requirement: V1 Visual roles have distinct acceptance evidence

Live spectator scenes, recorded policy evaluation and policy-consumed sensor tensors SHALL have separate identities and checks. A displayed scene SHALL NOT establish camera/depth/segmentation/history correctness, and a recording SHALL NOT establish that a live viewer works. Screenshots alone SHALL NOT establish optimization or learning improvement.

#### Scenario: A state-only actor has a visible environment
- **WHEN** its spectator or evaluation recording displays the robot
- **THEN** the result remains labelled as state-policy viewing
- **AND** no image-policy or sensor-training support is inferred from the display

### Requirement: V2 TensorBoard previews work in the real browser

TensorBoard acceptance SHALL open the candidate's actual events in a browser, inspect advancing scalar values and select animated evaluation previews at identified source frame counts. Downloadable MP4 files SHALL remain distinct from the native animated-image presentation. Reopening retained logs SHALL preserve usable results after training exits.

#### Scenario: Training stages are compared
- **WHEN** users select initial, intermediate and final qualified evaluation events
- **THEN** the browser displays changing preview frames from the corresponding identified policies
- **AND** a static thumbnail, copied file beside events or missing source-frame identity is not counted as equivalent proof

### Requirement: V3 MLflow retrieval and logger combinations are exercised

Acceptance SHALL inspect actual MLflow runs and retrieve selected GIF/MP4/metadata artifacts through the supported UI or authenticated artifact route. TensorBoard-only, MLflow-only, both and intentionally disabled logging SHALL be exercised. Selected payloads SHALL match across enabled backends without duplicate delivery or reliance on an unconfigured writer.

#### Scenario: Only MLflow is enabled
- **WHEN** a real recorded evaluation completes
- **THEN** its run has the expected metrics, status and retrievable selected artifacts
- **AND** the downloaded media plays without constructing a TensorBoard writer

#### Scenario: Both tracking backends are enabled
- **WHEN** the same evaluation is published
- **THEN** corresponding finalized artifacts and metadata identify the same captured rollout in both backends
- **AND** absent inline MP4 support is distinguished from successful downloadable playback

### Requirement: V4 Recorded frames preserve episode and sensor meaning

Recorded evaluation SHALL retain correct reset, transition and terminal scene identity, source timing, bounds and capture-limit labels. Sensor cases SHALL additionally expose bounded labelled raw/processed diagnostic samples and verify actual units, labels and history/reset order. Uniform or stale output SHALL be diagnosed with task-specific checks, not accepted merely because a codec decodes it.

#### Scenario: An episode autoresets during recording
- **WHEN** its terminal observation and scene are retained
- **THEN** they belong to the ending episode rather than the next reset
- **AND** continuing streams preserve their own declared history

#### Scenario: A depth policy is inspected
- **WHEN** raw and processed diagnostic samples are retrieved
- **THEN** units, invalid-depth treatment and history/channel ordering are identifiable
- **AND** an attractive spectator image does not substitute for these checks

### Requirement: V5 Isaac spectator acceptance uses real training and replay

The qualified Isaac spectator SHALL be exercised during ordinary remote CLI training and separately enabled fresh-process replay. Browser navigation, display pause, selection, disconnect/reconnect and lifecycle cleanup SHALL preserve the existing nonmutating contract. Exact private worker access and viewer/backend support limits SHALL be recorded.

#### Scenario: A browser disconnects and reconnects
- **WHEN** it observes a running remote training job
- **THEN** collection and updates continue without client presence
- **AND** reconnecting displays current state without resetting physics, policy history or budgets

#### Scenario: The viewer-requested job is cancelled
- **WHEN** the owned remote job receives cancellation
- **THEN** its native resources and viewer listener close with truthful status
- **AND** no unrelated job is stopped and the viewer port can be reused by a later owned attempt

### Requirement: V6 Native MuJoCo viewing is qualified on remote workers

The supported MuJoCo reference SHALL demonstrate an actual live native viewer while its remote public CLI training executes, using a prepared worker display and private access path. Human viewing and RGB recording SHALL use compatible separate environment selections. Viewer availability SHALL NOT be inferred from offscreen rendering, and native interactive controls SHALL NOT be advertised as read-only without proof.

#### Scenario: Human viewing and evaluation recording are requested
- **WHEN** the MuJoCo training job uses its native viewer and its saved policy is later recorded
- **THEN** the live session shows actual advancing training state and a fresh evaluation uses the compatible recording mode
- **AND** the actor's declared observation/action semantics remain unchanged

#### Scenario: Reacher uses the stock Ray worker image
- **WHEN** the remote training job requests live Reacher viewing
- **THEN** an X11/TigerVNC sidecar provides the display without changing the Ray image
- **AND** the viewer uses an authenticated pod-local display and private port forwarding, with no public listener
- **AND** the Ray job, not the worker image or display sidecar, installs its application Python dependencies

### Requirement: V7 Native Gazebo viewing requires a real ROS simulation

The Gazebo viewing journey SHALL run the real simulation, controllers and reset/clock interfaces on prepared remote resources. Its GUI SHALL show the robot driven by the actual CLI policy while real updates are evidenced separately. Camera navigation, private access and cleanup SHALL be checked; fake transport or virtual-display startup alone SHALL NOT qualify the journey.

#### Scenario: The GUI connects to the wrong world
- **WHEN** its world, robot or simulation namespace differs from the training case
- **THEN** visual acceptance fails even if the scene animates
- **AND** the evidence identifies the mismatched worker/world/session

### Requirement: V8 Visual controls and resource costs remain explicit

Viewer mutation capabilities SHALL be disclosed. Sessions accepting simulation pause/reset/force edits SHALL be labelled debug and excluded from quantitative acceptance. Bounded comparisons SHALL record viewing/recording overhead, frame limits and memory/disk use. An unavailable renderer or remote display SHALL block the selected viewing claim without enabling a different unqualified backend.

#### Scenario: A native GUI permits a physical intervention
- **WHEN** a user changes simulation state through that GUI
- **THEN** the session is not represented as read-only or used as a frozen quality trial
- **AND** independent headless evaluation remains available without inheriting the intervention

#### Scenario: A browser uses software rendering
- **WHEN** it displays a qualified spectator through an approved private route
- **THEN** the result identifies browser-side software rendering
- **AND** it does not claim server RTX, native sensor or universal renderer qualification

### Requirement: V9 The authored task is visible without changing its physics

Acceptance SHALL exercise the authored task's live training view and separately enabled fresh evaluation view on the declared physics backend, using project-supplied camera settings. Evidence SHALL identify the scene edit, task, purpose, physics, viewer and renderer where applicable. Sensor diagnostics, video captured from an episode, and a new rollout of the policy SHALL be labelled separately.

#### Scenario: The engineer follows the authored target through training and evaluation
- **WHEN** the worker trains and later evaluates the exported policy with live viewing enabled for each purpose
- **THEN** both private views show the identified edited scene and advancing state, without adding a camera pose keyed to that task name in NexuML
- **AND** any claimed video is from the identified episode on its declared backend

#### Scenario: A camera-policy recipe has sensor output but no spectator video renderer
- **WHEN** the run retains raw or processed sensor diagnostics but cannot record the requested spectator video
- **THEN** only the diagnostics are reported as delivered and the requested recording remains unsupported or blocked
- **AND** a screen capture or recording from another physics backend does not qualify that video

### Requirement: V10 Arena has private live viewing and remote headless results

The same Arena evaluation workflow SHALL support headless scoring and private live GUI observation on Ray. Qualification SHALL match policy, scene, physics and starting conditions across modes without requiring the viewer for scoring. In headless mode, the suite summary, per-episode results and a selected failure diagnostic SHALL be retrievable remotely in MLflow after worker cleanup.

#### Scenario: The engineer watches a live Arena evaluation
- **WHEN** the engineer connects through the approved private route to a viewer-enabled Ray job
- **THEN** the GUI shows the advancing policy rollout in the actual evaluation worker, with camera navigation that does not intervene in physics
- **AND** the viewer is released when the job ends; a recording, different process or display startup alone cannot satisfy the live gate

#### Scenario: The benchmark runs without a live viewer
- **WHEN** the engineer opens a completed headless evaluation after worker removal
- **THEN** the identified scores, episode rows and selected failure diagnostic remain accessible, independently of the live GUI
