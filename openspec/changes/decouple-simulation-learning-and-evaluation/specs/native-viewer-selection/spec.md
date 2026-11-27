## ADDED Requirements

### Requirement: Isaac exposes four typed native viewer modes

One typed configuration with ordinary CLI overrides SHALL select headless, viser, webrtc or local for each qualified Isaac runtime and purpose. It SHALL reuse the native launcher and Viser settings without a competing headless flag or replacement renderer. Training and evaluation SHALL default to headless. Selection SHALL preserve actor/task semantics; a training viewer SHALL NOT enable a play or evaluation viewer. Unsupported choices SHALL fail without fallback.

#### Scenario: Headless policy requires cameras
- **WHEN** an environment has qualified offscreen camera observations but no interactive viewer is requested
- **THEN** required sensors and any requested qualified offscreen recording remain active without opening an interactive listener or desktop.

### Requirement: Viser retains private spectator guarantees

Viser SHALL reuse the existing native read-only training/evaluation implementation and satisfy V1–V5 of read-only-isaac-visualization. It SHALL remain loopback-bound, sharing disabled, with requested endpoint verification and user-managed private TCP forwarding. Qualified play controls SHALL be explicitly scoped and SHALL NOT weaken the training/evaluation denial boundary.

#### Scenario: A browser sends a reset command to a training spectator
- **WHEN** the command reaches the actual viewer dispatch path
- **THEN** no simulation mutation occurs and the existing private spectator and cleanup guarantees remain intact.

### Requirement: Native WebRTC and local Kit viewing use prepared infrastructure

WebRTC SHALL use native Kit streaming with qualified GPU, encoder, graphics and private connectivity. Local mode SHALL use a native Kit window on a prepared display. Neither mode SHALL provision networking, forwarding, VNC or graphical sessions, expose a viewer publicly by default or silently substitute another viewer.

#### Scenario: A remote operator uses the native streaming client
- **WHEN** an approved private connection provides the required native TCP and UDP reachability and a standalone graphics pod starts WebRTC play outside Ray training
- **THEN** the client displays the actual live policy rollout and the job exposes and reports the configured supported signaling and media endpoints without managing the external network.
- **AND** the existing Isaac image may be used as a comparison runtime, not as a requirement for installing the Python stack.

#### Scenario: A prepared local or remote graphical display is selected
- **WHEN** native display and graphics prerequisites are present
- **THEN** a real Kit window displays the live rollout; an absent or incompatible display produces an actionable error, not a headless fallback.
- **AND** local workstation, X11 and TigerVNC paths are documented and qualified separately where supported.

### Requirement: Viewer purposes and ownership remain explicit

All training and scientific evaluation viewers SHALL be nonmutating; local and WebRTC modes SHALL be rejected for these purposes until qualified to enforce this. Explicit simulation play may use approved native controls. On stop, failure or cancellation, the viewer SHALL release owned applications, streams, listeners and threads without modifying borrowed infrastructure.

#### Scenario: A native Kit viewer cannot deny benchmark mutation
- **WHEN** it is requested for formal evaluation
- **THEN** the unsupported purpose fails explicitly instead of running an interactive benchmark or silently switching viewer.

#### Scenario: A native display, browser or stream is qualified
- **WHEN** a viewer is submitted for acceptance
- **THEN** evidence shows the actual live policy rollout and cleanup; application startup, endpoint reachability or fake tests alone do not qualify it.


### Requirement: Distributed visualization is optional and role scoped

Distributed training SHALL be headless by default without disabling policy
sensors. One selected rank or collector MAY provide a private read-only
Viser of its actual rollout. Independent snapshot play remains complementary,
and Kit/WebRTC SHALL NOT be required for training.

#### Scenario: An operator selects one live training spectator
- **WHEN** a qualified distributed training run uses the selected rank's private Viser
- **THEN** Only that rank shows actual transitions, owns a unique endpoint and limits the visible subset. Disconnect or cancellation preserves learning semantics and native ownership.

#### Scenario: An operator requests an external native viewer
- **THEN** Use explicit prepared independent playback or a compatible local profile. Qualify WebRTC client UDP, forwarded Windows X11 and other external displays independently of proven headless/Viser and Kit/TigerVNC. No provisioning or fallback.
