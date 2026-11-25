## ADDED Requirements

### Requirement: Isaac exposes four typed native viewer modes

NexuML SHALL support explicit headless, viser, webrtc and local viewer selection for qualified Isaac runtime/purpose combinations using one typed configuration and ordinary CLI overrides. It SHALL reuse the native launcher and existing Viser settings, with no competing headless flag or replacement renderer. Headless SHALL remain the train/evaluation default and preserve offscreen policy sensors and qualified recording. Viewer selection SHALL not change actor/task semantics. Training viewer activation SHALL not implicitly activate playback/evaluation; unsupported choices SHALL fail without fallback.

#### Scenario: Headless policy requires cameras
- **WHEN** an environment has qualified offscreen camera observations but no interactive viewer is requested
- **THEN** required sensors remain active without opening an interactive listener or desktop.

### Requirement: Viser retains private spectator guarantees

Viser SHALL reuse the existing native read-only training/evaluation implementation and satisfy V1–V5 of read-only-isaac-visualization. It SHALL remain loopback-bound, sharing disabled, with requested endpoint verification and user-managed private TCP forwarding. Qualified play controls SHALL be explicitly scoped and SHALL NOT weaken the training/evaluation denial boundary.

#### Scenario: A browser sends a reset command to a training spectator
- **WHEN** the command reaches the actual viewer dispatch path
- **THEN** no simulation mutation occurs and the existing private spectator and cleanup guarantees remain intact.

### Requirement: Native WebRTC and local Kit viewing use prepared infrastructure

WebRTC SHALL reuse native Kit streaming and the existing CI-distributed Isaac image, expose supported configured signaling/media endpoints and require qualified native GPU/encoder/graphics prerequisites and approved private connectivity. Local SHALL launch the native Kit window on a prepared graphical display; local workstation, X11 and TigerVNC paths SHALL be documented and qualified separately where supported. Neither mode SHALL provision VPN/NetBird, routes, firewall, SSH or Kubernetes forwarding, VNC servers or graphical sessions. No automatic public exposure or silent substitution of a viewer is allowed.

#### Scenario: A remote operator uses the native streaming client
- **WHEN** an approved private connection provides the required native TCP and UDP reachability and the qualified worker starts WebRTC play
- **THEN** the client displays the actual live policy rollout and the job reports its connection details without managing the external network.

#### Scenario: A prepared local or remote graphical display is selected
- **WHEN** native display and graphics prerequisites are present
- **THEN** a real Kit window displays the live rollout; an absent or incompatible display produces an actionable error, not a headless fallback.

### Requirement: Viewer purposes and ownership remain explicit

All viewer modes SHALL keep training and scientific evaluation nonmutating. Native local/WebRTC viewer purposes that cannot enforce this SHALL be rejected for train/evaluate until qualified, while explicit simulation play may use approved native controls. Lifecycle SHALL release owned applications, streams/listeners and threads on stop/failure/cancellation without modifying borrowed infrastructure. Native display, browser and streaming acceptance SHALL require actual live policy rollout and cleanup evidence; application startup, endpoint reachability or fake tests SHALL NOT suffice.

#### Scenario: A native Kit viewer cannot deny benchmark mutation
- **WHEN** it is requested for formal evaluation
- **THEN** the unsupported purpose fails explicitly instead of running an interactive benchmark or silently switching viewer.
