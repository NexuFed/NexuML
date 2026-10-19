# Spec Delta

## Purpose

Provide a reproducible Isaac-first NexuML physical-learning experience together with maintained CPU MuJoCo, Meta-World and GPU MuJoCo Playground references, without making the entire NexuML distribution depend on one mutually incompatible native package set.

## ADDED Requirements

### Requirement: Base NexuML remains simulator independent

Core NexuML, supervised workflows, discovery, portable contracts and exported inference SHALL work without importing Isaac, MuJoCo, Meta-World, JAX, ROS, simulator graphics libraries or downloading models. NexuML library extras SHALL own concrete optional requirements; root extras SHALL forward to the library. Each requested native workflow SHALL report its missing optional dependency explicitly.

#### Scenario: A base-only wheel is installed
- **WHEN** a user imports NexuML, discovers all documented scenario IDs, loads a portable policy or trains an unrelated supervised model
- **THEN** no native physics backend or model checkpoint is allocated and dependency metadata does not force a robotics stack

### Requirement: Compatibility is a tested selection matrix, not an impossible universal extra

The implementation SHALL maintain one versioned matrix for CPython, Torch/CUDA, TorchRL/TensorDict, Isaac Lab/Sim/Newton, MuJoCo, Meta-World, MuJoCo Playground/MJX/MJWarp, JAX and LeRobot. The initial target is the existing consumer-managed Python 3.12/CUDA-13 Isaac-compatible primary environment, subject to native verification. The matrix SHALL label every combination tested, blocked, optional or not attempted, with exact upstream source and lock identities.

#### Scenario: Meta-World pins MuJoCo 3.3.0 while Newton requires another release family
- **WHEN** complete dependency resolution detects the known incompatible native requirements
- **THEN** both constraints, chosen supported alternatives and the incompatibility are recorded
- **AND** the implementation SHALL NOT force conflicting packages into one interpreter, silently downgrade the primary stack, patch upstream metadata, install with --no-deps or mark either native reference complete

### Requirement: One normal developer experience with explicit opt-in incompatible lanes

NexuVibe SHALL own primary Python, backend wheel indexes and lock. Compatible workflows SHALL use ordinary NexuML session/configuration interfaces there. After maintained alternatives fail, an approved pinned optional benchmark lane MAY host an incompatible native stack with a tested policy/data/metrics boundary. Such a lane SHALL remain opt-in, preserve primary acceptance and never serialize native handles.

#### Scenario: A benchmark requires an isolated dependency lane
- **WHEN** the user selects that benchmark explicitly and its approved installed lane is available
- **THEN** the same NexuML-facing scenario, policy artifact contract, metrics and evaluation command remain usable through the documented process boundary where technically supported
- **AND** training support, inference-only support and cross-runtime checkpoint conversion each have independent evidence or are explicitly blocked
- **AND** the separately launched lane records its own lock, interpreter/source identity, protocol and reproducibility proof without replacing the primary environment or becoming a default dependency

#### Scenario: No validated boundary exists for the incompatible lane
- **WHEN** its task is requested without a verified environment or policy-interop path
- **THEN** a precise unsupported/blocked diagnostic is returned; the task is not falsely reported as an integrated NexuML learner

### Requirement: Temporary constraint exceptions are narrow and reproducible

Temporary consumer-owned overrides SHALL have source/API rationale, exception records, exact selections and focused validation. The five recorded conflicts are initial cases; the authorized campaign MAY evaluate further documented cases without blanket dependency suppression. Upstream metadata SHALL remain unmodified; --no-deps and blind physics overrides remain forbidden. Resolution SHALL NOT imply security/ABI/native/model compatibility.

#### Scenario: A primary constraint exception is evaluated
- **WHEN** maintained alternatives retain a recorded conflict, including a further case evaluated under campaign authorization
- **THEN** the record identifies affected package/source, original and replacement bounds, exact selected versions, reason, upstream issue/source reference, owner, lock, affected workflows, test plan, removal condition and rollback inputs
- **AND** a complete solver result is labeled resolution with declared exceptions, not native compatibility
- **AND** configuration belongs to the owning consumer without silently changing published NexuML requirements or authorizing installation, downloads or another environment

#### Scenario: A replacement resolves but affected behavior fails
- **WHEN** native startup/version checks/task stepping or real-model processor/inference/training/resume/export/evaluation checks fail or cannot run
- **THEN** failures or missing prerequisites remain visible and the affected acceptance stays incomplete
- **AND** imports, tiny substitutes and solver success do not close the required OvPhysX/Isaac or SmolVLA/GR00T/selected pi0.5 workflow gates

#### Scenario: The newly approved numeric, HTTP or USD selection is evaluated
- **WHEN** an exception selects NumPy 2.3.1, urllib3 >=2.8.0 or Lab-matched usd-exchange 3.0.1
- **THEN** its record requires NumPy/OpenCV/native ABI and numeric parity, preserved Lab security floor and tested HTTP/S3 paths, or matching isolated-importer/OpenUSD/schema/asset/native parity respectively
- **AND** no other ABI, physics, Python, CUDA or driver requirement is waived and no competing pxr provider is installed

#### Scenario: A dependency checker reports original bounds
- **WHEN** a consumer override leaves an upstream dependency-check discrepancy
- **THEN** the exact recorded mismatch and unmodified checker result remain visible, with all other dependencies verified
- **AND** unexpected discrepancies block acceptance; the installation is not described as unconditionally clean

### Requirement: Exception evidence expires with its selection

Every affected version/source change SHALL requalify the exception and relevant tests. When maintained upstream releases resolve the conflict, the exception SHALL be removed and ordinary resolution, integrity and behavior reverified before the primary lock changes. Failed validation SHALL retain rollback inputs and block the selection rather than broaden the exception automatically.

#### Scenario: An upstream fix becomes available
- **WHEN** the recorded removal condition is satisfied by a maintained selection
- **THEN** the override is removed and ordinary resolution/dependency checks plus affected behavior tests are rerun
- **AND** an unrelated version change cannot inherit the old exception evidence or an unrecorded wider override

### Requirement: Package-wide Python support is not reduced solely for one benchmark extra

The baseline supported Python range SHALL be determined by NexuML core and other agreed package dependencies, not made globally >=3.12,<3.14 merely to accommodate an optional Meta-World release. Compatible interpreter constraints SHALL be documented per optional integration/lock. If packaging cannot encode incompatible extras precisely, dependency conflicts SHALL be enforced by verified installation profiles and negative tests instead of silently narrowing every user's Python eligibility.

#### Scenario: A core-only user selects Python 3.14
- **WHEN** the base published package remains compatible with that interpreter
- **THEN** it can install and operate independently of Meta-World
- **AND** requesting an unsupported Meta-World native lane fails explicitly

### Requirement: Published extras have truthful and predictable semantics

The optional Physical AI stack SHALL live in nexuml-library[robots], forwarded by nexuml[robots], outside both all closures. Preserve unrelated extras and explicit incompatible-provider selections rather than promise an impossible union. The development consumer SHALL select robots explicitly. Wheel/sdist metadata, uv resolution, clean and editable installs SHALL agree; static discovery SHALL NOT download assets/models.

#### Scenario: A user installs all without robots
- **WHEN** the base or all package selection is installed without an explicit robotics selection
- **THEN** it does not acquire the new Isaac/LeRobot/native robotics stack through a hidden transitive all dependency
- **AND** requesting robots resolves its declared integrations or returns the real upstream compatibility failure without silently omitting packages

#### Scenario: A user requests a conflicting extras combination
- **WHEN** an optional dependency selection cannot be jointly resolved
- **THEN** documented resolution diagnostics list compatible alternatives and no incomplete environment is described as successful

### Requirement: Published compatibility is distinct from environment pins

NexuML distributions SHALL use genuine language/API minimums and only evidence-backed incompatibility bounds. Development Python/CUDA/Torch choices, exact lock versions and backend indexes SHALL NOT become universal package restrictions. The consuming root uv project SHALL own concrete versions, sources, Python/backend/index selection, approved overrides and its lock. Broad ranges SHALL NOT imply tested support for every combination.

#### Scenario: A consumer selects another otherwise compatible environment
- **WHEN** its Python, Torch or backend selection differs from the development reference but satisfies genuine requirements
- **THEN** package metadata does not force the reference pins or CUDA index merely to match the development machine
- **AND** actual upstream wheel/API/ABI limits and runtime control/security validation still apply; untested combinations are labeled unverified

#### Scenario: A bound or exact pin is proposed for a published requirement
- **WHEN** optional metadata is implemented or a dependency is revised
- **THEN** each restrictive bound has a concrete language/API/ABI/security incompatibility reason or is kept only in consumer configuration
- **AND** consumer-root override configuration remains separate from published requirements and unmodified dependency-check evidence

### Requirement: Hardware and rendering prerequisites are independently verified

NexuML SHALL check CPU native MuJoCo, CPU OSMesa, NVIDIA EGL and GPU-native MuJoCo Playground separately from Isaac/renderer/ROS readiness. A compute-visible GPU does not prove EGL/video capability; a successful simulator import does not prove native batched rollout correctness.

#### Scenario: Headless CPU and GPU renderers are configured
- **WHEN** their supported reference runs execute
- **THEN** actually rendered nonempty frames and encoder outputs are verified for each claimed profile and state-only training remains renderer-free

### Requirement: Provenance and adoption never rewrite historical evidence

Retain earlier completed Gymnasium, Isaac EA and MuJoCo reports as historical evidence with their original package/source/asset identities. A migrated release, physics preset or environment identity SHALL receive new native validation and fresh learning/export/evaluation proof. No CI skip, resolver smoke or substitute robot counts as completion.

#### Scenario: A historical SO-101 actor is replayed in another physics version
- **WHEN** control rate, joint order, action scaling, asset or reset settings differ
- **THEN** NexuML rejects incompatible semantics or performs a separately tested explicit migration

### Requirement: Cross-repository setup and destructive operations require separate authority

NexuML SHALL document implementation handoffs for the NexuVibe-owned DevContainer/lock and any independently scoped NexuFL compatibility. Planning does not authorize silently rewriting another repository, restarting active containers, downloading licensed weights, scheduling paid GPU runs or actuating hardware. Runtime readiness and manual approvals remain visible separate gates.

#### Scenario: The requested primary runtime cannot be provisioned
- **WHEN** native OS, driver, quota or user approval constraints block clean setup
- **THEN** code/static planning can proceed, but native readiness remains incomplete, with explicit next actions and retained old environments
