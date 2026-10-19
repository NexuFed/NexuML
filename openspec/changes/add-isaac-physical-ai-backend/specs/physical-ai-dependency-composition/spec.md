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

NexuVibe SHALL own the primary Python selection, backend wheel index and primary lock. All compatible workflows SHALL run there and be available through the ordinary NexuML session/configuration interface. If a required optional upstream benchmark truly cannot resolve with that primary stack after testing maintained alternatives, an explicit pinned, approved, separately launched process or execution profile MAY host it. Such a lane SHALL have its own recorded lock, interpreter/source identity, protocol and reproducibility proof, and SHALL NOT replace the primary environment or silently become a default dependency. Native runtime handles SHALL never be serialized across the boundary.

#### Scenario: A benchmark requires an isolated dependency lane
- **WHEN** the user selects that benchmark explicitly and its approved installed lane is available
- **THEN** the same NexuML-facing scenario, policy artifact contract, metrics and evaluation command remain usable through the documented process boundary where technically supported
- **AND** training support, inference-only support and cross-runtime checkpoint conversion each have independent evidence or are explicitly blocked

#### Scenario: No validated boundary exists for the incompatible lane
- **WHEN** its task is requested without a verified environment or policy-interop path
- **THEN** a precise unsupported/blocked diagnostic is returned; the task is not falsely reported as an integrated NexuML learner

### Requirement: Package-wide Python support is not reduced solely for one benchmark extra

The baseline supported Python range SHALL be determined by NexuML core and other agreed package dependencies, not made globally >=3.12,<3.14 merely to accommodate an optional Meta-World release. Compatible interpreter constraints SHALL be documented per optional integration/lock. If packaging cannot encode incompatible extras precisely, dependency conflicts SHALL be enforced by verified installation profiles and negative tests instead of silently narrowing every user's Python eligibility.

#### Scenario: A core-only user selects Python 3.14
- **WHEN** the base published package remains compatible with that interpreter
- **THEN** it can install and operate independently of Meta-World
- **AND** requesting an unsupported Meta-World native lane fails explicitly

### Requirement: Published extras have truthful and predictable semantics

Do not advertise an unrestricted all extra as the union of conflicting Isaac/Newton, legacy Meta-World and MJX/JAX pins. Either preserve all as an explicitly compatible tested subset and expose named opt-in extras, or deprecate its misleading robotics meaning with a documented migration. Wheel/sdist metadata, uv resolution, clean installs and editable installs SHALL agree. Default package selections SHALL not cause large asset or model downloads during static discovery.

#### Scenario: A user requests a conflicting extras combination
- **WHEN** an optional dependency selection cannot be jointly resolved
- **THEN** documented resolution diagnostics list compatible alternatives and no incomplete environment is described as successful

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
