# Spec Delta

## Purpose

Make the selected MuJoCo and Meta-World stack installable from NexuML's own package metadata and composable with NexuFL in the consumer's single uv-managed development environment, with explicit interpreter and backend compatibility.

## ADDED Requirements

### Requirement: NexuML owns simulator dependency declarations

MuJoCo and Meta-World requirements SHALL be declared in NexuML's package manifests through optional library extras and root forwarding extras. Both distributions' `all` selections SHALL include the selected stack transitively. Consumers SHALL NOT need duplicate direct simulator declarations or a manual post-install package injection.

#### Scenario: NexuVibe consumes the existing local package selections
- **WHEN** its existing `nexuml[all,dev]`, `nexuml-library[all]` and NexuFL selections are resolved
- **THEN** MuJoCo and Meta-World are included from NexuML metadata without adding them directly to NexuVibe or NexuFL
- **AND** wheel metadata preserves the same dependency relationship as editable installation

### Requirement: Python compatibility is explicit and temporary

Both NexuML distributions SHALL declare `>=3.12,<3.14` for the selected stack, with matching classifiers, packaging assertions, CI and documentation. Unsupported interpreters SHALL fail clearly rather than succeeding with a requested simulator silently omitted. The upstream reason for the upper bound SHALL be documented.

#### Scenario: A user selects Python 3.14
- **WHEN** the changed NexuML packages are requested on that interpreter
- **THEN** resolution rejects the unsupported combination through package metadata
- **AND** guidance distinguishes Meta-World's Python limit and MuJoCo pin from current MuJoCo releases' broader support

#### Scenario: A supported interpreter is selected
- **WHEN** the complete consumer composition resolves on Python 3.12 or 3.13
- **THEN** simulator packages are genuinely included rather than hidden behind omission markers
- **AND** the tested interpreter patch and all selected package versions are recorded

### Requirement: One shared consumer environment contains the complete stack

Acceptance SHALL use one consumer-owned uv lock and one environment containing the changed NexuML core/library, NexuFL and selected simulator stack. A dedicated MuJoCo venv, deployment lock/profile or worker container SHALL NOT substitute for successful joint resolution. The repository's existing development lock SHALL remain maintained.

#### Scenario: Joint installation is verified
- **WHEN** the user-prepared shared environment is checked
- **THEN** dependency integrity and imports for NexuML, NexuFL, Torch, TorchRL, TensorDict, Gymnasium, MuJoCo and Meta-World succeed in that same interpreter
- **AND** import origins identify the intended changed NexuML checkout rather than a different local checkout or stale wheel

#### Scenario: Joint requirements conflict
- **WHEN** the complete consumer composition has no compatible resolution
- **THEN** acceptance remains blocked with the conflicting requirements recorded
- **AND** no dependency override, `--no-deps` install, unsupported upstream patch or second training environment is used to manufacture success

### Requirement: Consumer backend ownership remains intact

NexuML's simulator extras SHALL NOT select a consumer's CUDA/ROCm wheel family or require GPU rendering for state-only training. NexuVibe SHALL continue selecting its existing mutually exclusive backend. Installation evidence SHALL identify the chosen backend separately from physics and rendering capability.

#### Scenario: CPU training is selected
- **WHEN** the consumer selects its CPU backend and rendering is disabled
- **THEN** the references train without requiring CUDA, GPU graphics or a display

#### Scenario: NVIDIA recording is selected
- **WHEN** the consumer selects its supported NVIDIA backend and EGL recording
- **THEN** headless graphics and actual frame rendering are verified independently of Torch's compute visibility
- **AND** toolkit version alone does not choose or replace the installed PyTorch wheel family

### Requirement: Ordinary installations retain optional boundaries

Installing NexuML without the simulator extras SHALL NOT require MuJoCo, Meta-World, rendering or system middleware. Base import, supervised operation and static scenario discovery SHALL remain usable without simulator imports, while selected native execution SHALL report focused missing-prerequisite diagnostics.

#### Scenario: Simulation packages are unavailable
- **WHEN** a base user imports NexuML, uses a supervised reference or discovers the library
- **THEN** simulation and benchmark imports are not required
- **AND** invoking a missing MuJoCo or Meta-World runtime names the appropriate install extra without silently substituting another environment

### Requirement: Shared environment preparation remains user controlled

This repository-local change SHALL document the compatible interpreter, system graphics prerequisites and shared-resolution checks without editing sibling repositories or replacing the user's active environment. User-owned setup and graphics access SHALL be explicit completion prerequisites; incomplete preparation SHALL remain a visible blocker.

#### Scenario: NexuVibe still requires Python 3.14
- **WHEN** its consumer manifest/interpreter have not yet been changed by the user to a supported combination
- **THEN** the implementation reports the required interpreter/consumer preparation and leaves shared-environment acceptance incomplete
- **AND** it does not write NexuVibe files, rebuild the DevContainer or introduce an isolated MuJoCo environment

#### Scenario: Documentation describes installation
- **WHEN** users follow the new workflow instructions
- **THEN** instructions distinguish normal optional package installation from the shared development workflow and identify the authoritative consumer lock
- **AND** setup does not silently upgrade or rewrite the selected shared lock
