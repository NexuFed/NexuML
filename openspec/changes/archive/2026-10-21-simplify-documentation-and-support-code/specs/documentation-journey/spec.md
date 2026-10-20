## ADDED Requirements

### Requirement: Single authoritative procedure per task

Human-facing documentation SHALL keep each complete procedure in one canonical location. Entry pages SHALL provide a short working path and links, not duplicate advanced instructions, implementation inventories, or verification histories.

#### Scenario: User starts Studio

- **WHEN** a user opens the Studio README
- **THEN** they can find installation, launch, one first-run example, and links to further help without reading hosting, API internals, or release engineering instructions

#### Scenario: User needs advanced details

- **WHEN** a user needs hosting, manual API access, or contributor setup
- **THEN** the entry guide links to the authoritative procedure
- **AND** authentication, trusted-code, filesystem, and process-ownership limits remain available with the relevant procedure

#### Scenario: User consults API documentation

- **WHEN** a reader needs REST request or response details
- **THEN** the guide points to the existing authenticated generated contract
- **AND** the handwritten reference retains necessary WebSocket, conflict, and lifecycle rules without repeating the UI manual

### Requirement: Documentation without hard-coded version numbers

Human-facing documentation SHALL omit explicit product, runtime, dependency, operating-system, and protocol compatibility numbers from prose, badges, and installation examples. Machine-readable metadata, executable contract values, and generated references SHALL remain authoritative and unchanged by this editorial rule.

#### Scenario: Maintainer reviews documentation

- **WHEN** root/package READMEs, authored site pages, and skill bodies/references are reviewed
- **THEN** they contain no hard-coded release numbers, minimum-version ranges, numbered compatibility blocks, pinned image tags, or wheel/tarball release filenames
- **AND** technical metadata in skill frontmatter and example package manifests is not removed

#### Scenario: An example needs an executable contract value

- **WHEN** a working example contains a registered component identity, required configuration version field, or existing endpoint path
- **THEN** its executable meaning is preserved rather than deleting required fields or changing the protocol
- **AND** IP addresses, ports, numerical parameters, release-tag placeholders, and dates are not misclassified as release-number claims

#### Scenario: Reader selects a container image

- **WHEN** the container instructions require an explicit image tag
- **THEN** they show how to select and supply a tag from the registry without hard-coding a release or inventing an unsupported default tag
