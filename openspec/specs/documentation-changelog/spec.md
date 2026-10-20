# Spec: documentation-changelog

## Purpose

Provide a curated, chronological record of NexuML updates that publishes independently of immutable documentation releases and remains easy to author as individual MDX entries.

## Requirements

### Requirement: Validated editorial update entries
The changelog SHALL accept committed MDX entries with validated publication date, title, category, summary, optional release version, links, and optional media. Invalid entries SHALL fail documentation validation before publication.

#### Scenario: Contributor adds a valid entry
- **WHEN** a contributor adds one entry following the documented template
- **THEN** validation accepts its metadata and rich MDX body without requiring edits to an index, sidebar, or route list

#### Scenario: Entry metadata is invalid
- **WHEN** an entry has a missing required field, invalid date/category, unsafe link, conflicting route, or nonexistent local media asset
- **THEN** validation fails with the offending entry and field or asset identified

#### Scenario: Entry describes an identified release
- **WHEN** an entry is categorized as a release announcement
- **THEN** it provides the exact release identifier and a matching tag or GitHub Release link
- **AND** an ordinary product update is not required to invent a release version

### Requirement: Automatically ordered changelog routes
The site SHALL generate a global changelog index and stable dated entry pages from the entries. The index SHALL order entries newest first, with deterministic ordering for equal dates, and derive any date/version grouping from metadata rather than maintained lists.

#### Scenario: New entry changes the index
- **WHEN** a newer valid entry is added and the site is rebuilt
- **THEN** the index places it before older entries and links to its detail page
- **AND** previously published entry URLs remain unchanged

#### Scenario: Entries share a date
- **WHEN** multiple entries have the same publication date
- **THEN** successive builds use the same documented tie-break order

#### Scenario: Reader opens an entry directly
- **WHEN** a reader opens a dated entry URL under the configured site mount
- **THEN** the page shows its title, date, category, summary, rich body, and available release links
- **AND** canonical URLs, navigation, local media, and search work under that mount

### Requirement: Rich accessible changelog content
Changelog entries SHALL support formatted prose, code, supported MDX components, screenshots, and short video content within the documentation branding and accessible navigation.

#### Scenario: Entry contains rich media
- **WHEN** a reader opens an entry containing a screenshot and a short video
- **THEN** the image has meaningful alternative text and the video has keyboard-operable controls and appropriate captions or a text alternative
- **AND** the media works on desktop and mobile without obscuring the article or forcing horizontal page scrolling

### Requirement: Main-published global changelog
The global `/changelog/` section SHALL publish from validated main commits independently of versioned documentation. Release-tag publication SHALL NOT replace the global changelog with the tag's older editorial content.

#### Scenario: Main adds an editorial update without a release
- **WHEN** a valid changelog entry is merged to main and publication succeeds
- **THEN** the entry appears at the global changelog URL without requiring a new release tag or changing `latest`
- **AND** existing release snapshots remain unchanged

#### Scenario: Release snapshot is published later
- **WHEN** a release tag whose source predates newer changelog entries is published
- **THEN** those newer global entries and their media remain available and unchanged

#### Scenario: Reader enters from versioned documentation
- **WHEN** a reader selects Changelog from a new versioned documentation site
- **THEN** the link opens the global changelog rather than a release-frozen or duplicated timeline
- **AND** the global changelog exposes a clear link back to documentation and version selection

### Requirement: Curated updates coexist with GitHub Releases
The project SHALL retain automated categorized GitHub Release notes while maintaining changelog prose as reviewed, committed editorial content. Contributor instructions SHALL include a sample/template, metadata rules, preview/validation commands, and the main-publication procedure.

#### Scenario: Maintainer publishes a GitHub Release
- **WHEN** a maintainer uses the existing automated release-note generation
- **THEN** the existing categories and exclusions still apply
- **AND** changelog links can identify that release without replacing the editorial body with raw commits or pull requests

#### Scenario: Contributor follows the update guide
- **WHEN** a contributor follows the documented publishing procedure
- **THEN** they can add one MDX entry, preview and validate it, and publish it through the normal reviewed main workflow
- **AND** templates/test examples are not accidentally published as real product announcements
