# dataset-downloads Specification

## Purpose
TBD - created by archiving change normalize-dataset-roots-downloads. Update Purpose after archive.
## Requirements
### Requirement: AudioSet Hugging Face download
The system SHALL support opt-in AudioSet download from Hugging Face dataset `agkphysics/AudioSet` and SHALL NOT use YouTube or `yt-dlp` for this download path.

#### Scenario: AudioSet download requested
- **WHEN** AudioSet data is missing and the user enables AudioSet download
- **THEN** the system downloads the configured AudioSet splits from Hugging Face instead of invoking YouTube or `yt-dlp`

#### Scenario: AudioSet download not requested
- **WHEN** AudioSet data is missing and download is disabled
- **THEN** the system fails with an actionable message instead of starting a large download automatically

### Requirement: AudioSet Hugging Face layout
The system SHALL store Hugging Face-downloaded AudioSet data under the resolved root `audioset_hf/full` with split folders `bal_train`, `eval`, and `unbal_train`, plus metadata required by the AudioSet dataset loader.

#### Scenario: AudioSet root under global data root
- **WHEN** `NEXUML_DATA_ROOT=/mnt/local` and AudioSet uses its default root
- **THEN** the AudioSet data builder targets `/mnt/local/audioset_hf/full`

#### Scenario: AudioSet split folders are available
- **WHEN** the Hugging Face AudioSet download completes
- **THEN** the local dataset root contains `bal_train`, `eval`, `unbal_train`, and metadata usable by `AudiosetDataset`

### Requirement: AudioSet local layout validation
The system SHALL validate that `AudiosetDataset` can initialize and read the current Hugging Face-derived local layout, tolerating corrupt media according to its loader behavior.

#### Scenario: Existing AudioSet layout is usable
- **WHEN** the user has an AudioSet root containing `bal_train`, `eval`, `metadata`, and `unbal_train`
- **THEN** the AudioSet data builder and dataset class can construct fit and test datasets without relying on YouTube-derived folder layouts

#### Scenario: Corrupt AudioSet media exists
- **WHEN** individual AudioSet media files are corrupt or unreadable
- **THEN** the loader handles them according to existing corrupt-file tolerance without failing dataset discovery or initialization

