# S3 WebDataset and DALI

## Purpose

Provide the shared-data path required by Ray workers using NexuML's existing WebDataset export and DALI loading abstractions.

## ADDED Requirements

### Requirement: WebDataset stores canonical tensor payloads as NumPy
The default WebDataset exporter SHALL serialize tensor components as `.npy` with `allow_pickle=False`. It SHALL preserve tensor shape and dtype and SHALL NOT reconstruct PNG/WAV/MP4 media from already-decoded tensors in the generic export path.

#### Scenario: Tensor data is exported
- **WHEN** a TensorDict component is written to WebDataset
- **THEN** the tar member contains a NumPy `.npy` payload preserving shape and dtype without pickle

### Requirement: Every tar shard has a DALI index
The WebDataset exporter SHALL generate one sibling `.idx` for every closed tar shard by invoking the `wds2idx` utility distributed with DALI. NexuML SHALL NOT maintain its own DALI `.idx` format, compatibility profile, or index-version implementation.

#### Scenario: A shard closes
- **WHEN** `shard-000123.tar` is closed
- **THEN** `wds2idx` produces `shard-000123.idx` before the shard is considered complete or uploaded

#### Scenario: DALI index tooling is unavailable
- **WHEN** index generation is enabled and `wds2idx` is not installed
- **THEN** export fails with a concise error identifying the DALI extra/tool requirement rather than silently omitting the index

### Requirement: S3 export is streaming and credential-free in config
WebDataset export SHALL accept S3-compatible object storage and upload each closed `.tar`/`.idx` pair before proceeding, so local disk usage is bounded by the current shard. S3 endpoint/region/profile MAY be configured, but credentials SHALL remain in the standard boto3/AWS/provider identity chain and SHALL NOT be serialized in scenario configuration.

#### Scenario: Large dataset is exported to S3
- **WHEN** multiple shards are produced for an S3 destination
- **THEN** each closed tar/index pair is uploaded and its local temporary files are removed before later shards complete

### Requirement: Export metadata remains simple
The existing export `config.yaml` and metadata table SHALL remain the dataset metadata authority. S3 WebDataset support SHALL record shard and index paths there and SHALL NOT introduce generation pointers, publication manifests, integrity-protocol hierarchies, or a generic storage abstraction.

#### Scenario: S3 export completes
- **WHEN** all shards are uploaded
- **THEN** `config.yaml` and metadata are uploaded to the dataset prefix and are sufficient to discover the shard/index pairs

### Requirement: DALI remains the loading and sharding authority
An S3-backed exported WebDataset SHALL expose its shard and index references as `s3://` URLs to the existing DALI WebDataset loader. The loader SHALL continue to use DALI `shard_id=global_rank` and `num_shards=world_size`; NexuML SHALL NOT implement a second Ray-specific data sharding path.

#### Scenario: Four Ray workers load one S3 dataset
- **WHEN** the workers have ranks 0..3 and world size 4
- **THEN** each worker builds the existing DALI WebDataset reader with its own rank as `shard_id` and 4 as `num_shards`

### Requirement: Direct DALI S3 support is tested, not abstracted preemptively
The implementation SHALL pass `s3://` tar/index paths directly to DALI and SHALL include an optional real S3-compatible DALI integration test. NEX-154 SHALL NOT add a direct/cache capability framework or node-cache subsystem before a concrete target-environment limitation demonstrates that it is required.

#### Scenario: Target DALI environment supports direct S3 WebDataset paths
- **WHEN** the optional integration environment is available
- **THEN** DALI reads the exported S3 tar/index pair directly and yields the expected tensors

#### Scenario: Target DALI environment rejects direct S3 WebDataset paths
- **WHEN** the real integration test demonstrates that the selected DALI/object-store combination cannot consume the S3 tar/index paths
- **THEN** the limitation is reported explicitly and any follow-up adapter is designed from that measured failure rather than activating a prebuilt generic cache framework
