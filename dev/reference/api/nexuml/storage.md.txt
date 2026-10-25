# nexuml.storage

*module*

Storage helpers for remote dataset transport.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/storage/__init__.py#L1-L1)

## nexuml.storage.S3Client

*class*

```python
class S3Client
```

Re-exported from: `nexuml.storage.s3`

Lazy boto3 wrapper for the few operations NexuML actually needs.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/storage/s3.py#L50-L134)

### nexuml.storage.S3Client.endpoint_url

*attribute* · *instance attribute*

```python
endpoint_url = endpoint_url
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/storage/s3.py#L61-L61)

### nexuml.storage.S3Client.profile

*attribute* · *instance attribute*

```python
profile = profile
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/storage/s3.py#L63-L63)

### nexuml.storage.S3Client.region

*attribute* · *instance attribute*

```python
region = region
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/storage/s3.py#L62-L62)

### nexuml.storage.S3Client.download_file

*method*

```python
def download_file(source: str | S3Path, destination: str | Path) -> None
```

Download one object to a local path.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/storage/s3.py#L116-L119)

### nexuml.storage.S3Client.list

*method*

```python
def list(prefix: str | S3Path) -> builtins.list[S3Path]
```

List object paths below an S3 prefix.

**Returns**

- (`builtins.list[S3Path]`) — S3 paths in provider order.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/storage/s3.py#L121-L134)

### nexuml.storage.S3Client.read_bytes

*method*

```python
def read_bytes(uri: str | S3Path) -> bytes
```

Read one object into memory.

**Returns**

- (`bytes`) — Object bytes.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/storage/s3.py#L93-L101)

### nexuml.storage.S3Client.upload_bytes

*method*

```python
def upload_bytes(data: bytes, destination: str | S3Path, *, exclusive: bool = False) -> None
```

Upload one in-memory object to S3.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/storage/s3.py#L108-L114)

### nexuml.storage.S3Client.upload_file

*method*

```python
def upload_file(source: str | Path, destination: str | S3Path) -> None
```

Upload one local file to S3.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/storage/s3.py#L103-L106)

## nexuml.storage.S3Path

*class* · *dataclass*

```python
class S3Path
```

Re-exported from: `nexuml.storage.s3`

Minimal path-like wrapper around an ``s3://`` object key.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/storage/s3.py#L13-L47)

### nexuml.storage.S3Path.bucket

*attribute* · *instance attribute*

```python
bucket: str
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/storage/s3.py#L17-L17)

### nexuml.storage.S3Path.key

*attribute* · *class attribute* · *instance attribute*

```python
key: str = ''
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/storage/s3.py#L18-L18)

### nexuml.storage.S3Path.suffix

*property*

```python
suffix: str
```

Object-key suffix.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/storage/s3.py#L41-L43)

### nexuml.storage.S3Path.parse

*method* · *classmethod*

```python
def parse(value: str) -> 'S3Path'
```

Parse one S3 URI.

**Returns**

- (`'S3Path'`) — Parsed S3 path.

**Raises**

- `ValueError` — If *value* is not a plain ``s3://bucket/key`` URI.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/storage/s3.py#L20-L33)

## nexuml.storage.is_s3_uri

*function*

```python
def is_s3_uri(value: object) -> bool
```

Re-exported from: `nexuml.storage.s3`

Return whether *value* is an S3 URI.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/storage/s3.py#L137-L139)

**Modules**

- `nexuml.storage.s3`
