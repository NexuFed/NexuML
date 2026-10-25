# nexuml.api.files

*module*

Authorized transport file access, separate from NexuML dataset path policies.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/files.py#L1-L1)

## nexuml.api.files.Files

*class*

```python
class Files
```

One selected directory boundary and serialized revision-checked writes.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/files.py#L12-L73)

### nexuml.api.files.Files.directory

*attribute* · *instance attribute*

```python
directory = directory
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/files.py#L16-L16)

### nexuml.api.files.Files.lock

*attribute* · *instance attribute*

```python
lock = threading.Lock()
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/files.py#L17-L17)

### nexuml.api.files.Files.path

*method*

```python
def path(value: str) -> Path
```

Resolve a path and reject traversal/symlink escapes.

**Returns**

- (`Path`) — Canonical path within the selected working directory.

**Raises**

- `ApiError` — When a path escapes the authorized directory.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/files.py#L19-L34)

### nexuml.api.files.Files.revision

*method* · *staticmethod*

```python
def revision(path: Path) -> str | None
```

Hash file bytes for conflict detection.

**Returns**

- (`str | None`) — Current revision, or None for a new file.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/files.py#L36-L43)

### nexuml.api.files.Files.save

*method*

```python
def save(path: Path, text: str, base_revision: str | None) -> str
```

Atomically save only against the supplied current revision.

**Returns**

- (`str`) — Revision of the saved file.

**Raises**

- `ApiError` — When the source changed after load.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/files.py#L45-L73)
