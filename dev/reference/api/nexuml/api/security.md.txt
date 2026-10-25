# nexuml.api.security

*module*

Loopback trust boundary shared by HTTP and WebSocket transports.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/security.py#L1-L1)

## nexuml.api.security.INTERFACE_VERSION

*attribute* · *module attribute*

```python
INTERFACE_VERSION = 3
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/security.py#L16-L16)

## nexuml.api.security.LOOPBACK_HOSTS

*attribute* · *module attribute*

```python
LOOPBACK_HOSTS = {'127.0.0.1', 'localhost', '::1'}
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/security.py#L18-L18)

## nexuml.api.security.MAX_BODY

*attribute* · *module attribute*

```python
MAX_BODY = 2 * 1024 * 1024
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/security.py#L17-L17)

## nexuml.api.security.Settings

*class* · *dataclass*

```python
class Settings
```

Explicit selected-runtime settings; credentials are never serialized.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/security.py#L37-L87)

### nexuml.api.security.Settings.directory

*attribute* · *instance attribute*

```python
directory: Path
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/security.py#L41-L41)

### nexuml.api.security.Settings.origin

*attribute* · *instance attribute*

```python
origin: str
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/security.py#L42-L42)

### nexuml.api.security.Settings.token

*attribute* · *instance attribute*

```python
token: str
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/security.py#L43-L43)

### nexuml.api.security.Settings.matches

*method*

```python
def matches(value: str) -> bool
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/security.py#L70-L71)

### nexuml.api.security.Settings.permits

*method*

```python
def permits(headers) -> bool
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/security.py#L73-L87)

## nexuml.api.security.TransportSecurity

*class*

```python
class TransportSecurity
```

Authenticate before routing, including docs and WebSocket upgrades.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/security.py#L90-L152)

### nexuml.api.security.TransportSecurity.app

*attribute* · *instance attribute*

```python
app = app
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/security.py#L94-L94)

### nexuml.api.security.TransportSecurity.settings

*attribute* · *instance attribute*

```python
settings = settings
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/security.py#L95-L95)

### nexuml.api.security.TransportSecurity.reject

*method* · *async* · *staticmethod*

```python
async def reject(scope, receive, send, status, code, message)
```

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/security.py#L149-L152)

## nexuml.api.security.authenticate_websocket

*function* · *async*

```python
async def authenticate_websocket(socket: WebSocket, settings: Settings) -> bool
```

Accept only a bounded, timely first authentication frame.

**Returns**

- (`bool`) — Whether authentication succeeded; failures close the socket without data.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/security.py#L155-L177)

## nexuml.api.security.read_token

*function*

```python
def read_token(path: Path | None) -> str
```

Read an explicit private file or the runtime environment.

**Returns**

- (`str`) — Private token, never included in errors.

**Raises**

- `ValueError` — If a token file is not private on POSIX.

[View source](https://github.com/NexuFed/NexuML/blob/f54bedddeadd6bea3168c52441dc09bb2d77a22b/src/nexuml/api/security.py#L21-L34)
