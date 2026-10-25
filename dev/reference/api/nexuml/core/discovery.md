# nexuml.core.discovery

*module*

Core discovery module with decorator-based registration and scanning.

Provides decorators (@layer, @data_source, @scenario, @eval_algorithm) that attach
metadata to objects, and a Scanner that collects decorated objects from imported
modules. Supports entry-point and local-root discovery.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L1-L1)

## nexuml.core.discovery.DEFAULT_CONFIG_DIR

*attribute* · *module attribute*

```python
DEFAULT_CONFIG_DIR = Path.home() / '.config' / 'nexuml'
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L390-L390)

## nexuml.core.discovery.DEFAULT_CONFIG_PATH

*attribute* · *module attribute*

```python
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / 'libraries.json'
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L391-L391)

## nexuml.core.discovery.DISCOVERED_ATTR

*attribute* · *module attribute*

```python
DISCOVERED_ATTR = '__nexuml_discovered__'
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L37-L37)

## nexuml.core.discovery.DefinitionT

*attribute* · *module attribute*

```python
DefinitionT = TypeVar('DefinitionT', bound=ComponentDefinition)
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L130-L130)

## nexuml.core.discovery.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L31-L31)

## nexuml.core.discovery.DiscoveredItem

*class*

```python
class DiscoveredItem
```

Metadata for a decorated discovery item.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L40-L63)

### nexuml.core.discovery.DiscoveredItem.key

*attribute* · *instance attribute*

```python
key = key
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L54-L54)

### nexuml.core.discovery.DiscoveredItem.kind

*attribute* · *instance attribute*

```python
kind = kind
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L53-L53)

### nexuml.core.discovery.DiscoveredItem.module

*attribute* · *instance attribute*

```python
module = module
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L56-L56)

### nexuml.core.discovery.DiscoveredItem.obj

*attribute* · *instance attribute*

```python
obj = obj
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L55-L55)

### nexuml.core.discovery.DiscoveredItem.origin

*attribute* · *instance attribute*

```python
origin = origin
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L57-L57)

## nexuml.core.discovery.DiscoveryError

*class*

```python
class DiscoveryError
```

A failure encountered while importing or registering a discovery item.

Collected instead of raised so that one broken module cannot hide every
other scenario/layer/dataset/eval algorithm in the registry.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L66-L101)

### nexuml.core.discovery.DiscoveryError.error_type

*attribute* · *instance attribute*

```python
error_type = error_type
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L86-L86)

### nexuml.core.discovery.DiscoveryError.key

*attribute* · *instance attribute*

```python
key = key
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L89-L89)

### nexuml.core.discovery.DiscoveryError.message

*attribute* · *instance attribute*

```python
message = message
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L87-L87)

### nexuml.core.discovery.DiscoveryError.module

*attribute* · *instance attribute*

```python
module = module
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L84-L84)

### nexuml.core.discovery.DiscoveryError.phase

*attribute* · *instance attribute*

```python
phase = phase
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L85-L85)

### nexuml.core.discovery.DiscoveryError.traceback

*attribute* · *instance attribute*

```python
traceback = traceback
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L88-L88)

### nexuml.core.discovery.DiscoveryError.short

*method*

```python
def short() -> str
```

One-line, human-readable summary for tables/log lines.

**Returns**

- (`str`) — Compact ``module (key): error_type: message`` string.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L91-L98)

## nexuml.core.discovery.LibraryConfig

*class*

```python
class LibraryConfig
```

User-level configuration for local library roots.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L394-L433)

### nexuml.core.discovery.LibraryConfig.roots

*attribute* · *instance attribute*

```python
roots = list(roots or [])
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L400-L400)

### nexuml.core.discovery.LibraryConfig.add_root

*method*

```python
def add_root(root: str) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L425-L428)

### nexuml.core.discovery.LibraryConfig.load

*method* · *classmethod*

```python
def load(path: Path | None = None) -> 'LibraryConfig'
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L402-L415)

### nexuml.core.discovery.LibraryConfig.remove_root

*method*

```python
def remove_root(root: str) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L430-L433)

### nexuml.core.discovery.LibraryConfig.save

*method*

```python
def save(path: Path | None = None) -> None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L417-L420)

## nexuml.core.discovery.Scanner

*class*

```python
class Scanner
```

Collects decorated objects from imported modules.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L244-L355)

### nexuml.core.discovery.Scanner.errors

*property*

```python
errors: list[DiscoveryError]
```

Import/walk failures collected during scanning.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L350-L352)

### nexuml.core.discovery.Scanner.items

*property*

```python
items: list[DiscoveredItem]
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L346-L347)

### nexuml.core.discovery.Scanner.by_kind

*method*

```python
def by_kind(kind: str) -> list[DiscoveredItem]
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L354-L355)

### nexuml.core.discovery.Scanner.scan_module

*method*

```python
def scan_module(module: Any) -> list[DiscoveredItem]
```

Scan a single already-imported module for decorated objects.

**Returns**

- (`list[DiscoveredItem]`) — Newly discovered items found in *module*.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L269-L291)

### nexuml.core.discovery.Scanner.scan_package

*method*

```python
def scan_package(package_path: str) -> list[DiscoveredItem]
```

Import a package (if possible) and recursively scan its modules.

Resilient by design: a module that fails to import (SyntaxError,
NameError, validation errors at import time, etc.) is recorded as a
:class:`DiscoveryError` and skipped, so the remaining modules in the
package are still discovered. Inspect :attr:`errors` afterwards to see
what was skipped and why.

**Returns**

- (`list[DiscoveredItem]`) — Newly discovered items across all modules in the package.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L293-L332)

### nexuml.core.discovery.Scanner.scan_packages

*method*

```python
def scan_packages(package_paths: list[str]) -> list[DiscoveredItem]
```

Scan multiple package paths.

**Returns**

- (`list[DiscoveredItem]`) — Combined list of discovered items from all packages.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L334-L343)

## nexuml.core.discovery.data_source

*function*

```python
def data_source(key: str, *, version: str = '1') -> Callable[[type[DefinitionT]], type[DefinitionT]]
```

Register a typed data-source definition.

**Returns**

- (`Callable[[type[DefinitionT]], type[DefinitionT]]`) — Definition class decorator.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L164-L174)

## nexuml.core.discovery.discover_entry_point_packages

*function*

```python
def discover_entry_point_packages(group: str = 'nexuml.libraries') -> list[str]
```

Return package names advertised via importlib.metadata entry points.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L363-L382)

## nexuml.core.discovery.discover_library_packages

*function*

```python
def discover_library_packages(include_entry_points: bool = True, include_local_roots: bool = True) -> list[str]
```

Return package paths from installed entry points and configured roots.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L488-L499)

## nexuml.core.discovery.discover_local_packages

*function*

```python
def discover_local_packages(config: LibraryConfig | None = None) -> list[str]
```

Discover all importable package paths under configured local roots.

**Returns**

- (`list[str]`) — Dotted package path strings discovered under all configured roots.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L475-L485)

## nexuml.core.discovery.eval_algorithm

*function*

```python
def eval_algorithm(key: str, *, version: str = '1') -> Callable[[type[DefinitionT]], type[DefinitionT]]
```

Register a typed evaluation-algorithm definition.

**Returns**

- (`Callable[[type[DefinitionT]], type[DefinitionT]]`) — Definition class decorator.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L191-L204)

## nexuml.core.discovery.execution_backend

*function*

```python
def execution_backend(key: str, *, version: str = '1') -> Callable[[type[DefinitionT]], type[DefinitionT]]
```

Register installation-owned execution settings and behavior.

**Returns**

- (`Callable[[type[DefinitionT]], type[DefinitionT]]`) — Definition class decorator.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L223-L236)

## nexuml.core.discovery.layer

*function*

```python
def layer(key: str, *, version: str = '1') -> Callable[[type[DefinitionT]], type[DefinitionT]]
```

Register a typed layer definition under a stable persisted identity.

**Returns**

- (`Callable[[type[DefinitionT]], type[DefinitionT]]`) — Definition class decorator.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L155-L161)

## nexuml.core.discovery.loader_backend

*function*

```python
def loader_backend(key: str, *, version: str = '1') -> Callable[[type[DefinitionT]], type[DefinitionT]]
```

Register a typed loader-backend definition.

**Returns**

- (`Callable[[type[DefinitionT]], type[DefinitionT]]`) — Definition class decorator.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L207-L220)

## nexuml.core.discovery.register_items

*function*

```python
def register_items(items: list[DiscoveredItem], register: Callable[[str, Any], None], errors: list[DiscoveryError]) -> None
```

Register each discovered item, collecting (not raising) failures.

Mirrors the resilience of :meth:`Scanner.scan_package`: a single key
conflict or bad item must not wipe out every other item in the registry.
Appends a :class:`DiscoveryError` (phase="register") for each failure.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L507-L538)

## nexuml.core.discovery.scan_all

*function*

```python
def scan_all(extra_package_paths: list[str] | None = None, include_entry_points: bool = True, include_local_roots: bool = True) -> Scanner
```

Scan built-in packages, entry points, and local roots for decorated items.

No persistent object cache is used; this always performs fresh imports.

**Returns**

- (`Scanner`) — Scanner populated with all discovered items.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L541-L562)

## nexuml.core.discovery.scenario

*function*

```python
def scenario(key: str) -> Callable[[Any], Any]
```

Decorator to register a scenario function.

**Returns**

- (`Callable[[Any], Any]`) — Decorator that attaches discovery metadata to *fn*.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/discovery.py#L177-L188)
