# nexuml.api.config_routes

*module*

HTTP presentation of existing discovery and configuration boundaries.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/config_routes.py#L1-L1)

## nexuml.api.config_routes.ConfigInput

*class* · *pydantic model*

```python
class ConfigInput(BaseModel)
```

Bases: `BaseModel`

Editable portable config; stage_order is transport metadata only.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/config_routes.py#L16-L22)

### nexuml.api.config_routes.ConfigInput.data

*attribute* · *pydantic field*

```python
data: dict[str, Any] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/config_routes.py#L20-L20)

### nexuml.api.config_routes.ConfigInput.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/config_routes.py#L19-L19)

### nexuml.api.config_routes.ConfigInput.stage_order

*attribute* · *pydantic field*

```python
stage_order: list[str] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/config_routes.py#L22-L22)

### nexuml.api.config_routes.ConfigInput.yaml

*attribute* · *pydantic field*

```python
yaml: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/config_routes.py#L21-L21)

## nexuml.api.config_routes.LayoutInput

*class* · *pydantic model*

```python
class LayoutInput(PathInput)
```

Bases: `nexuml.api.config_routes.PathInput`

Visual metadata remains separate from executable configuration.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/config_routes.py#L39-L44)

### nexuml.api.config_routes.LayoutInput.base_revision

*attribute* · *pydantic field*

```python
base_revision: str | None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/config_routes.py#L42-L42)

### nexuml.api.config_routes.LayoutInput.layout

*attribute* · *pydantic field*

```python
layout: dict[str, Any]
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/config_routes.py#L44-L44)

### nexuml.api.config_routes.LayoutInput.semantic_revision

*attribute* · *pydantic field*

```python
semantic_revision: str
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/config_routes.py#L43-L43)

### nexuml.api.config_routes.LayoutInput.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.api.config_routes.PathInput`

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/config_routes.py#L35-L35)

### nexuml.api.config_routes.LayoutInput.path

*attribute* · *pydantic field*

```python
path: str
```

Inherited from: `nexuml.api.config_routes.PathInput`

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/config_routes.py#L36-L36)

## nexuml.api.config_routes.PathInput

*class* · *pydantic model*

```python
class PathInput(BaseModel)
```

Bases: `BaseModel`

Explicit authorized path, not arbitrary filesystem browsing.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/config_routes.py#L32-L36)

### nexuml.api.config_routes.PathInput.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/config_routes.py#L35-L35)

### nexuml.api.config_routes.PathInput.path

*attribute* · *pydantic field*

```python
path: str
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/config_routes.py#L36-L36)

## nexuml.api.config_routes.SaveInput

*class* · *pydantic model*

```python
class SaveInput(ConfigInput)
```

Bases: `nexuml.api.config_routes.ConfigInput`

Revision-protected file save; new files require a null base_revision.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/config_routes.py#L25-L29)

### nexuml.api.config_routes.SaveInput.base_revision

*attribute* · *pydantic field*

```python
base_revision: str | None
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/config_routes.py#L29-L29)

### nexuml.api.config_routes.SaveInput.path

*attribute* · *pydantic field*

```python
path: str
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/config_routes.py#L28-L28)

### nexuml.api.config_routes.SaveInput.data

*attribute* · *pydantic field*

```python
data: dict[str, Any] | None = None
```

Inherited from: `nexuml.api.config_routes.ConfigInput`

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/config_routes.py#L20-L20)

### nexuml.api.config_routes.SaveInput.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid')
```

Inherited from: `nexuml.api.config_routes.ConfigInput`

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/config_routes.py#L19-L19)

### nexuml.api.config_routes.SaveInput.stage_order

*attribute* · *pydantic field*

```python
stage_order: list[str] | None = None
```

Inherited from: `nexuml.api.config_routes.ConfigInput`

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/config_routes.py#L22-L22)

### nexuml.api.config_routes.SaveInput.yaml

*attribute* · *pydantic field*

```python
yaml: str | None = None
```

Inherited from: `nexuml.api.config_routes.ConfigInput`

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/config_routes.py#L21-L21)

## nexuml.api.config_routes.config_router

*function*

```python
def config_router(settings: Settings) -> APIRouter
```

Register thin synchronous routes (FastAPI runs them outside its event loop).

**Returns**

- (`APIRouter`) — Router over the selected working directory and existing framework calls.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/config_routes.py#L47-L152)
