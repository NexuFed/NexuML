# nexuml.execution.definitions

*module*

Core execution definitions; importing settings never imports cluster runtimes.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L1-L1)

## nexuml.execution.definitions.ConfigHandoff

*class* · *pydantic model*

```python
class ConfigHandoff(BaseModel)
```

Bases: `BaseModel`

Explicit shared filesystem or S3 destination for one frozen worker config.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L182-L208)

### nexuml.execution.definitions.ConfigHandoff.destination

*attribute* · *pydantic field*

```python
destination: str
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L186-L186)

### nexuml.execution.definitions.ConfigHandoff.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid', frozen=True)
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L185-L185)

### nexuml.execution.definitions.ConfigHandoff.worker_directory

*attribute* · *pydantic field*

```python
worker_directory: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L187-L187)

### nexuml.execution.definitions.ConfigHandoff.validate_destination

*method* · *pydantic validator*

```python
def validate_destination(value: str | None) -> str | None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L189-L208)

## nexuml.execution.definitions.KubernetesExecution

*class* · *pydantic model*

```python
class KubernetesExecution(ExecutionBackendDefinition)
```

Bases: `nexuml.core.components.ExecutionBackendDefinition`

Infrastructure-owned template and named training resource bindings.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L211-L270)

### nexuml.execution.definitions.KubernetesExecution.api_version

*attribute* · *class attribute*

```python
api_version: str
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L221-L221)

### nexuml.execution.definitions.KubernetesExecution.capabilities

*attribute* · *pydantic field*

```python
capabilities = {'status': True, 'logs': True, 'metrics': False, 'cancellation': True, 'resume': False, 'artifacts': False, 'process_ownership': False, 'native_reference': True}
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L224-L233)

### nexuml.execution.definitions.KubernetesExecution.container

*attribute* · *pydantic field*

```python
container: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L240-L240)

### nexuml.execution.definitions.KubernetesExecution.context

*attribute* · *pydantic field*

```python
context: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L234-L234)

### nexuml.execution.definitions.KubernetesExecution.dependencies

*attribute* · *pydantic field*

```python
dependencies = ('kubernetes')
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L214-L214)

### nexuml.execution.definitions.KubernetesExecution.handoff

*attribute* · *pydantic field*

```python
handoff: ConfigHandoff | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L239-L239)

### nexuml.execution.definitions.KubernetesExecution.namespace

*attribute* · *pydantic field*

```python
namespace: str = 'default'
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L235-L237)

### nexuml.execution.definitions.KubernetesExecution.plural

*attribute* · *class attribute*

```python
plural: str
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L223-L223)

### nexuml.execution.definitions.KubernetesExecution.presentation

*attribute* · *pydantic field*

```python
presentation = {'target': ['context', 'namespace'], 'setup': ['template', 'handoff'], 'resources': ['worker_group', 'container', 'replicas', 'resources'], 'expert': ['ray']}
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L215-L220)

### nexuml.execution.definitions.KubernetesExecution.replicas

*attribute* · *pydantic field*

```python
replicas: int | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L242-L242)

### nexuml.execution.definitions.KubernetesExecution.resource_kind

*attribute* · *class attribute*

```python
resource_kind: str
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L222-L222)

### nexuml.execution.definitions.KubernetesExecution.resources

*attribute* · *pydantic field*

```python
resources: dict[str, str]
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L243-L243)

### nexuml.execution.definitions.KubernetesExecution.template

*attribute* · *pydantic field*

```python
template: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L238-L238)

### nexuml.execution.definitions.KubernetesExecution.worker_group

*attribute* · *pydantic field*

```python
worker_group: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L241-L241)

### nexuml.execution.definitions.KubernetesExecution.kind

*attribute* · *pydantic field*

```python
kind = 'execution_backend'
```

Inherited from: `nexuml.core.components.ExecutionBackendDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L169-L169)

### nexuml.execution.definitions.KubernetesExecution.label

*attribute* · *class attribute*

```python
label: str = ''
```

Inherited from: `nexuml.core.components.ExecutionBackendDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L170-L170)

### nexuml.execution.definitions.KubernetesExecution.component_name

*attribute* · *class attribute*

```python
component_name: str
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L34-L34)

### nexuml.execution.definitions.KubernetesExecution.component_version

*attribute* · *class attribute*

```python
component_version: str = '1'
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L35-L35)

### nexuml.execution.definitions.KubernetesExecution.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid', frozen=True, validate_default=True, allow_inf_nan=False, arbitrary_types_allowed=False)
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L25-L31)

### nexuml.execution.definitions.KubernetesExecution.cancel

*method*

```python
def cancel(reference: Any) -> dict[str, Any]
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L267-L270)

### nexuml.execution.definitions.KubernetesExecution.discover

*method*

```python
def discover(**kwargs: Any) -> Any
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L257-L260)

### nexuml.execution.definitions.KubernetesExecution.inspect

*method*

```python
def inspect(reference: Any) -> dict[str, Any]
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L262-L265)

### nexuml.execution.definitions.KubernetesExecution.preflight

*method*

```python
def preflight(scenario: Any, **kwargs: Any) -> dict[str, Any]
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L245-L249)

### nexuml.execution.definitions.KubernetesExecution.run

*method*

```python
def run(scenario: Any, **kwargs: Any) -> Any
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L251-L255)

## nexuml.execution.definitions.KubernetesJobExecution

*class* · *pydantic model*

```python
class KubernetesJobExecution(KubernetesExecution)
```

Bases: `nexuml.execution.definitions.KubernetesExecution`

Submit one Kubernetes training pod.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L287-L294)

### nexuml.execution.definitions.KubernetesJobExecution.api_version

*attribute* · *pydantic field*

```python
api_version = 'batch/v1'
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L292-L292)

### nexuml.execution.definitions.KubernetesJobExecution.label

*attribute* · *pydantic field*

```python
label = 'Kubernetes Job'
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L291-L291)

### nexuml.execution.definitions.KubernetesJobExecution.plural

*attribute* · *pydantic field*

```python
plural = 'jobs'
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L294-L294)

### nexuml.execution.definitions.KubernetesJobExecution.resource_kind

*attribute* · *pydantic field*

```python
resource_kind = 'Job'
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L293-L293)

### nexuml.execution.definitions.KubernetesJobExecution.capabilities

*attribute* · *pydantic field*

```python
capabilities = {'status': True, 'logs': True, 'metrics': False, 'cancellation': True, 'resume': False, 'artifacts': False, 'process_ownership': False, 'native_reference': True}
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L224-L233)

### nexuml.execution.definitions.KubernetesJobExecution.container

*attribute* · *pydantic field*

```python
container: str | None = None
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L240-L240)

### nexuml.execution.definitions.KubernetesJobExecution.context

*attribute* · *pydantic field*

```python
context: str | None = None
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L234-L234)

### nexuml.execution.definitions.KubernetesJobExecution.dependencies

*attribute* · *pydantic field*

```python
dependencies = ('kubernetes')
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L214-L214)

### nexuml.execution.definitions.KubernetesJobExecution.handoff

*attribute* · *pydantic field*

```python
handoff: ConfigHandoff | None = None
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L239-L239)

### nexuml.execution.definitions.KubernetesJobExecution.namespace

*attribute* · *pydantic field*

```python
namespace: str = 'default'
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L235-L237)

### nexuml.execution.definitions.KubernetesJobExecution.presentation

*attribute* · *pydantic field*

```python
presentation = {'target': ['context', 'namespace'], 'setup': ['template', 'handoff'], 'resources': ['worker_group', 'container', 'replicas', 'resources'], 'expert': ['ray']}
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L215-L220)

### nexuml.execution.definitions.KubernetesJobExecution.replicas

*attribute* · *pydantic field*

```python
replicas: int | None = None
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L242-L242)

### nexuml.execution.definitions.KubernetesJobExecution.resources

*attribute* · *pydantic field*

```python
resources: dict[str, str]
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L243-L243)

### nexuml.execution.definitions.KubernetesJobExecution.template

*attribute* · *pydantic field*

```python
template: str | None = None
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L238-L238)

### nexuml.execution.definitions.KubernetesJobExecution.worker_group

*attribute* · *pydantic field*

```python
worker_group: str | None = None
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L241-L241)

### nexuml.execution.definitions.KubernetesJobExecution.kind

*attribute* · *pydantic field*

```python
kind = 'execution_backend'
```

Inherited from: `nexuml.core.components.ExecutionBackendDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L169-L169)

### nexuml.execution.definitions.KubernetesJobExecution.component_name

*attribute* · *class attribute*

```python
component_name: str
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L34-L34)

### nexuml.execution.definitions.KubernetesJobExecution.component_version

*attribute* · *class attribute*

```python
component_version: str = '1'
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L35-L35)

### nexuml.execution.definitions.KubernetesJobExecution.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid', frozen=True, validate_default=True, allow_inf_nan=False, arbitrary_types_allowed=False)
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L25-L31)

### nexuml.execution.definitions.KubernetesJobExecution.cancel

*method*

```python
def cancel(reference: Any) -> dict[str, Any]
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L267-L270)

### nexuml.execution.definitions.KubernetesJobExecution.discover

*method*

```python
def discover(**kwargs: Any) -> Any
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L257-L260)

### nexuml.execution.definitions.KubernetesJobExecution.inspect

*method*

```python
def inspect(reference: Any) -> dict[str, Any]
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L262-L265)

### nexuml.execution.definitions.KubernetesJobExecution.preflight

*method*

```python
def preflight(scenario: Any, **kwargs: Any) -> dict[str, Any]
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L245-L249)

### nexuml.execution.definitions.KubernetesJobExecution.run

*method*

```python
def run(scenario: Any, **kwargs: Any) -> Any
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L251-L255)

## nexuml.execution.definitions.LocalExecution

*class* · *pydantic model*

```python
class LocalExecution(ExecutionBackendDefinition)
```

Bases: `nexuml.core.components.ExecutionBackendDefinition`

Run the existing NexuML session in the current process.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L12-L48)

### nexuml.execution.definitions.LocalExecution.capabilities

*attribute* · *pydantic field*

```python
capabilities = {'status': True, 'logs': True, 'metrics': True, 'cancellation': True, 'resume': True, 'artifacts': True, 'process_ownership': True}
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L17-L25)

### nexuml.execution.definitions.LocalExecution.label

*attribute* · *pydantic field*

```python
label = 'Local'
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L16-L16)

### nexuml.execution.definitions.LocalExecution.dependencies

*attribute* · *class attribute*

```python
dependencies: tuple[str, ...] = ()
```

Inherited from: `nexuml.core.components.ExecutionBackendDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L171-L171)

### nexuml.execution.definitions.LocalExecution.kind

*attribute* · *pydantic field*

```python
kind = 'execution_backend'
```

Inherited from: `nexuml.core.components.ExecutionBackendDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L169-L169)

### nexuml.execution.definitions.LocalExecution.presentation

*attribute* · *class attribute*

```python
presentation: dict[str, list[str]] = {}
```

Inherited from: `nexuml.core.components.ExecutionBackendDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L173-L173)

### nexuml.execution.definitions.LocalExecution.component_name

*attribute* · *class attribute*

```python
component_name: str
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L34-L34)

### nexuml.execution.definitions.LocalExecution.component_version

*attribute* · *class attribute*

```python
component_version: str = '1'
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L35-L35)

### nexuml.execution.definitions.LocalExecution.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid', frozen=True, validate_default=True, allow_inf_nan=False, arbitrary_types_allowed=False)
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L25-L31)

### nexuml.execution.definitions.LocalExecution.discover

*method*

```python
def discover(**kwargs: Any) -> Any
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L45-L48)

### nexuml.execution.definitions.LocalExecution.run

*method*

```python
def run(scenario: Any, **kwargs: Any) -> Any
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L27-L43)

### nexuml.execution.definitions.LocalExecution.cancel

*method*

```python
def cancel(reference: Any) -> dict[str, Any]
```

Inherited from: `nexuml.core.components.ExecutionBackendDefinition`

Refuse to confuse submitter termination with native cancellation.

**Raises**

- `NotImplementedError` — When native cancellation cannot be confirmed.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L208-L214)

### nexuml.execution.definitions.LocalExecution.inspect

*method*

```python
def inspect(reference: Any) -> dict[str, Any]
```

Inherited from: `nexuml.core.components.ExecutionBackendDefinition`

Refuse native observation that the backend has not implemented.

**Raises**

- `NotImplementedError` — When no native inspection is available.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L200-L206)

### nexuml.execution.definitions.LocalExecution.preflight

*method*

```python
def preflight(scenario: Any, **kwargs: Any) -> dict[str, Any]
```

Inherited from: `nexuml.core.components.ExecutionBackendDefinition`

Validate a request without allocating workers.

**Returns**

- (`dict[str, Any]`) — Backend-specific review information.

**Raises**

- `ValueError` — If requested resume or exports are unsupported.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L175-L188)

## nexuml.execution.definitions.PyTorchJobExecution

*class* · *pydantic model*

```python
class PyTorchJobExecution(KubernetesExecution)
```

Bases: `nexuml.execution.definitions.KubernetesExecution`

Submit the served Kubeflow Training Operator PyTorchJob resource.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L297-L304)

### nexuml.execution.definitions.PyTorchJobExecution.api_version

*attribute* · *pydantic field*

```python
api_version = 'kubeflow.org/v1'
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L302-L302)

### nexuml.execution.definitions.PyTorchJobExecution.label

*attribute* · *pydantic field*

```python
label = 'Kubernetes PyTorchJob'
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L301-L301)

### nexuml.execution.definitions.PyTorchJobExecution.plural

*attribute* · *pydantic field*

```python
plural = 'pytorchjobs'
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L304-L304)

### nexuml.execution.definitions.PyTorchJobExecution.resource_kind

*attribute* · *pydantic field*

```python
resource_kind = 'PyTorchJob'
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L303-L303)

### nexuml.execution.definitions.PyTorchJobExecution.capabilities

*attribute* · *pydantic field*

```python
capabilities = {'status': True, 'logs': True, 'metrics': False, 'cancellation': True, 'resume': False, 'artifacts': False, 'process_ownership': False, 'native_reference': True}
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L224-L233)

### nexuml.execution.definitions.PyTorchJobExecution.container

*attribute* · *pydantic field*

```python
container: str | None = None
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L240-L240)

### nexuml.execution.definitions.PyTorchJobExecution.context

*attribute* · *pydantic field*

```python
context: str | None = None
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L234-L234)

### nexuml.execution.definitions.PyTorchJobExecution.dependencies

*attribute* · *pydantic field*

```python
dependencies = ('kubernetes')
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L214-L214)

### nexuml.execution.definitions.PyTorchJobExecution.handoff

*attribute* · *pydantic field*

```python
handoff: ConfigHandoff | None = None
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L239-L239)

### nexuml.execution.definitions.PyTorchJobExecution.namespace

*attribute* · *pydantic field*

```python
namespace: str = 'default'
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L235-L237)

### nexuml.execution.definitions.PyTorchJobExecution.presentation

*attribute* · *pydantic field*

```python
presentation = {'target': ['context', 'namespace'], 'setup': ['template', 'handoff'], 'resources': ['worker_group', 'container', 'replicas', 'resources'], 'expert': ['ray']}
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L215-L220)

### nexuml.execution.definitions.PyTorchJobExecution.replicas

*attribute* · *pydantic field*

```python
replicas: int | None = None
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L242-L242)

### nexuml.execution.definitions.PyTorchJobExecution.resources

*attribute* · *pydantic field*

```python
resources: dict[str, str]
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L243-L243)

### nexuml.execution.definitions.PyTorchJobExecution.template

*attribute* · *pydantic field*

```python
template: str | None = None
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L238-L238)

### nexuml.execution.definitions.PyTorchJobExecution.worker_group

*attribute* · *pydantic field*

```python
worker_group: str | None = None
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L241-L241)

### nexuml.execution.definitions.PyTorchJobExecution.kind

*attribute* · *pydantic field*

```python
kind = 'execution_backend'
```

Inherited from: `nexuml.core.components.ExecutionBackendDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L169-L169)

### nexuml.execution.definitions.PyTorchJobExecution.component_name

*attribute* · *class attribute*

```python
component_name: str
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L34-L34)

### nexuml.execution.definitions.PyTorchJobExecution.component_version

*attribute* · *class attribute*

```python
component_version: str = '1'
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L35-L35)

### nexuml.execution.definitions.PyTorchJobExecution.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid', frozen=True, validate_default=True, allow_inf_nan=False, arbitrary_types_allowed=False)
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L25-L31)

### nexuml.execution.definitions.PyTorchJobExecution.cancel

*method*

```python
def cancel(reference: Any) -> dict[str, Any]
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L267-L270)

### nexuml.execution.definitions.PyTorchJobExecution.discover

*method*

```python
def discover(**kwargs: Any) -> Any
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L257-L260)

### nexuml.execution.definitions.PyTorchJobExecution.inspect

*method*

```python
def inspect(reference: Any) -> dict[str, Any]
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L262-L265)

### nexuml.execution.definitions.PyTorchJobExecution.preflight

*method*

```python
def preflight(scenario: Any, **kwargs: Any) -> dict[str, Any]
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L245-L249)

### nexuml.execution.definitions.PyTorchJobExecution.run

*method*

```python
def run(scenario: Any, **kwargs: Any) -> Any
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L251-L255)

## nexuml.execution.definitions.RayClusterExecution

*class* · *pydantic model*

```python
class RayClusterExecution(ExecutionBackendDefinition)
```

Bases: `nexuml.core.components.ExecutionBackendDefinition`

Ray placement configuration; training semantics stay in TrainingSpec.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L91-L179)

### nexuml.execution.definitions.RayClusterExecution.capabilities

*attribute* · *pydantic field*

```python
capabilities = {'status': True, 'logs': True, 'metrics': True, 'cancellation': False, 'resume': False, 'artifacts': False, 'process_ownership': False}
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L102-L110)

### nexuml.execution.definitions.RayClusterExecution.dependencies

*attribute* · *pydantic field*

```python
dependencies = ('ray')
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L96-L96)

### nexuml.execution.definitions.RayClusterExecution.label

*attribute* · *pydantic field*

```python
label = 'RayCluster'
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L95-L95)

### nexuml.execution.definitions.RayClusterExecution.presentation

*attribute* · *pydantic field*

```python
presentation = {'target': ['target'], 'resources': ['workers', 'resources_per_worker'], 'expert': ['storage_path']}
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L97-L101)

### nexuml.execution.definitions.RayClusterExecution.resources_per_worker

*attribute* · *pydantic field*

```python
resources_per_worker: dict[str, float]
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L113-L113)

### nexuml.execution.definitions.RayClusterExecution.storage_path

*attribute* · *pydantic field*

```python
storage_path: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L114-L114)

### nexuml.execution.definitions.RayClusterExecution.target

*attribute* · *pydantic field*

```python
target: RayClusterTarget
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L111-L111)

### nexuml.execution.definitions.RayClusterExecution.workers

*attribute* · *pydantic field*

```python
workers: int | tuple[int, int] = 1
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L112-L112)

### nexuml.execution.definitions.RayClusterExecution.kind

*attribute* · *pydantic field*

```python
kind = 'execution_backend'
```

Inherited from: `nexuml.core.components.ExecutionBackendDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L169-L169)

### nexuml.execution.definitions.RayClusterExecution.component_name

*attribute* · *class attribute*

```python
component_name: str
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L34-L34)

### nexuml.execution.definitions.RayClusterExecution.component_version

*attribute* · *class attribute*

```python
component_version: str = '1'
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L35-L35)

### nexuml.execution.definitions.RayClusterExecution.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid', frozen=True, validate_default=True, allow_inf_nan=False, arbitrary_types_allowed=False)
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L25-L31)

### nexuml.execution.definitions.RayClusterExecution.discover

*method*

```python
def discover(**kwargs: Any) -> Any
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L176-L179)

### nexuml.execution.definitions.RayClusterExecution.preflight

*method*

```python
def preflight(scenario: Any, **kwargs: Any) -> dict[str, Any]
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L146-L169)

### nexuml.execution.definitions.RayClusterExecution.run

*method*

```python
def run(scenario: Any, **kwargs: Any) -> Any
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L171-L174)

### nexuml.execution.definitions.RayClusterExecution.validate_resources

*method* · *pydantic validator*

```python
def validate_resources(value: dict[str, float]) -> dict[str, float]
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L139-L144)

### nexuml.execution.definitions.RayClusterExecution.validate_storage

*method* · *pydantic validator*

```python
def validate_storage(value: str | None) -> str | None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L116-L125)

### nexuml.execution.definitions.RayClusterExecution.validate_workers

*method* · *pydantic validator*

```python
def validate_workers(value: int | tuple[int, int]) -> int | tuple[int, int]
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L127-L137)

### nexuml.execution.definitions.RayClusterExecution.cancel

*method*

```python
def cancel(reference: Any) -> dict[str, Any]
```

Inherited from: `nexuml.core.components.ExecutionBackendDefinition`

Refuse to confuse submitter termination with native cancellation.

**Raises**

- `NotImplementedError` — When native cancellation cannot be confirmed.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L208-L214)

### nexuml.execution.definitions.RayClusterExecution.inspect

*method*

```python
def inspect(reference: Any) -> dict[str, Any]
```

Inherited from: `nexuml.core.components.ExecutionBackendDefinition`

Refuse native observation that the backend has not implemented.

**Raises**

- `NotImplementedError` — When no native inspection is available.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L200-L206)

## nexuml.execution.definitions.RayClusterTarget

*class* · *pydantic model*

```python
class RayClusterTarget(BaseModel)
```

Bases: `BaseModel`

An explicit existing Ray cluster and its native runtime environment.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L51-L88)

### nexuml.execution.definitions.RayClusterTarget.address

*attribute* · *pydantic field*

```python
address: str = 'auto'
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L55-L55)

### nexuml.execution.definitions.RayClusterTarget.context

*attribute* · *pydantic field*

```python
context: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L59-L59)

### nexuml.execution.definitions.RayClusterTarget.dashboard_address

*attribute* · *pydantic field*

```python
dashboard_address: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L56-L56)

### nexuml.execution.definitions.RayClusterTarget.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid', frozen=True)
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L54-L54)

### nexuml.execution.definitions.RayClusterTarget.namespace

*attribute* · *pydantic field*

```python
namespace: str = 'default'
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L60-L60)

### nexuml.execution.definitions.RayClusterTarget.py_executable

*attribute* · *pydantic field*

```python
py_executable: str | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L58-L58)

### nexuml.execution.definitions.RayClusterTarget.working_dir

*attribute* · *pydantic field*

```python
working_dir: str | None = '.'
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L57-L57)

### nexuml.execution.definitions.RayClusterTarget.validate_address

*method* · *pydantic validator*

```python
def validate_address(value: str | None) -> str | None
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L62-L88)

## nexuml.execution.definitions.RayJobExecution

*class* · *pydantic model*

```python
class RayJobExecution(KubernetesExecution)
```

Bases: `nexuml.execution.definitions.KubernetesExecution`

Submit a KubeRay RayJob; its template owns the cluster and cleanup policy.

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L273-L284)

### nexuml.execution.definitions.RayJobExecution.api_version

*attribute* · *pydantic field*

```python
api_version = 'ray.io/v1'
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L278-L278)

### nexuml.execution.definitions.RayJobExecution.capabilities

*attribute* · *pydantic field*

```python
capabilities = {: KubernetesExecution.capabilities, 'cancellation': False, 'logs': False}
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L281-L281)

### nexuml.execution.definitions.RayJobExecution.label

*attribute* · *pydantic field*

```python
label = 'RayJob'
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L277-L277)

### nexuml.execution.definitions.RayJobExecution.plural

*attribute* · *pydantic field*

```python
plural = 'rayjobs'
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L280-L280)

### nexuml.execution.definitions.RayJobExecution.ray

*attribute* · *pydantic field*

```python
ray: RayClusterExecution
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L282-L284)

### nexuml.execution.definitions.RayJobExecution.resource_kind

*attribute* · *pydantic field*

```python
resource_kind = 'RayJob'
```

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L279-L279)

### nexuml.execution.definitions.RayJobExecution.container

*attribute* · *pydantic field*

```python
container: str | None = None
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L240-L240)

### nexuml.execution.definitions.RayJobExecution.context

*attribute* · *pydantic field*

```python
context: str | None = None
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L234-L234)

### nexuml.execution.definitions.RayJobExecution.dependencies

*attribute* · *pydantic field*

```python
dependencies = ('kubernetes')
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L214-L214)

### nexuml.execution.definitions.RayJobExecution.handoff

*attribute* · *pydantic field*

```python
handoff: ConfigHandoff | None = None
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L239-L239)

### nexuml.execution.definitions.RayJobExecution.namespace

*attribute* · *pydantic field*

```python
namespace: str = 'default'
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L235-L237)

### nexuml.execution.definitions.RayJobExecution.presentation

*attribute* · *pydantic field*

```python
presentation = {'target': ['context', 'namespace'], 'setup': ['template', 'handoff'], 'resources': ['worker_group', 'container', 'replicas', 'resources'], 'expert': ['ray']}
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L215-L220)

### nexuml.execution.definitions.RayJobExecution.replicas

*attribute* · *pydantic field*

```python
replicas: int | None = None
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L242-L242)

### nexuml.execution.definitions.RayJobExecution.resources

*attribute* · *pydantic field*

```python
resources: dict[str, str]
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L243-L243)

### nexuml.execution.definitions.RayJobExecution.template

*attribute* · *pydantic field*

```python
template: str | None = None
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L238-L238)

### nexuml.execution.definitions.RayJobExecution.worker_group

*attribute* · *pydantic field*

```python
worker_group: str | None = None
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L241-L241)

### nexuml.execution.definitions.RayJobExecution.kind

*attribute* · *pydantic field*

```python
kind = 'execution_backend'
```

Inherited from: `nexuml.core.components.ExecutionBackendDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L169-L169)

### nexuml.execution.definitions.RayJobExecution.component_name

*attribute* · *class attribute*

```python
component_name: str
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L34-L34)

### nexuml.execution.definitions.RayJobExecution.component_version

*attribute* · *class attribute*

```python
component_version: str = '1'
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L35-L35)

### nexuml.execution.definitions.RayJobExecution.model_config

*attribute* · *class attribute* · *instance attribute*

```python
model_config = ConfigDict(extra='forbid', frozen=True, validate_default=True, allow_inf_nan=False, arbitrary_types_allowed=False)
```

Inherited from: `nexuml.core.components.ComponentDefinition`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/core/components.py#L25-L31)

### nexuml.execution.definitions.RayJobExecution.cancel

*method*

```python
def cancel(reference: Any) -> dict[str, Any]
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L267-L270)

### nexuml.execution.definitions.RayJobExecution.discover

*method*

```python
def discover(**kwargs: Any) -> Any
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L257-L260)

### nexuml.execution.definitions.RayJobExecution.inspect

*method*

```python
def inspect(reference: Any) -> dict[str, Any]
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L262-L265)

### nexuml.execution.definitions.RayJobExecution.preflight

*method*

```python
def preflight(scenario: Any, **kwargs: Any) -> dict[str, Any]
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L245-L249)

### nexuml.execution.definitions.RayJobExecution.run

*method*

```python
def run(scenario: Any, **kwargs: Any) -> Any
```

Inherited from: `nexuml.execution.definitions.KubernetesExecution`

[View source](https://github.com/NexuFed/NexuML/blob/0ac41f9fdf4b1f0bdb4182261c550f9fa6b08dba/src/nexuml/execution/definitions.py#L251-L255)
