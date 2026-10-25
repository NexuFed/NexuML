# nexuml.api.errors

*module*

Actionable HTTP errors without leaking request inputs.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/errors.py#L1-L1)

## nexuml.api.errors.ApiError

*class*

```python
class ApiError(Exception)
```

Bases: `Exception`

Transport failure; existing NexuML errors remain authoritative.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/errors.py#L4-L11)

### nexuml.api.errors.ApiError.code

*attribute* · *instance attribute*

```python
code = code
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/errors.py#L9-L9)

### nexuml.api.errors.ApiError.fields

*attribute* · *instance attribute*

```python
fields = fields or []
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/errors.py#L11-L11)

### nexuml.api.errors.ApiError.message

*attribute* · *instance attribute*

```python
message = message
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/errors.py#L10-L10)

### nexuml.api.errors.ApiError.status

*attribute* · *instance attribute*

```python
status = status
```

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/errors.py#L8-L8)
