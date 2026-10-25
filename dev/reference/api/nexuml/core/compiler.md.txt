# nexuml.core.compiler

*module*

Compiler: transforms ScenarioSpec into a runnable CompiledPipeline.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/compiler.py#L1-L1)

## nexuml.core.compiler.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger(__name__)
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/compiler.py#L18-L18)

## nexuml.core.compiler.compile

*function*

```python
def compile(scenario: ScenarioSpec) -> CompiledPipeline
```

Compile a ScenarioSpec into a runnable CompiledPipeline.

> **Steps**
>
> 1. Iterate pipeline stages in order
> 2. For each LayerSpec: resolve meta_in, materialize its definition, capture meta_out
> 3. Run dummy forward for shape propagation
> 4. Return assembled CompiledPipeline

**Returns**

- (`CompiledPipeline`) — Compiled pipeline ready for training or inference.

**Raises**

- `TypeError` — If a definition does not build a ``PipelineLayer``.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/core/compiler.py#L21-L130)
