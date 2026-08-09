# Optimize Anything

The Optimize Anything backend is experimental and isolated under:

```python
from pydantic_gepa.experimental.optimize_anything import (
    PydanticOptimizeAnythingAdapter,
    PydanticOptimizeAnythingOptimizer,
)
```

It does not replace the standard GEPA backend.

## Common API selection

```python
pipeline = Optimization.from_examples(
    examples=train,
    val_examples=validation,
    task=task,
    score=score,
    components=components,
    injections=injections,
    backend="optimize_anything",
    optimization_objective="Maximize support-routing accuracy.",
    background="The candidate contains agent instructions and tool descriptions.",
)

result = pipeline.run(config=config)
```

The same examples, candidate model, injection system, evaluation harness, and
result type are retained. Only the backend adapter and optimization objective
change.

## Low-level adapter

Wrap a standard `PydanticGEPAAdapter` in
`PydanticOptimizeAnythingAdapter`, then construct
`PydanticOptimizeAnythingOptimizer`. This is useful for direct control over the
experimental engine but is not required for common use.

## Configuration

Use `GEPAConfig`. Unsupported extra keyword options are rejected by the
Optimize Anything backend rather than forwarded blindly. Objective and
background may be set on pipeline construction or overridden for a run.

## Stability boundary

The upstream Optimize Anything API is less stable than standard GEPA. Keeping
the backend in `experimental` means:

- imports explicitly acknowledge instability;
- standard API behavior can remain compatible;
- backend-specific changes do not redefine common candidates or results;
- applications can test both backends without duplicating evaluation code.

Do not persist experimental backend implementation objects. Persist normalized
candidates and `stable_dump()` results.

See [`examples/experimental_optimize_anything.py`](https://github.com/vcoderun/pydantic-gepa/blob/main/examples/experimental_optimize_anything.py).
