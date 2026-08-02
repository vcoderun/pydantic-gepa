# Advanced Adapters

Most applications do not implement an adapter. Advanced integrations can use:

- `pydantic_gepa.evaluation` for typed contexts, evaluators, encoders, and
  cache stores;
- `pydantic_gepa.runtime` for candidate scope and factory isolation;
- `pydantic_gepa.state` for custom durable stores;
- `pydantic_gepa.events` for typed lifecycle events;
- `pydantic_gepa.integrations.pydantic_ai` for reflection-model adaptation;
- `pydantic_gepa.experimental.optimize_anything` for the unstable backend.

The low-level `PydanticGEPAAdapter` and Pydantic Evals harness remain available
for migrations. New application code should begin with `Example`, callables,
components, and typed configuration.

## Experimental Optimize Anything

The unstable backend lives only under
`pydantic_gepa.experimental.optimize_anything`. It consumes the same
`GEPAConfig`, reflection model, evaluation adapter, candidate codecs, run
store, and normalized result model as the standard backend.

```python
from pydantic_gepa import CallableReflectionModel
from pydantic_gepa.configuration import BudgetConfig, GEPAConfig, ReflectionConfig
from pydantic_gepa.experimental.optimize_anything import (
    PydanticOptimizeAnythingAdapter,
    PydanticOptimizeAnythingOptimizer,
)

optimizer = PydanticOptimizeAnythingOptimizer(
    adapter=PydanticOptimizeAnythingAdapter(adapter=standard_adapter),
    initial_candidate=initial_candidate,
    optimization_objective="Increase routing accuracy.",
)

result = optimizer.optimize(
    trainset=train_cases,
    valset=validation_cases,
    config=GEPAConfig(
        budget=BudgetConfig(max_metric_calls=50),
        reflection=ReflectionConfig(model=CallableReflectionModel(reflect)),
    ),
)
```

Backend-specific option bags are intentionally rejected. Selection,
reflection, merge, tracking, progress, cache, run, and budget behavior are
configured through the shared typed configuration. Raw text proposals from
Optimize Anything are normalized through each component's codec before
application and before the result is returned.
