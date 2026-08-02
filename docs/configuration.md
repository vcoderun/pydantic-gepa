# Typed GEPA Configuration

`GEPAConfig` is the supported configuration path for the standard GEPA backend.
It separates reflection, selection, merging, budgets, run behavior, tracking,
progress, and evaluation-set policy so unknown or conflicting options fail
before an optimization starts.

```python
from pydantic_gepa import GEPAConfig
from pydantic_gepa.configuration import (
    BudgetConfig,
    ProgressConfig,
    ReflectionConfig,
    SelectionConfig,
)

config = GEPAConfig(
    reflection=ReflectionConfig(
        model="openai:gpt-5-mini",
        model_kwargs={"temperature": 0.2},
        minibatch_size=4,
    ),
    selection=SelectionConfig(
        candidate="pareto",
        frontier="hybrid",
        component="round_robin",
    ),
    budget=BudgetConfig(max_metric_calls=100),
    progress=ProgressConfig(display_bar=True),
)

result = optimization.optimize(config=config)
```

## Callable Reflection

`CallableReflectionModel` adapts ordinary synchronous or asynchronous
callables to GEPA's synchronous language-model contract. It records requests,
token estimates or supplied usage, cost, retries, durations, and normalized
errors.

```python
from pydantic_gepa import CallableReflectionModel

reflection = CallableReflectionModel(
    lambda prompt: "A more precise candidate",
    retries=2,
)
```

## Pydantic AI Reflection

The optional integration accepts an existing string-output agent or constructs
one from a Pydantic AI model. Importing the generic configuration and runtime
modules does not import Pydantic AI.

```python
from pydantic_gepa.integrations.pydantic_ai import PydanticAIReflectionModel

reflection = PydanticAIReflectionModel.from_model(
    "openai:gpt-5-mini",
    max_output_tokens=2_000,
    timeout=30,
    retries=2,
)
```

## Legacy Options

`GEPAConfig.from_legacy_kwargs()` exists only as a migration path. It emits a
deprecation warning, maps every known standalone GEPA option, and rejects
unknown keys rather than forwarding an untyped option bag.
